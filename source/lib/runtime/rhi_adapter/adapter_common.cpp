#include "adapter_common.h"

#include "adapter_internal.h"
#include "adapter_test_hooks.h"
#include "rhi/rhi_internal.h"

#include <algorithm>
#include <cstdio>
#include <new>
#include <optional>
#include <utility>

namespace vernon::runtime::rhi_adapter {

void setBackendError(std::string &error, const char *message) noexcept {
    try {
        error = message;
    } catch (...) {
        error.clear();
    }
}

vernon::ProviderError providerError(vernon::RhiError error) noexcept {
    using vernon::ProviderErrorCode;
    using vernon::RhiErrorCode;
    switch (error.code) {
    case RhiErrorCode::InvalidArgument:
        return {ProviderErrorCode::InvalidArgument, error.context};
    case RhiErrorCode::Unsupported:
        return {ProviderErrorCode::Unsupported, error.context};
    case RhiErrorCode::ResourceExhausted:
        return {ProviderErrorCode::ResourceExhausted, error.context};
    case RhiErrorCode::LifecycleFailure:
        return {ProviderErrorCode::LifecycleFailure, error.context};
    case RhiErrorCode::BackendFailure:
    case RhiErrorCode::SynchronizationFailure:
        return {ProviderErrorCode::BackendFailure, error.context};
    }
    return {ProviderErrorCode::BackendFailure, error.context};
}

void recordProviderError(VernonRuntimeRhiAdapter &adapter, vernon::ProviderError error,
                         const char *diagnostic) noexcept {
    char rendered[384]{};
    if (error.context.operation) {
        std::snprintf(rendered, sizeof(rendered), "%s: %s", diagnostic, error.context.operation);
        setBackendError(adapter.error, rendered);
        return;
    }
    setBackendError(adapter.error, diagnostic);
}

VernonStatus providerStatus(VernonRuntimeRhiAdapter &adapter, RhiAdapterResult<void> result,
                            const char *diagnostic) noexcept {
    adapter.error.clear();
    if (result)
        return VERNON_STATUS_OK;
    const vernon::ProviderError error = std::move(result).error();
    recordProviderError(adapter, error, diagnostic);
    return vernon::toVernonStatus(error);
}

VernonStatus providerStatus(VernonRuntimeRhiAdapter &adapter, vernon::ProviderError error,
                            const char *diagnostic) noexcept {
    adapter.error.clear();
    recordProviderError(adapter, error, diagnostic);
    return vernon::toVernonStatus(error);
}

} // namespace vernon::runtime::rhi_adapter

using namespace vernon::runtime::rhi_adapter;

RhiAdapterBackendStorage::~RhiAdapterBackendStorage() {
    if (state)
        ops->destroy(state);
}

RhiAdapterBackendStorage::RhiAdapterBackendStorage(RhiAdapterBackendStorage &&other) noexcept
    : state(std::exchange(other.state, nullptr)), ops(std::exchange(other.ops, nullptr)) {}

RhiAdapterBackendStorage &RhiAdapterBackendStorage::operator=(RhiAdapterBackendStorage &&other) noexcept {
    if (this == &other)
        return *this;
    if (state)
        ops->destroy(state);
    state = std::exchange(other.state, nullptr);
    ops = std::exchange(other.ops, nullptr);
    return *this;
}

RhiAdapterResult<void> RhiAdapterBackendStorage::adopt(void *newState, const RhiAdapterBackendOps *newOps) noexcept {
    if (!newState || !newOps || state || ops)
        return RhiAdapterResult<void>{vernon::err(
            vernon::ProviderError{vernon::ProviderErrorCode::InvalidArgument, {"rhi_adapter_backend_adopt", 0, 0}})};
    state = newState;
    ops = newOps;
    return RhiAdapterResult<void>{vernon::ok()};
}

VernonRuntimeRhiAdapter::~VernonRuntimeRhiAdapter() {
    std::lock_guard<std::mutex> guard(retainedLeaseMutex);
    for (RetainedLeaseEntry *entry = retainedLeaseHead.get(); entry; entry = entry->next.get()) {
        auto released = entry->lease.release();
        (void)released;
    }
    retainedLeaseHead.reset();
}

extern "C" VernonRuntimeRhiAdapter *vernonRuntimeRhiAdapterCreateForDevice(VernonRhiDevice device,
                                                                           VernonRhiBackend backend) {
    auto deviceLease = vernon::rhi::retainDeviceLease(device);
    if (deviceLease.isErr())
        return nullptr;
    VernonRuntimeRhiAdapter *adapter = nullptr;
    switch (backend) {
#if defined(VERNON_HAS_CUDA_RHI)
    case VERNON_RHI_BACKEND_CUDA:
        adapter = vernon::runtime::createCudaRhiAdapter(device, backend);
        break;
#endif
#if defined(VERNON_HAS_VULKAN_RHI)
    case VERNON_RHI_BACKEND_VULKAN:
        adapter = vernon::runtime::createVulkanRhiAdapter(device, backend);
        break;
#endif
#if defined(VERNON_HAS_DIRECTX12_RHI)
    case VERNON_RHI_BACKEND_DIRECTX12:
        adapter = vernon::runtime::createDirectX12RhiAdapter(device, backend);
        break;
#endif
#if defined(VERNON_HAS_METAL_RHI)
    case VERNON_RHI_BACKEND_METAL:
        adapter = vernon::runtime::createMetalRhiAdapter(device, backend);
        break;
#endif
    case VERNON_RHI_BACKEND_OPENGL:
    case VERNON_RHI_BACKEND_OPENGL_ES:
        adapter = vernon::runtime::createOpenGLRhiAdapter(device, backend);
        break;
    default:
        break;
    }
    if (adapter) {
        adapter->rhiDevice = device;
        adapter->rhiBackend = backend;
        adapter->rhiDeviceLease.emplace(std::move(deviceLease).value());
    }
    return adapter;
}

extern "C" void vernonRuntimeRhiAdapterDestroy(VernonRuntimeRhiAdapter *adapter) { delete adapter; }

extern "C" const VernonRuntimeDeviceProvider *vernonRuntimeRhiAdapterGetProvider(VernonRuntimeRhiAdapter *adapter) {
    return adapter ? &adapter->provider : nullptr;
}

extern "C" VernonStatus vernonRuntimeRhiAdapterSynchronize(VernonRuntimeRhiAdapter *adapter) {
    if (!adapter || !adapter->backend.state || !adapter->backend.ops)
        return VERNON_STATUS_INVALID_ARGUMENT;
    adapter->error.clear();
    auto synchronized = adapter->backend.ops->synchronize(adapter->backend.state, adapter->error);
    if (synchronized)
        return VERNON_STATUS_OK;
    const vernon::ProviderError error = std::move(synchronized).error();
    if (adapter->error.empty())
        recordProviderError(*adapter, error, "RHI adapter synchronization failed");
    return vernon::toVernonStatus(error);
}

extern "C" VernonStringView vernonRuntimeRhiAdapterGetLastError(const VernonRuntimeRhiAdapter *adapter) {
    return adapter ? VernonStringView{adapter->error.data(), adapter->error.size()} : VernonStringView{};
}

namespace {

uint64_t resourceIdentity(const VernonRuntimeRhiAdapter &adapter, uint64_t resourceKind) {
    if (!adapter.backend.state || !adapter.backend.ops)
        return 0;
    return adapter.backend.ops->resourceIdentity(adapter.backend.state) | resourceKind;
}

} // namespace

namespace vernon::runtime::rhi_adapter {

namespace {

std::optional<VernonTextureFormat> providerTextureFormat(VernonRhiFormat format) {
    switch (format) {
    case VERNON_RHI_FORMAT_R8_UNORM:
        return VERNON_TEXTURE_R8_UNORM;
    case VERNON_RHI_FORMAT_RG8_UNORM:
        return VERNON_TEXTURE_RG8_UNORM;
    case VERNON_RHI_FORMAT_RGB8_UNORM:
        return VERNON_TEXTURE_RGB8_UNORM;
    case VERNON_RHI_FORMAT_RGBA8_UNORM:
        return VERNON_TEXTURE_RGBA8_UNORM;
    case VERNON_RHI_FORMAT_RGBA8_SRGB:
        return VERNON_TEXTURE_RGBA8_SRGB;
    case VERNON_RHI_FORMAT_R16_FLOAT:
        return VERNON_TEXTURE_R16_FLOAT;
    case VERNON_RHI_FORMAT_RGBA16_FLOAT:
        return VERNON_TEXTURE_RGBA16_FLOAT;
    case VERNON_RHI_FORMAT_R32_FLOAT:
        return VERNON_TEXTURE_R32_FLOAT;
    case VERNON_RHI_FORMAT_RGBA32_FLOAT:
        return VERNON_TEXTURE_RGBA32_FLOAT;
    case VERNON_RHI_FORMAT_R11G11B10_FLOAT:
        return VERNON_TEXTURE_R11G11B10_FLOAT;
    case VERNON_RHI_FORMAT_D32_FLOAT:
        return VERNON_TEXTURE_D32_FLOAT;
    case VERNON_RHI_FORMAT_D32_FLOAT_S8_UINT:
        return VERNON_TEXTURE_D32_FLOAT_S8_UINT;
    default:
        return std::nullopt;
    }
}

} // namespace

RhiAdapterResult<vernon::rhi::ResourceKind> resourceKind(const VernonRuntimeRhiAdapter &adapter,
                                                         VernonRuntimeProviderResourceReference resource) noexcept {
    const uint64_t encodedKind = resource.identity & kRhiResourceKindMask;
    if (resource.identity != resourceIdentity(adapter, encodedKind) || resource.resource.value == 0)
        return RhiAdapterResult<vernon::rhi::ResourceKind>{vernon::err(vernon::ProviderError{
            vernon::ProviderErrorCode::InvalidArgument, {"rhi_adapter_resource_kind", resource.resource.value, 0}})};
    if (encodedKind == kRhiBufferResource)
        return RhiAdapterResult<vernon::rhi::ResourceKind>{vernon::ok(vernon::rhi::ResourceKind::Buffer)};
    if (encodedKind == kRhiImageResource)
        return RhiAdapterResult<vernon::rhi::ResourceKind>{vernon::ok(vernon::rhi::ResourceKind::Image)};
    if (encodedKind == kRhiImageViewResource)
        return RhiAdapterResult<vernon::rhi::ResourceKind>{vernon::ok(vernon::rhi::ResourceKind::ImageView)};
    if (encodedKind == kRhiSamplerResource)
        return RhiAdapterResult<vernon::rhi::ResourceKind>{vernon::ok(vernon::rhi::ResourceKind::Sampler)};
    return RhiAdapterResult<vernon::rhi::ResourceKind>{vernon::err(vernon::ProviderError{
        vernon::ProviderErrorCode::InvalidArgument, {"rhi_adapter_resource_kind", resource.resource.value, 0}})};
}

RhiAdapterResult<void> retainRhiResource(VernonRuntimeRhiAdapter &adapter,
                                         VernonRuntimeProviderResourceReference resource) noexcept {
    auto kindResult = resourceKind(adapter, resource);
    if (!kindResult)
        return RhiAdapterResult<void>{vernon::err(std::move(kindResult).error())};
    const auto kind = std::move(kindResult).value();
    auto retained = vernon::rhi::retainResource(adapter.rhiDevice, kind, resource.resource.value);
    if (!retained)
        return RhiAdapterResult<void>{vernon::err(providerError(std::move(retained).error()))};
    auto lease = std::move(retained).value();
    auto entry = std::unique_ptr<VernonRuntimeRhiAdapter::RetainedLeaseEntry>(new (
        std::nothrow) VernonRuntimeRhiAdapter::RetainedLeaseEntry(kind, resource.resource.value, std::move(lease)));
    if (!entry) {
        auto released = lease.release();
        (void)released;
        return RhiAdapterResult<void>{
            vernon::err(vernon::ProviderError{vernon::ProviderErrorCode::ResourceExhausted,
                                              {"store_retained_rhi_resource", resource.resource.value, 0}})};
    }
    std::lock_guard<std::mutex> guard(adapter.retainedLeaseMutex);
    entry->next = std::move(adapter.retainedLeaseHead);
    adapter.retainedLeaseHead = std::move(entry);
    return RhiAdapterResult<void>{vernon::ok()};
}

RhiAdapterResult<void> releaseRetainedRhiResource(VernonRuntimeRhiAdapter &adapter,
                                                  VernonRuntimeProviderResourceReference resource) noexcept {
    auto kindResult = resourceKind(adapter, resource);
    if (!kindResult)
        return RhiAdapterResult<void>{vernon::err(std::move(kindResult).error())};
    const auto kind = std::move(kindResult).value();
    std::lock_guard<std::mutex> guard(adapter.retainedLeaseMutex);
    auto *entry = &adapter.retainedLeaseHead;
    while (*entry && ((*entry)->kind != kind || (*entry)->key != resource.resource.value || !(*entry)->lease.active()))
        entry = &(*entry)->next;
    if (!*entry)
        return RhiAdapterResult<void>{
            vernon::err(vernon::ProviderError{vernon::ProviderErrorCode::LifecycleFailure,
                                              {"release_retained_rhi_resource", resource.resource.value, 0}})};
    auto released = (*entry)->lease.release();
    if (!released)
        return RhiAdapterResult<void>{vernon::err(providerError(std::move(released).error()))};
    auto removed = std::move(*entry);
    *entry = std::move(removed->next);
    return RhiAdapterResult<void>{vernon::ok()};
}

RhiAdapterResult<uint64_t> resolveRhiResource(VernonRuntimeRhiAdapter &adapter,
                                              VernonRuntimeProviderResourceReference resource) noexcept {
    auto kindResult = resourceKind(adapter, resource);
    if (!kindResult)
        return RhiAdapterResult<uint64_t>{vernon::err(std::move(kindResult).error())};
    const auto kind = std::move(kindResult).value();
    {
        std::lock_guard<std::mutex> guard(adapter.retainedLeaseMutex);
        for (auto *entry = adapter.retainedLeaseHead.get(); entry; entry = entry->next.get()) {
            if (entry->kind != kind || entry->key != resource.resource.value || !entry->lease.active())
                continue;
            auto resolved =
                vernon::rhi::resolveResource(adapter.rhiDevice, kind, resource.resource.value, entry->lease);
            if (!resolved)
                return RhiAdapterResult<uint64_t>{vernon::err(providerError(std::move(resolved).error()))};
            return RhiAdapterResult<uint64_t>{vernon::ok(std::move(resolved).value())};
        }
    }
    auto retained = vernon::rhi::retainResource(adapter.rhiDevice, kind, resource.resource.value);
    if (!retained)
        return RhiAdapterResult<uint64_t>{vernon::err(providerError(std::move(retained).error()))};
    auto lease = std::move(retained).value();
    auto resolved = vernon::rhi::resolveResource(adapter.rhiDevice, kind, resource.resource.value, lease);
    auto released = lease.release();
    if (!resolved)
        return RhiAdapterResult<uint64_t>{vernon::err(providerError(std::move(resolved).error()))};
    if (!released)
        return RhiAdapterResult<uint64_t>{vernon::err(providerError(std::move(released).error()))};
    return RhiAdapterResult<uint64_t>{vernon::ok(std::move(resolved).value())};
}

RhiAdapterResult<uint64_t> resolveCommandRhiResource(VernonRuntimeRhiAdapter &adapter,
                                                     VernonRuntimeProviderObject encoder,
                                                     VernonRuntimeProviderResourceReference resource) noexcept {
    auto kind = resourceKind(adapter, resource);
    if (!kind)
        return RhiAdapterResult<uint64_t>{vernon::err(std::move(kind).error())};
    auto resolved = vernon::rhi::resolveCommandResource(adapter.rhiDevice, encoder.value, std::move(kind).value(),
                                                        resource.resource.value);
    if (!resolved)
        return RhiAdapterResult<uint64_t>{vernon::err(providerError(std::move(resolved).error()))};
    return RhiAdapterResult<uint64_t>{vernon::ok(std::move(resolved).value())};
}

VernonRuntimeRhiAdapter::RetainedLeaseEntry *findRetainedLease(VernonRuntimeRhiAdapter &adapter,
                                                               vernon::rhi::ResourceKind kind, uint64_t key) noexcept {
    for (auto *entry = adapter.retainedLeaseHead.get(); entry; entry = entry->next.get())
        if (entry->kind == kind && entry->key == key && entry->lease.active())
            return entry;
    return nullptr;
}

template <typename Operation>
RhiAdapterResult<void> withDescriptorLease(VernonRuntimeRhiAdapter &adapter, vernon::rhi::ResourceKind kind,
                                           uint64_t key, Operation &&operation) noexcept {
    {
        std::lock_guard<std::mutex> guard(adapter.retainedLeaseMutex);
        if (auto *retained = findRetainedLease(adapter, kind, key))
            return operation(retained->lease);
    }
    auto retained = vernon::rhi::retainResource(adapter.rhiDevice, kind, key);
    if (!retained)
        return RhiAdapterResult<void>{vernon::err(providerError(std::move(retained).error()))};
    auto lease = std::move(retained).value();
    auto result = operation(lease);
    auto released = lease.release();
    if (!result)
        return result;
    if (!released)
        return RhiAdapterResult<void>{vernon::err(providerError(std::move(released).error()))};
    return RhiAdapterResult<void>{vernon::ok()};
}

RhiAdapterResult<void> describeRhiImage(VernonRuntimeRhiAdapter &adapter,
                                        VernonRuntimeProviderResourceReference resource,
                                        VernonRhiImageDescriptor &descriptor) noexcept {
    auto kindResult = resourceKind(adapter, resource);
    if (!kindResult)
        return RhiAdapterResult<void>{vernon::err(std::move(kindResult).error())};
    const auto kind = std::move(kindResult).value();
    if (kind == vernon::rhi::ResourceKind::Image) {
        return withDescriptorLease(
            adapter, kind, resource.resource.value,
            [&](vernon::rhi::RetainedRhiResourceLease &lease) noexcept -> RhiAdapterResult<void> {
                auto described =
                    vernon::rhi::describeImageResource(adapter.rhiDevice, resource.resource.value, descriptor, lease);
                if (!described)
                    return RhiAdapterResult<void>{vernon::err(providerError(std::move(described).error()))};
                return RhiAdapterResult<void>{vernon::ok()};
            });
    }
    if (kind != vernon::rhi::ResourceKind::ImageView)
        return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
            vernon::ProviderErrorCode::InvalidArgument, {"describe_rhi_image", resource.resource.value, 0}})};
    return withDescriptorLease(
        adapter, kind, resource.resource.value,
        [&](vernon::rhi::RetainedRhiResourceLease &lease) noexcept -> RhiAdapterResult<void> {
            VernonRhiImageViewDescriptor view{};
            uint64_t parentKey = 0;
            auto described = vernon::rhi::describeImageViewResource(adapter.rhiDevice, resource.resource.value, view,
                                                                    descriptor, parentKey, lease);
            if (!described)
                return RhiAdapterResult<void>{vernon::err(providerError(std::move(described).error()))};
            return RhiAdapterResult<void>{vernon::ok()};
        });
}

RhiAdapterResult<void> describeProviderImage(VernonRuntimeRhiAdapter &adapter,
                                             VernonRuntimeProviderResourceReference resource,
                                             VernonRuntimeProviderImageDescription &description) noexcept {
    if (description.struct_size < sizeof(description))
        return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
            vernon::ProviderErrorCode::InvalidArgument, {"describe_provider_image", resource.resource.value, 0}})};
    VernonRhiImageDescriptor source{};
    VernonRhiImageViewDescriptor sourceView{};
    uint64_t parentKey = 0;
    auto kindResult = resourceKind(adapter, resource);
    if (!kindResult)
        return RhiAdapterResult<void>{vernon::err(std::move(kindResult).error())};
    const auto kind = std::move(kindResult).value();
    const bool view = kind == vernon::rhi::ResourceKind::ImageView;
    auto described = withDescriptorLease(
        adapter, kind, resource.resource.value,
        [&](vernon::rhi::RetainedRhiResourceLease &lease) noexcept -> RhiAdapterResult<void> {
            auto result =
                view ? vernon::rhi::describeImageViewResource(adapter.rhiDevice, resource.resource.value, sourceView,
                                                              source, parentKey, lease)
                     : vernon::rhi::describeImageResource(adapter.rhiDevice, resource.resource.value, source, lease);
            if (!result)
                return RhiAdapterResult<void>{vernon::err(providerError(std::move(result).error()))};
            return RhiAdapterResult<void>{vernon::ok()};
        });
    if (!described)
        return described;
    const auto format = providerTextureFormat(source.format);
    if (!format)
        return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
            vernon::ProviderErrorCode::Unsupported, {"describe_provider_image_format", source.format, 0}})};
    const VernonTextureDimension dimension = source.dimension == VERNON_RHI_IMAGE_3D     ? VERNON_TEXTURE_3D
                                             : source.dimension == VERNON_RHI_IMAGE_CUBE ? VERNON_TEXTURE_CUBE
                                                                                         : VERNON_TEXTURE_2D;
    uint32_t usage = 0;
    usage |= (source.usage & VERNON_RHI_IMAGE_SAMPLED) ? VERNON_IMAGE_SAMPLED : 0;
    usage |= (source.usage & VERNON_RHI_IMAGE_STORAGE) ? VERNON_IMAGE_STORAGE : 0;
    usage |= (source.usage & VERNON_RHI_IMAGE_COLOR_ATTACHMENT) ? VERNON_IMAGE_COLOR_ATTACHMENT : 0;
    usage |= (source.usage & VERNON_RHI_IMAGE_DEPTH_STENCIL_ATTACHMENT) ? VERNON_IMAGE_DEPTH_STENCIL_ATTACHMENT : 0;
    usage |= (source.usage & VERNON_RHI_IMAGE_TRANSFER_SOURCE) ? VERNON_IMAGE_TRANSFER_SOURCE : 0;
    usage |= (source.usage & VERNON_RHI_IMAGE_TRANSFER_DESTINATION) ? VERNON_IMAGE_TRANSFER_DESTINATION : 0;
    const uint32_t aspects = *format == VERNON_TEXTURE_D32_FLOAT ? VERNON_IMAGE_ASPECT_DEPTH
                             : *format == VERNON_TEXTURE_D32_FLOAT_S8_UINT
                                 ? VERNON_IMAGE_ASPECT_DEPTH | VERNON_IMAGE_ASPECT_STENCIL
                                 : VERNON_IMAGE_ASPECT_COLOR;
    description.image = {dimension,
                         {source.width, source.height, source.depth},
                         *format,
                         source.mip_levels,
                         source.array_layers,
                         source.sample_count,
                         usage};
    const VernonTextureDimension viewDimension = !view                                         ? dimension
                                                 : sourceView.dimension == VERNON_RHI_IMAGE_2D ? VERNON_TEXTURE_2D
                                                 : sourceView.dimension == VERNON_RHI_IMAGE_3D ? VERNON_TEXTURE_3D
                                                                                               : VERNON_TEXTURE_CUBE;
    const auto viewFormat = view ? providerTextureFormat(sourceView.format) : format;
    if (!viewFormat)
        return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
            vernon::ProviderErrorCode::Unsupported, {"describe_provider_image_view_format", sourceView.format, 0}})};
    const uint32_t viewAspects =
        !view ? aspects
              : ((sourceView.aspects & VERNON_RHI_IMAGE_ASPECT_COLOR) ? VERNON_IMAGE_ASPECT_COLOR : 0) |
                    ((sourceView.aspects & VERNON_RHI_IMAGE_ASPECT_DEPTH) ? VERNON_IMAGE_ASPECT_DEPTH : 0) |
                    ((sourceView.aspects & VERNON_RHI_IMAGE_ASPECT_STENCIL) ? VERNON_IMAGE_ASPECT_STENCIL : 0);
    description.view = {viewDimension,
                        *viewFormat,
                        {view ? sourceView.base_mip_level : 0, view ? sourceView.mip_level_count : source.mip_levels,
                         view ? sourceView.base_array_layer : 0,
                         view ? sourceView.array_layer_count : source.array_layers, viewAspects}};
    description.parent_identity = parentKey ? parentKey : resource.resource.value;
    description.resource_kind = view ? VERNON_RUNTIME_PROVIDER_IMAGE_VIEW : VERNON_RUNTIME_PROVIDER_IMAGE_OWNER;
    return RhiAdapterResult<void>{vernon::ok()};
}

VernonStatus retainRhiResourceCallback(void *data, VernonRuntimeProviderResourceReference resource) {
    if (!data)
        return VERNON_STATUS_INVALID_ARGUMENT;
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    return providerStatus(adapter, retainRhiResource(adapter, resource),
                          "RHI adapter could not retain the provider resource");
}

void releaseRhiResourceCallback(void *data, VernonRuntimeProviderResourceReference resource) {
    if (!data)
        return;
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    auto released = releaseRetainedRhiResource(adapter, resource);
    if (!released)
        recordProviderError(adapter, std::move(released).error(),
                            "RHI adapter release callback found no releasable retained resource lease");
}

VernonStatus describeProviderImageCallback(void *data, VernonRuntimeProviderResourceReference resource,
                                           VernonRuntimeProviderImageDescription *description) {
    if (!data || !description)
        return VERNON_STATUS_INVALID_ARGUMENT;
    auto &adapter = *static_cast<VernonRuntimeRhiAdapter *>(data);
    return providerStatus(adapter, describeProviderImage(adapter, resource, *description),
                          "RHI adapter could not describe the provider image");
}

const VernonRuntimeProviderResourceReference *providerBindingResource(const VernonRuntimeProviderBindingValue &value) {
    switch (value.kind) {
    case VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER:
    case VERNON_RUNTIME_PROVIDER_VERTEX_BUFFER:
        return &value.payload.buffer.resource;
    case VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE:
    case VERNON_RUNTIME_PROVIDER_STORAGE_IMAGE:
        return &value.payload.image.view;
    case VERNON_RUNTIME_PROVIDER_SAMPLER:
        return &value.payload.sampler.resource;
    case VERNON_RUNTIME_PROVIDER_INLINE_VALUE:
    case VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER:
        return nullptr;
    }
    return nullptr;
}

RhiAdapterResult<uint64_t> nativeCommandEncoder(VernonRuntimeRhiAdapter &adapter,
                                                VernonRuntimeProviderObject encoder) noexcept {
    const uint64_t native = vernon::rhi::commandEncoderNative(adapter.rhiDevice, encoder.value, adapter.rhiBackend);
    if (!native)
        return RhiAdapterResult<uint64_t>{vernon::err(vernon::ProviderError{
            vernon::ProviderErrorCode::InvalidArgument, {"native_command_encoder", encoder.value, 0}})};
    return RhiAdapterResult<uint64_t>{vernon::ok(native)};
}

bool commandEncoderRendering(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder) {
    return vernon::rhi::commandEncoderRendering(adapter.rhiDevice, encoder.value);
}

bool commandEncoderHasRenderingDescriptor(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder) {
    return vernon::rhi::commandEncoderHasRenderingDescriptor(adapter.rhiDevice, encoder.value);
}

RhiAdapterResult<void> commandColorOperations(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder,
                                              size_t index, VernonRhiLoadOperation &load,
                                              VernonRhiStoreOperation &store, float clear[4]) noexcept {
    if (!vernon::rhi::commandColorOperations(adapter.rhiDevice, encoder.value, index, load, store, clear))
        return RhiAdapterResult<void>{vernon::err(
            vernon::ProviderError{vernon::ProviderErrorCode::InvalidArgument,
                                  {"command_color_operations", encoder.value, static_cast<uint32_t>(index)}})};
    return RhiAdapterResult<void>{vernon::ok()};
}

RhiAdapterResult<void> commandDepthOperations(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder,
                                              VernonRhiLoadOperation &depthLoad, VernonRhiStoreOperation &depthStore,
                                              VernonRhiLoadOperation &stencilLoad,
                                              VernonRhiStoreOperation &stencilStore, float &clearDepth,
                                              uint32_t &clearStencil) noexcept {
    if (!vernon::rhi::commandDepthOperations(adapter.rhiDevice, encoder.value, depthLoad, depthStore, stencilLoad,
                                             stencilStore, clearDepth, clearStencil))
        return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
            vernon::ProviderErrorCode::InvalidArgument, {"command_depth_operations", encoder.value, 0}})};
    return RhiAdapterResult<void>{vernon::ok()};
}

RhiAdapterResult<int> claimCommandRendering(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder,
                                            uint32_t backendKind) noexcept {
    const int claim = vernon::rhi::claimCommandRendering(adapter.rhiDevice, encoder.value, backendKind);
    if (claim < 0)
        return RhiAdapterResult<int>{vernon::err(vernon::ProviderError{
            vernon::ProviderErrorCode::LifecycleFailure, {"claim_command_rendering", encoder.value, backendKind}})};
    return RhiAdapterResult<int>{vernon::ok(claim)};
}

RhiAdapterResult<uint64_t> commandRenderingObject(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder,
                                                  uint64_t candidate) noexcept {
    const uint64_t object = vernon::rhi::commandRenderingObject(adapter.rhiDevice, encoder.value, candidate);
    if (!object)
        return RhiAdapterResult<uint64_t>{vernon::err(vernon::ProviderError{
            vernon::ProviderErrorCode::LifecycleFailure, {"command_rendering_object", encoder.value, 0}})};
    return RhiAdapterResult<uint64_t>{vernon::ok(object)};
}

RhiAdapterResult<void> recordProviderCommand(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder,
                                             bool draw) noexcept {
    if (!vernon::rhi::recordProviderCommand(adapter.rhiDevice, encoder.value, draw))
        return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
            vernon::ProviderErrorCode::LifecycleFailure, {"record_provider_command", encoder.value, draw}})};
    return RhiAdapterResult<void>{vernon::ok()};
}

RhiAdapterResult<void> recordCommandWriteResource(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder,
                                                  VernonRuntimeProviderResourceReference resource) noexcept {
    auto kind = resourceKind(adapter, resource);
    if (!kind)
        return RhiAdapterResult<void>{vernon::err(std::move(kind).error())};
    if (!vernon::rhi::recordCommandWriteResource(adapter.rhiDevice, encoder.value, std::move(kind).value(),
                                                 resource.resource.value))
        return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
            vernon::ProviderErrorCode::LifecycleFailure, {"record_command_write_resource", encoder.value, 0}})};
    return RhiAdapterResult<void>{vernon::ok()};
}

RhiAdapterResult<void> retainCommandResource(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder,
                                             VernonRuntimeProviderResourceReference resource) noexcept {
    auto kind = resourceKind(adapter, resource);
    if (!kind)
        return RhiAdapterResult<void>{vernon::err(std::move(kind).error())};
    auto retained = vernon::rhi::retainCommandResource(adapter.rhiDevice, encoder.value, std::move(kind).value(),
                                                       resource.resource.value);
    if (!retained)
        return RhiAdapterResult<void>{vernon::err(providerError(std::move(retained).error()))};
    return RhiAdapterResult<void>{vernon::ok()};
}

bool validCommonDrawDescriptor(const VernonRuntimeProviderDrawDescriptor *descriptor) {
    return descriptor && descriptor->struct_size >= sizeof(*descriptor) && descriptor->instance_count != 0 &&
           (descriptor->vertex_count != 0 || descriptor->index_count != 0) &&
           (descriptor->color_attachment_count != 0 || descriptor->depth_stencil_view.resource.value != 0) &&
           descriptor->color_attachment_count <= VERNON_RUNTIME_PROVIDER_MAX_COLOR_ATTACHMENTS &&
           (descriptor->color_attachment_count == 0 || descriptor->color_attachments) &&
           ((descriptor->index_count != 0) == (descriptor->index_buffer.resource.value != 0));
}

RhiAdapterResult<void> deferCommandCleanup(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder,
                                           void *context, uint64_t object, void (*cleanup)(void *, uint64_t)) noexcept {
    if (!vernon::rhi::deferCommandCleanup(adapter.rhiDevice, encoder.value, context, object, cleanup))
        return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{vernon::ProviderErrorCode::ResourceExhausted,
                                                                        {"defer_command_cleanup", encoder.value, 0}})};
    return RhiAdapterResult<void>{vernon::ok()};
}

RhiAdapterResult<void> deferCommandRollback(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder,
                                            void *context, uint64_t object,
                                            void (*rollback)(void *, uint64_t)) noexcept {
    if (!vernon::rhi::deferCommandRollback(adapter.rhiDevice, encoder.value, context, object, rollback))
        return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{vernon::ProviderErrorCode::ResourceExhausted,
                                                                        {"defer_command_rollback", encoder.value, 0}})};
    return RhiAdapterResult<void>{vernon::ok()};
}

RhiAdapterResult<void> setCommandRenderingTargets(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder,
                                                  const uint64_t *colors, const uint64_t *resources, size_t colorCount,
                                                  uint64_t depth, uint64_t depthResource) noexcept {
    if (!vernon::rhi::setCommandRenderingTargets(adapter.rhiDevice, encoder.value, colors, resources, colorCount, depth,
                                                 depthResource))
        return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
            vernon::ProviderErrorCode::LifecycleFailure, {"set_command_rendering_targets", encoder.value, 0}})};
    return RhiAdapterResult<void>{vernon::ok()};
}

} // namespace vernon::runtime::rhi_adapter

extern "C" VernonStatus vernonRuntimeRhiAdapterReferenceBuffer(const VernonRuntimeRhiAdapter *adapter,
                                                               VernonRhiBuffer buffer, uint64_t offset, uint64_t size,
                                                               VernonRuntimeProviderResourceReference *output) {
    if (!adapter || !output)
        return VERNON_STATUS_INVALID_ARGUMENT;
    const uint64_t resource = vernon::rhi::bufferResource(adapter->rhiDevice, buffer);
    if (!resource)
        return VERNON_STATUS_INVALID_ARGUMENT;
    output->identity = resourceIdentity(*adapter, kRhiBufferResource);
    output->resource = {resource};
    output->offset = offset;
    output->size = size;
    return VERNON_STATUS_OK;
}

extern "C" VernonStatus vernonRuntimeRhiAdapterReferenceImage(const VernonRuntimeRhiAdapter *adapter,
                                                              VernonRhiImage image,
                                                              VernonRuntimeProviderResourceReference *output) {
    if (!adapter || !output)
        return VERNON_STATUS_INVALID_ARGUMENT;
    const uint64_t resource = vernon::rhi::imageResource(adapter->rhiDevice, image);
    if (!resource)
        return VERNON_STATUS_INVALID_ARGUMENT;
    *output = {resourceIdentity(*adapter, kRhiImageResource), {resource}, 0, 0};
    return VERNON_STATUS_OK;
}

extern "C" VernonStatus vernonRuntimeRhiAdapterReferenceImageView(const VernonRuntimeRhiAdapter *adapter,
                                                                  VernonRhiImageView view,
                                                                  VernonRuntimeProviderResourceReference *output) {
    if (!adapter || !output)
        return VERNON_STATUS_INVALID_ARGUMENT;
    const uint64_t resource = vernon::rhi::imageViewResource(adapter->rhiDevice, view);
    if (!resource)
        return VERNON_STATUS_INVALID_ARGUMENT;
    *output = {resourceIdentity(*adapter, kRhiImageViewResource), {resource}, 0, 0};
    return VERNON_STATUS_OK;
}

extern "C" VernonStatus vernonRuntimeRhiAdapterReferenceSampler(const VernonRuntimeRhiAdapter *adapter,
                                                                VernonRhiSampler sampler,
                                                                VernonRuntimeProviderResourceReference *output) {
    if (!adapter || !output)
        return VERNON_STATUS_INVALID_ARGUMENT;
    const uint64_t resource = vernon::rhi::samplerResource(adapter->rhiDevice, sampler);
    if (!resource)
        return VERNON_STATUS_INVALID_ARGUMENT;
    *output = {resourceIdentity(*adapter, kRhiSamplerResource), {resource}, 0, 0};
    return VERNON_STATUS_OK;
}

extern "C" VernonStatus vernonRuntimeRhiAdapterReferenceCommandEncoder(const VernonRuntimeRhiAdapter *adapter,
                                                                       VernonRhiCommandEncoder encoder,
                                                                       VernonRuntimeProviderObject *output) {
    if (!adapter || !output)
        return VERNON_STATUS_INVALID_ARGUMENT;
    output->value = vernon::rhi::commandEncoderKey(adapter->rhiDevice, encoder);
    return output->value ? VERNON_STATUS_OK : VERNON_STATUS_INVALID_ARGUMENT;
}

namespace vernon::runtime {

RhiAdapterPreparationStats getRhiAdapterPreparationStats(const VernonRuntimeRhiAdapter &adapter) {
    RhiAdapterPreparationStats result{adapter.shaderPreparations.load(std::memory_order_relaxed),
                                      adapter.layoutPreparations.load(std::memory_order_relaxed),
                                      adapter.pipelinePreparations.load(std::memory_order_relaxed),
                                      adapter.bindingCreations.load(std::memory_order_relaxed),
                                      adapter.bindingSnapshotCreations.load(std::memory_order_relaxed),
                                      adapter.dispatches.load(std::memory_order_relaxed),
                                      adapter.livePreparedPipelines.load(std::memory_order_relaxed),
                                      adapter.lastStencilReference.load(std::memory_order_relaxed),
                                      adapter.lastDrawIndexed.load(std::memory_order_relaxed)};
    return result;
}

size_t getRhiAdapterRetainedLeaseCount(const VernonRuntimeRhiAdapter &adapter) {
    std::lock_guard<std::mutex> guard(adapter.retainedLeaseMutex);
    size_t count = 0;
    for (auto *entry = adapter.retainedLeaseHead.get(); entry; entry = entry->next.get())
        count += entry->lease.active();
    return count;
}

} // namespace vernon::runtime
