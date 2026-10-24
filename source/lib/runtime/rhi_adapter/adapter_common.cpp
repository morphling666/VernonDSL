#include "adapter_common.h"

#include "VernonFormats.hpp"
#include "adapter_internal.h"
#include "adapter_test_hooks.h"
#include "rhi/rhi_internal.h"
#include "runtime/provider_image_description.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <iterator>
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
    case RhiErrorCode::SurfaceOutOfDate:
    case RhiErrorCode::SurfaceLost:
        return {ProviderErrorCode::BackendFailure, error.context};
    }
    return {ProviderErrorCode::BackendFailure, error.context};
}

void recordProviderError(VernonRuntimeRhiAdapter &adapter, vernon::ProviderError error,
                         const char *diagnostic) noexcept {
    char rendered[384]{};
    if (error.context.operation) {
        if (error.context.value || error.context.detail)
            std::snprintf(rendered, sizeof(rendered), "%s: %s (%llu, %u)", diagnostic, error.context.operation,
                          static_cast<unsigned long long>(error.context.value), error.context.detail);
        else
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
    if (!adapter)
        return nullptr;
    adapter->rhiDevice = device;
    adapter->rhiBackend = backend;
    adapter->rhiDeviceLease.emplace(std::move(deviceLease).value());
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
    const vernon::formats::Format *entry = vernon::formats::find(format);
    return entry ? std::optional<VernonTextureFormat>(entry->texture) : std::nullopt;
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
    auto retained = vernon::rhi::retainResource(adapter.rhiDevice, kind, resource.resource.value);
    if (!retained)
        return RhiAdapterResult<void>{vernon::err(providerError(std::move(retained).error()))};
    auto lease = std::move(retained).value();
    auto described =
        view ? vernon::rhi::describeImageViewResource(adapter.rhiDevice, resource.resource.value, sourceView, source,
                                                      parentKey, lease)
             : vernon::rhi::describeImageResource(adapter.rhiDevice, resource.resource.value, source, lease);
    if (!described)
        return RhiAdapterResult<void>{vernon::err(providerError(std::move(described).error()))};
    const auto format = providerTextureFormat(source.format);
    if (!format)
        return RhiAdapterResult<void>{vernon::err(
            vernon::ProviderError{vernon::ProviderErrorCode::Unsupported,
                                  {"describe_provider_image_format", static_cast<uint64_t>(source.format), 0}})};
    const auto dimension = static_cast<VernonTextureDimension>(source.dimension);
    uint32_t usage = 0;
    usage |= (source.usage & VERNON_RHI_IMAGE_SAMPLED) ? VERNON_IMAGE_SAMPLED : 0;
    usage |= (source.usage & VERNON_RHI_IMAGE_STORAGE) ? VERNON_IMAGE_STORAGE : 0;
    usage |= (source.usage & VERNON_RHI_IMAGE_COLOR_ATTACHMENT) ? VERNON_IMAGE_COLOR_ATTACHMENT : 0;
    usage |= (source.usage & VERNON_RHI_IMAGE_DEPTH_STENCIL_ATTACHMENT) ? VERNON_IMAGE_DEPTH_STENCIL_ATTACHMENT : 0;
    usage |= (source.usage & VERNON_RHI_IMAGE_TRANSFER_SOURCE) ? VERNON_IMAGE_TRANSFER_SOURCE : 0;
    usage |= (source.usage & VERNON_RHI_IMAGE_TRANSFER_DESTINATION) ? VERNON_IMAGE_TRANSFER_DESTINATION : 0;
    const uint32_t aspects = vernon::formats::aspects(*format);
    description.image = {dimension,
                         {source.width, source.height, source.depth},
                         *format,
                         source.mip_levels,
                         source.array_layers,
                         source.sample_count,
                         usage};
    const VernonTextureDimension viewDimension =
        view ? static_cast<VernonTextureDimension>(sourceView.dimension) : dimension;
    const auto viewFormat = view ? providerTextureFormat(sourceView.format) : format;
    if (!viewFormat)
        return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{
            vernon::ProviderErrorCode::Unsupported,
            {"describe_provider_image_view_format", static_cast<uint64_t>(sourceView.format), 0}})};
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
    auto native = vernon::rhi::commandEncoderNative(adapter.rhiDevice, encoder.value, adapter.rhiBackend);
    if (!native)
        return RhiAdapterResult<uint64_t>{vernon::err(providerError(std::move(native).error()))};
    return RhiAdapterResult<uint64_t>{vernon::ok(std::move(native).value())};
}

RhiAdapterResult<bool> commandEncoderRendering(VernonRuntimeRhiAdapter &adapter,
                                               VernonRuntimeProviderObject encoder) noexcept {
    auto rendering = vernon::rhi::commandEncoderRendering(adapter.rhiDevice, encoder.value);
    if (!rendering)
        return RhiAdapterResult<bool>{vernon::err(providerError(std::move(rendering).error()))};
    return RhiAdapterResult<bool>{vernon::ok(std::move(rendering).value())};
}

RhiAdapterResult<vernon::rhi::OpenedRendering> openCommandRendering(VernonRuntimeRhiAdapter &adapter,
                                                                    VernonRuntimeProviderObject encoder) noexcept {
    auto opened = vernon::rhi::openCommandRendering(adapter.rhiDevice, encoder.value);
    if (!opened)
        return RhiAdapterResult<vernon::rhi::OpenedRendering>{vernon::err(providerError(std::move(opened).error()))};
    return RhiAdapterResult<vernon::rhi::OpenedRendering>{vernon::ok(std::move(opened).value())};
}

namespace {

RhiAdapterResult<uint64_t> attachmentView(const VernonRuntimeRhiAdapter &adapter,
                                          VernonRuntimeProviderResourceReference view) noexcept {
    auto kind = resourceKind(adapter, view);
    if (!kind)
        return RhiAdapterResult<uint64_t>{vernon::err(std::move(kind).error())};
    if (kind.value() != vernon::rhi::ResourceKind::ImageView)
        return RhiAdapterResult<uint64_t>{vernon::err(vernon::ProviderError{
            vernon::ProviderErrorCode::InvalidArgument, {"rendering_attachment_is_not_an_image_view", 0, 0}})};
    return RhiAdapterResult<uint64_t>{vernon::ok(view.resource.value)};
}

static_assert(VERNON_RUNTIME_PROVIDER_LOAD_CLEAR == static_cast<int>(VERNON_RHI_LOAD_CLEAR) &&
              VERNON_RUNTIME_PROVIDER_LOAD_PRESERVE == static_cast<int>(VERNON_RHI_LOAD_PRESERVE) &&
              VERNON_RUNTIME_PROVIDER_LOAD_DISCARD == static_cast<int>(VERNON_RHI_LOAD_DISCARD) &&
              VERNON_RUNTIME_PROVIDER_STORE_PRESERVE == static_cast<int>(VERNON_RHI_STORE_PRESERVE) &&
              VERNON_RUNTIME_PROVIDER_STORE_DISCARD == static_cast<int>(VERNON_RHI_STORE_DISCARD));

} // namespace

RhiAdapterResult<void> beginEncoderRendering(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder,
                                             const VernonRuntimeProviderRenderingDescriptor &descriptor) noexcept {
    const auto invalid = [](const char *operation) {
        return RhiAdapterResult<void>{
            vernon::err(vernon::ProviderError{vernon::ProviderErrorCode::InvalidArgument, {operation, 0, 0}})};
    };
    if (descriptor.struct_size < sizeof(descriptor) ||
        descriptor.color_attachment_count > VERNON_RUNTIME_PROVIDER_MAX_COLOR_ATTACHMENTS ||
        (descriptor.color_attachment_count && !descriptor.color_attachments) ||
        (descriptor.sampled_image_count && !descriptor.sampled_images) || descriptor.render_area[0] > INT32_MAX ||
        descriptor.render_area[1] > INT32_MAX)
        return invalid("begin_rendering_descriptor");
    std::vector<vernon::rhi::RenderingRead> reads;
    try {
        reads.reserve(descriptor.sampled_image_count);
    } catch (const std::bad_alloc &) {
        return RhiAdapterResult<void>{vernon::err(vernon::ProviderError{vernon::ProviderErrorCode::ResourceExhausted,
                                                                        {"begin_rendering_sampled_images", 0, 0}})};
    }
    for (size_t index = 0; index < descriptor.sampled_image_count; ++index) {
        const VernonRuntimeProviderResourceReference image = descriptor.sampled_images[index];
        auto kind = resourceKind(adapter, image);
        if (!kind)
            return RhiAdapterResult<void>{vernon::err(std::move(kind).error())};
        if (kind.value() != vernon::rhi::ResourceKind::Image && kind.value() != vernon::rhi::ResourceKind::ImageView)
            return invalid("rendering_sampled_image_is_not_an_image");
        reads.push_back({kind.value(), image.resource.value});
    }
    vernon::rhi::RenderingScope scope;
    scope.colorCount = descriptor.color_attachment_count;
    for (size_t index = 0; index < descriptor.color_attachment_count; ++index) {
        const auto &source = descriptor.color_attachments[index];
        auto view = attachmentView(adapter, source.view);
        if (!view)
            return RhiAdapterResult<void>{vernon::err(std::move(view).error())};
        auto &target = scope.colors[index];
        target.view = view.value();
        target.location = source.location;
        target.load = static_cast<VernonRhiLoadOperation>(source.load_operation);
        target.store = static_cast<VernonRhiStoreOperation>(source.store_operation);
        target.clear = source.clear_color;
    }
    if (descriptor.depth_stencil_attachment) {
        const auto &source = *descriptor.depth_stencil_attachment;
        auto view = attachmentView(adapter, source.view);
        if (!view)
            return RhiAdapterResult<void>{vernon::err(std::move(view).error())};
        scope.depth.view = view.value();
        scope.depth.depthLoad = static_cast<VernonRhiLoadOperation>(source.depth_load_operation);
        scope.depth.depthStore = static_cast<VernonRhiStoreOperation>(source.depth_store_operation);
        scope.depth.stencilLoad = static_cast<VernonRhiLoadOperation>(source.stencil_load_operation);
        scope.depth.stencilStore = static_cast<VernonRhiStoreOperation>(source.stencil_store_operation);
        scope.depth.clearDepth = source.clear_depth;
        scope.depth.clearStencil = source.clear_stencil;
        scope.hasDepth = true;
    }
    scope.x = static_cast<int32_t>(descriptor.render_area[0]);
    scope.y = static_cast<int32_t>(descriptor.render_area[1]);
    scope.width = descriptor.render_area[2];
    scope.height = descriptor.render_area[3];
    scope.layers = descriptor.layers;
    auto begun =
        vernon::rhi::beginEncoderRendering(adapter.rhiDevice, encoder.value, scope, reads.data(), reads.size());
    if (!begun)
        return RhiAdapterResult<void>{vernon::err(providerError(std::move(begun).error()))};
    return RhiAdapterResult<void>{vernon::ok()};
}

RhiAdapterResult<void> endEncoderRendering(VernonRuntimeRhiAdapter &adapter,
                                           VernonRuntimeProviderObject encoder) noexcept {
    auto ended = vernon::rhi::endEncoderRendering(adapter.rhiDevice, encoder.value);
    if (!ended)
        return RhiAdapterResult<void>{vernon::err(providerError(std::move(ended).error()))};
    return RhiAdapterResult<void>{vernon::ok()};
}

RhiAdapterResult<void> recordProviderCommand(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder,
                                             bool draw) noexcept {
    auto recorded = vernon::rhi::recordProviderCommand(adapter.rhiDevice, encoder.value, draw);
    if (!recorded)
        return RhiAdapterResult<void>{vernon::err(providerError(std::move(recorded).error()))};
    return RhiAdapterResult<void>{vernon::ok()};
}

RhiAdapterResult<void> recordCommandWriteResource(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder,
                                                  VernonRuntimeProviderResourceReference resource) noexcept {
    auto kind = resourceKind(adapter, resource);
    if (!kind)
        return RhiAdapterResult<void>{vernon::err(std::move(kind).error())};
    auto recorded = vernon::rhi::recordCommandWriteResource(adapter.rhiDevice, encoder.value, std::move(kind).value(),
                                                            resource.resource.value);
    if (!recorded)
        return RhiAdapterResult<void>{vernon::err(providerError(std::move(recorded).error()))};
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

RhiAdapterResult<vernon::rhi::CommandTransient> writeCommandTransient(VernonRuntimeRhiAdapter &adapter,
                                                                      VernonRuntimeProviderObject encoder,
                                                                      const void *data, uint64_t size,
                                                                      uint64_t alignment) noexcept {
    auto written = vernon::rhi::writeCommandTransient(adapter.rhiDevice, encoder.value, data, size, alignment);
    if (!written)
        return RhiAdapterResult<vernon::rhi::CommandTransient>{vernon::err(providerError(std::move(written).error()))};
    return RhiAdapterResult<vernon::rhi::CommandTransient>{vernon::ok(written.value())};
}

RhiAdapterResult<vernon::rhi::CommandTransientReservation> reserveCommandTransient(VernonRuntimeRhiAdapter &adapter,
                                                                                   VernonRuntimeProviderObject encoder,
                                                                                   uint64_t size,
                                                                                   uint64_t alignment) noexcept {
    auto reserved = vernon::rhi::reserveCommandTransient(adapter.rhiDevice, encoder.value, size, alignment);
    if (!reserved)
        return RhiAdapterResult<vernon::rhi::CommandTransientReservation>{
            vernon::err(providerError(std::move(reserved).error()))};
    return RhiAdapterResult<vernon::rhi::CommandTransientReservation>{vernon::ok(reserved.value())};
}

RhiAdapterResult<void> deferCommandCleanup(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder,
                                           void *context, uint64_t object, void (*cleanup)(void *, uint64_t)) noexcept {
    auto deferred = vernon::rhi::deferCommandCleanup(adapter.rhiDevice, encoder.value, context, object, cleanup);
    if (!deferred)
        return RhiAdapterResult<void>{vernon::err(providerError(std::move(deferred).error()))};
    return RhiAdapterResult<void>{vernon::ok()};
}

RhiAdapterResult<void> deferCommandRollback(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder,
                                            void *context, uint64_t object,
                                            void (*rollback)(void *, uint64_t)) noexcept {
    auto deferred = vernon::rhi::deferCommandRollback(adapter.rhiDevice, encoder.value, context, object, rollback);
    if (!deferred)
        return RhiAdapterResult<void>{vernon::err(providerError(std::move(deferred).error()))};
    return RhiAdapterResult<void>{vernon::ok()};
}

void releaseRetainedObject(RhiRetainedObject *object) noexcept {
    if (object && object->references.fetch_sub(1, std::memory_order_acq_rel) == 1)
        delete object;
}

namespace {

void releaseCommandObject(void *context, uint64_t) { releaseRetainedObject(static_cast<RhiRetainedObject *>(context)); }

} // namespace

RhiAdapterResult<void> retainForCommands(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder,
                                         RhiRetainedObject &object) noexcept {
    object.references.fetch_add(1, std::memory_order_relaxed);
    auto deferred = deferCommandCleanup(adapter, encoder, &object, 0, releaseCommandObject);
    if (!deferred)
        releaseRetainedObject(&object);
    return deferred;
}

RhiAdapterResult<uint64_t> acquireGroupResource(VernonRuntimeRhiAdapter &adapter, RhiGroup &group,
                                                VernonRuntimeProviderResourceReference resource,
                                                RhiImageBinding *image) noexcept {
    using Acquired = RhiAdapterResult<uint64_t>;
    auto kindResult = resourceKind(adapter, resource);
    if (!kindResult)
        return Acquired{vernon::err(std::move(kindResult).error())};
    const vernon::rhi::ResourceKind kind = kindResult.value();
    if (image && kind != vernon::rhi::ResourceKind::Image && kind != vernon::rhi::ResourceKind::ImageView)
        return Acquired{vernon::err(vernon::ProviderError{vernon::ProviderErrorCode::InvalidArgument,
                                                          {"group_image_binding_is_not_an_image", 0, 0}})};
    auto retained = vernon::rhi::retainResource(adapter.rhiDevice, kind, resource.resource.value);
    if (!retained)
        return Acquired{vernon::err(providerError(std::move(retained).error()))};
    try {
        group.leases.push_back(std::move(retained).value());
    } catch (const std::bad_alloc &) {
        return Acquired{vernon::err(vernon::ProviderError{vernon::ProviderErrorCode::ResourceExhausted,
                                                          {"group_resource_lease_allocation", 0, 0}})};
    }
    auto &lease = group.leases.back();
    auto resolved = vernon::rhi::resolveResource(adapter.rhiDevice, kind, resource.resource.value, lease);
    if (!resolved)
        return Acquired{vernon::err(providerError(std::move(resolved).error()))};
    if (image) {
        VernonRhiImageDescriptor descriptor{};
        VernonRhiImageViewDescriptor view{};
        uint64_t parentKey = 0;
        auto described =
            kind == vernon::rhi::ResourceKind::Image
                ? vernon::rhi::describeImageResource(adapter.rhiDevice, resource.resource.value, descriptor, lease)
                : vernon::rhi::describeImageViewResource(adapter.rhiDevice, resource.resource.value, view, descriptor,
                                                         parentKey, lease);
        if (!described)
            return Acquired{vernon::err(providerError(std::move(described).error()))};
        *image = kind == vernon::rhi::ResourceKind::Image ? RhiImageBinding{descriptor.dimension, descriptor.format}
                                                          : RhiImageBinding{view.dimension, view.format};
    }
    return Acquired{vernon::ok(resolved.value())};
}

RhiAdapterResult<void> recordGroupWrites(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder,
                                         const RhiGroup &group) noexcept {
    for (const VernonRuntimeProviderResourceReference &resource : group.writes)
        if (auto recorded = recordCommandWriteResource(adapter, encoder, resource); !recorded)
            return recorded;
    return RhiAdapterResult<void>{vernon::ok()};
}

RhiAdapterResult<void> beginRhiCommands(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder,
                                        RhiCommands &commands) noexcept {
    auto native = nativeCommandEncoder(adapter, encoder);
    if (!native)
        return RhiAdapterResult<void>{vernon::err(std::move(native).error())};
    auto rendering = commandEncoderRendering(adapter, encoder);
    if (!rendering)
        return RhiAdapterResult<void>{vernon::err(std::move(rendering).error())};
    if (rendering.value())
        return providerFailure(vernon::ProviderErrorCode::InvalidArgument, "begin_commands_inside_a_rendering_scope");
    commands.encoder = encoder;
    commands.native = native.value();
    commands.rendering = false;
    return RhiAdapterResult<void>{vernon::ok()};
}

} // namespace vernon::runtime::rhi_adapter

extern "C" VernonStatus vernonRuntimeRhiAdapterReferenceBuffer(const VernonRuntimeRhiAdapter *adapter,
                                                               VernonRhiBuffer buffer, uint64_t offset, uint64_t size,
                                                               VernonRuntimeProviderResourceReference *output) {
    if (!adapter || !output)
        return VERNON_STATUS_INVALID_ARGUMENT;
    auto resource = vernon::rhi::bufferResource(adapter->rhiDevice, buffer);
    if (!resource)
        return vernon::toVernonStatus(std::move(resource).error());
    const uint64_t resourceKey = std::move(resource).value();
    output->identity = resourceIdentity(*adapter, kRhiBufferResource);
    output->resource = {resourceKey};
    output->offset = offset;
    output->size = size;
    return VERNON_STATUS_OK;
}

extern "C" VernonStatus vernonRuntimeRhiAdapterReferenceImage(const VernonRuntimeRhiAdapter *adapter,
                                                              VernonRhiImage image,
                                                              VernonRuntimeProviderResourceReference *output) {
    if (!adapter || !output)
        return VERNON_STATUS_INVALID_ARGUMENT;
    auto resource = vernon::rhi::imageResource(adapter->rhiDevice, image);
    if (!resource)
        return vernon::toVernonStatus(std::move(resource).error());
    *output = {resourceIdentity(*adapter, kRhiImageResource), {std::move(resource).value()}, 0, 0};
    return VERNON_STATUS_OK;
}

extern "C" VernonStatus vernonRuntimeRhiAdapterReferenceImageView(const VernonRuntimeRhiAdapter *adapter,
                                                                  VernonRhiImageView view,
                                                                  VernonRuntimeProviderResourceReference *output) {
    if (!adapter || !output)
        return VERNON_STATUS_INVALID_ARGUMENT;
    auto resource = vernon::rhi::imageViewResource(adapter->rhiDevice, view);
    if (!resource)
        return vernon::toVernonStatus(std::move(resource).error());
    *output = {resourceIdentity(*adapter, kRhiImageViewResource), {std::move(resource).value()}, 0, 0};
    return VERNON_STATUS_OK;
}

extern "C" VernonStatus vernonRuntimeRhiAdapterReferenceSampler(const VernonRuntimeRhiAdapter *adapter,
                                                                VernonRhiSampler sampler,
                                                                VernonRuntimeProviderResourceReference *output) {
    if (!adapter || !output)
        return VERNON_STATUS_INVALID_ARGUMENT;
    auto resource = vernon::rhi::samplerResource(adapter->rhiDevice, sampler);
    if (!resource)
        return vernon::toVernonStatus(std::move(resource).error());
    *output = {resourceIdentity(*adapter, kRhiSamplerResource), {std::move(resource).value()}, 0, 0};
    return VERNON_STATUS_OK;
}

extern "C" VernonStatus vernonRuntimeRhiAdapterReferenceCommandEncoder(const VernonRuntimeRhiAdapter *adapter,
                                                                       VernonRhiCommandEncoder encoder,
                                                                       VernonRuntimeProviderObject *output) {
    if (!adapter || !output)
        return VERNON_STATUS_INVALID_ARGUMENT;
    auto key = vernon::rhi::commandEncoderKey(adapter->rhiDevice, encoder);
    if (!key)
        return vernon::toVernonStatus(std::move(key).error());
    output->value = std::move(key).value();
    return VERNON_STATUS_OK;
}

namespace vernon::runtime {

RhiAdapterPreparationStats getRhiAdapterPreparationStats(const VernonRuntimeRhiAdapter &adapter) {
    RhiAdapterPreparationStats result{adapter.shaderPreparations.load(std::memory_order_relaxed),
                                      adapter.layoutPreparations.load(std::memory_order_relaxed),
                                      adapter.pipelinePreparations.load(std::memory_order_relaxed),
                                      adapter.groupCreations.load(std::memory_order_relaxed),
                                      adapter.dispatches.load(std::memory_order_relaxed),
                                      adapter.livePreparedPipelines.load(std::memory_order_relaxed),
                                      adapter.lastStencilReference.load(std::memory_order_relaxed),
                                      adapter.lastDrawIndexed.load(std::memory_order_relaxed)};
    return result;
}

} // namespace vernon::runtime
