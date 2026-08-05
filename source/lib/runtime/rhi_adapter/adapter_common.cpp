#include "adapter_common.h"

#include "adapter_internal.h"
#include "adapter_test_hooks.h"
#include "rhi/rhi_internal.h"

#include <new>
#include <optional>
#include <utility>

namespace vernon::runtime::rhi_adapter {

VernonStatus fail(VernonRuntimeRhiAdapter &adapter, std::string message, VernonStatus status) {
    adapter.error = std::move(message);
    return status;
}

void setBackendError(std::string &error, const char *message) noexcept {
    try {
        error = message;
    } catch (...) {
        error.clear();
    }
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

bool RhiAdapterBackendStorage::adopt(void *newState, const RhiAdapterBackendOps *newOps) noexcept {
    if (!newState || !newOps || state || ops)
        return false;
    state = newState;
    ops = newOps;
    return true;
}

VernonRuntimeRhiAdapter::~VernonRuntimeRhiAdapter() = default;

extern "C" VernonRuntimeRhiAdapter *vernonRuntimeRhiAdapterCreateCuda(uint32_t deviceIndex) {
#if defined(VERNON_HAS_CUDA_RHI)
    return vernon::runtime::createOwnedCudaRhiAdapter(deviceIndex);
#else
    (void)deviceIndex;
    return nullptr;
#endif
}

extern "C" VernonRuntimeRhiAdapter *vernonRuntimeRhiAdapterCreateForDevice(VernonRhiDevice device,
                                                                           VernonRhiBackend backend) {
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
    return adapter->backend.ops->synchronize(adapter->backend.state, adapter->error);
}

extern "C" void vernonRuntimeRhiAdapterInvalidateState(VernonRuntimeRhiAdapter *adapter) {
    if (!adapter || !adapter->backend.state || !adapter->backend.ops)
        return;
    adapter->backend.ops->invalidate(adapter->backend.state);
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

std::optional<vernon::rhi::ResourceKind> resourceKind(const VernonRuntimeRhiAdapter &adapter,
                                                      VernonRuntimeProviderResourceReference resource) {
    const uint64_t encodedKind = resource.identity & kRhiResourceKindMask;
    if (resource.identity != resourceIdentity(adapter, encodedKind) || resource.resource.value == 0)
        return std::nullopt;
    if (encodedKind == kRhiBufferResource)
        return vernon::rhi::ResourceKind::Buffer;
    if (encodedKind == kRhiImageResource)
        return vernon::rhi::ResourceKind::Image;
    if (encodedKind == kRhiSamplerResource)
        return vernon::rhi::ResourceKind::Sampler;
    return std::nullopt;
}

} // namespace

bool retainRhiResource(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderResourceReference resource) {
    const auto kind = resourceKind(adapter, resource);
    return kind && vernon::rhi::retainResource(adapter.rhiDevice, *kind, resource.resource.value);
}

void releaseRhiResource(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderResourceReference resource) {
    const auto kind = resourceKind(adapter, resource);
    if (kind)
        vernon::rhi::releaseResource(adapter.rhiDevice, *kind, resource.resource.value);
}

uint64_t resolveRhiResource(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderResourceReference resource) {
    const auto kind = resourceKind(adapter, resource);
    return kind ? vernon::rhi::resolveResource(adapter.rhiDevice, *kind, resource.resource.value) : 0;
}

uint64_t nativeCommandEncoder(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder) {
    return vernon::rhi::commandEncoderNative(adapter.rhiDevice, encoder.value, adapter.rhiBackend);
}

bool commandEncoderRendering(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder) {
    return vernon::rhi::commandEncoderRendering(adapter.rhiDevice, encoder.value);
}

bool commandEncoderHasRenderingDescriptor(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder) {
    return vernon::rhi::commandEncoderHasRenderingDescriptor(adapter.rhiDevice, encoder.value);
}

bool commandColorOperations(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder, size_t index,
                            VernonRhiLoadOperation &load, VernonRhiStoreOperation &store, float clear[4]) {
    return vernon::rhi::commandColorOperations(adapter.rhiDevice, encoder.value, index, load, store, clear);
}

bool commandDepthOperations(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder,
                            VernonRhiLoadOperation &depthLoad, VernonRhiStoreOperation &depthStore,
                            VernonRhiLoadOperation &stencilLoad, VernonRhiStoreOperation &stencilStore,
                            float &clearDepth, uint32_t &clearStencil) {
    return vernon::rhi::commandDepthOperations(adapter.rhiDevice, encoder.value, depthLoad, depthStore, stencilLoad,
                                               stencilStore, clearDepth, clearStencil);
}

int claimCommandRendering(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder, uint32_t backendKind) {
    return vernon::rhi::claimCommandRendering(adapter.rhiDevice, encoder.value, backendKind);
}

uint64_t commandRenderingObject(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder,
                                uint64_t candidate) {
    return vernon::rhi::commandRenderingObject(adapter.rhiDevice, encoder.value, candidate);
}

bool recordProviderCommand(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder, bool draw) {
    return vernon::rhi::recordProviderCommand(adapter.rhiDevice, encoder.value, draw);
}

bool recordCommandWriteResource(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder,
                                VernonRuntimeProviderResourceReference resource) {
    const auto kind = resourceKind(adapter, resource);
    return kind &&
           vernon::rhi::recordCommandWriteResource(adapter.rhiDevice, encoder.value, *kind, resource.resource.value);
}

bool retainCommandResource(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder,
                           VernonRuntimeProviderResourceReference resource) {
    const auto kind = resourceKind(adapter, resource);
    return kind && vernon::rhi::retainCommandResource(adapter.rhiDevice, encoder.value, *kind, resource.resource.value);
}

bool deferCommandCleanup(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder, void *context,
                         uint64_t object, void (*cleanup)(void *, uint64_t)) {
    return vernon::rhi::deferCommandCleanup(adapter.rhiDevice, encoder.value, context, object, cleanup);
}

bool deferCommandRollback(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder, void *context,
                          uint64_t object, void (*rollback)(void *, uint64_t)) {
    return vernon::rhi::deferCommandRollback(adapter.rhiDevice, encoder.value, context, object, rollback);
}

bool setCommandRenderingTargets(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder,
                                const uint64_t *colors, const uint64_t *resources, size_t colorCount, uint64_t depth,
                                uint64_t depthResource) {
    return vernon::rhi::setCommandRenderingTargets(adapter.rhiDevice, encoder.value, colors, resources, colorCount,
                                                   depth, depthResource);
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

} // namespace vernon::runtime
