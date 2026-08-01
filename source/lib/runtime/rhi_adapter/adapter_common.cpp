#include "adapter_common.h"

#include "../../rhi/rhi_internal.h"
#include "adapter_test_hooks.h"

#include <new>
#include <optional>
#include <utility>

namespace vernon::runtime::rhi_adapter {

VernonStatus fail(VernonRuntimeRhiAdapter &adapter, std::string message, VernonStatus status) {
    adapter.error = std::move(message);
    return status;
}

} // namespace vernon::runtime::rhi_adapter

using namespace vernon::runtime::rhi_adapter;

extern "C" VernonRuntimeRhiAdapter *vernonRuntimeRhiAdapterCreateCuda(uint32_t deviceIndex) {
#if defined(VERNON_HAS_CUDA_RHI)
    auto adapter = std::unique_ptr<VernonRuntimeRhiAdapter>(new (std::nothrow) VernonRuntimeRhiAdapter());
    if (!adapter)
        return nullptr;
    try {
        adapter->ownedDevice = std::make_unique<vernon::rhi::cuda::DeviceState>();
    } catch (const std::bad_alloc &) {
        return nullptr;
    }
    adapter->device = adapter->ownedDevice.get();
    const vernon::rhi::cuda::Result result = adapter->device->initialize(deviceIndex);
    if (result != vernon::rhi::cuda::kSuccess) {
        adapter->error = vernon::rhi::cuda::describeResult(result, "CUDA RHI adapter initialization");
        return nullptr;
    }
    initializeCudaProvider(*adapter);
    return adapter.release();
#else
    (void)deviceIndex;
    return nullptr;
#endif
}

extern "C" VernonRuntimeRhiAdapter *vernonRuntimeRhiAdapterCreateForDevice(VernonRhiDevice device,
                                                                           VernonRhiBackend backend) {
    void *state = vernon::rhi::deviceState(device, backend);
    if (!state)
        return nullptr;
    VernonRuntimeRhiAdapter *adapter = nullptr;
    switch (backend) {
#if defined(VERNON_HAS_CUDA_RHI)
    case VERNON_RHI_BACKEND_CUDA:
        adapter = vernon::runtime::createBorrowedCudaRhiAdapter(*static_cast<vernon::rhi::cuda::DeviceState *>(state));
        break;
#endif
#if defined(VERNON_HAS_VULKAN_RHI)
    case VERNON_RHI_BACKEND_VULKAN:
        adapter =
            vernon::runtime::createBorrowedVulkanRhiAdapter(*static_cast<vernon::rhi::vulkan::DeviceState *>(state));
        break;
#endif
#if defined(VERNON_HAS_DIRECTX12_RHI)
    case VERNON_RHI_BACKEND_DIRECTX12:
        adapter = vernon::runtime::createBorrowedDirectX12RhiAdapter(
            *static_cast<vernon::rhi::directx12::DeviceState *>(state));
        break;
#endif
#if defined(VERNON_HAS_METAL_RHI)
    case VERNON_RHI_BACKEND_METAL:
        adapter =
            vernon::runtime::createBorrowedMetalRhiAdapter(*static_cast<vernon::rhi::metal::DeviceState *>(state));
        break;
#endif
    case VERNON_RHI_BACKEND_OPENGL:
    case VERNON_RHI_BACKEND_OPENGL_ES:
        adapter =
            vernon::runtime::createBorrowedOpenGLRhiAdapter(*static_cast<vernon::rhi::opengl::DeviceState *>(state));
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
    if (!adapter)
        return VERNON_STATUS_INVALID_ARGUMENT;
    if (adapter->openGLDevice) {
        adapter->openGLDevice->makeCurrent();
        adapter->openGLDevice->driver.finish();
        return VERNON_STATUS_OK;
    }
#if defined(VERNON_HAS_DIRECTX12_RHI)
    if (adapter->directX12Device)
        return adapter->directX12Device->synchronize(adapter->error) ? VERNON_STATUS_OK : VERNON_STATUS_INTERNAL_ERROR;
#endif
#if defined(VERNON_HAS_VULKAN_RHI)
    if (adapter->vulkanDevice)
        return adapter->vulkanDevice->synchronize(adapter->error) ? VERNON_STATUS_OK : VERNON_STATUS_INTERNAL_ERROR;
#endif
#if defined(VERNON_HAS_METAL_RHI)
    if (adapter->metalDevice)
        return synchronizeMetalProvider(*adapter);
#endif
#if defined(VERNON_HAS_CUDA_RHI)
    if (adapter->device) {
        const auto result = adapter->device->synchronize();
        return result == vernon::rhi::cuda::kSuccess
                   ? VERNON_STATUS_OK
                   : fail(*adapter, vernon::rhi::cuda::describeResult(result, "cuStreamSynchronize"),
                          VERNON_STATUS_INTERNAL_ERROR);
    }
#endif
    return fail(*adapter, "adapter has no device", VERNON_STATUS_UNSUPPORTED_TARGET);
}

extern "C" void vernonRuntimeRhiAdapterInvalidateState(VernonRuntimeRhiAdapter *adapter) {
    if (!adapter)
        return;
    adapter->openGLProgramValid = false;
    adapter->openGLVertexArrayValid = false;
    adapter->openGLFramebufferValid = false;
    adapter->openGLViewportValid = false;
    adapter->openGLScissorValid = false;
    adapter->openGLFramebufferSignatures.clear();
}

extern "C" VernonStringView vernonRuntimeRhiAdapterGetLastError(const VernonRuntimeRhiAdapter *adapter) {
    return adapter ? VernonStringView{adapter->error.data(), adapter->error.size()} : VernonStringView{};
}

namespace {

uint64_t resourceIdentity(const VernonRuntimeRhiAdapter &adapter, uint64_t directX12Kind) {
    switch (adapter.rhiBackend) {
#if defined(VERNON_HAS_CUDA_RHI)
    case VERNON_RHI_BACKEND_CUDA:
        return reinterpret_cast<uintptr_t>(adapter.device) | directX12Kind;
#endif
#if defined(VERNON_HAS_VULKAN_RHI)
    case VERNON_RHI_BACKEND_VULKAN:
        return reinterpret_cast<uintptr_t>(adapter.vulkanDevice) | directX12Kind;
#endif
#if defined(VERNON_HAS_DIRECTX12_RHI)
    case VERNON_RHI_BACKEND_DIRECTX12:
        return reinterpret_cast<uintptr_t>(adapter.directX12Device) | directX12Kind;
#endif
#if defined(VERNON_HAS_METAL_RHI)
    case VERNON_RHI_BACKEND_METAL:
        return reinterpret_cast<uintptr_t>(adapter.metalDevice) | directX12Kind;
#endif
    case VERNON_RHI_BACKEND_OPENGL:
    case VERNON_RHI_BACKEND_OPENGL_ES:
        return reinterpret_cast<uintptr_t>(adapter.openGLDevice) | directX12Kind;
    default:
        return 0;
    }
}

} // namespace

namespace vernon::runtime::rhi_adapter {

namespace {

std::optional<vernon::rhi::ResourceKind> resourceKind(const VernonRuntimeRhiAdapter &adapter,
                                                      VernonRuntimeProviderResourceReference resource) {
    const uint64_t encodedKind = resource.identity & kDirectX12ResourceKindMask;
    if (resource.identity != resourceIdentity(adapter, encodedKind) || resource.resource.value == 0)
        return std::nullopt;
    if (encodedKind == kDirectX12BufferResource)
        return vernon::rhi::ResourceKind::Buffer;
    if (encodedKind == kDirectX12ImageResource)
        return vernon::rhi::ResourceKind::Image;
    if (encodedKind == kDirectX12SamplerResource)
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
    output->identity = resourceIdentity(*adapter, kDirectX12BufferResource);
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
    *output = {resourceIdentity(*adapter, kDirectX12ImageResource), {resource}, 0, 0};
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
    *output = {resourceIdentity(*adapter, kDirectX12SamplerResource), {resource}, 0, 0};
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

VernonRuntimeRhiAdapter *createBorrowedCudaRhiAdapter(rhi::cuda::DeviceState &device) {
#if defined(VERNON_HAS_CUDA_RHI)
    auto adapter = std::unique_ptr<VernonRuntimeRhiAdapter>(new (std::nothrow) VernonRuntimeRhiAdapter());
    if (!adapter)
        return nullptr;
    adapter->device = &device;
    rhi_adapter::initializeCudaProvider(*adapter);
    return adapter.release();
#else
    (void)device;
    return nullptr;
#endif
}

#if defined(VERNON_HAS_CUDA_RHI)
rhi::cuda::DeviceState &cudaRhiAdapterDevice(VernonRuntimeRhiAdapter &adapter) { return *adapter.device; }
#endif

VernonRuntimeRhiAdapter *createBorrowedOpenGLRhiAdapter(rhi::opengl::DeviceState &device) {
    auto adapter = std::unique_ptr<VernonRuntimeRhiAdapter>(new (std::nothrow) VernonRuntimeRhiAdapter());
    if (!adapter)
        return nullptr;
    adapter->openGLDevice = &device;
    rhi_adapter::initializeOpenGLProvider(*adapter);
    return adapter.release();
}

#if defined(VERNON_HAS_DIRECTX12_RHI)
VernonRuntimeRhiAdapter *createBorrowedDirectX12RhiAdapter(rhi::directx12::DeviceState &device) {
    auto adapter = std::unique_ptr<VernonRuntimeRhiAdapter>(new (std::nothrow) VernonRuntimeRhiAdapter());
    if (!adapter)
        return nullptr;
    adapter->directX12Device = &device;
    rhi_adapter::initializeDirectX12Provider(*adapter);
    return adapter.release();
}

uint64_t directX12RhiAdapterResourceIdentity(const VernonRuntimeRhiAdapter &adapter) {
    return reinterpret_cast<uintptr_t>(adapter.directX12Device);
}

uint64_t directX12RhiAdapterImageIdentity(const VernonRuntimeRhiAdapter &adapter) {
    return directX12RhiAdapterResourceIdentity(adapter) | rhi_adapter::kDirectX12ImageResource;
}

uint64_t directX12RhiAdapterSamplerIdentity(const VernonRuntimeRhiAdapter &adapter) {
    return directX12RhiAdapterResourceIdentity(adapter) | rhi_adapter::kDirectX12SamplerResource;
}

uint64_t directX12RhiAdapterBufferIdentity(const VernonRuntimeRhiAdapter &adapter) {
    return directX12RhiAdapterResourceIdentity(adapter) | rhi_adapter::kDirectX12BufferResource;
}
#endif

#if defined(VERNON_HAS_VULKAN_RHI)
VernonRuntimeRhiAdapter *createBorrowedVulkanRhiAdapter(rhi::vulkan::DeviceState &device) {
    auto adapter = std::unique_ptr<VernonRuntimeRhiAdapter>(new (std::nothrow) VernonRuntimeRhiAdapter());
    if (!adapter)
        return nullptr;
    adapter->vulkanDevice = &device;
    rhi_adapter::initializeVulkanProvider(*adapter);
    return adapter.release();
}

uint64_t vulkanRhiAdapterResourceIdentity(const VernonRuntimeRhiAdapter &adapter) {
    return reinterpret_cast<uintptr_t>(adapter.vulkanDevice);
}
#endif

#if defined(VERNON_HAS_METAL_RHI)
VernonRuntimeRhiAdapter *createBorrowedMetalRhiAdapter(rhi::metal::DeviceState &device) {
    auto adapter = std::unique_ptr<VernonRuntimeRhiAdapter>(new (std::nothrow) VernonRuntimeRhiAdapter());
    if (!adapter)
        return nullptr;
    adapter->metalDevice = &device;
    rhi_adapter::initializeMetalProvider(*adapter);
    return adapter.release();
}
#endif

uint64_t openGLRhiAdapterResourceIdentity(const VernonRuntimeRhiAdapter &adapter) {
    return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(adapter.openGLDevice));
}

RhiAdapterPreparationStats getRhiAdapterPreparationStats(const VernonRuntimeRhiAdapter &adapter) {
    return {adapter.shaderPreparations.load(std::memory_order_relaxed),
            adapter.layoutPreparations.load(std::memory_order_relaxed),
            adapter.pipelinePreparations.load(std::memory_order_relaxed),
            adapter.bindingCreations.load(std::memory_order_relaxed),
            adapter.bindingSnapshotCreations.load(std::memory_order_relaxed),
            adapter.dispatches.load(std::memory_order_relaxed)};
}

} // namespace vernon::runtime
