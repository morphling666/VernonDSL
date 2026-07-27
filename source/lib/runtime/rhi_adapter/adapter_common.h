#ifndef VERNON_RUNTIME_RHI_ADAPTER_COMMON_H
#define VERNON_RUNTIME_RHI_ADAPTER_COMMON_H

#include "../../rhi/rhi_internal.h"
#include "adapter_internal.h"
#include "adapter_test_hooks.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

struct OpenGLFramebufferSignature {
    std::array<uint64_t, 18> values{};
    size_t count{};

    bool operator==(const OpenGLFramebufferSignature &other) const {
        return count == other.count && std::equal(values.begin(), values.begin() + count, other.values.begin());
    }
};

struct VernonRuntimeRhiAdapter {
    VernonRhiDevice rhiDevice{static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
    VernonRhiBackend rhiBackend{VERNON_RHI_BACKEND_CUDA};
#if defined(VERNON_HAS_CUDA_RHI)
    std::unique_ptr<vernon::rhi::cuda::DeviceState> ownedDevice;
    vernon::rhi::cuda::DeviceState *device{};
#endif
    vernon::rhi::opengl::DeviceState *openGLDevice{};
    uint64_t openGLProgram{};
    uint64_t openGLVertexArray{};
    uint64_t openGLFramebuffer{};
    uint64_t openGLFramebufferGeneration{};
    std::array<uint32_t, 4> openGLViewport{};
    std::array<uint32_t, 4> openGLScissor{};
    std::unordered_map<uint64_t, OpenGLFramebufferSignature> openGLFramebufferSignatures;
    bool openGLProgramValid{};
    bool openGLVertexArrayValid{};
    bool openGLFramebufferValid{};
    bool openGLViewportValid{};
    bool openGLScissorValid{};
#if defined(VERNON_HAS_DIRECTX12_RHI)
    vernon::rhi::directx12::DeviceState *directX12Device{};
#endif
#if defined(VERNON_HAS_VULKAN_RHI)
    vernon::rhi::vulkan::DeviceState *vulkanDevice{};
#endif
    VernonRuntimeDeviceProvider provider{};
    std::string error;
    std::atomic<size_t> shaderPreparations{};
    std::atomic<size_t> layoutPreparations{};
    std::atomic<size_t> pipelinePreparations{};
    std::atomic<size_t> bindingCreations{};
    std::atomic<size_t> dispatches{};
};

namespace vernon::runtime::rhi_adapter {

inline constexpr uint64_t kDirectX12ResourceKindMask = 3;
inline constexpr uint64_t kDirectX12ImageResource = 1;
inline constexpr uint64_t kDirectX12SamplerResource = 2;
inline constexpr uint64_t kDirectX12BufferResource = 3;

template <typename Object> VernonRuntimeProviderObject toHandle(Object *object) {
    return {static_cast<uint64_t>(reinterpret_cast<uintptr_t>(object))};
}

template <typename Object> Object *fromHandle(VernonRuntimeProviderObject handle) {
    return reinterpret_cast<Object *>(static_cast<uintptr_t>(handle.value));
}

VernonStatus fail(VernonRuntimeRhiAdapter &adapter, std::string message,
                  VernonStatus status = VERNON_STATUS_INVALID_ARGUMENT);
bool retainRhiResource(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderResourceReference resource);
void releaseRhiResource(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderResourceReference resource);
uint64_t resolveRhiResource(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderResourceReference resource);
uint64_t nativeCommandEncoder(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder);
bool commandEncoderRendering(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder);
bool commandEncoderHasRenderingDescriptor(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder);
bool commandColorOperations(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder, size_t index,
                            VernonRhiLoadOperation &load, VernonRhiStoreOperation &store, float clear[4]);
bool commandDepthOperations(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder,
                            VernonRhiLoadOperation &depthLoad, VernonRhiStoreOperation &depthStore,
                            VernonRhiLoadOperation &stencilLoad, VernonRhiStoreOperation &stencilStore,
                            float &clearDepth, uint32_t &clearStencil);
int claimCommandRendering(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder, uint32_t backendKind);
uint64_t commandRenderingObject(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder,
                                uint64_t candidate);
bool recordProviderCommand(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder, bool draw);
bool recordCommandWriteResource(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder,
                                VernonRuntimeProviderResourceReference resource);
bool retainCommandResource(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder,
                           VernonRuntimeProviderResourceReference resource);
bool deferCommandCleanup(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder, void *context,
                         uint64_t object, void (*cleanup)(void *, uint64_t));
bool deferCommandRollback(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder, void *context,
                          uint64_t object, void (*rollback)(void *, uint64_t));
bool setCommandRenderingTargets(VernonRuntimeRhiAdapter &adapter, VernonRuntimeProviderObject encoder,
                                const uint64_t *colors, const uint64_t *resources, size_t colorCount, uint64_t depth,
                                uint64_t depthResource);

#if defined(VERNON_HAS_CUDA_RHI)
void initializeCudaProvider(VernonRuntimeRhiAdapter &adapter);
#endif
void initializeOpenGLProvider(VernonRuntimeRhiAdapter &adapter);
#if defined(VERNON_HAS_DIRECTX12_RHI)
void initializeDirectX12Provider(VernonRuntimeRhiAdapter &adapter);
#endif
#if defined(VERNON_HAS_VULKAN_RHI)
void initializeVulkanProvider(VernonRuntimeRhiAdapter &adapter);
#endif

} // namespace vernon::runtime::rhi_adapter

#endif
