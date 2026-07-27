#ifndef VERNON_RUNTIME_RHI_ADAPTER_COMMON_H
#define VERNON_RUNTIME_RHI_ADAPTER_COMMON_H

#include "adapter_internal.h"
#include "adapter_test_hooks.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <deque>
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
    struct RetainedDirectX12Resource {
        // RHI slot addresses may be reused before RuntimeCore releases an old binding.
        // Keep native COM references in retain order so release never dereferences that slot.
        std::deque<ID3D12Resource *> resources;
    };
    vernon::rhi::directx12::DeviceState *directX12Device{};
    std::mutex directX12RetainedResourceMutex;
    std::unordered_map<uint64_t, RetainedDirectX12Resource> directX12RetainedResources;
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
