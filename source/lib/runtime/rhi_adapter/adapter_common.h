#ifndef VERNON_RUNTIME_RHI_ADAPTER_COMMON_H
#define VERNON_RUNTIME_RHI_ADAPTER_COMMON_H

#include "adapter_internal.h"
#include "adapter_test_hooks.h"

#include <atomic>
#include <memory>
#include <string>

struct VernonRuntimeRhiAdapter {
    VernonRhiDevice rhiDevice{static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
    VernonRhiBackend rhiBackend{VERNON_RHI_BACKEND_CUDA};
#if defined(VERNON_HAS_CUDA_RHI)
    std::unique_ptr<vernon::rhi::cuda::DeviceState> ownedDevice;
    vernon::rhi::cuda::DeviceState *device{};
#endif
    vernon::rhi::opengl::DeviceState *openGLDevice{};
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
