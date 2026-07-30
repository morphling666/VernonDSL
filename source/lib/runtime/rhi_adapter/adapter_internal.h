#ifndef VERNON_RUNTIME_RHI_ADAPTER_INTERNAL_H
#define VERNON_RUNTIME_RHI_ADAPTER_INTERNAL_H

#include "../../rhi/cuda_backend.h"
#if defined(_WIN32)
#include "../../rhi/directx12_backend.h"
#endif
#include "../../rhi/opengl_backend.h"
#if defined(VERNON_HAS_VULKAN_RUNTIME) || defined(VERNON_HAS_VULKAN_RHI)
#include "../../rhi/vulkan_backend.h"
#endif
#include "VernonRuntimeRHIAdapter.h"

namespace vernon::runtime {

VernonRuntimeRhiAdapter *createBorrowedCudaRhiAdapter(rhi::cuda::DeviceState &device);
rhi::cuda::DeviceState &cudaRhiAdapterDevice(VernonRuntimeRhiAdapter &adapter);
VernonRuntimeRhiAdapter *createBorrowedOpenGLRhiAdapter(rhi::opengl::DeviceState &device);
uint64_t openGLRhiAdapterResourceIdentity(const VernonRuntimeRhiAdapter &adapter);
#if defined(VERNON_HAS_VULKAN_RUNTIME) || defined(VERNON_HAS_VULKAN_RHI)
VernonRuntimeRhiAdapter *createBorrowedVulkanRhiAdapter(rhi::vulkan::DeviceState &device);
uint64_t vulkanRhiAdapterResourceIdentity(const VernonRuntimeRhiAdapter &adapter);
#endif
#if defined(_WIN32)
VernonRuntimeRhiAdapter *createBorrowedDirectX12RhiAdapter(rhi::directx12::DeviceState &device);
uint64_t directX12RhiAdapterResourceIdentity(const VernonRuntimeRhiAdapter &adapter);
uint64_t directX12RhiAdapterImageIdentity(const VernonRuntimeRhiAdapter &adapter);
uint64_t directX12RhiAdapterSamplerIdentity(const VernonRuntimeRhiAdapter &adapter);
uint64_t directX12RhiAdapterBufferIdentity(const VernonRuntimeRhiAdapter &adapter);
#endif

} // namespace vernon::runtime

#endif
