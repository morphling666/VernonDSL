#ifndef VERNON_RUNTIME_RHI_ADAPTER_INTERNAL_H
#define VERNON_RUNTIME_RHI_ADAPTER_INTERNAL_H

#include "VernonRuntimeRHIAdapter.h"

namespace vernon::runtime {

VernonRuntimeRhiAdapter *createOwnedCudaRhiAdapter(uint32_t deviceIndex);
VernonRuntimeRhiAdapter *createCudaRhiAdapter(VernonRhiDevice device, VernonRhiBackend backend);
VernonRuntimeRhiAdapter *createOpenGLRhiAdapter(VernonRhiDevice device, VernonRhiBackend backend);
#if defined(VERNON_HAS_VULKAN_RHI)
VernonRuntimeRhiAdapter *createVulkanRhiAdapter(VernonRhiDevice device, VernonRhiBackend backend);
#endif
#if defined(VERNON_HAS_DIRECTX12_RHI)
VernonRuntimeRhiAdapter *createDirectX12RhiAdapter(VernonRhiDevice device, VernonRhiBackend backend);
#endif
#if defined(VERNON_HAS_METAL_RHI)
VernonRuntimeRhiAdapter *createMetalRhiAdapter(VernonRhiDevice device, VernonRhiBackend backend);
#endif

} // namespace vernon::runtime

#endif
