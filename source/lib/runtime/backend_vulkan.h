#ifndef VERNON_RUNTIME_BACKEND_VULKAN_H
#define VERNON_RUNTIME_BACKEND_VULKAN_H

#include "VernonRuntime.h"
#include "VernonRuntimeCore.h"
#include "rhi_provider_pipeline.h"
#include "runtime_state.h"

#if defined(VERNON_HAS_VULKAN_RUNTIME)
#include "rhi/vulkan_backend.h"
#endif

struct VernonRuntimeRhiAdapter;
namespace vernon::runtime {

#if defined(VERNON_HAS_VULKAN_RUNTIME)
struct VulkanContextState {
    VernonRuntimeRhiAdapter *adapter{};
    uint32_t apiVersion{};
    uint32_t maxComputeWorkGroupInvocations{};
    uint32_t maxComputeWorkGroupSize[3]{};
    uint32_t defaultImplicitSamplerCreations{};
    uint32_t commandBufferAllocations{};
    uint32_t descriptorPoolCreations{};
    uint32_t stagingBufferAllocations{};
    bool dynamicRendering{};
};
inline VulkanContextState &vulkanState(VernonRuntimeContext &context) {
    return runtimeBackendState<VulkanContextState>(context);
}

inline const VulkanContextState &vulkanState(const VernonRuntimeContext &context) {
    return runtimeBackendState<VulkanContextState>(context);
}

struct VulkanPipelineState {
    RhiComputePipeline rhiCompute;
    RhiGraphicsPipeline rhiGraphics;
};
#endif

} // namespace vernon::runtime

#endif
