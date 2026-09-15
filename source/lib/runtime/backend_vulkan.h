#ifndef VERNON_RUNTIME_BACKEND_VULKAN_H
#define VERNON_RUNTIME_BACKEND_VULKAN_H

#include "VernonRuntime.h"
#include "VernonRuntimeCore.h"
#include "prepared_binding_plan.h"
#include "runtime_state.h"

#if defined(VERNON_HAS_VULKAN_RUNTIME)
#include "rhi/vulkan_backend.h"
#endif

#include <vector>

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
    VernonRuntimeCorePipeline *rhiComputePipeline{};
    VernonRuntimeCoreBindings *rhiComputeBindings{};
    PreparedComputeBindingPlan rhiComputeBindingPlan;
    std::vector<VernonRuntimeProviderBindingValue> rhiComputeValues;
    std::vector<int64_t> rhiComputeDescriptorValues;
    uint32_t rhiComputeWorkgroup[3]{1, 1, 1};
    VernonRuntimeCorePipeline *rhiGraphicsPipeline{};
    VernonRuntimeCoreBindings *rhiGraphicsBindings{};
    PreparedGraphicsVariant rhiGraphicsVariant;
    PreparedGraphicsBindingPlan rhiGraphicsBindingPlan;
    std::vector<VernonRuntimeProviderBindingValue> rhiGraphicsValues;
};
#endif

} // namespace vernon::runtime

#endif
