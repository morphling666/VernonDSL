#ifndef VERNON_RUNTIME_BACKEND_VULKAN_H
#define VERNON_RUNTIME_BACKEND_VULKAN_H

#include "VernonRuntime.h"
#include "VernonRuntimeCore.h"
#include "prepared_binding_plan.h"
#include "runtime_state.h"

#include <memory>
#include <mutex>
#include <string>
#include <vector>

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

// The SPIR-V of one shader stage, which the HAL device creates its own pipelines from.
struct VulkanShaderCode {
    std::vector<uint32_t> words;
    std::string entry;
};

// A graphics Stage has vertex and fragment code and `graphicsPlan`; a compute Stage has compute code and `computePlan`.
struct VulkanPipelineState {
    PreparedComputeBindingPlan computePlan;
    PreparedGraphicsBindingPlan graphicsPlan;
    VulkanShaderCode compute;
    VulkanShaderCode vertex;
    VulkanShaderCode fragment;
    // The HAL device's pipelines of the Stage, made on first use under `halMutex` (hal/vulkan/vulkan_device.cpp).
    std::mutex halMutex;
    std::shared_ptr<void> hal;
};
#endif

} // namespace vernon::runtime

#endif
