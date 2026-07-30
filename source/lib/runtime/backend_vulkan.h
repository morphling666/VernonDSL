#ifndef VERNON_RUNTIME_BACKEND_VULKAN_H
#define VERNON_RUNTIME_BACKEND_VULKAN_H

#include "VernonRuntime.h"
#include "VernonRuntimeCore.h"
#include "compute_launch_planner.h"
#include "pipeline_metadata.h"
#include "runtime_state.h"
#include "tensor_bridge.h"

#if defined(VERNON_HAS_VULKAN_RUNTIME)
#include "../rhi/vulkan_backend.h"
#endif

#include <string>
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
    struct Binding {
        enum Source {
            EXTERNAL_VERTEX,
            EXTERNAL_TEXTURE,
            EXTERNAL_SAMPLER,
            EXTERNAL_UNIFORM,
            EXTERNAL_STORAGE,
            IMPLICIT_SAMPLER,
            RESOLUTION
        };
        Source source{};
        uint32_t externalSlot{};
        TensorPackingLayout packing;
        std::vector<uint8_t> storage;
    };

    VernonRuntimeCorePipeline *rhiComputePipeline{};
    VernonRuntimeCoreBindings *rhiComputeBindings{};
    std::vector<VernonRuntimeProviderBindingLayoutEntry> rhiComputeLayout;
    std::vector<VernonRuntimeProviderBindingValue> rhiComputeValues;
    std::vector<uint64_t> rhiComputeResourceOffsets;
    std::vector<ComputeBindingSource> rhiComputeBindingSources;
    std::vector<int64_t> rhiComputeDescriptorValues;
    uint32_t rhiComputeWorkgroup[3]{1, 1, 1};
    VernonRuntimeCorePipeline *rhiGraphicsPipeline{};
    VernonRuntimeCoreBindings *rhiGraphicsBindings{};
    VernonRuntimeCoreGraphicsVariant *rhiGraphicsVariant{};
    std::vector<VernonRuntimeProviderBindingLayoutEntry> rhiGraphicsLayout;
    std::vector<VernonRuntimeProviderVertexAttribute> rhiGraphicsVertexAttributes;
    std::vector<VernonRuntimeProviderBindingValue> rhiGraphicsValues;
    std::vector<Binding> rhiGraphicsBindingPlan;
    std::vector<uint32_t> rhiGraphicsFormats;
    uint32_t rhiGraphicsDepthFormat{};
    uint64_t rhiGraphicsVertexLayoutIdentity{};
    uint32_t rhiGraphicsTopology{};
};
#endif

} // namespace vernon::runtime

#endif
