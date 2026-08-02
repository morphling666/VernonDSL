#ifndef VERNON_RUNTIME_BACKEND_METAL_H
#define VERNON_RUNTIME_BACKEND_METAL_H

#include "VernonRuntime.h"
#include "VernonRuntimeCore.h"
#include "compute_launch_planner.h"
#include "metal_runtime_capabilities.h"
#include "runtime_state.h"
#include "tensor_bridge.h"

#include <cstdint>
#include <vector>

struct VernonRuntimeRhiAdapter;

namespace vernon::runtime {

struct MetalContextState {
    VernonRuntimeRhiAdapter *adapter{};
    uint32_t maxComputeInvocations{};
    uint32_t maxComputeWorkGroupSize[3]{};
    RuntimeVersion operatingSystemVersion;
    uint32_t argumentBuffersTier{};
    bool argumentBufferEncodingSupported{};
};

inline MetalContextState &metalState(VernonRuntimeContext &context) {
    return runtimeBackendState<MetalContextState>(context);
}

inline const MetalContextState &metalState(const VernonRuntimeContext &context) {
    return runtimeBackendState<MetalContextState>(context);
}

struct MetalPipelineState {
    struct GraphicsBinding {
        enum Source {
            EXTERNAL_UNIFORM,
            EXTERNAL_VERTEX,
            EXTERNAL_TEXTURE,
            EXTERNAL_SAMPLER,
            EXTERNAL_STORAGE,
            IMPLICIT_SAMPLER,
            RESOLUTION
        };
        Source source{};
        uint32_t externalSlot{};
        uint32_t descriptorSet{UINT32_MAX};
        uint32_t descriptorBinding{UINT32_MAX};
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
    PreparedGraphicsVariant rhiGraphicsVariant;
    std::vector<VernonRuntimeProviderBindingLayoutEntry> rhiGraphicsLayout;
    std::vector<VernonRuntimeProviderVertexAttribute> rhiGraphicsVertexAttributes;
    std::vector<VernonRuntimeProviderBindingValue> rhiGraphicsValues;
    std::vector<GraphicsBinding> rhiGraphicsBindingPlan;
};

} // namespace vernon::runtime

#endif
