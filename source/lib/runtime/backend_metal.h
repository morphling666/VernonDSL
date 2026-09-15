#ifndef VERNON_RUNTIME_BACKEND_METAL_H
#define VERNON_RUNTIME_BACKEND_METAL_H

#include "VernonRuntime.h"
#include "VernonRuntimeCore.h"
#include "graphics_invocation_planner.h"
#include "metal_runtime_capabilities.h"
#include "prepared_binding_plan.h"
#include "runtime_state.h"

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

} // namespace vernon::runtime

#endif
