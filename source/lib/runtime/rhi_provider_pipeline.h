#ifndef VERNON_RUNTIME_RHI_PROVIDER_PIPELINE_H
#define VERNON_RUNTIME_RHI_PROVIDER_PIPELINE_H

#include "VernonRuntimeCore.h"
#include "compute_launch_planner.h"
#include "graphics_invocation_planner.h"
#include "prepared_binding_plan.h"
#include "stage_artifact.h"
#include "stage_binding_plan.h"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

struct VernonRuntimeContext;
struct VernonRuntimeRhiAdapter;

namespace vernon::runtime {

// Each Stage prepared on a RuntimeCore provider owns its provider objects and releases them on destruction. The binding
// set is created at the first invocation and updated in place afterwards; every invocation rewrites all values, so
// resolve never creates placeholder bindings.
struct RhiComputePipeline {
    RhiComputePipeline() = default;
    RhiComputePipeline(const RhiComputePipeline &) = delete;
    RhiComputePipeline &operator=(const RhiComputePipeline &) = delete;
    ~RhiComputePipeline();

    VernonRuntimeCorePipeline *pipeline{};
    VernonRuntimeCoreBindings *bindings{};
    PreparedComputeBindingPlan bindingPlan;
    std::vector<VernonRuntimeProviderBindingValue> values;
};

struct RhiGraphicsPipeline {
    RhiGraphicsPipeline() = default;
    RhiGraphicsPipeline(const RhiGraphicsPipeline &) = delete;
    RhiGraphicsPipeline &operator=(const RhiGraphicsPipeline &) = delete;
    ~RhiGraphicsPipeline();

    VernonRuntimeCorePipeline *pipeline{};
    VernonRuntimeCoreBindings *bindings{};
    PreparedGraphicsVariant variant;
    PreparedGraphicsBindingPlan bindingPlan;
    std::vector<VernonRuntimeProviderBindingValue> values;
};

using FillRhiComputeValues = bool (*)(const PreparedComputeBindingPlan &plan, const StageBindingPlan &projection,
                                      const PlannedComputeLaunch &launch,
                                      std::vector<VernonRuntimeProviderBindingValue> &values, std::string &error);
// Returns the provider's native attachment format, or nullopt when the provider cannot render to `format`.
using MapRhiAttachmentFormat = std::optional<uint32_t> (*)(VernonTextureFormat format);

// `name` prefixes diagnostics; `artifactKind` names the shader code handed to the provider.
struct RhiComputeTarget {
    std::string_view name;
    std::string_view artifactKind;
    FillRhiComputeValues fillValues{fillPreparedComputeBindingValues};
};

struct RhiGraphicsTarget {
    std::string_view name;
    std::string_view artifactKind;
    MapRhiAttachmentFormat colorFormat{};
    MapRhiAttachmentFormat depthFormat{};
};

// Remaps one prepared layout to the backend's native resource location before the pipeline is prepared.
using ResolveRhiComputeNativeBinding = bool (*)(const void *userData, const PreparedBindingSource &source,
                                                VernonRuntimeProviderBindingLayoutEntry &layout, std::string &error);

struct RhiComputeNativeBindingResolver {
    ResolveRhiComputeNativeBinding resolve{};
    const void *userData{};
};

bool prepareRhiComputePipeline(VernonRuntimeContext &context, VernonRuntimeRhiAdapter *adapter,
                               const RhiComputeTarget &target, const LoadedStageArtifact &stage,
                               const StageBindingPlan &plan, RhiComputePipeline &output,
                               RhiComputeNativeBindingResolver resolver = {});
VernonStatus invokeRhiComputePipeline(VernonRuntimeContext &context, VernonRuntimeRhiAdapter *adapter,
                                      const RhiComputeTarget &target, const StageBindingPlan &projection,
                                      const PlannedComputeLaunch &launch, RhiComputePipeline &pipeline);

bool prepareRhiGraphicsPipeline(VernonRuntimeContext &context, VernonRuntimeRhiAdapter *adapter,
                                const RhiGraphicsTarget &target, const LoadedStageArtifact &vertex,
                                const LoadedStageArtifact &fragment, const StageBindingPlan &plan,
                                const PreparedGraphicsBindingOptions &options, RhiGraphicsPipeline &output);
VernonStatus invokeRhiGraphicsPipeline(VernonRuntimeContext &context, VernonRuntimeRhiAdapter *adapter,
                                       const RhiGraphicsTarget &target,
                                       const VernonStageInvocationDescriptor &invocation,
                                       const PlannedGraphicsInvocation &plan, RhiGraphicsPipeline &pipeline);

} // namespace vernon::runtime

#endif
