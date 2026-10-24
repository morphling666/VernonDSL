#ifndef VERNON_RUNTIME_RHI_PROVIDER_PIPELINE_H
#define VERNON_RUNTIME_RHI_PROVIDER_PIPELINE_H

#include "VernonRuntimeCore.h"
#include "compute_launch_planner.h"
#include "graphics_invocation_planner.h"
#include "pipeline_groups.h"
#include "prepared_binding_plan.h"
#include "stage_artifact.h"
#include "stage_binding_plan.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

struct VernonRuntimeContext;
struct VernonRuntimeRhiAdapter;

namespace vernon::runtime {

// Each Stage prepared on a RuntimeCore provider owns its provider objects and releases them on destruction.
struct RhiComputePipeline {
    RhiComputePipeline() = default;
    RhiComputePipeline(const RhiComputePipeline &) = delete;
    RhiComputePipeline &operator=(const RhiComputePipeline &) = delete;
    ~RhiComputePipeline();

    VernonRuntimeCorePipeline *pipeline{};
    PipelineGroups groups;
    PreparedComputeBindingPlan bindingPlan;
};

struct RhiGraphicsPipeline {
    RhiGraphicsPipeline() = default;
    RhiGraphicsPipeline(const RhiGraphicsPipeline &) = delete;
    RhiGraphicsPipeline &operator=(const RhiGraphicsPipeline &) = delete;
    ~RhiGraphicsPipeline();

    VernonRuntimeCorePipeline *pipeline{};
    PipelineGroups groups;
    PreparedGraphicsVariant variant;
    PreparedGraphicsBindingPlan bindingPlan;
};

using FillRhiComputeValue = bool (*)(const PreparedComputeBindingPlan &plan, const StageBindingPlan &projection,
                                     const PlannedComputeLaunch &launch, size_t slot,
                                     VernonRuntimeProviderBindingValue &value, std::string &error);
// Returns the provider's native attachment format, or nullopt when the provider cannot render to `format`.
using MapRhiAttachmentFormat = std::optional<uint32_t> (*)(VernonTextureFormat format);

// `name` prefixes diagnostics; `artifactKind` names the shader code handed to the provider.
struct RhiComputeTarget {
    std::string_view name;
    std::string_view artifactKind;
    FillRhiComputeValue fillValue{fillPreparedComputeBindingValue};
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
// Dispatches on `commands`, whose transient groups `transient` keeps. `owners`, when given, holds the owner of each
// Stage parameter's argument; otherwise the launch owns them all.
VernonStatus invokeRhiComputePipeline(VernonRuntimeContext &context, VernonRuntimeRhiAdapter *adapter,
                                      const RhiComputeTarget &target, const StageBindingPlan &projection,
                                      VernonRuntimeCoreCommands *commands, GroupStore &transient,
                                      const PlannedComputeLaunch &launch, const ArgumentOwner *owners,
                                      RhiComputePipeline &pipeline);

bool prepareRhiGraphicsPipeline(VernonRuntimeContext &context, VernonRuntimeRhiAdapter *adapter,
                                const RhiGraphicsTarget &target, const LoadedStageArtifact &vertex,
                                const LoadedStageArtifact &fragment, const StageBindingPlan &plan,
                                const PreparedGraphicsBindingOptions &options, RhiGraphicsPipeline &output);
// Encodes one draw of `stream` on `commands`, inside its rendering scope.
VernonStatus invokeRhiGraphicsPipeline(VernonRuntimeContext &context, VernonRuntimeRhiAdapter *adapter,
                                       const RhiGraphicsTarget &target, VernonRuntimeCoreCommands *commands,
                                       const GraphicsCommand &command, GraphicsStream &stream,
                                       RhiGraphicsPipeline &pipeline);

} // namespace vernon::runtime

#endif
