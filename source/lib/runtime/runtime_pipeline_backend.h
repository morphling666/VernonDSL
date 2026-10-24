#ifndef VERNON_RUNTIME_PIPELINE_BACKEND_H
#define VERNON_RUNTIME_PIPELINE_BACKEND_H

#include "backend_stage_pipeline.h"
#include "compute_launch_planner.h"
#include "graphics_invocation_planner.h"
#include "pipeline_groups.h"
#include "prepared_binding_plan.h"
#include "runtime_state.h"

namespace vernon::runtime {

inline BackendPipelineResult backendPipelineResolutionFailure(BackendStageBuildInputs &inputs) {
    std::string diagnostic;
    if (inputs.context)
        diagnostic.swap(invocationDiagnostic(*inputs.context));
    return BackendPipelineResult{
        vernon::err(BackendPipelineFailure{BackendPipelineError::BackendResolutionFailed, std::move(diagnostic)})};
}

BackendPipelineResult resolveCpuPipeline(BackendStageBuildInputs &inputs, const StageBindingPlan &plan,
                                         VernonStageExecutable &pipeline);
void destroyCpuPipeline(VernonStageExecutable &pipeline);
// Compute invocations dispatch on `commands`, whose transient groups `transient` keeps.
VernonStatus invokeCpuComputePipeline(VernonStageExecutable &pipeline, VernonRuntimeCoreCommands *commands,
                                      GroupStore &transient, const PlannedComputeLaunch &plan,
                                      const ArgumentOwner *owners);

BackendPipelineResult resolveCudaPipeline(BackendStageBuildInputs &inputs, const StageBindingPlan &plan,
                                          VernonStageExecutable &pipeline);
VernonStatus invokeCudaComputePipeline(VernonStageExecutable &pipeline, VernonRuntimeCoreCommands *commands,
                                       GroupStore &transient, const PlannedComputeLaunch &plan,
                                       const ArgumentOwner *owners);

BackendPipelineResult resolveVulkanPipeline(BackendStageBuildInputs &inputs, const StageBindingPlan &plan,
                                            VernonStageExecutable &pipeline);
VernonStatus invokeVulkanGraphicsPipeline(VernonStageExecutable &pipeline, VernonRuntimeCoreCommands *commands,
                                          const GraphicsCommand &command, GraphicsStream &stream);
VernonStatus invokeVulkanComputePipeline(VernonStageExecutable &pipeline, VernonRuntimeCoreCommands *commands,
                                         GroupStore &transient, const PlannedComputeLaunch &plan,
                                         const ArgumentOwner *owners);

BackendPipelineResult resolveDirectX12Pipeline(BackendStageBuildInputs &inputs, const StageBindingPlan &plan,
                                               VernonStageExecutable &pipeline);
VernonStatus invokeDirectX12GraphicsPipeline(VernonStageExecutable &pipeline, VernonRuntimeCoreCommands *commands,
                                             const GraphicsCommand &command, GraphicsStream &stream);
VernonStatus invokeDirectX12ComputePipeline(VernonStageExecutable &pipeline, VernonRuntimeCoreCommands *commands,
                                            GroupStore &transient, const PlannedComputeLaunch &plan,
                                            const ArgumentOwner *owners);

BackendPipelineResult resolveMetalPipeline(BackendStageBuildInputs &inputs, const StageBindingPlan &plan,
                                           VernonStageExecutable &pipeline);
VernonStatus invokeMetalGraphicsPipeline(VernonStageExecutable &pipeline, VernonRuntimeCoreCommands *commands,
                                         const GraphicsCommand &command, GraphicsStream &stream);
VernonStatus invokeMetalComputePipeline(VernonStageExecutable &pipeline, VernonRuntimeCoreCommands *commands,
                                        GroupStore &transient, const PlannedComputeLaunch &plan,
                                        const ArgumentOwner *owners);

BackendPipelineResult resolveOpenGLPipeline(BackendStageBuildInputs &inputs, const StageBindingPlan &plan,
                                            VernonStageExecutable &pipeline);
VernonStatus invokeOpenGLGraphicsPipeline(VernonStageExecutable &pipeline, VernonRuntimeCoreCommands *commands,
                                          const GraphicsCommand &command, GraphicsStream &stream);
VernonStatus invokeOpenGLComputePipeline(VernonStageExecutable &pipeline, VernonRuntimeCoreCommands *commands,
                                         GroupStore &transient, const PlannedComputeLaunch &plan,
                                         const ArgumentOwner *owners);

} // namespace vernon::runtime

#endif
