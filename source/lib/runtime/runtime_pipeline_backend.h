#ifndef VERNON_RUNTIME_PIPELINE_BACKEND_H
#define VERNON_RUNTIME_PIPELINE_BACKEND_H

#include "backend_stage_pipeline.h"
#include "compute_launch_planner.h"
#include "graphics_invocation_planner.h"
#include "pipeline_metadata.h"
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

inline bool prepareComputeBindingPlan(VernonRuntimeContext &context, const LoadedStageArtifact &stage,
                                      const StageBindingPlan &plan, PreparedComputeBindingPlan &output) {
    std::string &diagnostic = invocationDiagnostic(context);
    ReflectedEntry reflection;
    return resolveStageReflection(stage, context.backend, reflection, diagnostic) &&
           buildPreparedComputeBindingPlan(plan, reflection, context.backend, output, diagnostic);
}

BackendPipelineResult resolveCudaPipeline(BackendStageBuildInputs &inputs, const StageBindingPlan &plan,
                                          VernonStageExecutable &pipeline);
BackendPipelineResult resolveVulkanPipeline(BackendStageBuildInputs &inputs, const StageBindingPlan &plan,
                                            VernonStageExecutable &pipeline);
BackendPipelineResult resolveDirectX12Pipeline(BackendStageBuildInputs &inputs, const StageBindingPlan &plan,
                                               VernonStageExecutable &pipeline);
BackendPipelineResult resolveMetalPipeline(BackendStageBuildInputs &inputs, const StageBindingPlan &plan,
                                           VernonStageExecutable &pipeline);
BackendPipelineResult resolveOpenGLPipeline(BackendStageBuildInputs &inputs, const StageBindingPlan &plan,
                                            VernonStageExecutable &pipeline);

} // namespace vernon::runtime

#endif
