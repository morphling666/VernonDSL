#ifndef VERNON_RUNTIME_PIPELINE_BACKEND_H
#define VERNON_RUNTIME_PIPELINE_BACKEND_H

#include "backend_stage_pipeline.h"
#include "compute_launch_planner.h"
#include "graphics_invocation_planner.h"
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
VernonStatus invokeCpuComputePipeline(VernonStageExecutable &pipeline, const PlannedComputeLaunch &plan);

BackendPipelineResult resolveCudaPipeline(BackendStageBuildInputs &inputs, const StageBindingPlan &plan,
                                          VernonStageExecutable &pipeline);
void destroyCudaPipeline(VernonStageExecutable &pipeline);
VernonStatus invokeCudaComputePipeline(VernonStageExecutable &pipeline, const PlannedComputeLaunch &plan);

void destroyVulkanPipeline(VernonStageExecutable &pipeline);
BackendPipelineResult resolveVulkanPipeline(BackendStageBuildInputs &inputs, const StageBindingPlan &plan,
                                            VernonStageExecutable &pipeline);
VernonStatus invokeVulkanGraphicsPipeline(VernonStageExecutable &pipeline,
                                          const VernonStageInvocationDescriptor &invocation,
                                          const PlannedGraphicsInvocation &plan);
VernonStatus invokeVulkanComputePipeline(VernonStageExecutable &pipeline, const PlannedComputeLaunch &plan);

void destroyDirectX12Pipeline(VernonStageExecutable &pipeline);
BackendPipelineResult resolveDirectX12Pipeline(BackendStageBuildInputs &inputs, const StageBindingPlan &plan,
                                               VernonStageExecutable &pipeline);
VernonStatus invokeDirectX12GraphicsPipeline(VernonStageExecutable &pipeline,
                                             const VernonStageInvocationDescriptor &invocation,
                                             const PlannedGraphicsInvocation &plan);
VernonStatus invokeDirectX12ComputePipeline(VernonStageExecutable &pipeline, const PlannedComputeLaunch &plan);

void destroyMetalPipeline(VernonStageExecutable &pipeline);
BackendPipelineResult resolveMetalPipeline(BackendStageBuildInputs &inputs, const StageBindingPlan &plan,
                                           VernonStageExecutable &pipeline);
VernonStatus invokeMetalGraphicsPipeline(VernonStageExecutable &pipeline,
                                         const VernonStageInvocationDescriptor &invocation,
                                         const PlannedGraphicsInvocation &plan);
VernonStatus invokeMetalComputePipeline(VernonStageExecutable &pipeline, const PlannedComputeLaunch &plan);

void destroyOpenGLPipeline(VernonStageExecutable &pipeline);
BackendPipelineResult resolveOpenGLPipeline(BackendStageBuildInputs &inputs, const StageBindingPlan &plan,
                                            VernonStageExecutable &pipeline);
VernonStatus invokeOpenGLGraphicsPipeline(VernonStageExecutable &pipeline,
                                          const VernonStageInvocationDescriptor &invocation,
                                          const PlannedGraphicsInvocation &plan);
VernonStatus invokeOpenGLComputePipeline(VernonStageExecutable &pipeline, const PlannedComputeLaunch &plan);

} // namespace vernon::runtime

#endif
