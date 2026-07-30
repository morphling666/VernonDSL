#ifndef VERNON_RUNTIME_PIPELINE_BACKEND_H
#define VERNON_RUNTIME_PIPELINE_BACKEND_H

#include "compute_launch_planner.h"
#include "graphics_invocation_planner.h"
#include "runtime_state.h"

namespace vernon::runtime {

bool resolveCpuPipeline(VernonPipelineBundle &bundle, const Variant &variant, VernonLoadedPipeline &pipeline);
void destroyCpuPipeline(VernonLoadedPipeline &pipeline);
VernonStatus invokeCpuComputePipeline(VernonLoadedPipeline &pipeline, const PlannedComputeLaunch &plan);

bool resolveCudaPipeline(VernonPipelineBundle &bundle, const Variant &variant, VernonLoadedPipeline &pipeline);
void destroyCudaPipeline(VernonLoadedPipeline &pipeline);
VernonStatus invokeCudaComputePipeline(VernonLoadedPipeline &pipeline, const PlannedComputeLaunch &plan);

void destroyVulkanPipeline(VernonLoadedPipeline &pipeline);
bool resolveVulkanPipeline(VernonPipelineBundle &bundle, const Variant &variant, VernonLoadedPipeline &pipeline);
VernonStatus invokeVulkanGraphicsPipeline(VernonLoadedPipeline &pipeline, const VernonPipelineInvocation &invocation,
                                          const PlannedGraphicsInvocation &plan);
VernonStatus invokeVulkanComputePipeline(VernonLoadedPipeline &pipeline, const PlannedComputeLaunch &plan);

void destroyDirectX12Pipeline(VernonLoadedPipeline &pipeline);
bool resolveDirectX12Pipeline(VernonPipelineBundle &bundle, const Variant &variant, VernonLoadedPipeline &pipeline);
VernonStatus invokeDirectX12GraphicsPipeline(VernonLoadedPipeline &pipeline, const VernonPipelineInvocation &invocation,
                                             const PlannedGraphicsInvocation &plan);
VernonStatus invokeDirectX12ComputePipeline(VernonLoadedPipeline &pipeline, const PlannedComputeLaunch &plan);

void destroyOpenGLPipeline(VernonLoadedPipeline &pipeline);
bool resolveOpenGLPipeline(VernonPipelineBundle &bundle, const Variant &variant, VernonLoadedPipeline &pipeline);
VernonStatus invokeOpenGLGraphicsPipeline(VernonLoadedPipeline &pipeline, const VernonPipelineInvocation &invocation,
                                          const PlannedGraphicsInvocation &plan);
VernonStatus invokeOpenGLComputePipeline(VernonLoadedPipeline &pipeline, const PlannedComputeLaunch &plan);

} // namespace vernon::runtime

#endif
