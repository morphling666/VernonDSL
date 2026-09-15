#ifndef VERNON_RUNTIME_PIPELINE_BACKEND_H
#define VERNON_RUNTIME_PIPELINE_BACKEND_H

#include "backend_stage_pipeline.h"
#include "compute_launch_planner.h"
#include "graphics_invocation_planner.h"
#include "prepared_binding_plan.h"
#include "runtime_state.h"

#include <optional>
#include <string_view>
#include <vector>

namespace vernon::runtime {

inline BackendPipelineResult backendPipelineResolutionFailure(BackendStageBuildInputs &inputs) {
    std::string diagnostic;
    if (inputs.context)
        diagnostic.swap(invocationDiagnostic(*inputs.context));
    return BackendPipelineResult{
        vernon::err(BackendPipelineFailure{BackendPipelineError::BackendResolutionFailed, std::move(diagnostic)})};
}

inline std::optional<VernonRuntimeProviderBindingKind> providerBindingKindForTransport(std::string_view transport) {
    if (transport == "storage_buffer")
        return VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER;
    if (transport == "uniform_buffer")
        return VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER;
    if (transport == "push_constant")
        return VERNON_RUNTIME_PROVIDER_INLINE_VALUE;
    return std::nullopt;
}

inline void configureComputeValueStorage(const ParameterUse &use, VernonRuntimeProviderBindingLayoutEntry &layout) {
    if (layout.kind == VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER && use.interfaceKind == "value" &&
        !use.tensorViewDescriptor)
        layout.interface_kind = VERNON_RUNTIME_PROVIDER_INTERFACE_UNIFORM;
}

inline bool bindComputeValueStorage(const VernonRuntimeProviderBindingLayoutEntry &layout,
                                    const ComputeLaunchArgument &argument, VernonRuntimeProviderBindingValue &value) {
    if (layout.kind != VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER ||
        layout.interface_kind != VERNON_RUNTIME_PROVIDER_INTERFACE_UNIFORM)
        return false;
    const auto *scalar = std::get_if<ComputeScalarArgument>(&argument);
    if (!scalar || !scalar->data || scalar->size != layout.element_size)
        return false;
    value.payload.inline_value.data = scalar->data;
    value.payload.inline_value.size = scalar->size;
    return true;
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
