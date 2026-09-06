#ifndef VERNON_RUNTIME_PIPELINE_BACKEND_H
#define VERNON_RUNTIME_PIPELINE_BACKEND_H

#include "backend_stage_pipeline.h"
#include "compute_launch_planner.h"
#include "graphics_invocation_planner.h"
#include "runtime_state.h"

#include <string_view>
#include <vector>

namespace vernon::runtime {

struct OpenGLNativeUniformShape {
    uint32_t scalarCount{};
    uint32_t matrixColumns{1};
};

bool resolveOpenGLNativeUniformShape(std::string_view dtype, const std::vector<uint64_t> &shape,
                                     OpenGLNativeUniformShape &result);

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

bool resolveCpuPipeline(BackendPipelineBundle &bundle, const Variant &variant, VernonStageExecutable &pipeline);
void destroyCpuPipeline(VernonStageExecutable &pipeline);
VernonStatus invokeCpuComputePipeline(VernonStageExecutable &pipeline, const PlannedComputeLaunch &plan);

bool resolveCudaPipeline(BackendPipelineBundle &bundle, const Variant &variant, VernonStageExecutable &pipeline);
void destroyCudaPipeline(VernonStageExecutable &pipeline);
VernonStatus invokeCudaComputePipeline(VernonStageExecutable &pipeline, const PlannedComputeLaunch &plan);

void destroyVulkanPipeline(VernonStageExecutable &pipeline);
bool resolveVulkanPipeline(BackendPipelineBundle &bundle, const Variant &variant, VernonStageExecutable &pipeline);
VernonStatus invokeVulkanGraphicsPipeline(VernonStageExecutable &pipeline,
                                          const VernonStageInvocationDescriptor &invocation,
                                          const PlannedGraphicsInvocation &plan);
VernonStatus invokeVulkanComputePipeline(VernonStageExecutable &pipeline, const PlannedComputeLaunch &plan);

void destroyDirectX12Pipeline(VernonStageExecutable &pipeline);
bool resolveDirectX12Pipeline(BackendPipelineBundle &bundle, const Variant &variant, VernonStageExecutable &pipeline);
VernonStatus invokeDirectX12GraphicsPipeline(VernonStageExecutable &pipeline,
                                             const VernonStageInvocationDescriptor &invocation,
                                             const PlannedGraphicsInvocation &plan);
VernonStatus invokeDirectX12ComputePipeline(VernonStageExecutable &pipeline, const PlannedComputeLaunch &plan);

void destroyMetalPipeline(VernonStageExecutable &pipeline);
bool resolveMetalPipeline(BackendPipelineBundle &bundle, const Variant &variant, VernonStageExecutable &pipeline);
VernonStatus invokeMetalGraphicsPipeline(VernonStageExecutable &pipeline,
                                         const VernonStageInvocationDescriptor &invocation,
                                         const PlannedGraphicsInvocation &plan);
VernonStatus invokeMetalComputePipeline(VernonStageExecutable &pipeline, const PlannedComputeLaunch &plan);

void destroyOpenGLPipeline(VernonStageExecutable &pipeline);
bool resolveOpenGLPipeline(BackendPipelineBundle &bundle, const Variant &variant, VernonStageExecutable &pipeline);
VernonStatus invokeOpenGLGraphicsPipeline(VernonStageExecutable &pipeline,
                                          const VernonStageInvocationDescriptor &invocation,
                                          const PlannedGraphicsInvocation &plan);
VernonStatus invokeOpenGLComputePipeline(VernonStageExecutable &pipeline, const PlannedComputeLaunch &plan);

} // namespace vernon::runtime

#endif
