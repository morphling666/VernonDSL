#ifndef VERNON_RUNTIME_PIPELINE_BACKEND_H
#define VERNON_RUNTIME_PIPELINE_BACKEND_H

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

bool resolveCpuPipeline(VernonProgramBundle &bundle, const Variant &variant, VernonProgramExecutable &pipeline);
void destroyCpuPipeline(VernonProgramExecutable &pipeline);
VernonStatus invokeCpuComputePipeline(VernonProgramExecutable &pipeline, const PlannedComputeLaunch &plan);

bool resolveCudaPipeline(VernonProgramBundle &bundle, const Variant &variant, VernonProgramExecutable &pipeline);
void destroyCudaPipeline(VernonProgramExecutable &pipeline);
VernonStatus invokeCudaComputePipeline(VernonProgramExecutable &pipeline, const PlannedComputeLaunch &plan);

void destroyVulkanPipeline(VernonProgramExecutable &pipeline);
bool resolveVulkanPipeline(VernonProgramBundle &bundle, const Variant &variant, VernonProgramExecutable &pipeline);
VernonStatus invokeVulkanGraphicsPipeline(VernonProgramExecutable &pipeline,
                                          const VernonProgramSubmitDescriptor &invocation,
                                          const PlannedGraphicsInvocation &plan);
VernonStatus invokeVulkanComputePipeline(VernonProgramExecutable &pipeline, const PlannedComputeLaunch &plan);

void destroyDirectX12Pipeline(VernonProgramExecutable &pipeline);
bool resolveDirectX12Pipeline(VernonProgramBundle &bundle, const Variant &variant, VernonProgramExecutable &pipeline);
VernonStatus invokeDirectX12GraphicsPipeline(VernonProgramExecutable &pipeline,
                                             const VernonProgramSubmitDescriptor &invocation,
                                             const PlannedGraphicsInvocation &plan);
VernonStatus invokeDirectX12ComputePipeline(VernonProgramExecutable &pipeline, const PlannedComputeLaunch &plan);

void destroyMetalPipeline(VernonProgramExecutable &pipeline);
bool resolveMetalPipeline(VernonProgramBundle &bundle, const Variant &variant, VernonProgramExecutable &pipeline);
VernonStatus invokeMetalGraphicsPipeline(VernonProgramExecutable &pipeline,
                                         const VernonProgramSubmitDescriptor &invocation,
                                         const PlannedGraphicsInvocation &plan);
VernonStatus invokeMetalComputePipeline(VernonProgramExecutable &pipeline, const PlannedComputeLaunch &plan);

void destroyOpenGLPipeline(VernonProgramExecutable &pipeline);
bool resolveOpenGLPipeline(VernonProgramBundle &bundle, const Variant &variant, VernonProgramExecutable &pipeline);
VernonStatus invokeOpenGLGraphicsPipeline(VernonProgramExecutable &pipeline,
                                          const VernonProgramSubmitDescriptor &invocation,
                                          const PlannedGraphicsInvocation &plan);
VernonStatus invokeOpenGLComputePipeline(VernonProgramExecutable &pipeline, const PlannedComputeLaunch &plan);

} // namespace vernon::runtime

#endif
