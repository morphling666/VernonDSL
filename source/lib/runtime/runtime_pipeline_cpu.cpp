#include "runtime_pipeline_backend.h"

#include "backend_cpu.h"
#include "compute_launch_planner.h"

#include <memory>
#include <utility>

namespace vernon::runtime {
namespace {

VernonStatus fail(VernonRuntimeContext &context, std::string error,
                  VernonStatus status = VERNON_STATUS_INVALID_ARGUMENT) {
    invocationDiagnostic(context) = std::move(error);
    return status;
}

} // namespace

bool resolveCpuPipeline(VernonPipelineBundle &bundle, const Variant &variant, VernonLoadedPipeline &pipeline) {
    auto state = std::make_unique<CpuPipelineState>();
    CpuKernelState kernel;
    ReflectedEntry reflection;
    if (!loadCpuNativeArtifact(*bundle.stages.at(variant.compute).cpuArtifact, kernel, reflection,
                               invocationDiagnostic(*bundle.context)) ||
        !prepareCpuComputePipeline(*bundle.context, std::move(kernel), std::move(reflection), *state))
        return false;
    installRuntimeBackendState(pipeline, state.release());
    return true;
}

void destroyCpuPipeline(VernonLoadedPipeline &pipeline) {
    CpuPipelineState &state = runtimeBackendState<CpuPipelineState>(pipeline);
    vernonRuntimeCoreBindingsDestroy(state.bindings);
    vernonRuntimeCorePipelineDestroy(state.pipeline);
}

VernonStatus invokeCpuComputePipeline(VernonLoadedPipeline &pipeline, const PlannedComputeLaunch &launch) {
    CpuPipelineState &state = runtimeBackendState<CpuPipelineState>(pipeline);
    for (size_t index = 0; index < state.layout.size(); ++index) {
        const auto &layout = state.layout[index];
        if (layout.argument_index >= launch.arguments.size())
            return fail(*pipeline.context, "CPU prepared argument index is invalid");
        const ComputeLaunchArgument &argument = launch.arguments[layout.argument_index];
        auto &value = state.values[index];
        value = {};
        value.slot = layout.slot;
        value.kind = layout.kind;
        if (layout.kind == VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER) {
            const auto *tensor = std::get_if<ComputeTensorArgument>(&argument);
            if (!tensor || !tensor->hostData || !tensor->hostSize)
                return fail(*pipeline.context, "CPU storage binding requires a host Tensor");
            value.payload.buffer.resource.identity = cpuProviderResourceIdentity(*pipeline.context);
            value.payload.buffer.resource.resource.value = reinterpret_cast<uintptr_t>(tensor->hostData);
            value.payload.buffer.resource.size = tensor->hostSize;
        } else {
            const auto *scalar = std::get_if<ComputeScalarArgument>(&argument);
            const auto *tensor = std::get_if<ComputeTensorArgument>(&argument);
            if (scalar && scalar->data && scalar->size) {
                value.payload.inline_value.data = scalar->data;
                value.payload.inline_value.size = scalar->size;
            } else if (tensor && tensor->tensorViewData && tensor->tensorViewSize == layout.element_size) {
                value.payload.inline_value.data = tensor->tensorViewData;
                value.payload.inline_value.size = tensor->tensorViewSize;
            } else {
                return fail(*pipeline.context, "CPU inline binding requires host data");
            }
        }
    }
    VernonStatus status =
        state.bindings ? vernonRuntimeCoreUpdateBindings(state.bindings, state.values.data(), state.values.size())
                       : vernonRuntimeCoreCreateBindings(state.pipeline, state.values.data(), state.values.size(),
                                                         &state.bindings);
    if (status != VERNON_STATUS_OK) {
        const VernonStringView providerError = cpuProviderLastError(*pipeline.context);
        return fail(*pipeline.context,
                    providerError.data ? std::string(providerError.data, providerError.size)
                                       : "failed to prepare CPU provider bindings",
                    status);
    }
    const uint32_t groups[3]{launch.grid.x, launch.grid.y, launch.grid.z};
    status = vernonRuntimeCoreEncodeDispatch(state.pipeline, state.bindings, launch.commandEncoder, groups, nullptr, 0);
    if (status != VERNON_STATUS_OK) {
        const VernonStringView providerError = cpuProviderLastError(*pipeline.context);
        return fail(*pipeline.context,
                    providerError.data ? std::string(providerError.data, providerError.size)
                                       : "failed to encode CPU provider dispatch",
                    status);
    }
    return VERNON_STATUS_OK;
}

} // namespace vernon::runtime
