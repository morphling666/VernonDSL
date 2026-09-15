#include "runtime_pipeline_backend.h"

#include "compute_launch_planner.h"

#if defined(VERNON_HAS_CUDA_RUNTIME)
#include "VernonRuntimeRHIAdapter.h"
#include "backend_cuda.h"

#include <algorithm>
#include <memory>
#include <utility>
#endif

namespace vernon::runtime {

#if defined(VERNON_HAS_CUDA_RUNTIME)
namespace {

VernonStatus fail(VernonRuntimeContext &context, std::string error,
                  VernonStatus status = VERNON_STATUS_INVALID_ARGUMENT) {
    invocationDiagnostic(context) = std::move(error);
    return status;
}

} // namespace
#endif

BackendPipelineResult resolveCudaPipeline(BackendStageBuildInputs &inputs, const StageBindingPlan &plan,
                                          VernonStageExecutable &pipeline) {
    const auto resolve = [&]() {
#if defined(VERNON_HAS_CUDA_RUNTIME)
        auto state = std::make_unique<CudaPipelineState>();
        const LoadedStageArtifact &stage = inputs.artifacts.at(plan.compute);
        ReflectedEntry reflection;
        if (!resolveStageReflection(stage, VERNON_RUNTIME_CUDA, reflection, invocationDiagnostic(*inputs.context)))
            return false;
        if (!buildPreparedComputeBindingPlan(plan, reflection, VERNON_RUNTIME_CUDA, state->bindingPlan,
                                             invocationDiagnostic(*inputs.context)))
            return false;
        state->values.resize(state->bindingPlan.size());
        state->descriptorValues.resize(state->bindingPlan.size());
        std::copy_n(stage.workgroup, 3, state->workgroup);
        const VernonRuntimeProviderShaderDescriptor shaderDescriptor{sizeof(VernonRuntimeProviderShaderDescriptor),
                                                                     VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE,
                                                                     {"ptx", 3},
                                                                     stage.source.data(),
                                                                     stage.source.size(),
                                                                     {stage.entry.data(), stage.entry.size()},
                                                                     {nullptr, 0},
                                                                     {0, 0, 0, 0}};
        VernonRuntimeCorePipelineDescriptor descriptor{};
        descriptor.struct_size = sizeof(descriptor);
        descriptor.kind = VERNON_RUNTIME_PROVIDER_COMPUTE_PIPELINE;
        descriptor.required_capabilities = VERNON_RUNTIME_PROVIDER_COMPUTE;
        descriptor.shaders = &shaderDescriptor;
        descriptor.shader_count = 1;
        descriptor.bindings = state->bindingPlan.layouts.data();
        descriptor.binding_count = state->bindingPlan.layouts.size();
        std::copy_n(state->workgroup, 3, descriptor.workgroup_size);
        const VernonStatus status = vernonRuntimeCorePreparePipeline(
            vernonRuntimeRhiAdapterGetProvider(cudaState(*inputs.context).adapter), &descriptor, &state->pipeline);
        if (status != VERNON_STATUS_OK) {
            const VernonStringView error = vernonRuntimeRhiAdapterGetLastError(cudaState(*inputs.context).adapter);
            invocationDiagnostic(*inputs.context) =
                error.data ? std::string(error.data, error.size) : "failed to prepare CUDA provider pipeline";
            return false;
        }
        installRuntimeBackendState(pipeline, state.release());
        return true;
#else
        (void)inputs;
        (void)plan;
        (void)pipeline;
        return false;
#endif
    };
    return resolve() ? BackendPipelineResult{vernon::ok()} : backendPipelineResolutionFailure(inputs);
}

void destroyCudaPipeline(VernonStageExecutable &pipeline) {
#if defined(VERNON_HAS_CUDA_RUNTIME)
    CudaPipelineState &state = runtimeBackendState<CudaPipelineState>(pipeline);
    vernonRuntimeCoreBindingsDestroy(state.bindings);
    vernonRuntimeCorePipelineDestroy(state.pipeline);
#else
    (void)pipeline;
#endif
}

VernonStatus invokeCudaComputePipeline(VernonStageExecutable &pipeline, const PlannedComputeLaunch &launch) {
#if defined(VERNON_HAS_CUDA_RUNTIME)
    CudaPipelineState &state = runtimeBackendState<CudaPipelineState>(pipeline);
    for (size_t index = 0; index < state.bindingPlan.size(); ++index) {
        const auto &layout = state.bindingPlan.layouts[index];
        if (layout.argument_index >= launch.arguments.size())
            return fail(*pipeline.context, "CUDA prepared argument index is invalid");
        const ComputeLaunchArgument &argument = launch.arguments[layout.argument_index];
        auto &value = state.values[index];
        value = {};
        value.slot = layout.slot;
        value.kind = layout.kind;
        const PreparedBindingSource &preparedSource = state.bindingPlan.sources[index];
        const ComputeBindingSource &source = preparedSource.source;
        if (source.kind != ComputeBindingSourceKind::Argument) {
            std::optional<int64_t> descriptor = computeBindingDescriptorValue(argument, source);
            if (!descriptor)
                return fail(*pipeline.context, "CUDA TensorView descriptor value is invalid");
            state.descriptorValues[index] = *descriptor;
            value.payload.inline_value.data = &state.descriptorValues[index];
            value.payload.inline_value.size = sizeof(int64_t);
            continue;
        }
        if (layout.kind == VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER) {
            const auto *tensor = std::get_if<ComputeTensorArgument>(&argument);
            const auto *packed = std::get_if<ComputeScalarArgument>(&argument);
            if (tensor && tensor->resource.resource.value) {
                value.payload.buffer.resource = tensor->resource;
                if (preparedSource.resourceOffset > value.payload.buffer.resource.size)
                    return fail(*pipeline.context, "CUDA aggregate storage leaf exceeds its Tensor resource");
                value.payload.buffer.resource.offset += preparedSource.resourceOffset;
                value.payload.buffer.resource.size -= preparedSource.resourceOffset;
            } else if (packed && packed->data && packed->size) {
                value.flags = VERNON_RUNTIME_PROVIDER_BINDING_HOST_STORAGE;
                value.payload.inline_value.data = packed->data;
                value.payload.inline_value.size = packed->size;
            } else {
                return fail(*pipeline.context, "CUDA prepared storage binding for argument #" +
                                                   std::to_string(layout.argument_index) +
                                                   " requires an RHI Tensor or packed host value");
            }
        } else {
            const auto *scalar = std::get_if<ComputeScalarArgument>(&argument);
            if (!scalar)
                return fail(*pipeline.context, "CUDA prepared inline binding for argument #" +
                                                   std::to_string(layout.argument_index) +
                                                   " received a resource value");
            if (!scalar->data || !scalar->size)
                return fail(*pipeline.context, "CUDA prepared inline binding for argument #" +
                                                   std::to_string(layout.argument_index) + " has no packed host data");
            value.payload.inline_value.data = scalar->data;
            value.payload.inline_value.size = scalar->size;
        }
    }
    VernonStatus status =
        state.bindings ? vernonRuntimeCoreUpdateBindings(state.bindings, state.values.data(), state.values.size())
                       : vernonRuntimeCoreCreateBindings(state.pipeline, state.values.data(), state.values.size(),
                                                         &state.bindings);
    if (status != VERNON_STATUS_OK) {
        const VernonStringView providerError =
            vernonRuntimeRhiAdapterGetLastError(cudaState(*pipeline.context).adapter);
        return fail(*pipeline.context,
                    providerError.data ? std::string(providerError.data, providerError.size)
                                       : "failed to prepare CUDA invocation bindings",
                    status);
    }
    const uint32_t groups[3]{launch.grid.x, launch.grid.y, launch.grid.z};
    status = vernonRuntimeCoreEncodeDispatch(state.pipeline, state.bindings, launch.commandEncoder, groups, nullptr, 0);
    if (status != VERNON_STATUS_OK) {
        const VernonStringView providerError =
            vernonRuntimeRhiAdapterGetLastError(cudaState(*pipeline.context).adapter);
        return fail(*pipeline.context,
                    providerError.data ? std::string(providerError.data, providerError.size)
                                       : "failed to encode CUDA provider dispatch",
                    status);
    }
    return VERNON_STATUS_OK;
#else
    (void)pipeline;
    (void)launch;
    return VERNON_STATUS_UNSUPPORTED_TARGET;
#endif
}

} // namespace vernon::runtime
