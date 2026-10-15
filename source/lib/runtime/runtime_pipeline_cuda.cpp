#include "runtime_pipeline_backend.h"

#include "compute_launch_planner.h"

#if defined(VERNON_HAS_CUDA_RUNTIME)
#include "backend_cuda.h"

#include <memory>
#include <string>
#include <variant>
#include <vector>
#endif

namespace vernon::runtime {

#if defined(VERNON_HAS_CUDA_RUNTIME)
namespace {

// CUDA passes packed host values as host-storage buffers, so storage bindings accept either an RHI Tensor or bytes.
bool fillCudaComputeValues(const PreparedComputeBindingPlan &plan, const StageBindingPlan &,
                           const PlannedComputeLaunch &launch, std::vector<VernonRuntimeProviderBindingValue> &values,
                           std::string &error) {
    values.resize(plan.size());
    for (size_t index = 0; index < plan.size(); ++index) {
        const auto &layout = plan.layouts[index];
        const PreparedBindingSource &preparedSource = plan.sources[index];
        auto &value = values[index];
        value = {};
        value.slot = layout.slot;
        value.kind = layout.kind;
        if (preparedSource.metadataCarrier) {
            if (bindComputeMetadataCarrier(layout, preparedSource, launch, value))
                continue;
            error = "metadata carrier payload does not match its compiled ABI";
            return false;
        }
        if (preparedSource.argumentIndex >= launch.arguments.size()) {
            error = "prepared argument index is invalid";
            return false;
        }
        const ComputeLaunchArgument &argument = launch.arguments[preparedSource.argumentIndex];
        const std::string argumentName = "#" + std::to_string(layout.argument_index);
        if (layout.kind == VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER) {
            const auto *tensor = std::get_if<ComputeTensorArgument>(&argument);
            const auto *packed = std::get_if<ComputeScalarArgument>(&argument);
            if (tensor && tensor->resource.resource.value) {
                value.payload.buffer.resource = tensor->resource;
                if (value.payload.buffer.resource.offset > value.payload.buffer.resource.size ||
                    preparedSource.resourceOffset >
                        value.payload.buffer.resource.size - value.payload.buffer.resource.offset) {
                    error = "aggregate storage leaf exceeds its Tensor resource";
                    return false;
                }
                value.payload.buffer.resource.offset += preparedSource.resourceOffset;
            } else if (packed && packed->data && packed->size) {
                value.flags = VERNON_RUNTIME_PROVIDER_BINDING_HOST_STORAGE;
                value.payload.inline_value.data = packed->data;
                value.payload.inline_value.size = packed->size;
            } else {
                error = "prepared storage binding for argument " + argumentName +
                        " requires an RHI Tensor or packed host value";
                return false;
            }
            continue;
        }
        const auto *scalar = std::get_if<ComputeScalarArgument>(&argument);
        if (!scalar) {
            error = "prepared inline binding for argument " + argumentName + " received a resource value";
            return false;
        }
        if (!scalar->data || !scalar->size) {
            error = "prepared inline binding for argument " + argumentName + " has no packed host data";
            return false;
        }
        value.payload.inline_value.data = scalar->data;
        value.payload.inline_value.size = scalar->size;
    }
    return true;
}

constexpr RhiComputeTarget kCudaCompute{"CUDA", "ptx", fillCudaComputeValues};

} // namespace
#endif

BackendPipelineResult resolveCudaPipeline(BackendStageBuildInputs &inputs, const StageBindingPlan &plan,
                                          VernonStageExecutable &pipeline) {
    const auto resolve = [&]() {
#if defined(VERNON_HAS_CUDA_RUNTIME)
        auto state = std::make_unique<CudaPipelineState>();
        if (!prepareRhiComputePipeline(*inputs.context, cudaState(*inputs.context).adapter, kCudaCompute,
                                       inputs.artifacts.at(plan.compute), plan, state->rhiCompute))
            return false;
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

VernonStatus invokeCudaComputePipeline(VernonStageExecutable &pipeline, const PlannedComputeLaunch &launch) {
#if defined(VERNON_HAS_CUDA_RUNTIME)
    return invokeRhiComputePipeline(*pipeline.context, cudaState(*pipeline.context).adapter, kCudaCompute,
                                    pipeline.bindingProjection, launch,
                                    runtimeBackendState<CudaPipelineState>(pipeline).rhiCompute);
#else
    (void)pipeline;
    (void)launch;
    return VERNON_STATUS_UNSUPPORTED_TARGET;
#endif
}

} // namespace vernon::runtime
