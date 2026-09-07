#include "runtime_pipeline_backend.h"

#include "compute_launch_planner.h"

#if defined(VERNON_HAS_CUDA_RUNTIME)
#include "VernonRuntimeRHIAdapter.h"
#include "backend_cuda.h"

#include <nlohmann/json.hpp>

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

bool resolveCudaPipeline(BackendStageBuildInputs &inputs, const StageBindingPlan &plan,
                         VernonStageExecutable &pipeline) {
#if defined(VERNON_HAS_CUDA_RUNTIME)
    auto state = std::make_unique<CudaPipelineState>();
    const LoadedStageArtifact &stage = inputs.artifacts.at(plan.compute);
    ReflectedEntry reflection;
    if (!resolveStageReflection(stage, VERNON_RUNTIME_CUDA, reflection, invocationDiagnostic(*inputs.context)))
        return false;
    uint32_t internalSlot = 0;
    for (const Parameter &parameter : plan.parameters)
        internalSlot = std::max(internalSlot, parameter.slot);
    std::vector<uint32_t> argumentBindings(reflection.arguments.size());
    uint32_t flattenedBinding = 0;
    for (size_t index = 0; index < reflection.arguments.size(); ++index) {
        argumentBindings[index] = flattenedBinding;
        if (reflection.arguments[index].kind != "builtin")
            flattenedBinding +=
                static_cast<uint32_t>(std::max(reflection.arguments[index].storageLeaves.size(), size_t{1}));
    }
    std::vector<uint32_t> descriptorBindings(reflection.arguments.size(), UINT32_MAX);
    for (size_t index = 0; index < reflection.arguments.size(); ++index)
        if (reflection.arguments[index].tensorViewDescriptor) {
            descriptorBindings[index] = flattenedBinding;
            flattenedBinding += 1 + 2 * reflection.arguments[index].tensorViewDescriptor->rank;
        }
    struct Candidate {
        VernonRuntimeProviderBindingLayoutEntry layout{};
        ComputeBindingSource source;
    };
    std::vector<Candidate> candidates;
    for (const Parameter &parameter : plan.parameters)
        for (const ParameterUse &use : parameter.uses) {
            if (use.stage != "compute" && use.stage != plan.compute)
                continue;
            if (use.index >= reflection.arguments.size()) {
                invocationDiagnostic(*inputs.context) = "CUDA parameter use exceeds reflected argument table";
                return false;
            }
            const ReflectedArgument &argument = reflection.arguments[use.index];
            const size_t leafCount = std::max(argument.storageLeaves.size(), size_t{1});
            for (size_t leafIndex = 0; leafIndex < leafCount; ++leafIndex) {
                Candidate candidate;
                auto &binding = candidate.layout;
                binding.slot = leafIndex == 0 ? parameter.slot : ++internalSlot;
                binding.binding = argumentBindings[use.index] + static_cast<uint32_t>(leafIndex);
                binding.kind = use.interfaceKind == "storage" ? VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER
                                                              : VERNON_RUNTIME_PROVIDER_INLINE_VALUE;
                binding.stage_mask = VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE;
                binding.access = parameter.access == "read" ? 1u : parameter.access == "write" ? 2u : 3u;
                binding.array_count = 1;
                binding.argument_index = use.index;
                binding.element_size = static_cast<uint32_t>(
                    argument.storageLeaves.empty()
                        ? (binding.kind == VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER ? argument.tensorElementSize
                                                                                  : argument.physical.size)
                        : argument.storageLeaves[leafIndex].elementSize);
                if (!binding.element_size) {
                    invocationDiagnostic(*inputs.context) = "CUDA reflected argument has zero element size";
                    return false;
                }
                candidate.source = {ComputeBindingSourceKind::Argument, use.index, 0};
                candidates.push_back(candidate);
            }
            if (use.tensorViewDescriptor) {
                uint32_t descriptorBinding = descriptorBindings[use.index];
                const auto addDescriptor = [&](ComputeBindingSourceKind kind, uint32_t dimension, uint32_t binding) {
                    Candidate candidate;
                    candidate.layout.slot = ++internalSlot;
                    candidate.layout.binding = binding;
                    candidate.layout.kind = VERNON_RUNTIME_PROVIDER_INLINE_VALUE;
                    candidate.layout.stage_mask = VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE;
                    candidate.layout.array_count = 1;
                    candidate.layout.argument_index = use.index;
                    candidate.layout.element_size = 8;
                    candidate.source = {kind, use.index, dimension};
                    candidates.push_back(candidate);
                };
                addDescriptor(ComputeBindingSourceKind::TensorOffset, 0, descriptorBinding++);
                for (uint32_t dimension = 0; dimension < use.tensorViewDescriptor->rank; ++dimension)
                    addDescriptor(ComputeBindingSourceKind::TensorExtent, dimension, descriptorBinding++);
                for (uint32_t dimension = 0; dimension < use.tensorViewDescriptor->rank; ++dimension)
                    addDescriptor(ComputeBindingSourceKind::TensorStride, dimension, descriptorBinding++);
            }
        }
    std::sort(candidates.begin(), candidates.end(),
              [](const auto &left, const auto &right) { return left.layout.slot < right.layout.slot; });
    for (const Candidate &candidate : candidates) {
        state->layout.push_back(candidate.layout);
        state->bindingSources.push_back(candidate.source);
    }
    state->values.resize(state->layout.size());
    state->descriptorValues.resize(state->layout.size());
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
    descriptor.bindings = state->layout.data();
    descriptor.binding_count = state->layout.size();
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
    for (size_t index = 0; index < state.layout.size(); ++index) {
        const auto &layout = state.layout[index];
        if (layout.argument_index >= launch.arguments.size())
            return fail(*pipeline.context, "CUDA prepared argument index is invalid");
        const ComputeLaunchArgument &argument = launch.arguments[layout.argument_index];
        auto &value = state.values[index];
        value = {};
        value.slot = layout.slot;
        value.kind = layout.kind;
        const ComputeBindingSource &source = state.bindingSources[index];
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
            if (!tensor || !tensor->resource.resource.value)
                return fail(*pipeline.context, "CUDA prepared storage binding requires an RHI Tensor");
            value.payload.buffer.resource = tensor->resource;
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
