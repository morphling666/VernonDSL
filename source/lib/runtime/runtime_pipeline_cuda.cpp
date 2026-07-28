#include "runtime_pipeline_backend.h"

#include "compute_launch_planner.h"

#if defined(VERNON_HAS_CUDA_RUNTIME)
#include "backend_cuda.h"
#include "rhi_adapter/adapter_internal.h"

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
    context.error = std::move(error);
    return status;
}

} // namespace
#endif

bool resolveCudaPipeline(VernonPipelineBundle &bundle, const Variant &variant, VernonLoadedPipeline &pipeline) {
#if defined(VERNON_HAS_CUDA_RUNTIME)
    auto state = std::make_unique<CudaPipelineState>();
    const Stage &stage = bundle.stages.at(variant.compute);
    ReflectedEntry reflection;
    const nlohmann::json parsed = nlohmann::json::parse(stage.reflection, nullptr, false);
    if (parsed.is_discarded() ||
        !parseReflection(parsed, stage.entry, reflection, VERNON_RUNTIME_CUDA, bundle.context->error))
        return false;
    uint32_t internalSlot = 0;
    for (const Parameter &parameter : variant.parameters)
        internalSlot = std::max(internalSlot, parameter.slot);
    std::vector<uint32_t> argumentBindings(reflection.arguments.size());
    uint32_t flattenedBinding = 0;
    for (size_t index = 0; index < reflection.arguments.size(); ++index) {
        argumentBindings[index] = flattenedBinding;
        if (reflection.arguments[index].kind != "builtin")
            flattenedBinding +=
                static_cast<uint32_t>(std::max(reflection.arguments[index].storageLeaves.size(), size_t{1}));
    }
    for (const Parameter &parameter : variant.parameters)
        for (const ParameterUse &use : parameter.uses) {
            if (use.stage != "compute" && use.stage != variant.compute)
                continue;
            if (use.index >= reflection.arguments.size()) {
                bundle.context->error = "CUDA parameter use exceeds reflected argument table";
                return false;
            }
            const ReflectedArgument &argument = reflection.arguments[use.index];
            const size_t leafCount = std::max(argument.storageLeaves.size(), size_t{1});
            for (size_t leafIndex = 0; leafIndex < leafCount; ++leafIndex) {
                VernonRuntimeProviderBindingLayoutEntry binding{};
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
                    bundle.context->error = "CUDA reflected argument has zero element size";
                    return false;
                }
                state->layout.push_back(binding);
            }
        }
    std::sort(state->layout.begin(), state->layout.end(),
              [](const auto &left, const auto &right) { return left.slot < right.slot; });
    state->values.resize(state->layout.size());
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
        vernonRuntimeRhiAdapterGetProvider(cudaState(*bundle.context).adapter), &descriptor, &state->pipeline);
    if (status != VERNON_STATUS_OK) {
        const VernonStringView error = vernonRuntimeRhiAdapterGetLastError(cudaState(*bundle.context).adapter);
        bundle.context->error =
            error.data ? std::string(error.data, error.size) : "failed to prepare CUDA provider pipeline";
        return false;
    }
    installRuntimeBackendState(pipeline, state.release());
    return true;
#else
    (void)bundle;
    (void)variant;
    (void)pipeline;
    return false;
#endif
}

void destroyCudaPipeline(VernonLoadedPipeline &pipeline) {
#if defined(VERNON_HAS_CUDA_RUNTIME)
    CudaPipelineState &state = runtimeBackendState<CudaPipelineState>(pipeline);
    vernonRuntimeCoreBindingsDestroy(state.bindings);
    vernonRuntimeCorePipelineDestroy(state.pipeline);
#else
    (void)pipeline;
#endif
}

VernonStatus invokeCudaComputePipeline(VernonLoadedPipeline &pipeline, const PlannedComputeLaunch &launch) {
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
        if (layout.kind == VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER) {
            if (argument.kind != ComputeLaunchArgumentKind::Tensor || !argument.resource.resource.value)
                return fail(*pipeline.context, "CUDA prepared storage binding requires an RHI Tensor");
            value.resource = argument.resource;
        } else {
            if (argument.kind != ComputeLaunchArgumentKind::Scalar || !argument.scalarData || !argument.scalarSize)
                return fail(*pipeline.context, "CUDA prepared inline binding requires host data");
            value.inline_data = argument.scalarData;
            value.inline_size = argument.scalarSize;
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
    const uint32_t groups[3]{(launch.grid.x - 1) / state.workgroup[0] + 1, (launch.grid.y - 1) / state.workgroup[1] + 1,
                             (launch.grid.z - 1) / state.workgroup[2] + 1};
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
