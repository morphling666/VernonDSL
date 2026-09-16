#include "runtime_pipeline_backend.h"

#if defined(VERNON_HAS_DIRECTX12_RUNTIME)
#include "VernonRuntimeRHIAdapter.h"
#include "backend_directx12.h"
#include "compute_launch_planner.h"
#include "pipeline_metadata.h"
#include "prepared_graphics_draw.h"
#include "rhi/rhi_internal.h"
#include "tensor_bridge.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cstring>
#include <string>
#include <utility>
#include <vector>
#endif

namespace vernon::runtime {

#if defined(VERNON_HAS_DIRECTX12_RUNTIME)
namespace {

VernonStatus fail(VernonRuntimeContext &context, std::string error,
                  VernonStatus status = VERNON_STATUS_INVALID_ARGUMENT) {
    invocationDiagnostic(context) = std::move(error);
    return status;
}

DXGI_FORMAT directX12TextureFormat(VernonTextureFormat format) {
    switch (format) {
    case VERNON_TEXTURE_RGBA8_UNORM:
        return DXGI_FORMAT_R8G8B8A8_UNORM;
    case VERNON_TEXTURE_RGBA8_SRGB:
        return DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
    case VERNON_TEXTURE_RGBA16_FLOAT:
        return DXGI_FORMAT_R16G16B16A16_FLOAT;
    case VERNON_TEXTURE_RGBA32_FLOAT:
        return DXGI_FORMAT_R32G32B32A32_FLOAT;
    case VERNON_TEXTURE_R8_UNORM:
        return DXGI_FORMAT_R8_UNORM;
    case VERNON_TEXTURE_R16_FLOAT:
        return DXGI_FORMAT_R16_FLOAT;
    case VERNON_TEXTURE_R32_FLOAT:
        return DXGI_FORMAT_R32_FLOAT;
    case VERNON_TEXTURE_RG8_UNORM:
        return DXGI_FORMAT_R8G8_UNORM;
    case VERNON_TEXTURE_R11G11B10_FLOAT:
        return DXGI_FORMAT_R11G11B10_FLOAT;
    case VERNON_TEXTURE_D32_FLOAT:
        return DXGI_FORMAT_D32_FLOAT;
    case VERNON_TEXTURE_D32_FLOAT_S8_UINT:
        return DXGI_FORMAT_D32_FLOAT_S8X24_UINT;
    case VERNON_TEXTURE_RGB8_UNORM:
        return DXGI_FORMAT_UNKNOWN;
    }
    return DXGI_FORMAT_UNKNOWN;
}

} // namespace
#endif

BackendPipelineResult resolveDirectX12Pipeline(BackendStageBuildInputs &inputs, const StageBindingPlan &plan,
                                               VernonStageExecutable &pipeline) {
    const auto resolve = [&]() {
#if defined(VERNON_HAS_DIRECTX12_RUNTIME)
        auto *pipelineState = new DirectX12PipelineState();
        if (!plan.compute.empty()) {
            const LoadedStageArtifact &stage = inputs.artifacts.at(plan.compute);
            ReflectedEntry reflection;
            if (!resolveStageReflection(stage, VERNON_RUNTIME_DIRECTX12, reflection,
                                        invocationDiagnostic(*inputs.context))) {
                delete pipelineState;
                return false;
            }
            if (!buildPreparedComputeBindingPlan(plan, reflection, VERNON_RUNTIME_DIRECTX12,
                                                 pipelineState->rhiComputeBindingPlan,
                                                 invocationDiagnostic(*inputs.context))) {
                delete pipelineState;
                return false;
            }
            pipelineState->rhiComputeValues.resize(pipelineState->rhiComputeBindingPlan.size());
            std::copy_n(stage.workgroup, 3, pipelineState->rhiComputeWorkgroup);
            const VernonRuntimeProviderShaderDescriptor shaderDescriptor{sizeof(VernonRuntimeProviderShaderDescriptor),
                                                                         VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE,
                                                                         {"dxil", 4},
                                                                         stage.binary.data(),
                                                                         stage.binary.size(),
                                                                         {stage.entry.data(), stage.entry.size()},
                                                                         {nullptr, 0},
                                                                         {0, 0, 0, 0}};
            VernonRuntimeCorePipelineDescriptor descriptor{};
            descriptor.struct_size = sizeof(descriptor);
            descriptor.kind = VERNON_RUNTIME_PROVIDER_COMPUTE_PIPELINE;
            descriptor.required_capabilities = VERNON_RUNTIME_PROVIDER_COMPUTE;
            descriptor.shaders = &shaderDescriptor;
            descriptor.shader_count = 1;
            descriptor.bindings = pipelineState->rhiComputeBindingPlan.layouts.data();
            descriptor.binding_count = pipelineState->rhiComputeBindingPlan.size();
            std::copy_n(pipelineState->rhiComputeWorkgroup, 3, descriptor.workgroup_size);
            const VernonStatus status = vernonRuntimeCorePreparePipeline(
                vernonRuntimeRhiAdapterGetProvider(directX12State(*inputs.context).adapter), &descriptor,
                &pipelineState->rhiComputePipeline);
            if (status != VERNON_STATUS_OK) {
                const VernonStringView providerError =
                    vernonRuntimeRhiAdapterGetLastError(directX12State(*inputs.context).adapter);
                invocationDiagnostic(*inputs.context) = providerError.data
                                                            ? std::string(providerError.data, providerError.size)
                                                            : "failed to prepare D3D12 provider compute pipeline";
                delete pipelineState;
                return false;
            }
        } else {
            if (!buildPreparedGraphicsBindingPlan(plan, {VERNON_RUNTIME_DIRECTX12},
                                                  pipelineState->rhiGraphicsBindingPlan,
                                                  invocationDiagnostic(*inputs.context))) {
                delete pipelineState;
                return false;
            }
            pipelineState->rhiGraphicsValues.resize(pipelineState->rhiGraphicsBindingPlan.size());
            const LoadedStageArtifact &vertex = inputs.artifacts.at(plan.vertex);
            const LoadedStageArtifact &fragment = inputs.artifacts.at(plan.fragment);
            const VernonRuntimeProviderShaderDescriptor shaders[2]{{sizeof(VernonRuntimeProviderShaderDescriptor),
                                                                    VERNON_RUNTIME_PROVIDER_STAGE_VERTEX,
                                                                    {"dxil", 4},
                                                                    vertex.binary.data(),
                                                                    vertex.binary.size(),
                                                                    {vertex.entry.data(), vertex.entry.size()},
                                                                    {},
                                                                    {0, 0, 0, 0}},
                                                                   {sizeof(VernonRuntimeProviderShaderDescriptor),
                                                                    VERNON_RUNTIME_PROVIDER_STAGE_FRAGMENT,
                                                                    {"dxil", 4},
                                                                    fragment.binary.data(),
                                                                    fragment.binary.size(),
                                                                    {fragment.entry.data(), fragment.entry.size()},
                                                                    {},
                                                                    {0, 0, 0, 0}}};
            VernonRuntimeCorePipelineDescriptor descriptor{};
            descriptor.struct_size = sizeof(descriptor);
            descriptor.kind = VERNON_RUNTIME_PROVIDER_GRAPHICS_PIPELINE;
            descriptor.required_capabilities = VERNON_RUNTIME_PROVIDER_GRAPHICS;
            descriptor.shaders = shaders;
            descriptor.shader_count = 2;
            descriptor.bindings = pipelineState->rhiGraphicsBindingPlan.layouts.data();
            descriptor.binding_count = pipelineState->rhiGraphicsBindingPlan.size();
            descriptor.vertex_attributes = pipelineState->rhiGraphicsBindingPlan.vertexAttributes.data();
            descriptor.vertex_attribute_count = pipelineState->rhiGraphicsBindingPlan.vertexAttributes.size();
            descriptor.topology = VERNON_TOPOLOGY_TRIANGLE_LIST;
            descriptor.sample_count = 1;
            const VernonStatus status = vernonRuntimeCorePreparePipeline(
                vernonRuntimeRhiAdapterGetProvider(directX12State(*inputs.context).adapter), &descriptor,
                &pipelineState->rhiGraphicsPipeline);
            if (status != VERNON_STATUS_OK) {
                const VernonStringView providerError =
                    vernonRuntimeRhiAdapterGetLastError(directX12State(*inputs.context).adapter);
                invocationDiagnostic(*inputs.context) = providerError.data
                                                            ? std::string(providerError.data, providerError.size)
                                                            : "failed to prepare D3D12 provider graphics pipeline";
                delete pipelineState;
                return false;
            }
        }
        installRuntimeBackendState(pipeline, pipelineState);
        return true;
#else
        return false;
#endif
    };
    return resolve() ? BackendPipelineResult{vernon::ok()} : backendPipelineResolutionFailure(inputs);
}

void destroyDirectX12Pipeline(VernonStageExecutable &pipeline) {
#if defined(VERNON_HAS_DIRECTX12_RUNTIME)
    DirectX12PipelineState &state = runtimeBackendState<DirectX12PipelineState>(pipeline);
    vernonRuntimeCoreBindingsDestroy(state.rhiComputeBindings);
    vernonRuntimeCorePipelineDestroy(state.rhiComputePipeline);
    vernonRuntimeCoreBindingsDestroy(state.rhiGraphicsBindings);
    destroyGraphicsVariant(state.rhiGraphicsVariant);
    vernonRuntimeCorePipelineDestroy(state.rhiGraphicsPipeline);
#else
    (void)pipeline;
#endif
}

VernonStatus invokeDirectX12GraphicsPipeline(VernonStageExecutable &pipeline,
                                             const VernonStageInvocationDescriptor &invocation,
                                             const PlannedGraphicsInvocation &plan) {
#if defined(VERNON_HAS_DIRECTX12_RUNTIME)
    DirectX12PipelineState &state = runtimeBackendState<DirectX12PipelineState>(pipeline);
    if (!state.rhiGraphicsPipeline)
        return VERNON_STATUS_OK;
    const auto &adapter = *directX12State(*pipeline.context).adapter;
    if (!fillPreparedGraphicsBindingValues(state.rhiGraphicsBindingPlan, plan, state.rhiGraphicsValues,
                                           invocationDiagnostic(*pipeline.context)))
        return VERNON_STATUS_INVALID_ARGUMENT;
    VernonStatus status =
        state.rhiGraphicsBindings
            ? vernonRuntimeCoreUpdateBindings(state.rhiGraphicsBindings, state.rhiGraphicsValues.data(),
                                              state.rhiGraphicsValues.size())
            : vernonRuntimeCoreCreateBindings(state.rhiGraphicsPipeline, state.rhiGraphicsValues.data(),
                                              state.rhiGraphicsValues.size(), &state.rhiGraphicsBindings);
    if (status != VERNON_STATUS_OK) {
        const VernonStringView providerError =
            vernonRuntimeRhiAdapterGetLastError(directX12State(*pipeline.context).adapter);
        return fail(*pipeline.context,
                    providerError.data ? std::string(providerError.data, providerError.size)
                                       : "failed to update D3D12 RHI graphics bindings",
                    status);
    }
    std::vector<uint32_t> formats;
    formats.reserve(plan.attachments.size());
    for (size_t index = 0; index < plan.attachments.size(); ++index) {
        const DXGI_FORMAT format = directX12TextureFormat(plan.attachmentFormats[index]);
        if (format == DXGI_FORMAT_UNKNOWN)
            return fail(*pipeline.context, "D3D12 RHI color attachment format is unsupported");
        formats.push_back(static_cast<uint32_t>(format));
    }
    const uint32_t depthFormat = !plan.depthAttachment
                                     ? 0
                                     : static_cast<uint32_t>(plan.depthFormat == VERNON_TEXTURE_D32_FLOAT_S8_UINT
                                                                 ? DXGI_FORMAT_D32_FLOAT_S8X24_UINT
                                                                 : DXGI_FORMAT_D32_FLOAT);
    PreparedGraphicsDraw prepared;
    if (!prepareGraphicsDraw(invocation, plan, std::move(formats), depthFormat, state.rhiGraphicsBindingPlan.layouts,
                             state.rhiGraphicsValues, prepared, invocationDiagnostic(*pipeline.context)))
        return VERNON_STATUS_INVALID_ARGUMENT;
    status = ensureGraphicsVariant(state.rhiGraphicsPipeline, prepared.variantKey, state.rhiGraphicsVariant);
    if (status != VERNON_STATUS_OK) {
        const VernonStringView providerError =
            vernonRuntimeRhiAdapterGetLastError(directX12State(*pipeline.context).adapter);
        return fail(*pipeline.context,
                    providerError.data ? std::string(providerError.data, providerError.size)
                                       : "failed to prepare D3D12 RHI graphics variant",
                    status);
    }
    status = vernonRuntimeCoreEncodeGraphicsVariantDrawInvocation(state.rhiGraphicsVariant.handle,
                                                                  state.rhiGraphicsBindings, &prepared.invocation);
    if (status != VERNON_STATUS_OK) {
        const VernonStringView providerError =
            vernonRuntimeRhiAdapterGetLastError(directX12State(*pipeline.context).adapter);
        return fail(*pipeline.context,
                    providerError.data ? std::string(providerError.data, providerError.size)
                                       : "failed to encode D3D12 RHI draw",
                    status);
    }
    return VERNON_STATUS_OK;
#else
    (void)pipeline;
    (void)invocation;
    (void)plan;
    return VERNON_STATUS_UNSUPPORTED_TARGET;
#endif
}

VernonStatus invokeDirectX12ComputePipeline(VernonStageExecutable &pipeline, const PlannedComputeLaunch &launch) {
#if defined(VERNON_HAS_DIRECTX12_RUNTIME)
    DirectX12PipelineState &state = runtimeBackendState<DirectX12PipelineState>(pipeline);
    VernonRuntimeRhiAdapter &adapter = *directX12State(*pipeline.context).adapter;
    for (size_t index = 0; index < state.rhiComputeBindingPlan.size(); ++index) {
        const VernonRuntimeProviderBindingLayoutEntry &layout = state.rhiComputeBindingPlan.layouts[index];
        const PreparedBindingSource &preparedSource = state.rhiComputeBindingPlan.sources[index];
        VernonRuntimeProviderBindingValue &value = state.rhiComputeValues[index];
        value = {};
        value.slot = layout.slot;
        value.kind = layout.kind;
        if (preparedSource.metadataCarrier) {
            if (!bindComputeMetadataCarrier(layout, preparedSource, launch, value))
                return fail(*pipeline.context, "D3D12 metadata carrier payload does not match its compiled ABI");
            continue;
        }
        if (preparedSource.argumentIndex >= launch.arguments.size())
            return fail(*pipeline.context, "D3D12 prepared argument index is invalid");
        const ComputeLaunchArgument &argument = launch.arguments[preparedSource.argumentIndex];
        if (layout.kind == VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER) {
            if (layout.interface_kind == VERNON_RUNTIME_PROVIDER_INTERFACE_UNIFORM) {
                if (!bindComputeValueStorage(layout, argument, value))
                    return fail(*pipeline.context, "D3D12 prepared Value storage binding has invalid bytes");
                continue;
            }
            const auto *tensor = std::get_if<ComputeTensorArgument>(&argument);
            if (!tensor || !tensor->resource.resource.value)
                return fail(*pipeline.context, "D3D12 prepared storage binding requires an RHI Tensor");
            value.payload.buffer.resource = tensor->resource;
            if (value.payload.buffer.resource.offset > value.payload.buffer.resource.size ||
                preparedSource.resourceOffset >
                    value.payload.buffer.resource.size - value.payload.buffer.resource.offset)
                return fail(*pipeline.context, "D3D12 aggregate storage leaf exceeds its Tensor resource");
            value.payload.buffer.resource.offset += preparedSource.resourceOffset;
        } else if (layout.kind == VERNON_RUNTIME_PROVIDER_STORAGE_IMAGE ||
                   layout.kind == VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE) {
            const auto *image = std::get_if<ComputeImageArgument>(&argument);
            if (!image || !image->view.resource.value)
                return fail(*pipeline.context, "D3D12 prepared image binding requires an RHI Texture");
            value.payload.image.view = image->view;
        } else {
            const auto *scalar = std::get_if<ComputeScalarArgument>(&argument);
            if (!scalar || !scalar->data || !scalar->size)
                return fail(*pipeline.context, "D3D12 prepared inline binding requires host data at argument " +
                                                   std::to_string(layout.argument_index) + " (kind " +
                                                   std::to_string(argument.index()) + ", size " +
                                                   std::to_string(scalar ? scalar->size : 0) + ")");
            value.payload.inline_value.data = scalar->data;
            value.payload.inline_value.size = scalar->size;
        }
    }
    VernonStatus status =
        state.rhiComputeBindings
            ? vernonRuntimeCoreUpdateBindings(state.rhiComputeBindings, state.rhiComputeValues.data(),
                                              state.rhiComputeValues.size())
            : vernonRuntimeCoreCreateBindings(state.rhiComputePipeline, state.rhiComputeValues.data(),
                                              state.rhiComputeValues.size(), &state.rhiComputeBindings);
    if (status != VERNON_STATUS_OK) {
        const VernonStringView providerError =
            vernonRuntimeRhiAdapterGetLastError(directX12State(*pipeline.context).adapter);
        return fail(*pipeline.context,
                    providerError.data ? std::string(providerError.data, providerError.size)
                                       : "failed to prepare D3D12 invocation bindings",
                    status);
    }
    const uint32_t groups[3]{launch.grid.x, launch.grid.y, launch.grid.z};
    status = vernonRuntimeCoreEncodeDispatch(state.rhiComputePipeline, state.rhiComputeBindings, launch.commandEncoder,
                                             groups, nullptr, 0);
    if (status != VERNON_STATUS_OK) {
        const VernonStringView providerError =
            vernonRuntimeRhiAdapterGetLastError(directX12State(*pipeline.context).adapter);
        return fail(*pipeline.context,
                    providerError.data ? std::string(providerError.data, providerError.size)
                                       : "failed to encode D3D12 provider dispatch",
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
