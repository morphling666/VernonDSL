#include "runtime_pipeline_backend.h"

#if defined(VERNON_HAS_VULKAN_RUNTIME)
#include "VernonRuntimeRHIAdapter.h"
#include "backend_vulkan.h"
#include "compute_launch_planner.h"
#include "pipeline_metadata.h"
#include "prepared_graphics_draw.h"
#include "tensor_bridge.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <utility>
#include <vector>
#endif

namespace vernon::runtime {

#if defined(VERNON_HAS_VULKAN_RUNTIME)
namespace {

VernonStatus fail(VernonRuntimeContext &context, std::string error,
                  VernonStatus status = VERNON_STATUS_INVALID_ARGUMENT) {
    invocationDiagnostic(context) = std::move(error);
    return status;
}

VkFormat vulkanTextureFormat(VernonTextureFormat format) {
    switch (format) {
    case VERNON_TEXTURE_RGBA8_UNORM:
        return VK_FORMAT_R8G8B8A8_UNORM;
    case VERNON_TEXTURE_RGBA8_SRGB:
        return VK_FORMAT_R8G8B8A8_SRGB;
    case VERNON_TEXTURE_RGBA16_FLOAT:
        return VK_FORMAT_R16G16B16A16_SFLOAT;
    case VERNON_TEXTURE_RGBA32_FLOAT:
        return VK_FORMAT_R32G32B32A32_SFLOAT;
    case VERNON_TEXTURE_R8_UNORM:
        return VK_FORMAT_R8_UNORM;
    case VERNON_TEXTURE_R16_FLOAT:
        return VK_FORMAT_R16_SFLOAT;
    case VERNON_TEXTURE_R32_FLOAT:
        return VK_FORMAT_R32_SFLOAT;
    case VERNON_TEXTURE_RG8_UNORM:
        return VK_FORMAT_R8G8_UNORM;
    case VERNON_TEXTURE_RGB8_UNORM:
        return VK_FORMAT_R8G8B8_UNORM;
    case VERNON_TEXTURE_R11G11B10_FLOAT:
        return VK_FORMAT_B10G11R11_UFLOAT_PACK32;
    case VERNON_TEXTURE_D32_FLOAT:
        return VK_FORMAT_D32_SFLOAT;
    case VERNON_TEXTURE_D32_FLOAT_S8_UINT:
        return VK_FORMAT_D32_SFLOAT_S8_UINT;
    }
    return VK_FORMAT_UNDEFINED;
}

} // namespace
#endif

BackendPipelineResult resolveVulkanPipeline(BackendStageBuildInputs &inputs, const StageBindingPlan &plan,
                                            VernonStageExecutable &pipeline) {
    const auto resolve = [&]() {
#if defined(VERNON_HAS_VULKAN_RUNTIME)
        auto *state = new VulkanPipelineState();
        if (!plan.compute.empty()) {
            const LoadedStageArtifact &stage = inputs.artifacts.at(plan.compute);
            ReflectedEntry reflection;
            if (!resolveStageReflection(stage, VERNON_RUNTIME_VULKAN, reflection,
                                        invocationDiagnostic(*inputs.context))) {
                delete state;
                return false;
            }
            if (!buildPreparedComputeBindingPlan(plan, reflection, VERNON_RUNTIME_VULKAN, state->rhiComputeBindingPlan,
                                                 invocationDiagnostic(*inputs.context))) {
                delete state;
                return false;
            }
            state->rhiComputeValues.resize(state->rhiComputeBindingPlan.size());
            state->rhiComputeDescriptorValues.resize(state->rhiComputeBindingPlan.size());
            std::copy_n(stage.workgroup, 3, state->rhiComputeWorkgroup);
            const VernonRuntimeProviderShaderDescriptor shader{sizeof(VernonRuntimeProviderShaderDescriptor),
                                                               VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE,
                                                               {"spirv", 5},
                                                               stage.binary.data(),
                                                               stage.binary.size(),
                                                               {stage.entry.data(), stage.entry.size()},
                                                               {},
                                                               {0, 0, 0, 0}};
            VernonRuntimeCorePipelineDescriptor descriptor{};
            descriptor.struct_size = sizeof(descriptor);
            descriptor.kind = VERNON_RUNTIME_PROVIDER_COMPUTE_PIPELINE;
            descriptor.required_capabilities = VERNON_RUNTIME_PROVIDER_COMPUTE;
            descriptor.shaders = &shader;
            descriptor.shader_count = 1;
            descriptor.bindings = state->rhiComputeBindingPlan.layouts.data();
            descriptor.binding_count = state->rhiComputeBindingPlan.layouts.size();
            std::copy_n(state->rhiComputeWorkgroup, 3, descriptor.workgroup_size);
            const VernonStatus status = vernonRuntimeCorePreparePipeline(
                vernonRuntimeRhiAdapterGetProvider(vulkanState(*inputs.context).adapter), &descriptor,
                &state->rhiComputePipeline);
            if (status != VERNON_STATUS_OK) {
                const VernonStringView providerError =
                    vernonRuntimeRhiAdapterGetLastError(vulkanState(*inputs.context).adapter);
                invocationDiagnostic(*inputs.context) = providerError.data
                                                            ? std::string(providerError.data, providerError.size)
                                                            : "failed to prepare Vulkan provider compute pipeline";
                delete state;
                return false;
            }
        }
        if (!plan.vertex.empty()) {
            const LoadedStageArtifact &vertex = inputs.artifacts.at(plan.vertex);
            const LoadedStageArtifact &fragment = inputs.artifacts.at(plan.fragment);
            if (!buildPreparedGraphicsBindingPlan(plan, {VERNON_RUNTIME_VULKAN}, state->rhiGraphicsBindingPlan,
                                                  invocationDiagnostic(*inputs.context))) {
                delete state;
                return false;
            }
            state->rhiGraphicsValues.resize(state->rhiGraphicsBindingPlan.size());
            std::vector<VernonRuntimeProviderShaderDescriptor> shaders{{sizeof(VernonRuntimeProviderShaderDescriptor),
                                                                        VERNON_RUNTIME_PROVIDER_STAGE_VERTEX,
                                                                        {"spirv", 5},
                                                                        vertex.binary.data(),
                                                                        vertex.binary.size(),
                                                                        {vertex.entry.data(), vertex.entry.size()},
                                                                        {},
                                                                        {0, 0, 0, 0}}};
            shaders.push_back({sizeof(VernonRuntimeProviderShaderDescriptor),
                               VERNON_RUNTIME_PROVIDER_STAGE_FRAGMENT,
                               {"spirv", 5},
                               fragment.binary.data(),
                               fragment.binary.size(),
                               {fragment.entry.data(), fragment.entry.size()},
                               {},
                               {0, 0, 0, 0}});
            VernonRuntimeCorePipelineDescriptor descriptor{};
            descriptor.struct_size = sizeof(descriptor);
            descriptor.kind = VERNON_RUNTIME_PROVIDER_GRAPHICS_PIPELINE;
            descriptor.required_capabilities = VERNON_RUNTIME_PROVIDER_GRAPHICS;
            descriptor.shaders = shaders.data();
            descriptor.shader_count = shaders.size();
            descriptor.bindings = state->rhiGraphicsBindingPlan.layouts.data();
            descriptor.binding_count = state->rhiGraphicsBindingPlan.size();
            descriptor.vertex_attributes = state->rhiGraphicsBindingPlan.vertexAttributes.data();
            descriptor.vertex_attribute_count = state->rhiGraphicsBindingPlan.vertexAttributes.size();
            descriptor.topology = VERNON_TOPOLOGY_TRIANGLE_LIST;
            descriptor.sample_count = 1;
            const VernonStatus status = vernonRuntimeCorePreparePipeline(
                vernonRuntimeRhiAdapterGetProvider(vulkanState(*inputs.context).adapter), &descriptor,
                &state->rhiGraphicsPipeline);
            if (status != VERNON_STATUS_OK) {
                const VernonStringView providerError =
                    vernonRuntimeRhiAdapterGetLastError(vulkanState(*inputs.context).adapter);
                invocationDiagnostic(*inputs.context) = providerError.data
                                                            ? std::string(providerError.data, providerError.size)
                                                            : "failed to prepare Vulkan provider graphics pipeline";
                delete state;
                return false;
            }
        }
        installRuntimeBackendState(pipeline, state);
        return true;
#else
        return false;
#endif
    };
    return resolve() ? BackendPipelineResult{vernon::ok()} : backendPipelineResolutionFailure(inputs);
}

void destroyVulkanPipeline(VernonStageExecutable &pipeline) {
#if defined(VERNON_HAS_VULKAN_RUNTIME)
    VulkanPipelineState &state = runtimeBackendState<VulkanPipelineState>(pipeline);
    vernonRuntimeCoreBindingsDestroy(state.rhiComputeBindings);
    vernonRuntimeCorePipelineDestroy(state.rhiComputePipeline);
    vernonRuntimeCoreBindingsDestroy(state.rhiGraphicsBindings);
    destroyGraphicsVariant(state.rhiGraphicsVariant);
    vernonRuntimeCorePipelineDestroy(state.rhiGraphicsPipeline);
#else
    (void)pipeline;
#endif
}

VernonStatus invokeVulkanGraphicsPipeline(VernonStageExecutable &pipeline,
                                          const VernonStageInvocationDescriptor &invocation,
                                          const PlannedGraphicsInvocation &plan) {
#if defined(VERNON_HAS_VULKAN_RUNTIME)
    VulkanPipelineState &state = runtimeBackendState<VulkanPipelineState>(pipeline);
    if (!state.rhiGraphicsPipeline)
        return VERNON_STATUS_OK;
    const auto &adapter = *vulkanState(*pipeline.context).adapter;
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
            vernonRuntimeRhiAdapterGetLastError(vulkanState(*pipeline.context).adapter);
        return fail(*pipeline.context,
                    providerError.data ? std::string(providerError.data, providerError.size)
                                       : "failed to update Vulkan RHI graphics bindings",
                    status);
    }
    std::vector<uint32_t> formats;
    for (size_t index = 0; index < plan.attachments.size(); ++index) {
        const VkFormat format = vulkanTextureFormat(plan.attachmentFormats[index]);
        if (format == VK_FORMAT_UNDEFINED)
            return fail(*pipeline.context, "Vulkan RHI color attachment format is unsupported");
        formats.push_back(static_cast<uint32_t>(format));
    }
    const uint32_t depthFormat =
        !plan.depthAttachment
            ? 0
            : static_cast<uint32_t>(plan.depthFormat == VERNON_TEXTURE_D32_FLOAT_S8_UINT ? VK_FORMAT_D32_SFLOAT_S8_UINT
                                                                                         : VK_FORMAT_D32_SFLOAT);
    PreparedGraphicsDraw prepared;
    if (!prepareGraphicsDraw(invocation, plan, std::move(formats), depthFormat, state.rhiGraphicsBindingPlan.layouts,
                             state.rhiGraphicsValues, prepared, invocationDiagnostic(*pipeline.context)))
        return VERNON_STATUS_INVALID_ARGUMENT;
    status = ensureGraphicsVariant(state.rhiGraphicsPipeline, prepared.variantKey, state.rhiGraphicsVariant);
    if (status != VERNON_STATUS_OK) {
        const VernonStringView providerError =
            vernonRuntimeRhiAdapterGetLastError(vulkanState(*pipeline.context).adapter);
        return fail(*pipeline.context,
                    providerError.data ? std::string(providerError.data, providerError.size)
                                       : "failed to prepare Vulkan RHI graphics variant",
                    status);
    }
    prepared.invocation.render_area[2] = plan.attachmentWidth;
    prepared.invocation.render_area[3] = plan.attachmentHeight;
    status = vernonRuntimeCoreEncodeGraphicsVariantDrawInvocation(state.rhiGraphicsVariant.handle,
                                                                  state.rhiGraphicsBindings, &prepared.invocation);
    if (status != VERNON_STATUS_OK) {
        const VernonStringView providerError =
            vernonRuntimeRhiAdapterGetLastError(vulkanState(*pipeline.context).adapter);
        return fail(*pipeline.context,
                    providerError.data ? std::string(providerError.data, providerError.size)
                                       : "failed to encode Vulkan RHI draw",
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

VernonStatus invokeVulkanComputePipeline(VernonStageExecutable &pipeline, const PlannedComputeLaunch &launch) {
#if defined(VERNON_HAS_VULKAN_RUNTIME)
    VulkanPipelineState &state = runtimeBackendState<VulkanPipelineState>(pipeline);
    for (size_t index = 0; index < state.rhiComputeBindingPlan.size(); ++index) {
        const auto &layout = state.rhiComputeBindingPlan.layouts[index];
        const ComputeLaunchArgument &argument = launch.arguments[layout.argument_index];
        auto &value = state.rhiComputeValues[index];
        value = {};
        value.slot = layout.slot;
        value.kind = layout.kind;
        const PreparedBindingSource &preparedSource = state.rhiComputeBindingPlan.sources[index];
        const ComputeBindingSource &source = preparedSource.source;
        if (source.kind != ComputeBindingSourceKind::Argument) {
            std::optional<int64_t> descriptor = computeBindingDescriptorValue(argument, source);
            if (!descriptor || *descriptor < INT32_MIN || *descriptor > INT32_MAX)
                return fail(*pipeline.context, "Vulkan TensorView descriptor exceeds the shader index range");
            state.rhiComputeDescriptorValues[index] = static_cast<int32_t>(*descriptor);
            value.payload.inline_value.data = &state.rhiComputeDescriptorValues[index];
            value.payload.inline_value.size = sizeof(int32_t);
            continue;
        }
        if (layout.kind == VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER) {
            if (layout.interface_kind == VERNON_RUNTIME_PROVIDER_INTERFACE_UNIFORM) {
                if (!bindComputeValueStorage(layout, argument, value))
                    return fail(*pipeline.context, "Vulkan prepared Value storage binding has invalid bytes");
                continue;
            }
            const auto *tensor = std::get_if<ComputeTensorArgument>(&argument);
            if (!tensor || !tensor->resource.resource.value)
                return fail(*pipeline.context, "Vulkan prepared storage binding requires an RHI Tensor");
            value.payload.buffer.resource = tensor->resource;
            const uint64_t leafOffset = preparedSource.resourceOffset;
            if (leafOffset > value.payload.buffer.resource.size)
                return fail(*pipeline.context, "Vulkan aggregate storage leaf exceeds its Tensor resource");
            value.payload.buffer.resource.offset += leafOffset;
            value.payload.buffer.resource.size -= leafOffset;
        } else if (layout.kind == VERNON_RUNTIME_PROVIDER_STORAGE_IMAGE ||
                   layout.kind == VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE) {
            const auto *image = std::get_if<ComputeImageArgument>(&argument);
            if (!image || !image->view.resource.value)
                return fail(*pipeline.context, "Vulkan prepared image binding requires an RHI Texture");
            value.payload.image.view = image->view;
        } else {
            const auto *scalar = std::get_if<ComputeScalarArgument>(&argument);
            if (!scalar || !scalar->data) {
                std::string parameterContract;
                for (const Parameter &parameter : pipeline.bindingProjection.parameters)
                    for (const ParameterUse &use : parameter.uses)
                        if (use.index == layout.argument_index)
                            parameterContract = " for parameter '" + parameter.name + "' (interface " +
                                                use.interfaceKind + ", transport " + use.transport + ")";
                return fail(*pipeline.context, "Vulkan prepared inline binding requires host data at argument " +
                                                   std::to_string(layout.argument_index) + " (kind " +
                                                   std::to_string(argument.index()) + ", size " +
                                                   std::to_string(scalar ? scalar->size : 0) + ")" + parameterContract);
            }
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
    if (status != VERNON_STATUS_OK)
        return fail(*pipeline.context, "failed to prepare Vulkan invocation bindings", status);
    const uint32_t groups[3]{launch.grid.x, launch.grid.y, launch.grid.z};
    status = vernonRuntimeCoreEncodeDispatch(state.rhiComputePipeline, state.rhiComputeBindings, launch.commandEncoder,
                                             groups, nullptr, 0);
    return status == VERNON_STATUS_OK ? status
                                      : fail(*pipeline.context, "failed to encode Vulkan provider dispatch", status);
#else
    (void)pipeline;
    (void)launch;
    return VERNON_STATUS_UNSUPPORTED_TARGET;
#endif
}

} // namespace vernon::runtime
