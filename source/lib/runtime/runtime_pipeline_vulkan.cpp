#include "runtime_pipeline_backend.h"

#if defined(VERNON_HAS_VULKAN_RUNTIME)
#include "VernonRuntimeRHIAdapter.h"
#include "backend_vulkan.h"
#include "compute_launch_planner.h"
#include "pipeline_metadata.h"
#include "prepared_graphics_draw.h"
#include "rhi/rhi_internal.h"
#include "tensor_bridge.h"

#include <nlohmann/json.hpp>

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

bool resolveVulkanPipeline(BackendStageBuildInputs &inputs, const StageBindingPlan &plan,
                           VernonStageExecutable &pipeline) {
#if defined(VERNON_HAS_VULKAN_RUNTIME)
    auto *state = new VulkanPipelineState();
    if (!plan.compute.empty()) {
        const LoadedStageArtifact &stage = inputs.artifacts.at(plan.compute);
        ReflectedEntry reflection;
        if (!resolveStageReflection(stage, VERNON_RUNTIME_VULKAN, reflection, invocationDiagnostic(*inputs.context))) {
            delete state;
            return false;
        }
        struct Candidate {
            VernonRuntimeProviderBindingLayoutEntry layout{};
            uint64_t resourceOffset{};
            ComputeBindingSource source;
        };
        std::vector<Candidate> candidates;
        uint32_t internalSlot = 0;
        std::vector<uint32_t> argumentBindings(reflection.arguments.size());
        uint32_t flattenedBinding = 0;
        for (size_t index = 0; index < reflection.arguments.size(); ++index) {
            argumentBindings[index] = flattenedBinding;
            if (reflection.arguments[index].kind != "builtin")
                flattenedBinding +=
                    static_cast<uint32_t>(std::max(reflection.arguments[index].storageLeaves.size(), size_t{1}));
        }
        for (const Parameter &parameter : plan.parameters)
            internalSlot = std::max(internalSlot, parameter.slot);
        for (const Parameter &parameter : plan.parameters)
            for (const ParameterUse &use : parameter.uses) {
                if (use.stage != "compute" && use.stage != plan.compute)
                    continue;
                if (use.index >= reflection.arguments.size()) {
                    invocationDiagnostic(*inputs.context) = "Vulkan parameter use exceeds reflected argument table";
                    delete state;
                    return false;
                }
                const ReflectedArgument &argument = reflection.arguments[use.index];
                const size_t leafCount = std::max(argument.storageLeaves.size(), size_t{1});
                for (size_t leafIndex = 0; leafIndex < leafCount; ++leafIndex) {
                    Candidate candidate;
                    candidate.layout.slot = leafIndex == 0 ? parameter.slot : ++internalSlot;
                    candidate.layout.set = argument.descriptorSet;
                    candidate.layout.binding = argument.storageLeaves.size() <= 1
                                                   ? argumentBindings[use.index] + static_cast<uint32_t>(leafIndex)
                                                   : argument.storageLeaves[leafIndex].binding;
                    candidate.layout.kind = parameter.kind == "image"   ? (parameter.bindingRole == "sampled"
                                                                               ? VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE
                                                                               : VERNON_RUNTIME_PROVIDER_STORAGE_IMAGE)
                                            : argument.kind == "tensor" ? VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER
                                                                        : VERNON_RUNTIME_PROVIDER_INLINE_VALUE;
                    configureComputeValueStorage(use, candidate.layout);
                    candidate.layout.access = parameter.access == "read" ? 1u : parameter.access == "write" ? 2u : 3u;
                    if (parameter.kind == "image" && !configureImageBindingLayout(parameter, candidate.layout))
                        return false;
                    candidate.layout.stage_mask = VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE;
                    candidate.layout.array_count = 1;
                    candidate.layout.argument_index = use.index;
                    candidate.layout.element_size = static_cast<uint32_t>(
                        argument.storageLeaves.empty() ? (parameter.kind == "image"   ? 1
                                                          : argument.kind == "tensor" ? argument.tensorElementSize
                                                                                      : argument.physical.size)
                                                       : argument.storageLeaves[leafIndex].elementSize);
                    // Aggregate lowering already folds each leaf's byte offset into the shader index.
                    // Every leaf descriptor must therefore retain the base address of the original AoS buffer.
                    candidate.resourceOffset = 0;
                    candidate.source = {ComputeBindingSourceKind::Argument, use.index, 0};
                    candidates.push_back(candidate);
                }
                if (use.tensorViewDescriptor) {
                    const auto addDescriptor = [&](ComputeBindingSourceKind kind, uint32_t dimension,
                                                   uint32_t binding) {
                        Candidate candidate;
                        candidate.layout.slot = ++internalSlot;
                        candidate.layout.set = argument.descriptorSet;
                        candidate.layout.binding = binding;
                        candidate.layout.kind = VERNON_RUNTIME_PROVIDER_INLINE_VALUE;
                        candidate.layout.stage_mask = VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE;
                        candidate.layout.array_count = 1;
                        candidate.layout.argument_index = use.index;
                        candidate.layout.element_size = 4;
                        candidate.source = {kind, use.index, dimension};
                        candidates.push_back(candidate);
                    };
                    addDescriptor(ComputeBindingSourceKind::TensorOffset, 0, use.tensorViewDescriptor->offsetBinding);
                    for (uint32_t dimension = 0; dimension < use.tensorViewDescriptor->rank; ++dimension)
                        addDescriptor(ComputeBindingSourceKind::TensorExtent, dimension,
                                      use.tensorViewDescriptor->extentBindings[dimension]);
                    for (uint32_t dimension = 0; dimension < use.tensorViewDescriptor->rank; ++dimension)
                        addDescriptor(ComputeBindingSourceKind::TensorStride, dimension,
                                      use.tensorViewDescriptor->strideBindings[dimension]);
                }
            }
        std::sort(candidates.begin(), candidates.end(),
                  [](const auto &left, const auto &right) { return left.layout.slot < right.layout.slot; });
        for (const auto &candidate : candidates) {
            state->rhiComputeLayout.push_back(candidate.layout);
            state->rhiComputeResourceOffsets.push_back(candidate.resourceOffset);
            state->rhiComputeBindingSources.push_back(candidate.source);
        }
        state->rhiComputeValues.resize(candidates.size());
        state->rhiComputeDescriptorValues.resize(candidates.size());
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
        descriptor.bindings = state->rhiComputeLayout.data();
        descriptor.binding_count = state->rhiComputeLayout.size();
        std::copy_n(state->rhiComputeWorkgroup, 3, descriptor.workgroup_size);
        const VernonStatus status =
            vernonRuntimeCorePreparePipeline(vernonRuntimeRhiAdapterGetProvider(vulkanState(*inputs.context).adapter),
                                             &descriptor, &state->rhiComputePipeline);
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
        struct Candidate {
            VernonRuntimeProviderBindingLayoutEntry layout{};
            std::vector<VernonRuntimeProviderVertexAttribute> attributes;
            VulkanPipelineState::Binding binding;
        };
        std::vector<Candidate> candidates;
        uint32_t nextProviderSlot = 0;
        uint32_t vertexBinding = 0;
        bool supported = true;
        const auto addUse = [&](const Parameter &parameter, const ParameterUse &use, bool internal) {
            if (use.stage != "vertex" && use.stage != "fragment")
                return false;
            Candidate candidate;
            if (nextProviderSlot == UINT32_MAX)
                return false;
            candidate.layout.slot = nextProviderSlot++;
            candidate.layout.argument_index = use.index;
            candidate.layout.stage_mask =
                use.stage == "vertex" ? VERNON_RUNTIME_PROVIDER_STAGE_VERTEX : VERNON_RUNTIME_PROVIDER_STAGE_FRAGMENT;
            candidate.layout.array_count = 1;
            candidate.binding.externalSlot = parameter.slot;
            if (parameter.kind == "sampler") {
                if (use.sampledImageBindings.empty())
                    return false;
                for (size_t index = 0; index < use.sampledImageBindings.size(); ++index) {
                    const SampledImageBinding &binding = use.sampledImageBindings[index];
                    Candidate sampler = candidate;
                    if (index != 0) {
                        if (nextProviderSlot == UINT32_MAX)
                            return false;
                        sampler.layout.slot = nextProviderSlot++;
                    }
                    sampler.layout.kind = VERNON_RUNTIME_PROVIDER_SAMPLER;
                    sampler.layout.interface_kind = VERNON_RUNTIME_PROVIDER_INTERFACE_RESOURCE;
                    sampler.layout.set = binding.descriptorSet;
                    sampler.layout.binding = binding.binding;
                    sampler.binding.source = internal ? VulkanPipelineState::Binding::IMPLICIT_SAMPLER
                                                      : VulkanPipelineState::Binding::EXTERNAL_SAMPLER;
                    candidates.push_back(std::move(sampler));
                }
                return true;
            }
            if (parameter.kind == "tensor" && use.interfaceKind == "uniform") {
                const auto &shape = use.shape.empty() ? parameter.shape : use.shape;
                const std::optional<VernonDataType> dtype = pipelineDataType(use.dtype);
                if (!dtype)
                    return false;
                if (!use.interfacePlan || !use.interfacePlan->root)
                    return false;
                candidate.layout.kind = use.transport == "storage_buffer"   ? VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER
                                        : use.transport == "uniform_buffer" ? VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER
                                                                            : VERNON_RUNTIME_PROVIDER_INLINE_VALUE;
                const uint64_t physicalSize = use.interfacePlan->root->size;
                if (!physicalSize || physicalSize > UINT32_MAX ||
                    ((candidate.layout.kind == VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER ||
                      candidate.layout.kind == VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER) &&
                     use.binding == UINT32_MAX))
                    return false;
                candidate.layout.element_size = static_cast<uint32_t>(physicalSize);
                candidate.layout.interface_kind = VERNON_RUNTIME_PROVIDER_INTERFACE_UNIFORM;
                const uint64_t physicalAlignment = use.interfacePlan->root->alignment;
                if (!physicalAlignment || physicalAlignment > UINT32_MAX)
                    return false;
                candidate.layout.element_alignment = static_cast<uint32_t>(physicalAlignment);
                candidate.layout.binding = use.binding;
                candidate.layout.set = use.descriptorSet;
                candidate.binding.source = internal ? VulkanPipelineState::Binding::RESOLUTION
                                                    : VulkanPipelineState::Binding::EXTERNAL_UNIFORM;
                const ValueLayout &canonical = parameter.valueLayout ? *parameter.valueLayout : parameter.elementLayout;
                std::optional<TensorCopyPlan> packing =
                    parameter.valueLayout
                        ? compileWholeValueCopyPlan(pipelineValueLayout(canonical), *use.interfacePlan->root)
                        : compileElementStreamCopyPlan(pipelineValueLayout(canonical), shape, *use.interfacePlan->root);
                if (!packing || packing->elementSize != canonical.byteSize) {
                    invocationDiagnostic(*inputs.context) =
                        !packing ? "Vulkan graphics value packing is incompatible with the canonical layout"
                                 : "Vulkan graphics value packing scalar size does not match its dtype";
                    return false;
                }
                candidate.binding.packing = std::move(*packing);
                candidate.binding.storage.resize(candidate.layout.element_size);
            } else if (parameter.kind == "tensor" && use.interfaceKind == "input" && use.stage == "vertex" &&
                       use.location != UINT32_MAX && !use.attributeLeaves.empty()) {
                candidate.layout.kind = VERNON_RUNTIME_PROVIDER_VERTEX_BUFFER;
                candidate.layout.element_size = parameter.elementLayout.byteSize;
                candidate.layout.interface_kind = VERNON_RUNTIME_PROVIDER_INTERFACE_VERTEX_INPUT;
                candidate.layout.binding = vertexBinding++;
                candidate.layout.divisor = use.divisor;
                for (const AttributeLeaf &leaf : use.attributeLeaves) {
                    const std::optional<VernonDataType> dtype = pipelineDataType(leaf.dtype);
                    if (!dtype)
                        return false;
                    candidate.attributes.push_back({candidate.layout.binding, use.location + leaf.locationOffset,
                                                    static_cast<uint32_t>(*dtype), leaf.componentCount,
                                                    leaf.byteOffset});
                }
                candidate.binding.source = VulkanPipelineState::Binding::EXTERNAL_VERTEX;
            } else if (parameter.kind == "image" && use.interfaceKind == "resource") {
                candidate.layout.kind = parameter.bindingRole == "sampled" ? VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE
                                                                           : VERNON_RUNTIME_PROVIDER_STORAGE_IMAGE;
                if (!configureImageBindingLayout(parameter, candidate.layout))
                    return false;
                candidate.layout.interface_kind = VERNON_RUNTIME_PROVIDER_INTERFACE_RESOURCE;
                candidate.layout.set = use.descriptorSet;
                candidate.layout.binding = use.binding;
                candidate.binding.source = VulkanPipelineState::Binding::EXTERNAL_TEXTURE;
            } else {
                return false;
            }
            candidates.push_back(std::move(candidate));
            return true;
        };
        for (const Parameter &parameter : plan.parameters) {
            for (const ParameterUse &use : parameter.uses)
                if (!addUse(parameter, use, false)) {
                    supported = false;
                    break;
                }
            if (!supported)
                break;
        }
        for (const Parameter &parameter : plan.runtimeParameters)
            for (const ParameterUse &use : parameter.uses)
                if (!supported || !addUse(parameter, use, true)) {
                    supported = false;
                    break;
                }
        if (!supported) {
            std::string &diagnostic = invocationDiagnostic(*inputs.context);
            if (diagnostic.empty())
                diagnostic = "Vulkan RuntimeCore graphics path does not support this parameter layout";
            for (const Parameter &parameter : plan.parameters)
                for (const ParameterUse &use : parameter.uses)
                    diagnostic += " [" + parameter.name + ": kind=" + parameter.kind +
                                  ", interface=" + use.interfaceKind + ", stage=" + use.stage + ", dtype=" + use.dtype +
                                  ", transport=" + use.transport +
                                  ", plan=" + (use.interfacePlan && use.interfacePlan->root ? "yes" : "no") + "]";
            delete state;
            return false;
        }
        std::sort(candidates.begin(), candidates.end(),
                  [](const auto &left, const auto &right) { return left.layout.slot < right.layout.slot; });
        for (auto &candidate : candidates) {
            state->rhiGraphicsLayout.push_back(candidate.layout);
            state->rhiGraphicsVertexAttributes.insert(state->rhiGraphicsVertexAttributes.end(),
                                                      candidate.attributes.begin(), candidate.attributes.end());
            state->rhiGraphicsBindingPlan.push_back(std::move(candidate.binding));
        }
        state->rhiGraphicsValues.resize(candidates.size());
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
        descriptor.bindings = state->rhiGraphicsLayout.data();
        descriptor.binding_count = state->rhiGraphicsLayout.size();
        descriptor.vertex_attributes = state->rhiGraphicsVertexAttributes.data();
        descriptor.vertex_attribute_count = state->rhiGraphicsVertexAttributes.size();
        descriptor.topology = VERNON_TOPOLOGY_TRIANGLE_LIST;
        descriptor.sample_count = 1;
        const VernonStatus status =
            vernonRuntimeCorePreparePipeline(vernonRuntimeRhiAdapterGetProvider(vulkanState(*inputs.context).adapter),
                                             &descriptor, &state->rhiGraphicsPipeline);
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
    for (size_t index = 0; index < state.rhiGraphicsLayout.size(); ++index) {
        const auto &layout = state.rhiGraphicsLayout[index];
        auto &prepared = state.rhiGraphicsBindingPlan[index];
        auto &value = state.rhiGraphicsValues[index];
        value = {};
        value.slot = layout.slot;
        value.kind = layout.kind;
        if (prepared.source == VulkanPipelineState::Binding::EXTERNAL_VERTEX) {
            const auto found = plan.arguments.find(prepared.externalSlot);
            if (found == plan.arguments.end() || found->second->kind != VERNON_PROGRAM_TENSOR ||
                found->second->tensor.storage != VERNON_TENSOR_RHI_RESOURCE ||
                !found->second->tensor.resource.resource.value || !found->second->tensor.byte_strides ||
                found->second->tensor.byte_strides[0] <= 0)
                return fail(*pipeline.context, "Vulkan RHI vertex argument is missing or invalid");
            const VernonTensorView &tensor = found->second->tensor;
            value.payload.buffer.resource = tensor.resource;
            value.payload.buffer.resource.offset += tensor.byte_offset;
            value.payload.buffer.stride = static_cast<uint32_t>(tensor.byte_strides[0]);
        } else if (prepared.source == VulkanPipelineState::Binding::EXTERNAL_STORAGE) {
            const auto found = plan.arguments.find(prepared.externalSlot);
            if (found == plan.arguments.end() || found->second->kind != VERNON_PROGRAM_TENSOR ||
                found->second->tensor.storage != VERNON_TENSOR_RHI_RESOURCE ||
                !found->second->tensor.resource.resource.value)
                return fail(*pipeline.context, "Vulkan RHI storage argument is missing");
            const VernonTensorView &tensor = found->second->tensor;
            value.payload.buffer.resource = tensor.resource;
            value.payload.buffer.resource.offset += tensor.byte_offset;
        } else if (prepared.source == VulkanPipelineState::Binding::EXTERNAL_TEXTURE) {
            const auto found = plan.arguments.find(prepared.externalSlot);
            if (found == plan.arguments.end() || found->second->kind != VERNON_PROGRAM_IMAGE ||
                !found->second->image.view.resource.value)
                return fail(*pipeline.context, "Vulkan RHI image argument is missing");
            value.payload.image.view = found->second->image.view;
        } else if (prepared.source == VulkanPipelineState::Binding::EXTERNAL_SAMPLER) {
            const auto found = plan.arguments.find(prepared.externalSlot);
            if (found == plan.arguments.end() || found->second->kind != VERNON_PROGRAM_SAMPLER ||
                !found->second->resource.resource.value)
                return fail(*pipeline.context, "Vulkan RHI sampler argument is missing");
            value.payload.sampler.resource = found->second->resource;
        } else if (prepared.source == VulkanPipelineState::Binding::IMPLICIT_SAMPLER) {
            const auto sampled = plan.sampledResources.find({layout.set, layout.binding});
            if (sampled == plan.sampledResources.end())
                return fail(*pipeline.context, "Vulkan RHI implicit sampler binding is missing");
            if (sampled->second.samplerResource.resource.value) {
                value.payload.sampler.resource = sampled->second.samplerResource;
            } else {
                value.flags = VERNON_RUNTIME_PROVIDER_BINDING_DEFAULT_RESOURCE;
            }
        } else if (prepared.source == VulkanPipelineState::Binding::RESOLUTION) {
            std::memcpy(prepared.storage.data(), plan.resolution.data(),
                        std::min(prepared.storage.size(), sizeof(plan.resolution)));
            value.payload.inline_value.data = prepared.storage.data();
            value.payload.inline_value.size = prepared.storage.size();
        } else {
            const auto found = plan.arguments.find(prepared.externalSlot);
            if (found == plan.arguments.end() || found->second->kind != VERNON_PROGRAM_TENSOR)
                return fail(*pipeline.context, "Vulkan RHI uniform argument is missing");
            const VernonTensorView &tensor = found->second->tensor;
            const std::optional<std::vector<uint8_t>> packed = packTensor(tensor, prepared.packing);
            if (!packed || packed->size() != prepared.storage.size())
                return fail(*pipeline.context, "Vulkan RHI uniform Tensor is invalid");
            prepared.storage = *packed;
            value.payload.inline_value.data = prepared.storage.data();
            value.payload.inline_value.size = prepared.storage.size();
        }
    }
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
    if (!prepareGraphicsDraw(invocation, plan, std::move(formats), depthFormat, state.rhiGraphicsLayout,
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
    for (size_t index = 0; index < state.rhiComputeLayout.size(); ++index) {
        const auto &layout = state.rhiComputeLayout[index];
        const ComputeLaunchArgument &argument = launch.arguments[layout.argument_index];
        auto &value = state.rhiComputeValues[index];
        value = {};
        value.slot = layout.slot;
        value.kind = layout.kind;
        const ComputeBindingSource &source = state.rhiComputeBindingSources[index];
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
            const uint64_t leafOffset = state.rhiComputeResourceOffsets[index];
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
