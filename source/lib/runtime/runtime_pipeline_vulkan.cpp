#include "runtime_pipeline_backend.h"

#if defined(VERNON_HAS_VULKAN_RUNTIME)
#include "backend_vulkan.h"
#include "compute_launch_planner.h"
#include "rhi_adapter/adapter_internal.h"
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
    context.error = std::move(error);
    return status;
}

} // namespace
#endif

bool resolveVulkanPipeline(VernonPipelineBundle &bundle, const Variant &variant, VernonLoadedPipeline &pipeline) {
#if defined(VERNON_HAS_VULKAN_RUNTIME)
    auto *state = new VulkanPipelineState();
    if (!variant.compute.empty()) {
        const Stage &stage = bundle.stages.at(variant.compute);
        ReflectedEntry reflection;
        const nlohmann::json parsed = nlohmann::json::parse(stage.reflection, nullptr, false);
        if (parsed.is_discarded() || !parseReflection(parsed, stage.entry, reflection, bundle.context->error)) {
            delete state;
            return false;
        }
        struct Candidate {
            VernonRuntimeProviderBindingLayoutEntry layout{};
            uint64_t resourceOffset{};
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
        for (const Parameter &parameter : variant.parameters)
            internalSlot = std::max(internalSlot, parameter.slot);
        for (const Parameter &parameter : variant.parameters)
            for (const ParameterUse &use : parameter.uses) {
                if (use.stage != "compute" && use.stage != variant.compute)
                    continue;
                if (use.index >= reflection.arguments.size()) {
                    bundle.context->error = "Vulkan parameter use exceeds reflected argument table";
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
                    candidate.layout.kind = argument.kind == "tensor" ? VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER
                                                                      : VERNON_RUNTIME_PROVIDER_INLINE_VALUE;
                    candidate.layout.stage_mask = VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE;
                    candidate.layout.array_count = 1;
                    candidate.layout.argument_index = use.index;
                    candidate.layout.element_size = static_cast<uint32_t>(
                        argument.storageLeaves.empty()
                            ? (argument.kind == "tensor" ? argument.tensorElementSize : argument.physicalSize)
                            : argument.storageLeaves[leafIndex].elementSize);
                    // Aggregate lowering already folds each leaf's byte offset into the shader index.
                    // Every leaf descriptor must therefore retain the base address of the original AoS buffer.
                    candidate.resourceOffset = 0;
                    candidates.push_back(candidate);
                }
            }
        std::sort(candidates.begin(), candidates.end(),
                  [](const auto &left, const auto &right) { return left.layout.slot < right.layout.slot; });
        for (const auto &candidate : candidates) {
            state->rhiComputeLayout.push_back(candidate.layout);
            state->rhiComputeResourceOffsets.push_back(candidate.resourceOffset);
        }
        state->rhiComputeValues.resize(candidates.size());
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
            vernonRuntimeCorePreparePipeline(vernonRuntimeRhiAdapterGetProvider(vulkanState(*bundle.context).adapter),
                                             &descriptor, &state->rhiComputePipeline);
        if (status != VERNON_STATUS_OK) {
            const VernonStringView providerError =
                vernonRuntimeRhiAdapterGetLastError(vulkanState(*bundle.context).adapter);
            bundle.context->error = providerError.data ? std::string(providerError.data, providerError.size)
                                                       : "failed to prepare Vulkan provider compute pipeline";
            delete state;
            return false;
        }
    }
    if (!variant.vertex.empty()) {
        const Stage &vertex = bundle.stages.at(variant.vertex);
        const Stage &fragment = bundle.stages.at(variant.fragment);
        struct Candidate {
            VernonRuntimeProviderBindingLayoutEntry layout{};
            std::vector<VernonRuntimeProviderVertexAttribute> attributes;
            VulkanPipelineState::Binding binding;
        };
        std::vector<Candidate> candidates;
        uint32_t maximumExternalSlot = 0;
        uint32_t vertexBinding = 0;
        bool supported = true;
        const auto addParameter = [&](const Parameter &parameter, uint32_t slot, bool internal) {
            if (parameter.uses.size() != 1)
                return false;
            const ParameterUse &use = parameter.uses[0];
            Candidate candidate;
            candidate.layout.slot = slot;
            candidate.layout.stage_mask =
                use.stage == "vertex" ? VERNON_RUNTIME_PROVIDER_STAGE_VERTEX : VERNON_RUNTIME_PROVIDER_STAGE_FRAGMENT;
            candidate.layout.array_count = 1;
            candidate.binding.externalSlot = parameter.slot;
            if (parameter.kind == "tensor" && use.interfaceKind == "uniform") {
                const auto &shape = use.shape.empty() ? parameter.shape : use.shape;
                const std::optional<VernonDataType> dtype = pipelineDataType(parameter.dtype);
                uint64_t count = 1;
                for (uint64_t dimension : shape) {
                    if (!dimension || count > UINT32_MAX / dimension)
                        return false;
                    count *= dimension;
                }
                if (!dtype)
                    return false;
                const size_t elementSize = dataTypeSize(*dtype);
                candidate.layout.kind = use.uniformLayout && use.uniformLayout->storage == "uniform_buffer"
                                            ? VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER
                                            : VERNON_RUNTIME_PROVIDER_INLINE_VALUE;
                const uint64_t physicalSize = use.uniformLayout ? use.uniformLayout->size : count * elementSize;
                if (!physicalSize || physicalSize > UINT32_MAX ||
                    (candidate.layout.kind == VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER && use.binding == UINT32_MAX))
                    return false;
                candidate.layout.element_size = static_cast<uint32_t>(physicalSize);
                candidate.layout.interface_kind = VERNON_RUNTIME_PROVIDER_INTERFACE_UNIFORM;
                candidate.layout.element_count = static_cast<uint32_t>(count);
                candidate.layout.binding = use.binding;
                candidate.layout.set = use.descriptorSet;
                candidate.binding.source = internal ? VulkanPipelineState::Binding::RESOLUTION
                                                    : VulkanPipelineState::Binding::EXTERNAL_UNIFORM;
                candidate.binding.packing.dtype = *dtype;
                candidate.binding.packing.shape = shape;
                candidate.binding.packing.byteSize = static_cast<size_t>(physicalSize);
                if (use.uniformLayout) {
                    candidate.binding.packing.byteStrides.reserve(use.uniformLayout->byteStrides.size());
                    for (uint64_t stride : use.uniformLayout->byteStrides) {
                        if (stride > SIZE_MAX)
                            return false;
                        candidate.binding.packing.byteStrides.push_back(static_cast<size_t>(stride));
                    }
                } else {
                    candidate.binding.packing.byteStrides.resize(shape.size());
                    size_t stride = elementSize;
                    const bool columnMajorMatrix =
                        shape.size() == 2 && shape[0] == shape[1] && (shape[0] == 3 || shape[0] == 4);
                    if (columnMajorMatrix) {
                        candidate.binding.packing.byteStrides = {elementSize,
                                                                 static_cast<size_t>(shape[0]) * elementSize};
                    } else {
                        for (size_t dimension = shape.size(); dimension-- > 0;) {
                            candidate.binding.packing.byteStrides[dimension] = stride;
                            stride *= static_cast<size_t>(shape[dimension]);
                        }
                    }
                }
                if (candidate.binding.packing.byteStrides.size() != shape.size())
                    return false;
                candidate.binding.storage.resize(candidate.layout.element_size);
            } else if (parameter.kind == "tensor" && use.interfaceKind == "input" && use.stage == "vertex" &&
                       use.location != UINT32_MAX && !use.attributeLeaves.empty()) {
                const std::optional<VernonDataType> dtype = pipelineDataType(parameter.dtype);
                if (!dtype || dataTypeSize(*dtype) == 0)
                    return false;
                candidate.layout.kind = VERNON_RUNTIME_PROVIDER_VERTEX_BUFFER;
                candidate.layout.element_size = static_cast<uint32_t>(dataTypeSize(*dtype));
                candidate.layout.interface_kind = VERNON_RUNTIME_PROVIDER_INTERFACE_VERTEX_INPUT;
                candidate.layout.binding = vertexBinding++;
                candidate.layout.divisor = use.divisor;
                for (const AttributeLeaf &leaf : use.attributeLeaves)
                    candidate.attributes.push_back({candidate.layout.binding, use.location + leaf.locationOffset,
                                                    static_cast<uint32_t>(*dtype), leaf.componentCount,
                                                    leaf.byteOffset});
                candidate.binding.source = VulkanPipelineState::Binding::EXTERNAL_VERTEX;
            } else if (parameter.kind == "texture" && use.interfaceKind == "resource") {
                candidate.layout.kind = VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE;
                candidate.layout.interface_kind = VERNON_RUNTIME_PROVIDER_INTERFACE_RESOURCE;
                candidate.layout.binding = use.binding;
                candidate.binding.source = VulkanPipelineState::Binding::EXTERNAL_TEXTURE;
            } else if (parameter.kind == "sampler" && use.sampledTextureBindings.size() == 1) {
                candidate.layout.kind = VERNON_RUNTIME_PROVIDER_SAMPLER;
                candidate.layout.interface_kind = VERNON_RUNTIME_PROVIDER_INTERFACE_RESOURCE;
                candidate.layout.binding = use.sampledTextureBindings[0].binding;
                candidate.binding.source = internal ? VulkanPipelineState::Binding::IMPLICIT_SAMPLER
                                                    : VulkanPipelineState::Binding::EXTERNAL_SAMPLER;
            } else {
                return false;
            }
            candidates.push_back(std::move(candidate));
            return true;
        };
        for (const Parameter &parameter : variant.parameters) {
            maximumExternalSlot = std::max(maximumExternalSlot, parameter.slot);
            if (!addParameter(parameter, parameter.slot, false)) {
                supported = false;
                break;
            }
        }
        uint32_t internalSlot = maximumExternalSlot;
        for (const Parameter &parameter : variant.internalParameters)
            if (!supported || !addParameter(parameter, ++internalSlot, true)) {
                supported = false;
                break;
            }
        if (!supported) {
            bundle.context->error = "Vulkan RuntimeCore graphics path does not support this parameter layout";
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
        const VernonRuntimeProviderShaderDescriptor shaders[2]{{sizeof(VernonRuntimeProviderShaderDescriptor),
                                                                VERNON_RUNTIME_PROVIDER_STAGE_VERTEX,
                                                                {"spirv", 5},
                                                                vertex.binary.data(),
                                                                vertex.binary.size(),
                                                                {vertex.entry.data(), vertex.entry.size()},
                                                                {},
                                                                {0, 0, 0, 0}},
                                                               {sizeof(VernonRuntimeProviderShaderDescriptor),
                                                                VERNON_RUNTIME_PROVIDER_STAGE_FRAGMENT,
                                                                {"spirv", 5},
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
        descriptor.bindings = state->rhiGraphicsLayout.data();
        descriptor.binding_count = state->rhiGraphicsLayout.size();
        descriptor.vertex_attributes = state->rhiGraphicsVertexAttributes.data();
        descriptor.vertex_attribute_count = state->rhiGraphicsVertexAttributes.size();
        descriptor.topology = VERNON_TOPOLOGY_TRIANGLE_LIST;
        descriptor.sample_count = 1;
        const VernonStatus status =
            vernonRuntimeCorePreparePipeline(vernonRuntimeRhiAdapterGetProvider(vulkanState(*bundle.context).adapter),
                                             &descriptor, &state->rhiGraphicsPipeline);
        if (status != VERNON_STATUS_OK) {
            const VernonStringView providerError =
                vernonRuntimeRhiAdapterGetLastError(vulkanState(*bundle.context).adapter);
            bundle.context->error = providerError.data ? std::string(providerError.data, providerError.size)
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

void destroyVulkanPipeline(VernonLoadedPipeline &pipeline) {
#if defined(VERNON_HAS_VULKAN_RUNTIME)
    VulkanPipelineState &state = runtimeBackendState<VulkanPipelineState>(pipeline);
    vernonRuntimeCoreBindingsDestroy(state.rhiComputeBindings);
    vernonRuntimeCorePipelineDestroy(state.rhiComputePipeline);
    vernonRuntimeCoreBindingsDestroy(state.rhiGraphicsBindings);
    vernonRuntimeCoreGraphicsVariantDestroy(state.rhiGraphicsVariant);
    vernonRuntimeCorePipelineDestroy(state.rhiGraphicsPipeline);
#else
    (void)pipeline;
#endif
}

VernonStatus invokeVulkanGraphicsPipeline(VernonLoadedPipeline &pipeline, const VernonPipelineInvocation &invocation,
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
            if (found == plan.arguments.end() || found->second->kind != VERNON_PIPELINE_TENSOR ||
                (found->second->tensor.storage != VERNON_TENSOR_DEVICE &&
                 found->second->tensor.storage != VERNON_TENSOR_RHI_RESOURCE) ||
                (!found->second->tensor.buffer && !found->second->tensor.resource.resource.value) ||
                !found->second->tensor.byte_strides || found->second->tensor.byte_strides[0] <= 0)
                return fail(*pipeline.context, "Vulkan RHI vertex argument is missing or invalid");
            const VernonTensorView &tensor = found->second->tensor;
            if (tensor.storage == VERNON_TENSOR_RHI_RESOURCE) {
                value.resource = tensor.resource;
                value.resource.offset += tensor.byte_offset;
            } else {
                value.resource.identity = vulkanRhiAdapterResourceIdentity(adapter);
                value.resource.resource.value = reinterpret_cast<uintptr_t>(&vulkanBufferState(*tensor.buffer));
                value.resource.offset = tensor.byte_offset;
                value.resource.size = tensor.buffer->size - tensor.byte_offset;
            }
            value.stride = static_cast<uint32_t>(tensor.byte_strides[0]);
        } else if (prepared.source == VulkanPipelineState::Binding::EXTERNAL_TEXTURE) {
            const auto found = plan.arguments.find(prepared.externalSlot);
            if (found == plan.arguments.end() || found->second->kind != VERNON_PIPELINE_TEXTURE ||
                (!found->second->texture.texture && !found->second->texture.resource.resource.value))
                return fail(*pipeline.context, "Vulkan RHI texture argument is missing");
            if (found->second->texture.resource.resource.value)
                value.resource = found->second->texture.resource;
            else {
                value.resource.identity = vulkanRhiAdapterResourceIdentity(adapter);
                value.resource.resource.value =
                    reinterpret_cast<uintptr_t>(&vulkanTextureState(*found->second->texture.texture));
            }
        } else if (prepared.source == VulkanPipelineState::Binding::EXTERNAL_SAMPLER) {
            const auto found = plan.arguments.find(prepared.externalSlot);
            if (found == plan.arguments.end() || found->second->kind != VERNON_PIPELINE_SAMPLER ||
                (!found->second->sampler && !found->second->resource.resource.value))
                return fail(*pipeline.context, "Vulkan RHI sampler argument is missing");
            if (found->second->resource.resource.value)
                value.resource = found->second->resource;
            else {
                value.resource.identity = vulkanRhiAdapterResourceIdentity(adapter);
                value.resource.resource.value =
                    reinterpret_cast<uintptr_t>(&vulkanSamplerState(*found->second->sampler));
            }
        } else if (prepared.source == VulkanPipelineState::Binding::IMPLICIT_SAMPLER) {
            const auto sampled = plan.sampledResources.find({layout.set, layout.binding});
            if (sampled == plan.sampledResources.end())
                return fail(*pipeline.context, "Vulkan RHI implicit sampler binding is missing");
            if (sampled->second.sampler) {
                value.resource.identity = vulkanRhiAdapterResourceIdentity(adapter);
                value.resource.resource.value =
                    reinterpret_cast<uintptr_t>(&vulkanSamplerState(*sampled->second.sampler));
            } else if (sampled->second.samplerResource.resource.value) {
                value.resource = sampled->second.samplerResource;
            } else {
                value.flags = VERNON_RUNTIME_PROVIDER_BINDING_DEFAULT_RESOURCE;
            }
        } else if (prepared.source == VulkanPipelineState::Binding::RESOLUTION) {
            std::memcpy(prepared.storage.data(), plan.resolution.data(),
                        std::min(prepared.storage.size(), sizeof(plan.resolution)));
            value.inline_data = prepared.storage.data();
            value.inline_size = prepared.storage.size();
        } else {
            const auto found = plan.arguments.find(prepared.externalSlot);
            if (found == plan.arguments.end() || found->second->kind != VERNON_PIPELINE_TENSOR)
                return fail(*pipeline.context, "Vulkan RHI uniform argument is missing");
            const VernonTensorView &tensor = found->second->tensor;
            const std::optional<std::vector<uint8_t>> packed = packTensor(tensor, prepared.packing);
            if (!packed || packed->size() != prepared.storage.size())
                return fail(*pipeline.context, "Vulkan RHI uniform Tensor is invalid");
            prepared.storage = *packed;
            value.inline_data = prepared.storage.data();
            value.inline_size = prepared.storage.size();
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
    if (plan.attachments.empty() || plan.attachments.size() > 8)
        return fail(*pipeline.context, "Vulkan RHI draw requires one to eight color attachments");
    std::array<VernonRuntimeProviderColorAttachment, 8> attachments{};
    VernonRuntimeProviderResourceReference depthAttachment{};
    std::vector<uint32_t> formats;
    for (size_t index = 0; index < plan.attachments.size(); ++index) {
        attachments[index].location = plan.attachments[index]->location;
        if (plan.attachments[index]->resource.resource.value) {
            attachments[index].image = plan.attachments[index]->resource;
            const auto *image = reinterpret_cast<const rhi::vulkan::Image *>(
                static_cast<uintptr_t>(plan.attachments[index]->resource.resource.value));
            formats.push_back(static_cast<uint32_t>(image->format));
        } else {
            auto &texture = vulkanTextureState(*plan.attachments[index]->texture);
            attachments[index].image.identity = vulkanRhiAdapterResourceIdentity(adapter);
            attachments[index].image.resource.value = reinterpret_cast<uintptr_t>(&texture);
            formats.push_back(static_cast<uint32_t>(texture.format));
        }
    }
    if (plan.depthAttachment) {
        if (plan.depthAttachment->resource.resource.value)
            depthAttachment = plan.depthAttachment->resource;
        else {
            auto &texture = vulkanTextureState(*plan.depthAttachment->texture);
            depthAttachment.identity = vulkanRhiAdapterResourceIdentity(adapter);
            depthAttachment.resource.value = reinterpret_cast<uintptr_t>(&texture);
        }
    }
    uint64_t vertexIdentity = 1469598103934665603ull;
    std::vector<uint32_t> vertexStrides;
    for (size_t index = 0; index < state.rhiGraphicsLayout.size(); ++index) {
        if (state.rhiGraphicsLayout[index].kind != VERNON_RUNTIME_PROVIDER_VERTEX_BUFFER)
            continue;
        const uint32_t binding = state.rhiGraphicsLayout[index].binding;
        if (vertexStrides.size() <= binding)
            vertexStrides.resize(binding + 1);
        vertexStrides[binding] = state.rhiGraphicsValues[index].stride;
        vertexIdentity = (vertexIdentity ^ state.rhiGraphicsValues[index].stride) * 1099511628211ull;
    }
    const uint32_t depthFormat = plan.depthAttachment ? static_cast<uint32_t>(VK_FORMAT_D32_SFLOAT) : 0;
    if (!state.rhiGraphicsVariant || state.rhiGraphicsFormats != formats ||
        state.rhiGraphicsDepthFormat != depthFormat || state.rhiGraphicsTopology != invocation.topology ||
        state.rhiGraphicsVertexLayoutIdentity != vertexIdentity) {
        vernonRuntimeCoreGraphicsVariantDestroy(state.rhiGraphicsVariant);
        state.rhiGraphicsVariant = nullptr;
        const VernonRuntimeCoreGraphicsCompatibility compatibility{sizeof(VernonRuntimeCoreGraphicsCompatibility),
                                                                   static_cast<uint32_t>(invocation.topology),
                                                                   formats.data(),
                                                                   formats.size(),
                                                                   depthFormat,
                                                                   1,
                                                                   vertexStrides.data(),
                                                                   vertexStrides.size(),
                                                                   vertexIdentity,
                                                                   {0, 0, 0, 0}};
        status = vernonRuntimeCorePrepareGraphicsVariant(state.rhiGraphicsPipeline, &compatibility,
                                                         &state.rhiGraphicsVariant);
        if (status != VERNON_STATUS_OK) {
            const VernonStringView providerError =
                vernonRuntimeRhiAdapterGetLastError(vulkanState(*pipeline.context).adapter);
            return fail(*pipeline.context,
                        providerError.data ? std::string(providerError.data, providerError.size)
                                           : "failed to prepare Vulkan RHI graphics variant",
                        status);
        }
        state.rhiGraphicsFormats = std::move(formats);
        state.rhiGraphicsDepthFormat = depthFormat;
        state.rhiGraphicsTopology = invocation.topology;
        state.rhiGraphicsVertexLayoutIdentity = vertexIdentity;
    }
    const bool hasViewport = invocation.viewport[2] && invocation.viewport[3];
    VernonRuntimeCoreDrawInvocation draw{};
    draw.struct_size = sizeof(draw);
    draw.vertex_count = plan.vertexCount;
    draw.instance_count = plan.instanceCount;
    draw.color_attachments = attachments.data();
    draw.color_attachment_count = plan.attachments.size();
    draw.depth_stencil_attachment = depthAttachment;
    draw.viewport[0] = hasViewport ? invocation.viewport[0] : 0;
    draw.viewport[1] = hasViewport ? invocation.viewport[1] : 0;
    draw.viewport[2] = hasViewport ? invocation.viewport[2] : plan.attachmentWidth;
    draw.viewport[3] = hasViewport ? invocation.viewport[3] : plan.attachmentHeight;
    draw.topology = invocation.topology;
    status = vernonRuntimeCoreEncodeGraphicsVariantDrawInvocation(state.rhiGraphicsVariant, state.rhiGraphicsBindings,
                                                                  &draw);
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

VernonStatus invokeVulkanComputePipeline(VernonLoadedPipeline &pipeline, const PlannedComputeLaunch &launch) {
#if defined(VERNON_HAS_VULKAN_RUNTIME)
    VulkanPipelineState &state = runtimeBackendState<VulkanPipelineState>(pipeline);
    const uint64_t identity = vulkanRhiAdapterResourceIdentity(*vulkanState(*pipeline.context).adapter);
    for (size_t index = 0; index < state.rhiComputeLayout.size(); ++index) {
        const auto &layout = state.rhiComputeLayout[index];
        const ComputeLaunchArgument &argument = launch.arguments[layout.argument_index];
        auto &value = state.rhiComputeValues[index];
        value = {};
        value.slot = layout.slot;
        value.kind = layout.kind;
        if (layout.kind == VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER) {
            if (argument.kind != ComputeLaunchArgumentKind::Tensor ||
                (!argument.buffer && !argument.resource.resource.value))
                return fail(*pipeline.context, "Vulkan prepared storage binding requires a device Tensor");
            if (argument.resource.resource.value) {
                value.resource = argument.resource;
                const uint64_t leafOffset = state.rhiComputeResourceOffsets[index];
                if (leafOffset > value.resource.size)
                    return fail(*pipeline.context, "Vulkan aggregate storage leaf exceeds its Tensor resource");
                value.resource.offset += leafOffset;
                value.resource.size -= leafOffset;
            } else {
                const uint64_t leafOffset = state.rhiComputeResourceOffsets[index];
                if (leafOffset > argument.buffer->size)
                    return fail(*pipeline.context, "Vulkan aggregate storage leaf exceeds its Tensor buffer");
                value.resource.identity = identity;
                value.resource.resource.value = reinterpret_cast<uintptr_t>(&vulkanBufferState(*argument.buffer));
                value.resource.offset = leafOffset;
                value.resource.size = argument.buffer->size - leafOffset;
            }
        } else {
            if (argument.kind != ComputeLaunchArgumentKind::Scalar || !argument.scalarData)
                return fail(*pipeline.context, "Vulkan prepared inline binding requires host data");
            value.inline_data = argument.scalarData;
            value.inline_size = argument.scalarSize;
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
    const uint32_t groups[3]{(launch.grid.x - 1) / state.rhiComputeWorkgroup[0] + 1,
                             (launch.grid.y - 1) / state.rhiComputeWorkgroup[1] + 1,
                             (launch.grid.z - 1) / state.rhiComputeWorkgroup[2] + 1};
    status =
        vernonRuntimeCoreEncodeDispatch(state.rhiComputePipeline, state.rhiComputeBindings, {}, groups, nullptr, 0);
    return status == VERNON_STATUS_OK ? status
                                      : fail(*pipeline.context, "failed to encode Vulkan provider dispatch", status);
#else
    (void)pipeline;
    (void)launch;
    return VERNON_STATUS_UNSUPPORTED_TARGET;
#endif
}

} // namespace vernon::runtime
