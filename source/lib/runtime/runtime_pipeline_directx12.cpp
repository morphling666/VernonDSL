#include "runtime_pipeline_backend.h"

#if defined(VERNON_HAS_DIRECTX12_RUNTIME)
#include "../rhi/rhi_internal.h"
#include "backend_directx12.h"
#include "compute_launch_planner.h"
#include "rhi_adapter/adapter_internal.h"
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
    context.error = std::move(error);
    return status;
}

} // namespace
#endif

bool resolveDirectX12Pipeline(VernonPipelineBundle &bundle, const Variant &variant, VernonLoadedPipeline &pipeline) {
#if defined(VERNON_HAS_DIRECTX12_RUNTIME)
    auto *pipelineState = new DirectX12PipelineState();
    if (!variant.compute.empty()) {
        const Stage &stage = bundle.stages.at(variant.compute);
        ReflectedEntry reflection;
        const nlohmann::json parsed = nlohmann::json::parse(stage.reflection, nullptr, false);
        if (parsed.is_discarded() ||
            !parseReflection(parsed, stage.entry, reflection, VERNON_RUNTIME_DIRECTX12, bundle.context->error)) {
            delete pipelineState;
            return false;
        }
        struct BindingCandidate {
            VernonRuntimeProviderBindingLayoutEntry layout{};
            uint64_t resourceOffset{};
        };
        std::vector<BindingCandidate> candidates;
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
                    bundle.context->error = "D3D12 parameter use exceeds reflected argument table";
                    delete pipelineState;
                    return false;
                }
                const ReflectedArgument &argument = reflection.arguments[use.index];
                const size_t leafCount = std::max(argument.storageLeaves.size(), size_t{1});
                for (size_t leafIndex = 0; leafIndex < leafCount; ++leafIndex) {
                    BindingCandidate candidate;
                    candidate.layout.slot = leafIndex == 0 ? parameter.slot : ++internalSlot;
                    candidate.layout.set = argument.descriptorSet;
                    // DirectX compute lowering assigns the flattened UAV ABI
                    // in source argument order, independent of canonical
                    // PipelineAsset slot ordering.
                    candidate.layout.binding = argumentBindings[use.index] + static_cast<uint32_t>(leafIndex);
                    candidate.layout.kind = argument.kind == "tensor" ? VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER
                                                                      : VERNON_RUNTIME_PROVIDER_INLINE_VALUE;
                    candidate.layout.stage_mask = VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE;
                    candidate.layout.access = parameter.access == "read" ? 1u : parameter.access == "write" ? 2u : 3u;
                    candidate.layout.array_count = 1;
                    candidate.layout.argument_index = use.index;
                    candidate.layout.element_size = static_cast<uint32_t>(
                        argument.storageLeaves.empty()
                            ? (argument.kind == "tensor" ? argument.tensorElementSize : argument.physical.size)
                            : argument.storageLeaves[leafIndex].elementSize);
                    // Aggregate lowering already folds each leaf's byte offset into the shader index.
                    // Every leaf descriptor must therefore retain the base address of the original AoS buffer.
                    candidate.resourceOffset = 0;
                    if (candidate.layout.binding == UINT32_MAX || candidate.layout.element_size == 0) {
                        bundle.context->error = "D3D12 reflected compute binding is incomplete";
                        delete pipelineState;
                        return false;
                    }
                    candidates.push_back(candidate);
                }
            }
        std::sort(candidates.begin(), candidates.end(),
                  [](const auto &left, const auto &right) { return left.layout.slot < right.layout.slot; });
        pipelineState->rhiComputeLayout.reserve(candidates.size());
        pipelineState->rhiComputeResourceOffsets.reserve(candidates.size());
        for (const BindingCandidate &candidate : candidates) {
            pipelineState->rhiComputeLayout.push_back(candidate.layout);
            pipelineState->rhiComputeResourceOffsets.push_back(candidate.resourceOffset);
        }
        pipelineState->rhiComputeValues.resize(candidates.size());
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
        descriptor.bindings = pipelineState->rhiComputeLayout.data();
        descriptor.binding_count = pipelineState->rhiComputeLayout.size();
        std::copy_n(pipelineState->rhiComputeWorkgroup, 3, descriptor.workgroup_size);
        const VernonStatus status = vernonRuntimeCorePreparePipeline(
            vernonRuntimeRhiAdapterGetProvider(directX12State(*bundle.context).adapter), &descriptor,
            &pipelineState->rhiComputePipeline);
        if (status != VERNON_STATUS_OK) {
            const VernonStringView providerError =
                vernonRuntimeRhiAdapterGetLastError(directX12State(*bundle.context).adapter);
            bundle.context->error = providerError.data ? std::string(providerError.data, providerError.size)
                                                       : "failed to prepare D3D12 provider compute pipeline";
            delete pipelineState;
            return false;
        }
    } else {
        struct GraphicsCandidate {
            VernonRuntimeProviderBindingLayoutEntry layout{};
            std::vector<VernonRuntimeProviderVertexAttribute> attributes;
            DirectX12PipelineState::GraphicsBinding binding;
        };
        std::vector<GraphicsCandidate> candidates;
        uint32_t maximumExternalSlot = 0;
        uint32_t vertexInputSlot = 0;
        bool supported = true;
        for (const Parameter &parameter : variant.parameters) {
            maximumExternalSlot = std::max(maximumExternalSlot, parameter.slot);
            if (parameter.uses.size() != 1) {
                supported = false;
                break;
            }
            const ParameterUse &use = parameter.uses[0];
            GraphicsCandidate candidate;
            candidate.layout.slot = parameter.slot;
            candidate.layout.argument_index = use.index;
            candidate.layout.stage_mask =
                use.stage == "vertex" ? VERNON_RUNTIME_PROVIDER_STAGE_VERTEX : VERNON_RUNTIME_PROVIDER_STAGE_FRAGMENT;
            candidate.layout.array_count = 1;
            candidate.binding.externalSlot = parameter.slot;
            if (parameter.kind == "tensor" && use.interfaceKind == "uniform" && use.physicalValueLayout &&
                use.physicalValueLayout->transport == "storage_buffer" && parameter.elementLayout.byteSize &&
                use.binding != UINT32_MAX) {
                candidate.layout.kind = VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER;
                candidate.layout.element_size = parameter.elementLayout.byteSize;
                candidate.layout.interface_kind = VERNON_RUNTIME_PROVIDER_INTERFACE_RESOURCE;
                candidate.layout.binding = use.binding;
                candidate.layout.set = use.descriptorSet;
                candidate.binding.source = DirectX12PipelineState::GraphicsBinding::EXTERNAL_STORAGE;
            } else if (parameter.kind == "tensor" && use.interfaceKind == "uniform") {
                const auto &shape = use.shape.empty() ? parameter.shape : use.shape;
                const std::optional<VernonDataType> dtype = pipelineDataType(use.dtype);
                uint64_t valueCount = 1;
                for (uint64_t dimension : shape) {
                    if (!dimension || valueCount > UINT32_MAX / dimension) {
                        supported = false;
                        break;
                    }
                    valueCount *= dimension;
                }
                if (!supported || !dtype) {
                    supported = false;
                    break;
                }
                const size_t elementSize = dataTypeSize(*dtype);
                candidate.layout.kind = use.physicalValueLayout->transport == "uniform_buffer"
                                            ? VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER
                                            : VERNON_RUNTIME_PROVIDER_INLINE_VALUE;
                const uint64_t physicalSize = use.physicalValueLayout->size;
                if (!physicalSize || physicalSize > UINT32_MAX ||
                    (candidate.layout.kind == VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER && use.binding == UINT32_MAX)) {
                    supported = false;
                    break;
                }
                candidate.layout.element_size = static_cast<uint32_t>(physicalSize);
                candidate.layout.interface_kind = VERNON_RUNTIME_PROVIDER_INTERFACE_UNIFORM;
                candidate.layout.element_count = static_cast<uint32_t>(valueCount);
                candidate.layout.vector_count = shape.size() == 2 ? static_cast<uint32_t>(shape[0]) : 1;
                const uint64_t physicalAlignment = use.physicalValueLayout->alignment;
                if (!physicalAlignment || physicalAlignment > UINT32_MAX) {
                    supported = false;
                    break;
                }
                candidate.layout.element_alignment = static_cast<uint32_t>(physicalAlignment);
                candidate.layout.binding = use.binding;
                candidate.layout.set = use.descriptorSet;
                candidate.binding.source = DirectX12PipelineState::GraphicsBinding::EXTERNAL_UNIFORM;
                candidate.binding.packing.elementSize = elementSize;
                candidate.binding.packing.shape = shape;
                candidate.binding.packing.byteSize = static_cast<size_t>(physicalSize);
                for (uint64_t stride : use.physicalValueLayout->byteStrides) {
                    if (stride > SIZE_MAX) {
                        supported = false;
                        break;
                    }
                    candidate.binding.packing.byteStrides.push_back(static_cast<size_t>(stride));
                }
                if (!supported || candidate.binding.packing.byteStrides.size() != shape.size()) {
                    supported = false;
                    break;
                }
                candidate.binding.storage.resize(candidate.layout.element_size);
            } else if (parameter.kind == "tensor" && use.interfaceKind == "input" && use.stage == "vertex" &&
                       use.location != UINT32_MAX && !use.attributeLeaves.empty()) {
                candidate.layout.kind = VERNON_RUNTIME_PROVIDER_VERTEX_BUFFER;
                candidate.layout.element_size = parameter.elementLayout.byteSize;
                candidate.layout.interface_kind = VERNON_RUNTIME_PROVIDER_INTERFACE_VERTEX_INPUT;
                candidate.layout.binding = vertexInputSlot++;
                candidate.layout.divisor = use.divisor;
                for (const AttributeLeaf &leaf : use.attributeLeaves) {
                    const std::optional<VernonDataType> dtype = pipelineDataType(leaf.dtype);
                    if (!dtype) {
                        supported = false;
                        break;
                    }
                    candidate.attributes.push_back({candidate.layout.binding, use.location + leaf.locationOffset,
                                                    static_cast<uint32_t>(*dtype), leaf.componentCount,
                                                    leaf.byteOffset});
                }
                if (!supported)
                    break;
                candidate.binding.source = DirectX12PipelineState::GraphicsBinding::EXTERNAL_VERTEX;
            } else if (parameter.kind == "texture" && use.interfaceKind == "resource" && use.descriptorSet == 0 &&
                       use.binding != UINT32_MAX) {
                candidate.layout.kind = VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE;
                candidate.layout.interface_kind = VERNON_RUNTIME_PROVIDER_INTERFACE_RESOURCE;
                candidate.layout.binding = use.binding;
                candidate.binding.source = DirectX12PipelineState::GraphicsBinding::EXTERNAL_TEXTURE;
            } else if (parameter.kind == "sampler" && use.sampledTextureBindings.size() == 1 &&
                       use.sampledTextureBindings[0].descriptorSet == 0) {
                candidate.layout.kind = VERNON_RUNTIME_PROVIDER_SAMPLER;
                candidate.layout.interface_kind = VERNON_RUNTIME_PROVIDER_INTERFACE_RESOURCE;
                candidate.layout.binding = use.sampledTextureBindings[0].binding;
                candidate.binding.source = DirectX12PipelineState::GraphicsBinding::EXTERNAL_SAMPLER;
            } else {
                supported = false;
                break;
            }
            candidates.push_back(candidate);
        }
        uint32_t internalSlot = maximumExternalSlot;
        for (const Parameter &parameter : variant.internalParameters) {
            if (!supported || parameter.uses.size() != 1 || internalSlot == UINT32_MAX) {
                supported = false;
                break;
            }
            const ParameterUse &use = parameter.uses[0];
            GraphicsCandidate candidate;
            candidate.layout.slot = ++internalSlot;
            candidate.layout.argument_index = use.index;
            candidate.layout.stage_mask =
                use.stage == "vertex" ? VERNON_RUNTIME_PROVIDER_STAGE_VERTEX : VERNON_RUNTIME_PROVIDER_STAGE_FRAGMENT;
            candidate.layout.array_count = 1;
            if (parameter.source == "implicit_sampler" && parameter.kind == "sampler" &&
                use.sampledTextureBindings.size() == 1 && use.sampledTextureBindings[0].descriptorSet == 0) {
                candidate.layout.kind = VERNON_RUNTIME_PROVIDER_SAMPLER;
                candidate.layout.interface_kind = VERNON_RUNTIME_PROVIDER_INTERFACE_RESOURCE;
                candidate.layout.binding = use.sampledTextureBindings[0].binding;
                candidate.binding.source = DirectX12PipelineState::GraphicsBinding::IMPLICIT_SAMPLER;
            } else if (parameter.source == "system_value" && parameter.systemValue == "resolution" &&
                       parameter.kind == "tensor" && use.interfaceKind == "uniform" && use.physicalValueLayout) {
                const auto &shape = use.shape.empty() ? parameter.shape : use.shape;
                const std::optional<VernonDataType> dtype = pipelineDataType(use.dtype);
                const uint64_t physicalSize = use.physicalValueLayout->size;
                const uint64_t physicalAlignment = use.physicalValueLayout->alignment;
                if (shape != std::vector<uint64_t>{2} || dtype != VERNON_DATA_F32 ||
                    physicalSize != sizeof(float) * 2 || !physicalAlignment || physicalAlignment > UINT32_MAX ||
                    (use.physicalValueLayout->transport != "push_constant" &&
                     use.physicalValueLayout->transport != "uniform_buffer") ||
                    (use.physicalValueLayout->transport == "uniform_buffer" && use.binding == UINT32_MAX)) {
                    supported = false;
                    break;
                }
                candidate.layout.kind = use.physicalValueLayout->transport == "uniform_buffer"
                                            ? VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER
                                            : VERNON_RUNTIME_PROVIDER_INLINE_VALUE;
                candidate.layout.element_size = static_cast<uint32_t>(physicalSize);
                candidate.layout.interface_kind = VERNON_RUNTIME_PROVIDER_INTERFACE_UNIFORM;
                candidate.layout.element_count = 2;
                candidate.layout.vector_count = 1;
                candidate.layout.element_alignment = static_cast<uint32_t>(physicalAlignment);
                candidate.layout.binding = use.binding;
                candidate.layout.set = use.descriptorSet;
                candidate.binding.source = DirectX12PipelineState::GraphicsBinding::RESOLUTION;
                candidate.binding.storage.resize(candidate.layout.element_size);
            } else {
                supported = false;
                break;
            }
            candidates.push_back(candidate);
        }
        if (!supported) {
            bundle.context->error = "D3D12 RuntimeCore graphics path does not support this parameter layout";
            delete pipelineState;
            return false;
        }
        std::sort(candidates.begin(), candidates.end(),
                  [](const auto &left, const auto &right) { return left.layout.slot < right.layout.slot; });
        for (const auto &candidate : candidates) {
            pipelineState->rhiGraphicsLayout.push_back(candidate.layout);
            pipelineState->rhiGraphicsVertexAttributes.insert(pipelineState->rhiGraphicsVertexAttributes.end(),
                                                              candidate.attributes.begin(), candidate.attributes.end());
            pipelineState->rhiGraphicsBindingsPlan.push_back(candidate.binding);
        }
        pipelineState->rhiGraphicsValues.resize(candidates.size());
        const Stage &vertex = bundle.stages.at(variant.vertex);
        const Stage &fragment = bundle.stages.at(variant.fragment);
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
        descriptor.bindings = pipelineState->rhiGraphicsLayout.data();
        descriptor.binding_count = pipelineState->rhiGraphicsLayout.size();
        descriptor.vertex_attributes = pipelineState->rhiGraphicsVertexAttributes.data();
        descriptor.vertex_attribute_count = pipelineState->rhiGraphicsVertexAttributes.size();
        descriptor.topology = VERNON_TOPOLOGY_TRIANGLE_LIST;
        descriptor.sample_count = 1;
        const VernonStatus status = vernonRuntimeCorePreparePipeline(
            vernonRuntimeRhiAdapterGetProvider(directX12State(*bundle.context).adapter), &descriptor,
            &pipelineState->rhiGraphicsPipeline);
        if (status != VERNON_STATUS_OK) {
            const VernonStringView providerError =
                vernonRuntimeRhiAdapterGetLastError(directX12State(*bundle.context).adapter);
            bundle.context->error = providerError.data ? std::string(providerError.data, providerError.size)
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
}

void destroyDirectX12Pipeline(VernonLoadedPipeline &pipeline) {
#if defined(VERNON_HAS_DIRECTX12_RUNTIME)
    DirectX12PipelineState &state = runtimeBackendState<DirectX12PipelineState>(pipeline);
    vernonRuntimeCoreBindingsDestroy(state.rhiComputeBindings);
    vernonRuntimeCorePipelineDestroy(state.rhiComputePipeline);
    vernonRuntimeCoreBindingsDestroy(state.rhiGraphicsBindings);
    vernonRuntimeCoreGraphicsVariantDestroy(state.rhiGraphicsVariant);
    vernonRuntimeCorePipelineDestroy(state.rhiGraphicsPipeline);
#else
    (void)pipeline;
#endif
}

VernonStatus invokeDirectX12GraphicsPipeline(VernonLoadedPipeline &pipeline, const VernonPipelineInvocation &invocation,
                                             const PlannedGraphicsInvocation &plan) {
#if defined(VERNON_HAS_DIRECTX12_RUNTIME)
    DirectX12PipelineState &state = runtimeBackendState<DirectX12PipelineState>(pipeline);
    if (!state.rhiGraphicsPipeline)
        return VERNON_STATUS_OK;
    const auto &adapter = *directX12State(*pipeline.context).adapter;
    for (size_t index = 0; index < state.rhiGraphicsLayout.size(); ++index) {
        const auto &layout = state.rhiGraphicsLayout[index];
        auto &prepared = state.rhiGraphicsBindingsPlan[index];
        auto &value = state.rhiGraphicsValues[index];
        value = {};
        value.slot = layout.slot;
        value.kind = layout.kind;
        if (prepared.source == DirectX12PipelineState::GraphicsBinding::EXTERNAL_UNIFORM) {
            const auto found = plan.arguments.find(prepared.externalSlot);
            if (found == plan.arguments.end() || found->second->kind != VERNON_PIPELINE_TENSOR)
                return fail(*pipeline.context, "D3D12 RHI uniform argument is missing");
            const VernonTensorView &tensor = found->second->tensor;
            const std::optional<std::vector<uint8_t>> packed = packTensor(tensor, prepared.packing);
            if (!packed || packed->size() != prepared.storage.size())
                return fail(*pipeline.context, "D3D12 RHI uniform Tensor is invalid");
            prepared.storage = *packed;
            value.inline_data = prepared.storage.data();
            value.inline_size = prepared.storage.size();
        } else if (prepared.source == DirectX12PipelineState::GraphicsBinding::EXTERNAL_VERTEX) {
            const auto found = plan.arguments.find(prepared.externalSlot);
            if (found == plan.arguments.end() || found->second->kind != VERNON_PIPELINE_TENSOR ||
                found->second->tensor.storage != VERNON_TENSOR_RHI_RESOURCE ||
                !found->second->tensor.resource.resource.value || !found->second->tensor.byte_strides ||
                found->second->tensor.byte_strides[0] <= 0)
                return fail(*pipeline.context, "D3D12 RHI vertex argument is missing or invalid");
            const VernonTensorView &tensor = found->second->tensor;
            value.resource = tensor.resource;
            value.resource.offset += tensor.byte_offset;
            value.stride = static_cast<uint32_t>(tensor.byte_strides[0]);
        } else if (prepared.source == DirectX12PipelineState::GraphicsBinding::EXTERNAL_STORAGE) {
            const auto found = plan.arguments.find(prepared.externalSlot);
            if (found == plan.arguments.end() || found->second->kind != VERNON_PIPELINE_TENSOR ||
                found->second->tensor.storage != VERNON_TENSOR_RHI_RESOURCE ||
                !found->second->tensor.resource.resource.value)
                return fail(*pipeline.context, "D3D12 RHI storage argument is missing");
            const VernonTensorView &tensor = found->second->tensor;
            value.resource = tensor.resource;
            value.resource.offset += tensor.byte_offset;
        } else if (prepared.source == DirectX12PipelineState::GraphicsBinding::EXTERNAL_TEXTURE) {
            const auto found = plan.arguments.find(prepared.externalSlot);
            if (found == plan.arguments.end() || found->second->kind != VERNON_PIPELINE_TEXTURE ||
                !found->second->texture.resource.resource.value)
                return fail(*pipeline.context, "D3D12 RHI texture argument is missing");
            value.resource = found->second->texture.resource;
        } else if (prepared.source == DirectX12PipelineState::GraphicsBinding::EXTERNAL_SAMPLER) {
            const auto found = plan.arguments.find(prepared.externalSlot);
            if (found == plan.arguments.end() || found->second->kind != VERNON_PIPELINE_SAMPLER ||
                !found->second->resource.resource.value)
                return fail(*pipeline.context, "D3D12 RHI sampler argument is missing");
            value.resource = found->second->resource;
        } else if (prepared.source == DirectX12PipelineState::GraphicsBinding::IMPLICIT_SAMPLER) {
            const auto sampled = plan.sampledResources.find({layout.set, layout.binding});
            if (sampled == plan.sampledResources.end())
                return fail(*pipeline.context, "D3D12 RHI implicit sampler binding is missing");
            if (sampled->second.samplerResource.resource.value) {
                value.resource = sampled->second.samplerResource;
            } else {
                value.flags = VERNON_RUNTIME_PROVIDER_BINDING_DEFAULT_RESOURCE;
            }
        } else if (prepared.source == DirectX12PipelineState::GraphicsBinding::RESOLUTION) {
            std::memcpy(prepared.storage.data(), plan.resolution.data(),
                        std::min(prepared.storage.size(), sizeof(plan.resolution)));
            value.inline_data = prepared.storage.data();
            value.inline_size = prepared.storage.size();
        } else {
            return fail(*pipeline.context, "D3D12 RHI graphics binding source is invalid");
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
            vernonRuntimeRhiAdapterGetLastError(directX12State(*pipeline.context).adapter);
        return fail(*pipeline.context,
                    providerError.data ? std::string(providerError.data, providerError.size)
                                       : "failed to update D3D12 RHI graphics bindings",
                    status);
    }
    constexpr size_t maxAttachments = 8;
    if (plan.attachments.empty() || plan.attachments.size() > maxAttachments)
        return fail(*pipeline.context, "D3D12 RHI draw requires one to eight color attachments");
    std::array<VernonRuntimeProviderColorAttachment, maxAttachments> attachments{};
    VernonRuntimeProviderResourceReference depthAttachment{};
    std::vector<uint32_t> formats;
    formats.reserve(plan.attachments.size());
    for (size_t index = 0; index < plan.attachments.size(); ++index) {
        const VernonColorAttachment &source = *plan.attachments[index];
        attachments[index].location = source.location;
        attachments[index].image = source.resource;
        attachments[index].load_operation = source.load_operation;
        attachments[index].store_operation = source.store_operation;
        std::copy(std::begin(source.clear_color), std::end(source.clear_color), attachments[index].clear_color);
        const auto *image = reinterpret_cast<const rhi::directx12::Image *>(vernon::rhi::resolveResource(
            pipeline.context->rhiDevice, vernon::rhi::ResourceKind::Image, source.resource.resource.value));
        if (!image)
            return fail(*pipeline.context, "D3D12 RHI color attachment is stale");
        formats.push_back(static_cast<uint32_t>(image->format));
    }
    if (plan.depthAttachment) {
        depthAttachment = plan.depthAttachment->resource;
    }
    uint64_t vertexLayoutIdentity = 1469598103934665603ull;
    std::vector<uint32_t> vertexStrides;
    for (const auto &value : state.rhiGraphicsValues) {
        vertexLayoutIdentity ^= value.kind;
        vertexLayoutIdentity *= 1099511628211ull;
        vertexLayoutIdentity ^= value.stride;
        vertexLayoutIdentity *= 1099511628211ull;
    }
    for (size_t index = 0; index < state.rhiGraphicsLayout.size(); ++index)
        if (state.rhiGraphicsLayout[index].kind == VERNON_RUNTIME_PROVIDER_VERTEX_BUFFER) {
            const uint32_t binding = state.rhiGraphicsLayout[index].binding;
            if (vertexStrides.size() <= binding)
                vertexStrides.resize(binding + 1);
            vertexStrides[binding] = state.rhiGraphicsValues[index].stride;
        }
    const uint32_t depthFormat = plan.depthAttachment ? static_cast<uint32_t>(DXGI_FORMAT_D32_FLOAT) : 0;
    if (!state.rhiGraphicsVariant || state.rhiGraphicsFormats != formats ||
        state.rhiGraphicsDepthFormat != depthFormat || state.rhiGraphicsTopology != invocation.topology ||
        state.rhiGraphicsVertexLayoutIdentity != vertexLayoutIdentity) {
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
                                                                   vertexLayoutIdentity,
                                                                   {0, 0, 0, 0}};
        status = vernonRuntimeCorePrepareGraphicsVariant(state.rhiGraphicsPipeline, &compatibility,
                                                         &state.rhiGraphicsVariant);
        if (status != VERNON_STATUS_OK) {
            const VernonStringView providerError =
                vernonRuntimeRhiAdapterGetLastError(directX12State(*pipeline.context).adapter);
            return fail(*pipeline.context,
                        providerError.data ? std::string(providerError.data, providerError.size)
                                           : "failed to prepare D3D12 RHI graphics variant",
                        status);
        }
        state.rhiGraphicsFormats = std::move(formats);
        state.rhiGraphicsDepthFormat = depthFormat;
        state.rhiGraphicsTopology = invocation.topology;
        state.rhiGraphicsVertexLayoutIdentity = vertexLayoutIdentity;
    }
    const bool hasViewport = invocation.viewport[2] && invocation.viewport[3];
    VernonRuntimeCoreDrawInvocation draw{};
    draw.struct_size = sizeof(draw);
    draw.command_encoder = invocation.command_encoder;
    draw.vertex_count = plan.vertexCount;
    draw.instance_count = plan.instanceCount;
    draw.color_attachments = attachments.data();
    draw.color_attachment_count = plan.attachments.size();
    draw.depth_stencil_attachment = depthAttachment;
    draw.depth_load_operation =
        plan.depthAttachment ? static_cast<uint32_t>(plan.depthAttachment->load_operation) : VERNON_RHI_LOAD_DISCARD;
    draw.depth_store_operation =
        plan.depthAttachment ? static_cast<uint32_t>(plan.depthAttachment->store_operation) : VERNON_RHI_STORE_DISCARD;
    draw.clear_depth = plan.depthAttachment ? plan.depthAttachment->clear_depth : 1.0f;
    draw.viewport[0] = hasViewport ? invocation.viewport[0] : 0;
    draw.viewport[1] = hasViewport ? invocation.viewport[1] : 0;
    draw.viewport[2] = hasViewport ? invocation.viewport[2] : plan.attachmentWidth;
    draw.viewport[3] = hasViewport ? invocation.viewport[3] : plan.attachmentHeight;
    const bool hasScissor = invocation.scissor[2] && invocation.scissor[3];
    for (size_t index = 0; index < 4; ++index)
        draw.scissor[index] = hasScissor ? invocation.scissor[index] : draw.viewport[index];
    draw.topology = invocation.topology;
    if (plan.indexBinding) {
        draw.index_buffer = plan.indexBinding->resource;
        draw.index_buffer.offset += plan.indexBinding->offset;
        draw.index_count = plan.indexBinding->index_count;
        draw.index_type = plan.indexBinding->type;
    }
    status = vernonRuntimeCoreEncodeGraphicsVariantDrawInvocation(state.rhiGraphicsVariant, state.rhiGraphicsBindings,
                                                                  &draw);
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

VernonStatus invokeDirectX12ComputePipeline(VernonLoadedPipeline &pipeline, const PlannedComputeLaunch &launch) {
#if defined(VERNON_HAS_DIRECTX12_RUNTIME)
    DirectX12PipelineState &state = runtimeBackendState<DirectX12PipelineState>(pipeline);
    VernonRuntimeRhiAdapter &adapter = *directX12State(*pipeline.context).adapter;
    for (size_t index = 0; index < state.rhiComputeLayout.size(); ++index) {
        const VernonRuntimeProviderBindingLayoutEntry &layout = state.rhiComputeLayout[index];
        if (layout.argument_index >= launch.arguments.size())
            return fail(*pipeline.context, "D3D12 prepared argument index is invalid");
        const ComputeLaunchArgument &argument = launch.arguments[layout.argument_index];
        VernonRuntimeProviderBindingValue &value = state.rhiComputeValues[index];
        value = {};
        value.slot = layout.slot;
        value.kind = layout.kind;
        if (layout.kind == VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER) {
            if (argument.kind != ComputeLaunchArgumentKind::Tensor || !argument.resource.resource.value)
                return fail(*pipeline.context, "D3D12 prepared storage binding requires an RHI Tensor");
            value.resource = argument.resource;
            value.resource.offset += state.rhiComputeResourceOffsets[index];
        } else {
            if (argument.kind != ComputeLaunchArgumentKind::Scalar || !argument.scalarData || !argument.scalarSize)
                return fail(*pipeline.context, "D3D12 prepared inline binding requires host data at argument " +
                                                   std::to_string(layout.argument_index) + " (kind " +
                                                   std::to_string(static_cast<uint32_t>(argument.kind)) + ", size " +
                                                   std::to_string(argument.scalarSize) + ")");
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
    if (status != VERNON_STATUS_OK) {
        const VernonStringView providerError =
            vernonRuntimeRhiAdapterGetLastError(directX12State(*pipeline.context).adapter);
        return fail(*pipeline.context,
                    providerError.data ? std::string(providerError.data, providerError.size)
                                       : "failed to prepare D3D12 invocation bindings",
                    status);
    }
    const uint32_t groups[3]{(launch.grid.x - 1) / state.rhiComputeWorkgroup[0] + 1,
                             (launch.grid.y - 1) / state.rhiComputeWorkgroup[1] + 1,
                             (launch.grid.z - 1) / state.rhiComputeWorkgroup[2] + 1};
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
