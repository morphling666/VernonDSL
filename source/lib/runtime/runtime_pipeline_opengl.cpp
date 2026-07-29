#include "runtime_pipeline_backend.h"

#include "backend_opengl.h"
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

namespace vernon::runtime {
namespace {

VernonStatus fail(VernonRuntimeContext &context, std::string error,
                  VernonStatus status = VERNON_STATUS_INVALID_ARGUMENT) {
    context.error = std::move(error);
    return status;
}

} // namespace

bool resolveOpenGLPipeline(VernonPipelineBundle &bundle, const Variant &variant, VernonLoadedPipeline &pipeline) {
    auto *state = new OpenGLPipelineState();
    struct OpenGLBindingCandidate {
        VernonRuntimeProviderBindingLayoutEntry layout{};
        std::vector<VernonRuntimeProviderVertexAttribute> attributes;
        OpenGLPipelineState::InlineBinding binding;
        std::string name;
    };
    std::vector<OpenGLBindingCandidate> candidates;
    uint32_t computeInternalSlot = 0;
    for (const Parameter &parameter : variant.parameters)
        computeInternalSlot = std::max(computeInternalSlot, parameter.slot);
    if (!variant.compute.empty()) {
        const Stage &stage = bundle.stages.at(variant.compute);
        ReflectedEntry reflection;
        const nlohmann::json parsed = nlohmann::json::parse(stage.reflection, nullptr, false);
        if (parsed.is_discarded() ||
            !parseReflection(parsed, stage.entry, reflection, bundle.context->backend, bundle.context->error)) {
            delete state;
            return false;
        }
        for (const Parameter &parameter : variant.parameters)
            for (const ParameterUse &use : parameter.uses) {
                if (use.stage != "compute" && use.stage != variant.compute)
                    continue;
                if (use.index >= reflection.arguments.size() || use.binding == UINT32_MAX) {
                    bundle.context->error = "OpenGL compute parameter reflection is incomplete";
                    delete state;
                    return false;
                }
                const ReflectedArgument &argument = reflection.arguments[use.index];
                const size_t leafCount = std::max(argument.storageLeaves.size(), size_t{1});
                for (size_t leafIndex = 0; leafIndex < leafCount; ++leafIndex) {
                    OpenGLBindingCandidate candidate;
                    candidate.layout.slot = leafIndex == 0 ? parameter.slot : ++computeInternalSlot;
                    candidate.layout.binding =
                        argument.storageLeaves.empty() ? use.binding : argument.storageLeaves[leafIndex].binding;
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
                    candidate.binding.externalSlot = parameter.slot;
                    candidate.binding.source = argument.kind == "tensor"
                                                   ? OpenGLPipelineState::InlineBinding::EXTERNAL_STORAGE
                                                   : OpenGLPipelineState::InlineBinding::COMPUTE_INLINE;
                    if (candidate.layout.element_size == 0) {
                        bundle.context->error = "OpenGL compute parameter has zero element size";
                        delete state;
                        return false;
                    }
                    candidates.push_back(std::move(candidate));
                }
            }
        std::sort(candidates.begin(), candidates.end(),
                  [](const auto &left, const auto &right) { return left.layout.slot < right.layout.slot; });
        state->rhiLayout.reserve(candidates.size());
        state->rhiInlineBindings.reserve(candidates.size());
        state->rhiValues.resize(candidates.size());
        for (size_t index = 0; index < candidates.size(); ++index) {
            state->rhiLayout.push_back(candidates[index].layout);
            state->rhiInlineBindings.push_back(std::move(candidates[index].binding));
            state->rhiValues[index].slot = state->rhiLayout[index].slot;
            state->rhiValues[index].kind = state->rhiLayout[index].kind;
            if (state->rhiLayout[index].kind == VERNON_RUNTIME_PROVIDER_INLINE_VALUE) {
                state->rhiInlineBindings[index].storage.resize(state->rhiLayout[index].element_size);
                state->rhiValues[index].inline_data = state->rhiInlineBindings[index].storage.data();
                state->rhiValues[index].inline_size = state->rhiInlineBindings[index].storage.size();
            } else {
                state->rhiValues[index].flags = VERNON_RUNTIME_PROVIDER_BINDING_DEFAULT_RESOURCE;
            }
        }
        const VernonRuntimeProviderShaderDescriptor shader{sizeof(VernonRuntimeProviderShaderDescriptor),
                                                           VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE,
                                                           {"glsl", 4},
                                                           stage.source.data(),
                                                           stage.source.size(),
                                                           {stage.entry.data(), stage.entry.size()},
                                                           {},
                                                           {0, 0, 0, 0}};
        VernonRuntimeCorePipelineDescriptor descriptor{};
        descriptor.struct_size = sizeof(descriptor);
        descriptor.kind = VERNON_RUNTIME_PROVIDER_COMPUTE_PIPELINE;
        descriptor.required_capabilities = VERNON_RUNTIME_PROVIDER_COMPUTE;
        descriptor.shaders = &shader;
        descriptor.shader_count = 1;
        descriptor.bindings = state->rhiLayout.data();
        descriptor.binding_count = state->rhiLayout.size();
        std::copy_n(stage.workgroup, 3, descriptor.workgroup_size);
        const VernonStatus status = vernonRuntimeCorePreparePipeline(
            vernonRuntimeRhiAdapterGetProvider(openGLState(*bundle.context).adapter), &descriptor, &state->rhiPipeline);
        if (status != VERNON_STATUS_OK) {
            const VernonStringView providerError =
                vernonRuntimeRhiAdapterGetLastError(openGLState(*bundle.context).adapter);
            bundle.context->error = providerError.data ? std::string(providerError.data, providerError.size)
                                                       : "failed to prepare OpenGL provider compute pipeline";
            delete state;
            return false;
        }
        if (!state->rhiLayout.empty()) {
            const VernonStatus bindingStatus = vernonRuntimeCoreCreateBindings(
                state->rhiPipeline, state->rhiValues.data(), state->rhiValues.size(), &state->rhiBindings);
            if (bindingStatus != VERNON_STATUS_OK) {
                vernonRuntimeCorePipelineDestroy(state->rhiPipeline);
                delete state;
                return false;
            }
        }
        std::copy_n(stage.workgroup, 3, state->workgroup);
        installRuntimeBackendState(pipeline, state);
        return true;
    }
    bool useRhiGraphics = variant.compute.empty() && !variant.vertex.empty() && !variant.fragment.empty();
    std::string representationError;
    const auto graphicsStageMask = [](const std::string &stage) {
        return stage == "vertex"     ? uint32_t{VERNON_RUNTIME_PROVIDER_STAGE_VERTEX}
               : stage == "fragment" ? uint32_t{VERNON_RUNTIME_PROVIDER_STAGE_FRAGMENT}
                                     : uint32_t{0};
    };
    if (useRhiGraphics) {
        for (const Parameter &parameter : variant.parameters) {
            for (const ParameterUse &use : parameter.uses) {
                if (graphicsStageMask(use.stage) == 0) {
                    representationError = "OpenGL graphics parameter references an unsupported stage";
                    useRhiGraphics = false;
                    break;
                }
                if ((use.binding != UINT32_MAX && use.descriptorSet != 0) ||
                    std::any_of(use.sampledTextureBindings.begin(), use.sampledTextureBindings.end(),
                                [](const SampledTextureBinding &binding) { return binding.descriptorSet != 0; })) {
                    representationError = "OpenGL supports reflected resources only in descriptor set 0";
                    useRhiGraphics = false;
                    break;
                }
                OpenGLBindingCandidate candidate;
                candidate.layout.stage_mask = graphicsStageMask(use.stage);
                candidate.layout.array_count = 1;
                candidate.binding.externalSlot = parameter.slot;
                if (parameter.kind == "sampler") {
                    if (use.sampledTextureBindings.empty()) {
                        representationError = "OpenGL sampler has no reflected texture binding";
                        useRhiGraphics = false;
                        break;
                    }
                    for (const SampledTextureBinding &binding : use.sampledTextureBindings) {
                        if (binding.binding == UINT32_MAX) {
                            representationError = "OpenGL sampler reflection is incomplete";
                            useRhiGraphics = false;
                            break;
                        }
                        OpenGLBindingCandidate sampler = candidate;
                        sampler.layout.kind = VERNON_RUNTIME_PROVIDER_SAMPLER;
                        sampler.layout.interface_kind = VERNON_RUNTIME_PROVIDER_INTERFACE_RESOURCE;
                        sampler.layout.set = binding.descriptorSet;
                        sampler.layout.binding = binding.binding;
                        sampler.binding.source = OpenGLPipelineState::InlineBinding::EXTERNAL_SAMPLER;
                        candidates.push_back(std::move(sampler));
                    }
                    if (!useRhiGraphics)
                        break;
                    continue;
                }
                if (parameter.kind == "tensor" && use.interfaceKind == "uniform" && use.physicalValueLayout &&
                    use.physicalValueLayout->transport == "storage_buffer" && parameter.elementLayout.byteSize &&
                    use.binding != UINT32_MAX) {
                    candidate.layout.kind = VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER;
                    candidate.layout.element_size = parameter.elementLayout.byteSize;
                    candidate.layout.interface_kind = VERNON_RUNTIME_PROVIDER_INTERFACE_RESOURCE;
                    candidate.layout.binding = use.binding;
                    candidate.layout.set = use.descriptorSet;
                    candidate.binding.source = OpenGLPipelineState::InlineBinding::EXTERNAL_STORAGE;
                } else if (parameter.kind == "tensor" && use.interfaceKind == "uniform" && use.physicalValueLayout &&
                           (!use.uniformName.empty() || use.physicalValueLayout->transport == "uniform_buffer")) {
                    const auto &shape = use.shape.empty() ? parameter.shape : use.shape;
                    const std::optional<VernonDataType> dtype = pipelineDataType(use.dtype);
                    uint64_t valueCount = 1;
                    for (uint64_t dimension : shape) {
                        if (dimension == 0 || valueCount > UINT32_MAX / dimension) {
                            valueCount = 0;
                            break;
                        }
                        valueCount *= dimension;
                    }
                    const bool buffered = use.physicalValueLayout->transport == "uniform_buffer";
                    const bool scalarOrVector = shape.empty() || (shape.size() == 1 && shape[0] <= 4);
                    const bool floatingMatrix = use.dtype == "f32" && shape.size() == 2 && shape[0] >= 2 &&
                                                shape[0] <= 4 && shape[1] >= 2 && shape[1] <= 4;
                    const bool nativeInline =
                        ((use.dtype == "f32" || use.dtype == "i32" || use.dtype == "u32") && scalarOrVector) ||
                        floatingMatrix;
                    if (!dtype || valueCount == 0 || (!buffered && !nativeInline)) {
                        representationError = "OpenGL uniform layout is unsupported";
                        useRhiGraphics = false;
                        break;
                    }
                    const size_t elementSize = dataTypeSize(*dtype);
                    candidate.layout.kind =
                        buffered ? VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER : VERNON_RUNTIME_PROVIDER_INLINE_VALUE;
                    const uint64_t physicalSize = use.physicalValueLayout->size;
                    if (!physicalSize || physicalSize > UINT32_MAX || (buffered && use.binding == UINT32_MAX)) {
                        representationError = "OpenGL uniform reflection is incomplete";
                        useRhiGraphics = false;
                        break;
                    }
                    candidate.layout.element_size = static_cast<uint32_t>(physicalSize);
                    candidate.layout.interface_kind = VERNON_RUNTIME_PROVIDER_INTERFACE_UNIFORM;
                    candidate.layout.element_count = static_cast<uint32_t>(valueCount);
                    candidate.layout.vector_count = shape.size() == 2 ? static_cast<uint32_t>(shape[1]) : 1;
                    candidate.layout.numeric_type = static_cast<uint32_t>(*dtype);
                    candidate.layout.binding = use.binding;
                    candidate.layout.set = use.descriptorSet;
                    candidate.binding.source = OpenGLPipelineState::InlineBinding::EXTERNAL_UNIFORM;
                    candidate.name = use.uniformName;
                    candidate.binding.packing.elementSize = elementSize;
                    candidate.binding.packing.shape = shape;
                    candidate.binding.packing.byteSize = static_cast<size_t>(physicalSize);
                    for (uint64_t stride : use.physicalValueLayout->byteStrides) {
                        if (stride > SIZE_MAX) {
                            representationError = "OpenGL uniform stride exceeds the host size range";
                            useRhiGraphics = false;
                            break;
                        }
                        candidate.binding.packing.byteStrides.push_back(static_cast<size_t>(stride));
                    }
                    if (!useRhiGraphics || candidate.binding.packing.byteStrides.size() != shape.size()) {
                        if (representationError.empty())
                            representationError = "OpenGL uniform stride rank does not match its shape";
                        useRhiGraphics = false;
                        break;
                    }
                } else if (parameter.kind == "tensor" && use.interfaceKind == "input" && use.stage == "vertex" &&
                           use.location != UINT32_MAX && !use.attributeLeaves.empty()) {
                    candidate.layout.kind = VERNON_RUNTIME_PROVIDER_VERTEX_BUFFER;
                    candidate.layout.element_size = parameter.elementLayout.byteSize;
                    candidate.layout.interface_kind = VERNON_RUNTIME_PROVIDER_INTERFACE_VERTEX_INPUT;
                    candidate.layout.binding = parameter.slot;
                    candidate.layout.divisor = use.divisor;
                    for (const AttributeLeaf &leaf : use.attributeLeaves) {
                        const std::optional<VernonDataType> dtype = pipelineDataType(leaf.dtype);
                        if (!dtype) {
                            representationError = "OpenGL vertex attribute dtype is unsupported";
                            useRhiGraphics = false;
                            break;
                        }
                        candidate.attributes.push_back({candidate.layout.binding, use.location + leaf.locationOffset,
                                                        static_cast<uint32_t>(*dtype), leaf.componentCount,
                                                        leaf.byteOffset});
                    }
                    if (!useRhiGraphics)
                        break;
                    candidate.binding.source = OpenGLPipelineState::InlineBinding::EXTERNAL_VERTEX;
                } else if (parameter.kind == "texture" && use.interfaceKind == "resource" &&
                           use.binding != UINT32_MAX) {
                    candidate.layout.kind = VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE;
                    candidate.layout.interface_kind = VERNON_RUNTIME_PROVIDER_INTERFACE_RESOURCE;
                    candidate.layout.set = use.descriptorSet;
                    candidate.layout.binding = use.binding;
                    candidate.binding.source = OpenGLPipelineState::InlineBinding::EXTERNAL_TEXTURE;
                    candidate.name =
                        use.uniformName.empty() ? "main_arg_" + std::to_string(use.index) : use.uniformName;
                } else {
                    representationError = "OpenGL graphics parameter layout is unsupported";
                    useRhiGraphics = false;
                    break;
                }
                candidates.push_back(std::move(candidate));
            }
            if (!useRhiGraphics)
                break;
        }
    }
    if (useRhiGraphics) {
        for (const Parameter &parameter : variant.internalParameters) {
            for (const ParameterUse &use : parameter.uses) {
                if (graphicsStageMask(use.stage) == 0) {
                    representationError = "OpenGL internal parameter references an unsupported stage";
                    useRhiGraphics = false;
                    break;
                }
                const uint32_t stageMask = graphicsStageMask(use.stage);
                if (parameter.source == "implicit_sampler" && parameter.kind == "sampler" &&
                    !use.sampledTextureBindings.empty()) {
                    for (const SampledTextureBinding &binding : use.sampledTextureBindings) {
                        if (binding.descriptorSet != 0 || binding.binding == UINT32_MAX) {
                            representationError = "OpenGL supports reflected resources only in descriptor set 0";
                            useRhiGraphics = false;
                            break;
                        }
                        OpenGLBindingCandidate candidate;
                        candidate.layout.stage_mask = stageMask;
                        candidate.layout.array_count = 1;
                        candidate.layout.kind = VERNON_RUNTIME_PROVIDER_SAMPLER;
                        candidate.layout.interface_kind = VERNON_RUNTIME_PROVIDER_INTERFACE_RESOURCE;
                        candidate.layout.set = binding.descriptorSet;
                        candidate.layout.binding = binding.binding;
                        candidate.binding.source = OpenGLPipelineState::InlineBinding::IMPLICIT_SAMPLER;
                        candidates.push_back(std::move(candidate));
                    }
                } else if (parameter.source == "system_value" && parameter.systemValue == "resolution" &&
                           use.interfaceKind == "uniform" && !use.uniformName.empty()) {
                    OpenGLBindingCandidate candidate;
                    candidate.layout.stage_mask = stageMask;
                    candidate.layout.array_count = 1;
                    candidate.layout.kind = VERNON_RUNTIME_PROVIDER_INLINE_VALUE;
                    candidate.layout.element_size = 2 * sizeof(float);
                    candidate.layout.interface_kind = VERNON_RUNTIME_PROVIDER_INTERFACE_UNIFORM;
                    candidate.layout.element_count = 2;
                    candidate.layout.vector_count = 1;
                    candidate.layout.numeric_type = VERNON_RUNTIME_PROVIDER_F32;
                    candidate.binding.source = OpenGLPipelineState::InlineBinding::RESOLUTION;
                    candidate.name = use.uniformName;
                    candidates.push_back(std::move(candidate));
                } else {
                    representationError = "OpenGL internal parameter layout is unsupported";
                    useRhiGraphics = false;
                    break;
                }
            }
            if (!useRhiGraphics)
                break;
        }
    }
    if (useRhiGraphics) {
        if (candidates.size() > UINT32_MAX) {
            representationError = "OpenGL binding layout exceeds the provider slot range";
            useRhiGraphics = false;
        }
        if (useRhiGraphics) {
            state->rhiLayout.reserve(candidates.size());
            state->rhiInlineBindings.reserve(candidates.size());
            for (size_t index = 0; index < candidates.size(); ++index) {
                auto &candidate = candidates[index];
                candidate.layout.slot = static_cast<uint32_t>(index);
                candidate.layout.name = {candidate.name.data(), candidate.name.size()};
                state->rhiLayout.push_back(candidate.layout);
                state->rhiVertexAttributes.insert(state->rhiVertexAttributes.end(), candidate.attributes.begin(),
                                                  candidate.attributes.end());
                state->rhiInlineBindings.push_back(std::move(candidate.binding));
            }
        }
    }
    if (useRhiGraphics) {
        const Stage &vertex = bundle.stages.at(variant.vertex);
        const Stage &fragment = bundle.stages.at(variant.fragment);
        std::vector<VernonRuntimeProviderShaderDescriptor> shaders{{sizeof(VernonRuntimeProviderShaderDescriptor),
                                                                    VERNON_RUNTIME_PROVIDER_STAGE_VERTEX,
                                                                    {"glsl", 4},
                                                                    vertex.source.data(),
                                                                    vertex.source.size(),
                                                                    {vertex.entry.data(), vertex.entry.size()},
                                                                    {nullptr, 0},
                                                                    {0, 0, 0, 0}}};
        shaders.push_back({sizeof(VernonRuntimeProviderShaderDescriptor),
                           VERNON_RUNTIME_PROVIDER_STAGE_FRAGMENT,
                           {"glsl", 4},
                           fragment.source.data(),
                           fragment.source.size(),
                           {fragment.entry.data(), fragment.entry.size()},
                           {nullptr, 0},
                           {0, 0, 0, 0}});
        VernonRuntimeCorePipelineDescriptor descriptor{};
        descriptor.struct_size = sizeof(descriptor);
        descriptor.kind = VERNON_RUNTIME_PROVIDER_GRAPHICS_PIPELINE;
        descriptor.required_capabilities = VERNON_RUNTIME_PROVIDER_GRAPHICS;
        descriptor.shaders = shaders.data();
        descriptor.shader_count = shaders.size();
        descriptor.bindings = state->rhiLayout.data();
        descriptor.binding_count = state->rhiLayout.size();
        descriptor.vertex_attributes = state->rhiVertexAttributes.data();
        descriptor.vertex_attribute_count = state->rhiVertexAttributes.size();
        descriptor.topology = VERNON_TOPOLOGY_TRIANGLE_LIST;
        descriptor.sample_count = 1;
        const VernonStatus status = vernonRuntimeCorePreparePipeline(
            vernonRuntimeRhiAdapterGetProvider(openGLState(*bundle.context).adapter), &descriptor, &state->rhiPipeline);
        if (status != VERNON_STATUS_OK) {
            const VernonStringView providerError =
                vernonRuntimeRhiAdapterGetLastError(openGLState(*bundle.context).adapter);
            bundle.context->error = providerError.data ? std::string(providerError.data, providerError.size)
                                                       : "failed to prepare OpenGL RHI provider pipeline";
            delete state;
            return false;
        }
        if (!state->rhiLayout.empty()) {
            state->rhiValues.resize(state->rhiLayout.size());
            for (size_t index = 0; index < state->rhiLayout.size(); ++index) {
                const auto &layout = state->rhiLayout[index];
                auto &value = state->rhiValues[index];
                value.slot = layout.slot;
                value.kind = layout.kind;
                if (layout.kind == VERNON_RUNTIME_PROVIDER_INLINE_VALUE ||
                    layout.kind == VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER) {
                    state->rhiInlineBindings[index].storage.resize(layout.element_size);
                    value.inline_data = state->rhiInlineBindings[index].storage.data();
                    value.inline_size = state->rhiInlineBindings[index].storage.size();
                } else {
                    value.flags = VERNON_RUNTIME_PROVIDER_BINDING_DEFAULT_RESOURCE;
                }
            }
            const VernonStatus bindingStatus = vernonRuntimeCoreCreateBindings(
                state->rhiPipeline, state->rhiValues.data(), state->rhiValues.size(), &state->rhiBindings);
            if (bindingStatus != VERNON_STATUS_OK) {
                const VernonStringView providerError =
                    vernonRuntimeRhiAdapterGetLastError(openGLState(*bundle.context).adapter);
                bundle.context->error = providerError.data ? std::string(providerError.data, providerError.size)
                                                           : "failed to prepare OpenGL RHI bindings";
                vernonRuntimeCorePipelineDestroy(state->rhiPipeline);
                delete state;
                return false;
            }
        }
        installRuntimeBackendState(pipeline, state);
        return true;
    }
    bundle.context->error = representationError.empty()
                                ? "OpenGL PipelineAsset is not representable by the RuntimeCore provider"
                                : std::move(representationError);
    delete state;
    return false;
}

void destroyOpenGLPipeline(VernonLoadedPipeline &pipeline) {
    OpenGLPipelineState &state = runtimeBackendState<OpenGLPipelineState>(pipeline);
    vernonRuntimeCoreBindingsDestroy(state.rhiBindings);
    vernonRuntimeCorePipelineDestroy(state.rhiPipeline);
}

VernonStatus invokeOpenGLGraphicsPipeline(VernonLoadedPipeline &pipeline, const VernonPipelineInvocation &invocation,
                                          const PlannedGraphicsInvocation &plan) {
    OpenGLPipelineState &state = runtimeBackendState<OpenGLPipelineState>(pipeline);
    if (!state.rhiPipeline)
        return VERNON_STATUS_OK;
    for (size_t bindingIndex = 0; bindingIndex < state.rhiLayout.size(); ++bindingIndex) {
        const auto &layout = state.rhiLayout[bindingIndex];
        auto &prepared = state.rhiInlineBindings[bindingIndex];
        VernonRuntimeProviderBindingValue &value = state.rhiValues[bindingIndex];
        value.flags = 0;
        value.resource = {};
        value.stride = 0;
        if (prepared.source == OpenGLPipelineState::InlineBinding::EXTERNAL_VERTEX) {
            const auto found = plan.arguments.find(prepared.externalSlot);
            if (found == plan.arguments.end() || found->second->kind != VERNON_PIPELINE_TENSOR ||
                found->second->tensor.storage != VERNON_TENSOR_RHI_RESOURCE ||
                !found->second->tensor.resource.resource.value)
                return fail(*pipeline.context, "OpenGL RHI vertex argument is missing");
            const VernonTensorView &tensor = found->second->tensor;
            if (!tensor.byte_strides || tensor.byte_strides[0] <= 0 ||
                static_cast<uint64_t>(tensor.byte_strides[0]) > UINT32_MAX)
                return fail(*pipeline.context, "OpenGL RHI vertex Tensor strides are invalid");
            value.resource = tensor.resource;
            value.resource.offset += tensor.byte_offset;
            value.stride = static_cast<uint32_t>(tensor.byte_strides[0]);
            continue;
        }
        if (prepared.source == OpenGLPipelineState::InlineBinding::EXTERNAL_STORAGE) {
            const auto found = plan.arguments.find(prepared.externalSlot);
            if (found == plan.arguments.end() || found->second->kind != VERNON_PIPELINE_TENSOR ||
                found->second->tensor.storage != VERNON_TENSOR_RHI_RESOURCE ||
                !found->second->tensor.resource.resource.value)
                return fail(*pipeline.context, "OpenGL RHI storage argument is missing");
            const VernonTensorView &tensor = found->second->tensor;
            value.resource = tensor.resource;
            value.resource.offset += tensor.byte_offset;
            continue;
        }
        if (prepared.source == OpenGLPipelineState::InlineBinding::EXTERNAL_TEXTURE) {
            const auto found = plan.arguments.find(prepared.externalSlot);
            if (found == plan.arguments.end() || found->second->kind != VERNON_PIPELINE_TEXTURE ||
                !found->second->texture.resource.resource.value)
                return fail(*pipeline.context, "OpenGL RHI texture argument is missing");
            value.resource = found->second->texture.resource;
            value.stride = static_cast<uint32_t>(found->second->texture.dimension);
            continue;
        }
        if (prepared.source == OpenGLPipelineState::InlineBinding::EXTERNAL_SAMPLER) {
            const auto found = plan.arguments.find(prepared.externalSlot);
            if (found == plan.arguments.end() || found->second->kind != VERNON_PIPELINE_SAMPLER ||
                !found->second->resource.resource.value)
                return fail(*pipeline.context, "OpenGL RHI sampler argument is missing");
            value.resource = found->second->resource;
            continue;
        }
        if (prepared.source == OpenGLPipelineState::InlineBinding::IMPLICIT_SAMPLER) {
            const auto sampled = plan.sampledResources.find({layout.set, layout.binding});
            if (sampled == plan.sampledResources.end())
                return fail(*pipeline.context, "OpenGL RHI implicit sampler binding is missing");
            if (sampled->second.samplerResource.resource.value) {
                value.resource = sampled->second.samplerResource;
            } else {
                value.flags = VERNON_RUNTIME_PROVIDER_BINDING_DEFAULT_RESOURCE;
            }
            continue;
        }
        if (prepared.source == OpenGLPipelineState::InlineBinding::RESOLUTION) {
            std::memcpy(prepared.storage.data(), plan.resolution.data(), 2 * sizeof(float));
            continue;
        }
        const auto found = plan.arguments.find(prepared.externalSlot);
        if (found == plan.arguments.end() || found->second->kind != VERNON_PIPELINE_TENSOR)
            return fail(*pipeline.context, "OpenGL RHI uniform argument is missing");
        const VernonTensorView &tensor = found->second->tensor;
        const std::optional<std::vector<uint8_t>> packed = packTensor(tensor, prepared.packing);
        if (!packed || packed->size() != prepared.storage.size())
            return fail(*pipeline.context, "OpenGL RHI uniform Tensor is invalid");
        std::memcpy(prepared.storage.data(), packed->data(), packed->size());
    }
    if (state.rhiBindings) {
        const VernonStatus bindingStatus =
            vernonRuntimeCoreUpdateBindings(state.rhiBindings, state.rhiValues.data(), state.rhiValues.size());
        if (bindingStatus != VERNON_STATUS_OK) {
            const VernonStringView providerError =
                vernonRuntimeRhiAdapterGetLastError(openGLState(*pipeline.context).adapter);
            return fail(*pipeline.context,
                        providerError.data ? std::string(providerError.data, providerError.size)
                                           : "failed to update OpenGL RHI bindings",
                        bindingStatus);
        }
    }
    constexpr size_t maxAttachments = 8;
    if (plan.attachments.size() > maxAttachments)
        return fail(*pipeline.context, "OpenGL RHI draw supports at most eight color attachments");
    std::array<VernonRuntimeProviderColorAttachment, maxAttachments> attachments{};
    VernonRuntimeProviderResourceReference depthAttachment{};
    for (size_t index = 0; index < plan.attachments.size(); ++index) {
        const VernonColorAttachment &source = *plan.attachments[index];
        attachments[index].location = source.location;
        attachments[index].image = source.resource;
        attachments[index].load_operation = source.load_operation;
        attachments[index].store_operation = source.store_operation;
        std::copy(std::begin(source.clear_color), std::end(source.clear_color), attachments[index].clear_color);
    }
    if (plan.depthAttachment) {
        depthAttachment = plan.depthAttachment->resource;
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
    const VernonStatus status = vernonRuntimeCoreEncodeDrawInvocation(state.rhiPipeline, state.rhiBindings, &draw);
    if (status != VERNON_STATUS_OK) {
        const VernonStringView providerError =
            vernonRuntimeRhiAdapterGetLastError(openGLState(*pipeline.context).adapter);
        return fail(*pipeline.context,
                    providerError.data ? std::string(providerError.data, providerError.size)
                                       : "failed to encode OpenGL RHI draw",
                    status);
    }
    return VERNON_STATUS_OK;
}

VernonStatus invokeOpenGLComputePipeline(VernonLoadedPipeline &pipeline, const PlannedComputeLaunch &launch) {
    OpenGLPipelineState &state = runtimeBackendState<OpenGLPipelineState>(pipeline);
    if (!state.rhiPipeline)
        return fail(*pipeline.context, "OpenGL provider compute pipeline is not loaded");
    for (size_t index = 0; index < state.rhiLayout.size(); ++index) {
        const auto &layout = state.rhiLayout[index];
        if (layout.argument_index >= launch.arguments.size())
            return fail(*pipeline.context, "OpenGL prepared argument index is invalid");
        const ComputeLaunchArgument &argument = launch.arguments[layout.argument_index];
        auto &value = state.rhiValues[index];
        value = {};
        value.slot = layout.slot;
        value.kind = layout.kind;
        if (layout.kind == VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER) {
            if (argument.kind != ComputeLaunchArgumentKind::Tensor || !argument.resource.resource.value)
                return fail(*pipeline.context, "OpenGL prepared storage binding requires an RHI Tensor");
            value.resource = argument.resource;
        } else {
            if (argument.kind != ComputeLaunchArgumentKind::Scalar || !argument.scalarData || !argument.scalarSize)
                return fail(*pipeline.context, "OpenGL prepared inline binding requires host data");
            value.inline_data = argument.scalarData;
            value.inline_size = argument.scalarSize;
        }
    }
    VernonStatus status =
        state.rhiBindings
            ? vernonRuntimeCoreUpdateBindings(state.rhiBindings, state.rhiValues.data(), state.rhiValues.size())
            : vernonRuntimeCoreCreateBindings(state.rhiPipeline, state.rhiValues.data(), state.rhiValues.size(),
                                              &state.rhiBindings);
    if (status != VERNON_STATUS_OK)
        return fail(*pipeline.context, "failed to prepare OpenGL provider bindings", status);
    const uint32_t groups[3]{(launch.grid.x - 1) / state.workgroup[0] + 1, (launch.grid.y - 1) / state.workgroup[1] + 1,
                             (launch.grid.z - 1) / state.workgroup[2] + 1};
    status = vernonRuntimeCoreEncodeDispatch(state.rhiPipeline, state.rhiBindings, launch.commandEncoder, groups,
                                             nullptr, 0);
    return status == VERNON_STATUS_OK ? status
                                      : fail(*pipeline.context, "failed to encode OpenGL provider dispatch", status);
}

} // namespace vernon::runtime
