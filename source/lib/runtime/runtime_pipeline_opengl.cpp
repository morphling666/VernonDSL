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
        if (parsed.is_discarded() || !parseReflection(parsed, stage.entry, reflection, bundle.context->error)) {
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
                            ? (argument.kind == "tensor" ? argument.tensorElementSize : argument.physicalSize)
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
    uint32_t maximumExternalSlot = 0;
    if (useRhiGraphics) {
        for (const Parameter &parameter : variant.parameters) {
            maximumExternalSlot = std::max(maximumExternalSlot, parameter.slot);
            if (parameter.uses.size() != 1 ||
                (parameter.uses[0].stage != "vertex" && parameter.uses[0].stage != "fragment")) {
                useRhiGraphics = false;
                break;
            }
            const ParameterUse &use = parameter.uses[0];
            OpenGLBindingCandidate candidate;
            candidate.layout.slot = parameter.slot;
            candidate.layout.stage_mask =
                use.stage == "vertex" ? VERNON_RUNTIME_PROVIDER_STAGE_VERTEX : VERNON_RUNTIME_PROVIDER_STAGE_FRAGMENT;
            candidate.layout.array_count = 1;
            candidate.binding.externalSlot = parameter.slot;
            if (parameter.kind == "tensor" && use.interfaceKind == "uniform" && use.uniformLayout &&
                use.uniformLayout->storage == "storage_buffer" && parameter.elementLayout.byteSize &&
                use.binding != UINT32_MAX) {
                candidate.layout.kind = VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER;
                candidate.layout.element_size = parameter.elementLayout.byteSize;
                candidate.layout.interface_kind = VERNON_RUNTIME_PROVIDER_INTERFACE_RESOURCE;
                candidate.layout.binding = use.binding;
                candidate.layout.set = use.descriptorSet;
                candidate.binding.source = OpenGLPipelineState::InlineBinding::EXTERNAL_STORAGE;
            } else if (parameter.kind == "tensor" && use.interfaceKind == "uniform" &&
                       (!use.uniformName.empty() ||
                        (use.uniformLayout && use.uniformLayout->storage == "uniform_buffer"))) {
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
                const bool buffered = use.uniformLayout && use.uniformLayout->storage == "uniform_buffer";
                const bool nativeInline =
                    use.dtype == "f32" &&
                    (shape.empty() || (shape.size() == 1 && shape[0] <= 4) ||
                     (shape.size() == 2 && shape[0] >= 2 && shape[0] <= 4 && shape[1] >= 2 && shape[1] <= 4));
                if (!dtype || valueCount == 0 || (!buffered && !nativeInline)) {
                    useRhiGraphics = false;
                    break;
                }
                const size_t elementSize = dataTypeSize(*dtype);
                candidate.layout.kind =
                    buffered ? VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER : VERNON_RUNTIME_PROVIDER_INLINE_VALUE;
                const uint64_t physicalSize =
                    buffered && use.uniformLayout ? use.uniformLayout->size : valueCount * elementSize;
                if (!physicalSize || physicalSize > UINT32_MAX || (buffered && use.binding == UINT32_MAX)) {
                    useRhiGraphics = false;
                    break;
                }
                candidate.layout.element_size = static_cast<uint32_t>(physicalSize);
                candidate.layout.interface_kind = VERNON_RUNTIME_PROVIDER_INTERFACE_UNIFORM;
                candidate.layout.element_count = static_cast<uint32_t>(valueCount);
                candidate.layout.vector_count = shape.size() == 2 ? static_cast<uint32_t>(shape[1]) : 1;
                candidate.layout.binding = use.binding;
                candidate.layout.set = use.descriptorSet;
                candidate.binding.source = OpenGLPipelineState::InlineBinding::EXTERNAL_UNIFORM;
                candidate.name = use.uniformName;
                candidate.binding.packing.elementSize = elementSize;
                candidate.binding.packing.shape = shape;
                candidate.binding.packing.byteSize = static_cast<size_t>(physicalSize);
                if (use.uniformLayout && buffered) {
                    for (uint64_t stride : use.uniformLayout->byteStrides) {
                        if (stride > SIZE_MAX) {
                            useRhiGraphics = false;
                            break;
                        }
                        candidate.binding.packing.byteStrides.push_back(static_cast<size_t>(stride));
                    }
                } else if (shape.size() == 2) {
                    if (bundle.context->backend == VERNON_RUNTIME_OPENGL_ES) {
                        candidate.binding.packing.byteStrides = {elementSize,
                                                                 static_cast<size_t>(shape[0]) * elementSize};
                    } else {
                        candidate.binding.packing.byteStrides = {static_cast<size_t>(shape[1]) * elementSize,
                                                                 elementSize};
                        candidate.binding.transpose = true;
                    }
                } else {
                    candidate.binding.packing.byteStrides.resize(shape.size());
                    size_t stride = elementSize;
                    for (size_t dimension = shape.size(); dimension-- > 0;) {
                        candidate.binding.packing.byteStrides[dimension] = stride;
                        stride *= static_cast<size_t>(shape[dimension]);
                    }
                }
                if (!useRhiGraphics || candidate.binding.packing.byteStrides.size() != shape.size()) {
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
            } else if (parameter.kind == "texture" && use.interfaceKind == "resource" && use.descriptorSet == 0 &&
                       use.binding != UINT32_MAX) {
                candidate.layout.kind = VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE;
                candidate.layout.interface_kind = VERNON_RUNTIME_PROVIDER_INTERFACE_RESOURCE;
                candidate.layout.binding = use.binding;
                candidate.binding.source = OpenGLPipelineState::InlineBinding::EXTERNAL_TEXTURE;
                candidate.name = use.uniformName.empty() ? "main_arg_" + std::to_string(use.index) : use.uniformName;
            } else if (parameter.kind == "sampler" && use.sampledTextureBindings.size() == 1 &&
                       use.sampledTextureBindings[0].descriptorSet == 0 &&
                       use.sampledTextureBindings[0].binding != UINT32_MAX) {
                candidate.layout.kind = VERNON_RUNTIME_PROVIDER_SAMPLER;
                candidate.layout.interface_kind = VERNON_RUNTIME_PROVIDER_INTERFACE_RESOURCE;
                candidate.layout.binding = use.sampledTextureBindings[0].binding;
                candidate.binding.source = OpenGLPipelineState::InlineBinding::EXTERNAL_SAMPLER;
            } else {
                useRhiGraphics = false;
                break;
            }
            candidates.push_back(std::move(candidate));
        }
    }
    uint32_t internalSlot = maximumExternalSlot;
    if (useRhiGraphics && !variant.internalParameters.empty() && internalSlot == UINT32_MAX)
        useRhiGraphics = false;
    if (useRhiGraphics) {
        for (const Parameter &parameter : variant.internalParameters) {
            if (parameter.uses.size() != 1 || internalSlot == UINT32_MAX ||
                (parameter.uses[0].stage != "vertex" && parameter.uses[0].stage != "fragment")) {
                useRhiGraphics = false;
                break;
            }
            const ParameterUse &use = parameter.uses[0];
            OpenGLBindingCandidate candidate;
            candidate.layout.slot = ++internalSlot;
            candidate.layout.stage_mask =
                use.stage == "vertex" ? VERNON_RUNTIME_PROVIDER_STAGE_VERTEX : VERNON_RUNTIME_PROVIDER_STAGE_FRAGMENT;
            candidate.layout.array_count = 1;
            if (parameter.source == "implicit_sampler" && parameter.kind == "sampler" &&
                use.sampledTextureBindings.size() == 1 && use.sampledTextureBindings[0].descriptorSet == 0 &&
                use.sampledTextureBindings[0].binding != UINT32_MAX) {
                candidate.layout.kind = VERNON_RUNTIME_PROVIDER_SAMPLER;
                candidate.layout.interface_kind = VERNON_RUNTIME_PROVIDER_INTERFACE_RESOURCE;
                candidate.layout.binding = use.sampledTextureBindings[0].binding;
                candidate.binding.source = OpenGLPipelineState::InlineBinding::IMPLICIT_SAMPLER;
            } else if (parameter.source == "system_value" && parameter.systemValue == "resolution" &&
                       use.interfaceKind == "uniform" && !use.uniformName.empty()) {
                candidate.layout.kind = VERNON_RUNTIME_PROVIDER_INLINE_VALUE;
                candidate.layout.element_size = 2 * sizeof(float);
                candidate.layout.interface_kind = VERNON_RUNTIME_PROVIDER_INTERFACE_UNIFORM;
                candidate.layout.element_count = 2;
                candidate.layout.vector_count = 1;
                candidate.binding.source = OpenGLPipelineState::InlineBinding::RESOLUTION;
                candidate.name = use.uniformName;
            } else {
                useRhiGraphics = false;
                break;
            }
            candidates.push_back(std::move(candidate));
        }
    }
    if (useRhiGraphics) {
        std::sort(candidates.begin(), candidates.end(),
                  [](const auto &left, const auto &right) { return left.layout.slot < right.layout.slot; });
        for (size_t index = 1; index < candidates.size(); ++index)
            if (candidates[index - 1].layout.slot == candidates[index].layout.slot)
                useRhiGraphics = false;
        state->rhiLayout.reserve(candidates.size());
        state->rhiInlineBindings.reserve(candidates.size());
        for (auto &candidate : candidates) {
            candidate.layout.name = {candidate.name.data(), candidate.name.size()};
            state->rhiLayout.push_back(candidate.layout);
            state->rhiVertexAttributes.insert(state->rhiVertexAttributes.end(), candidate.attributes.begin(),
                                              candidate.attributes.end());
            state->rhiInlineBindings.push_back(std::move(candidate.binding));
        }
    }
    if (useRhiGraphics) {
        const Stage &vertex = bundle.stages.at(variant.vertex);
        const Stage &fragment = bundle.stages.at(variant.fragment);
        const VernonRuntimeProviderShaderDescriptor shaders[2]{{sizeof(VernonRuntimeProviderShaderDescriptor),
                                                                VERNON_RUNTIME_PROVIDER_STAGE_VERTEX,
                                                                {"glsl", 4},
                                                                vertex.source.data(),
                                                                vertex.source.size(),
                                                                {vertex.entry.data(), vertex.entry.size()},
                                                                {nullptr, 0},
                                                                {0, 0, 0, 0}},
                                                               {sizeof(VernonRuntimeProviderShaderDescriptor),
                                                                VERNON_RUNTIME_PROVIDER_STAGE_FRAGMENT,
                                                                {"glsl", 4},
                                                                fragment.source.data(),
                                                                fragment.source.size(),
                                                                {fragment.entry.data(), fragment.entry.size()},
                                                                {nullptr, 0},
                                                                {0, 0, 0, 0}}};
        VernonRuntimeCorePipelineDescriptor descriptor{};
        descriptor.struct_size = sizeof(descriptor);
        descriptor.kind = VERNON_RUNTIME_PROVIDER_GRAPHICS_PIPELINE;
        descriptor.required_capabilities = VERNON_RUNTIME_PROVIDER_GRAPHICS;
        descriptor.shaders = shaders;
        descriptor.shader_count = 2;
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
    bundle.context->error = "OpenGL PipelineAsset is not representable by the RuntimeCore provider";
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
        TensorPackingLayout packing = prepared.packing;
        bool transpose = prepared.transpose;
        const size_t elementSize = tensor.element_layout.byte_size;
        if (transpose && tensor.rank == 2 && tensor.byte_strides &&
            tensor.byte_strides[0] == static_cast<int64_t>(elementSize) &&
            tensor.byte_strides[1] == static_cast<int64_t>(tensor.shape[0] * elementSize)) {
            packing.byteStrides = {elementSize, static_cast<size_t>(tensor.shape[0]) * elementSize};
            transpose = false;
        }
        const std::optional<std::vector<uint8_t>> packed = packTensor(tensor, packing);
        if (!packed || packed->size() != prepared.storage.size())
            return fail(*pipeline.context, "OpenGL RHI uniform Tensor is invalid");
        std::memcpy(prepared.storage.data(), packed->data(), packed->size());
        if (transpose)
            value.flags = VERNON_RUNTIME_PROVIDER_BINDING_TRANSPOSE;
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
    }
    if (plan.depthAttachment) {
        depthAttachment = plan.depthAttachment->resource;
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
    status = vernonRuntimeCoreEncodeDispatch(state.rhiPipeline, state.rhiBindings, {}, groups, nullptr, 0);
    return status == VERNON_STATUS_OK ? status
                                      : fail(*pipeline.context, "failed to encode OpenGL provider dispatch", status);
}

} // namespace vernon::runtime
