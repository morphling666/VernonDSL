#include "runtime_pipeline_backend.h"

#include "VernonRuntimeRHIAdapter.h"
#include "backend_opengl.h"
#include "compute_launch_planner.h"
#include "pipeline_metadata.h"
#include "prepared_graphics_draw.h"
#include "tensor_bridge.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cstring>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace vernon::runtime {

bool resolveOpenGLNativeUniformShape(std::string_view dtype, const std::vector<uint64_t> &shape,
                                     OpenGLNativeUniformShape &result) {
    result = {};
    if (shape.empty()) {
        result.scalarCount = 1;
        return dtype == "f32" || dtype == "i32" || dtype == "u32";
    }
    if (shape.size() == 1 && shape[0] >= 1 && shape[0] <= 4) {
        result.scalarCount = static_cast<uint32_t>(shape[0]);
        return dtype == "f32" || dtype == "i32" || dtype == "u32";
    }
    if (dtype != "f32" || shape.size() != 2 || shape[0] < 2 || shape[0] > 4 || shape[1] < 2 || shape[1] > 4)
        return false;
    result.scalarCount = static_cast<uint32_t>(shape[0] * shape[1]);
    result.matrixColumns = static_cast<uint32_t>(shape[1]);
    return true;
}

namespace {

VernonStatus fail(VernonRuntimeContext &context, std::string error,
                  VernonStatus status = VERNON_STATUS_INVALID_ARGUMENT) {
    invocationDiagnostic(context) = std::move(error);
    return status;
}

} // namespace

bool resolveOpenGLPipeline(BackendStageBuildInputs &inputs, const StageBindingPlan &plan,
                           VernonStageExecutable &pipeline) {
    auto *state = new OpenGLPipelineState();
    struct OpenGLBindingCandidate {
        VernonRuntimeProviderBindingLayoutEntry layout{};
        std::vector<VernonRuntimeProviderVertexAttribute> attributes;
        OpenGLPipelineState::InlineBinding binding;
        ComputeBindingSource source;
        std::string name;
    };
    std::vector<OpenGLBindingCandidate> candidates;
    uint32_t computeInternalSlot = 0;
    for (const Parameter &parameter : plan.parameters)
        computeInternalSlot = std::max(computeInternalSlot, parameter.slot);
    if (!plan.compute.empty()) {
        const LoadedStageArtifact &stage = inputs.artifacts.at(plan.compute);
        ReflectedEntry reflection;
        if (!resolveStageReflection(stage, inputs.context->backend, reflection,
                                    invocationDiagnostic(*inputs.context))) {
            delete state;
            return false;
        }
        for (const Parameter &parameter : plan.parameters) {
            if (parameter.kind == "image" && parameter.bindingRole == "sampled") {
                invocationDiagnostic(*inputs.context) =
                    "OpenGL compute pipelines do not support sampled textures in the current language contract";
                delete state;
                return false;
            }
            for (const ParameterUse &use : parameter.uses) {
                if (use.stage != "compute" && use.stage != plan.compute)
                    continue;
                if (use.index >= reflection.arguments.size() || use.binding == UINT32_MAX) {
                    invocationDiagnostic(*inputs.context) = "OpenGL compute parameter reflection is incomplete";
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
                    candidate.layout.kind = parameter.kind == "image"   ? (parameter.bindingRole == "sampled"
                                                                               ? VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE
                                                                               : VERNON_RUNTIME_PROVIDER_STORAGE_IMAGE)
                                            : argument.kind == "tensor" ? VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER
                                                                        : VERNON_RUNTIME_PROVIDER_INLINE_VALUE;
                    configureComputeValueStorage(use, candidate.layout);
                    candidate.layout.stage_mask = VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE;
                    candidate.layout.access = parameter.access == "read" ? 1u : parameter.access == "write" ? 2u : 3u;
                    if (parameter.kind == "image" && !configureImageBindingLayout(parameter, candidate.layout))
                        return false;
                    candidate.layout.array_count = 1;
                    candidate.layout.argument_index = use.index;
                    candidate.layout.element_size = static_cast<uint32_t>(
                        argument.storageLeaves.empty() ? (parameter.kind == "image"   ? 1
                                                          : argument.kind == "tensor" ? argument.tensorElementSize
                                                                                      : argument.physical.size)
                                                       : argument.storageLeaves[leafIndex].elementSize);
                    candidate.binding.externalSlot = parameter.slot;
                    candidate.binding.source =
                        parameter.kind == "image"   ? OpenGLPipelineState::InlineBinding::EXTERNAL_TEXTURE
                        : argument.kind == "tensor" ? OpenGLPipelineState::InlineBinding::EXTERNAL_STORAGE
                                                    : OpenGLPipelineState::InlineBinding::COMPUTE_INLINE;
                    candidate.source = {ComputeBindingSourceKind::Argument, use.index, 0};
                    if (candidate.layout.element_size == 0) {
                        invocationDiagnostic(*inputs.context) = "OpenGL compute parameter has zero element size";
                        delete state;
                        return false;
                    }
                    candidates.push_back(std::move(candidate));
                }
                if (use.tensorViewDescriptor) {
                    const auto addDescriptor = [&](ComputeBindingSourceKind kind, uint32_t dimension,
                                                   uint32_t binding) {
                        OpenGLBindingCandidate candidate;
                        candidate.layout.slot = ++computeInternalSlot;
                        candidate.layout.binding = binding;
                        candidate.layout.kind = VERNON_RUNTIME_PROVIDER_INLINE_VALUE;
                        candidate.layout.stage_mask = VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE;
                        candidate.layout.access = 1;
                        candidate.layout.array_count = 1;
                        candidate.layout.argument_index = use.index;
                        candidate.layout.element_size = 4;
                        candidate.binding.externalSlot = parameter.slot;
                        candidate.binding.source = OpenGLPipelineState::InlineBinding::COMPUTE_INLINE;
                        candidate.source = {kind, use.index, dimension};
                        candidates.push_back(std::move(candidate));
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
        }
        std::sort(candidates.begin(), candidates.end(),
                  [](const auto &left, const auto &right) { return left.layout.slot < right.layout.slot; });
        state->rhiLayout.reserve(candidates.size());
        state->rhiInlineBindings.reserve(candidates.size());
        state->rhiValues.resize(candidates.size());
        for (size_t index = 0; index < candidates.size(); ++index) {
            state->rhiLayout.push_back(candidates[index].layout);
            state->rhiInlineBindings.push_back(std::move(candidates[index].binding));
            state->rhiComputeBindingSources.push_back(candidates[index].source);
            state->rhiValues[index].slot = state->rhiLayout[index].slot;
            state->rhiValues[index].kind = state->rhiLayout[index].kind;
            if (state->rhiLayout[index].kind == VERNON_RUNTIME_PROVIDER_INLINE_VALUE) {
                state->rhiInlineBindings[index].storage.resize(state->rhiLayout[index].element_size);
                state->rhiValues[index].payload.inline_value.data = state->rhiInlineBindings[index].storage.data();
                state->rhiValues[index].payload.inline_value.size = state->rhiInlineBindings[index].storage.size();
            } else {
                state->rhiValues[index].flags = VERNON_RUNTIME_PROVIDER_BINDING_DEFAULT_RESOURCE;
            }
        }
        state->rhiComputeDescriptorValues.resize(candidates.size());
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
            vernonRuntimeRhiAdapterGetProvider(openGLState(*inputs.context).adapter), &descriptor, &state->rhiPipeline);
        if (status != VERNON_STATUS_OK) {
            const VernonStringView providerError =
                vernonRuntimeRhiAdapterGetLastError(openGLState(*inputs.context).adapter);
            invocationDiagnostic(*inputs.context) = providerError.data && providerError.size
                                                        ? std::string(providerError.data, providerError.size)
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
    bool useRhiGraphics = plan.compute.empty() && !plan.vertex.empty() && !plan.fragment.empty();
    std::string representationError;
    const auto graphicsStageMask = [](const std::string &stage) {
        return stage == "vertex"     ? uint32_t{VERNON_RUNTIME_PROVIDER_STAGE_VERTEX}
               : stage == "fragment" ? uint32_t{VERNON_RUNTIME_PROVIDER_STAGE_FRAGMENT}
                                     : uint32_t{0};
    };
    if (useRhiGraphics) {
        for (const Parameter &parameter : plan.parameters) {
            for (const ParameterUse &use : parameter.uses) {
                if (graphicsStageMask(use.stage) == 0) {
                    representationError = "OpenGL graphics parameter references an unsupported stage";
                    useRhiGraphics = false;
                    break;
                }
                if ((use.binding != UINT32_MAX && use.descriptorSet != 0) ||
                    std::any_of(use.sampledImageBindings.begin(), use.sampledImageBindings.end(),
                                [](const SampledImageBinding &binding) { return binding.descriptorSet != 0; })) {
                    representationError = "OpenGL supports reflected resources only in descriptor set 0";
                    useRhiGraphics = false;
                    break;
                }
                OpenGLBindingCandidate candidate;
                candidate.layout.stage_mask = graphicsStageMask(use.stage);
                candidate.layout.array_count = 1;
                candidate.binding.externalSlot = parameter.slot;
                if (parameter.kind == "sampler") {
                    if (use.sampledImageBindings.empty()) {
                        representationError = "OpenGL sampler has no reflected image binding";
                        useRhiGraphics = false;
                        break;
                    }
                    for (const SampledImageBinding &binding : use.sampledImageBindings) {
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
                if (parameter.kind == "tensor" && use.interfaceKind == "uniform" && use.interfacePlan &&
                    use.interfacePlan->root &&
                    (!use.uniformName.empty() || use.transport == "uniform_buffer" ||
                     use.transport == "storage_buffer")) {
                    const auto &shape = use.shape.empty() ? parameter.shape : use.shape;
                    const std::optional<VernonDataType> dtype = pipelineDataType(use.dtype);
                    const uint64_t physicalSize = use.interfacePlan->root->size;
                    const bool buffered = use.transport == "uniform_buffer" || use.transport == "storage_buffer";
                    OpenGLNativeUniformShape nativeShape;
                    if (!dtype || (!buffered && !resolveOpenGLNativeUniformShape(use.dtype, shape, nativeShape))) {
                        representationError = "OpenGL uniform layout is unsupported";
                        useRhiGraphics = false;
                        break;
                    }
                    candidate.layout.kind = use.transport == "storage_buffer"   ? VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER
                                            : use.transport == "uniform_buffer" ? VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER
                                                                                : VERNON_RUNTIME_PROVIDER_INLINE_VALUE;
                    if (!physicalSize || physicalSize > UINT32_MAX || (buffered && use.binding == UINT32_MAX)) {
                        representationError = "OpenGL uniform reflection is incomplete";
                        useRhiGraphics = false;
                        break;
                    }
                    candidate.layout.element_size = static_cast<uint32_t>(physicalSize);
                    candidate.layout.interface_kind = VERNON_RUNTIME_PROVIDER_INTERFACE_UNIFORM;
                    candidate.layout.element_count = buffered ? 0 : nativeShape.scalarCount;
                    candidate.layout.vector_count = buffered ? 0 : nativeShape.matrixColumns;
                    candidate.layout.numeric_type = static_cast<uint32_t>(*dtype);
                    candidate.layout.binding = use.binding;
                    candidate.layout.set = use.descriptorSet;
                    candidate.binding.source = OpenGLPipelineState::InlineBinding::EXTERNAL_UNIFORM;
                    candidate.name = use.uniformName;
                    const ValueLayout &canonical =
                        parameter.valueLayout ? *parameter.valueLayout : parameter.elementLayout;
                    std::optional<TensorCopyPlan> packing =
                        parameter.valueLayout
                            ? compileWholeValueCopyPlan(pipelineValueLayout(canonical), *use.interfacePlan->root)
                            : compileElementStreamCopyPlan(pipelineValueLayout(canonical), shape,
                                                           *use.interfacePlan->root);
                    if (!packing || packing->elementSize != canonical.byteSize) {
                        representationError = "OpenGL interface plan does not match the canonical layout";
                        useRhiGraphics = false;
                        break;
                    }
                    candidate.binding.packing = std::move(*packing);
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
                } else if (parameter.kind == "image" && use.interfaceKind == "resource" && use.binding != UINT32_MAX) {
                    candidate.layout.kind = parameter.bindingRole == "sampled" ? VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE
                                                                               : VERNON_RUNTIME_PROVIDER_STORAGE_IMAGE;
                    if (!configureImageBindingLayout(parameter, candidate.layout)) {
                        representationError = "OpenGL graphics image parameter layout is incomplete";
                        useRhiGraphics = false;
                        break;
                    }
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
        for (const Parameter &parameter : plan.runtimeParameters) {
            for (const ParameterUse &use : parameter.uses) {
                if (graphicsStageMask(use.stage) == 0) {
                    representationError = "OpenGL internal parameter references an unsupported stage";
                    useRhiGraphics = false;
                    break;
                }
                const uint32_t stageMask = graphicsStageMask(use.stage);
                if (parameter.source == StageParameterSource::ImplicitSampler && parameter.kind == "sampler" &&
                    !use.sampledImageBindings.empty()) {
                    for (const SampledImageBinding &binding : use.sampledImageBindings) {
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
                } else if (parameter.source == StageParameterSource::Resolution && use.interfaceKind == "uniform" &&
                           !use.uniformName.empty()) {
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
        const LoadedStageArtifact &vertex = inputs.artifacts.at(plan.vertex);
        const LoadedStageArtifact &fragment = inputs.artifacts.at(plan.fragment);
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
            vernonRuntimeRhiAdapterGetProvider(openGLState(*inputs.context).adapter), &descriptor, &state->rhiPipeline);
        if (status != VERNON_STATUS_OK) {
            const VernonStringView providerError =
                vernonRuntimeRhiAdapterGetLastError(openGLState(*inputs.context).adapter);
            invocationDiagnostic(*inputs.context) = providerError.data && providerError.size
                                                        ? std::string(providerError.data, providerError.size)
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
                    value.payload.inline_value.data = state->rhiInlineBindings[index].storage.data();
                    value.payload.inline_value.size = state->rhiInlineBindings[index].storage.size();
                } else {
                    value.flags = VERNON_RUNTIME_PROVIDER_BINDING_DEFAULT_RESOURCE;
                }
            }
            const VernonStatus bindingStatus = vernonRuntimeCoreCreateBindings(
                state->rhiPipeline, state->rhiValues.data(), state->rhiValues.size(), &state->rhiBindings);
            if (bindingStatus != VERNON_STATUS_OK) {
                const VernonStringView providerError =
                    vernonRuntimeRhiAdapterGetLastError(openGLState(*inputs.context).adapter);
                invocationDiagnostic(*inputs.context) = providerError.data && providerError.size
                                                            ? std::string(providerError.data, providerError.size)
                                                            : "failed to prepare OpenGL RHI bindings";
                vernonRuntimeCorePipelineDestroy(state->rhiPipeline);
                delete state;
                return false;
            }
        }
        installRuntimeBackendState(pipeline, state);
        return true;
    }
    invocationDiagnostic(*inputs.context) = representationError.empty()
                                                ? "OpenGL ProgramAsset is not representable by the RuntimeCore provider"
                                                : std::move(representationError);
    delete state;
    return false;
}

void destroyOpenGLPipeline(VernonStageExecutable &pipeline) {
    OpenGLPipelineState &state = runtimeBackendState<OpenGLPipelineState>(pipeline);
    destroyGraphicsVariant(state.rhiGraphicsVariant);
    vernonRuntimeCoreBindingsDestroy(state.rhiBindings);
    vernonRuntimeCorePipelineDestroy(state.rhiPipeline);
}

VernonStatus invokeOpenGLGraphicsPipeline(VernonStageExecutable &pipeline,
                                          const VernonStageInvocationDescriptor &invocation,
                                          const PlannedGraphicsInvocation &plan) {
    OpenGLPipelineState &state = runtimeBackendState<OpenGLPipelineState>(pipeline);
    if (!state.rhiPipeline)
        return VERNON_STATUS_OK;
    for (size_t bindingIndex = 0; bindingIndex < state.rhiLayout.size(); ++bindingIndex) {
        const auto &layout = state.rhiLayout[bindingIndex];
        auto &prepared = state.rhiInlineBindings[bindingIndex];
        VernonRuntimeProviderBindingValue &value = state.rhiValues[bindingIndex];
        value = {};
        value.slot = layout.slot;
        value.kind = layout.kind;
        if (prepared.source == OpenGLPipelineState::InlineBinding::EXTERNAL_VERTEX) {
            const auto found = plan.arguments.find(prepared.externalSlot);
            if (found == plan.arguments.end() || found->second->kind != VERNON_PROGRAM_TENSOR ||
                found->second->tensor.storage != VERNON_TENSOR_RHI_RESOURCE ||
                !found->second->tensor.resource.resource.value)
                return fail(*pipeline.context, "OpenGL RHI vertex argument is missing");
            const VernonTensorView &tensor = found->second->tensor;
            if (!tensor.byte_strides || tensor.byte_strides[0] <= 0 ||
                static_cast<uint64_t>(tensor.byte_strides[0]) > UINT32_MAX)
                return fail(*pipeline.context, "OpenGL RHI vertex Tensor strides are invalid");
            value.payload.buffer.resource = tensor.resource;
            value.payload.buffer.resource.offset += tensor.byte_offset;
            value.payload.buffer.stride = static_cast<uint32_t>(tensor.byte_strides[0]);
            continue;
        }
        if (prepared.source == OpenGLPipelineState::InlineBinding::EXTERNAL_STORAGE) {
            const auto found = plan.arguments.find(prepared.externalSlot);
            if (found == plan.arguments.end() || found->second->kind != VERNON_PROGRAM_TENSOR ||
                found->second->tensor.storage != VERNON_TENSOR_RHI_RESOURCE ||
                !found->second->tensor.resource.resource.value)
                return fail(*pipeline.context, "OpenGL RHI storage argument is missing");
            const VernonTensorView &tensor = found->second->tensor;
            value.payload.buffer.resource = tensor.resource;
            value.payload.buffer.resource.offset += tensor.byte_offset;
            continue;
        }
        if (prepared.source == OpenGLPipelineState::InlineBinding::EXTERNAL_TEXTURE) {
            const auto found = plan.arguments.find(prepared.externalSlot);
            if (found == plan.arguments.end() || found->second->kind != VERNON_PROGRAM_IMAGE ||
                !found->second->image.view.resource.value)
                return fail(*pipeline.context, "OpenGL RHI image argument is missing");
            value.payload.image.view = found->second->image.view;
            continue;
        }
        if (prepared.source == OpenGLPipelineState::InlineBinding::EXTERNAL_SAMPLER) {
            const auto found = plan.arguments.find(prepared.externalSlot);
            if (found == plan.arguments.end() || found->second->kind != VERNON_PROGRAM_SAMPLER ||
                !found->second->resource.resource.value)
                return fail(*pipeline.context, "OpenGL RHI sampler argument is missing");
            value.payload.sampler.resource = found->second->resource;
            continue;
        }
        if (prepared.source == OpenGLPipelineState::InlineBinding::IMPLICIT_SAMPLER) {
            const auto sampled = plan.sampledResources.find({layout.set, layout.binding});
            if (sampled == plan.sampledResources.end())
                return fail(*pipeline.context, "OpenGL RHI implicit sampler binding is missing");
            if (sampled->second.samplerResource.resource.value) {
                value.payload.sampler.resource = sampled->second.samplerResource;
            } else {
                value.flags = VERNON_RUNTIME_PROVIDER_BINDING_DEFAULT_RESOURCE;
            }
            continue;
        }
        if (prepared.source == OpenGLPipelineState::InlineBinding::RESOLUTION) {
            std::memcpy(prepared.storage.data(), plan.resolution.data(), 2 * sizeof(float));
            value.payload.inline_value.data = prepared.storage.data();
            value.payload.inline_value.size = prepared.storage.size();
            continue;
        }
        const auto found = plan.arguments.find(prepared.externalSlot);
        if (found == plan.arguments.end() || found->second->kind != VERNON_PROGRAM_TENSOR)
            return fail(*pipeline.context, "OpenGL RHI uniform argument is missing");
        const VernonTensorView &tensor = found->second->tensor;
        const std::optional<std::vector<uint8_t>> packed = packTensor(tensor, prepared.packing);
        if (!packed || packed->size() != prepared.storage.size())
            return fail(*pipeline.context, "OpenGL RHI uniform Tensor is invalid");
        std::memcpy(prepared.storage.data(), packed->data(), packed->size());
        value.payload.inline_value.data = prepared.storage.data();
        value.payload.inline_value.size = prepared.storage.size();
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
    std::vector<uint32_t> formats;
    formats.reserve(plan.attachments.size());
    for (VernonTextureFormat format : plan.attachmentFormats)
        formats.push_back(static_cast<uint32_t>(format));
    const uint32_t depthFormat = !plan.depthAttachment ? 0
                                 : plan.depthFormat == VERNON_TEXTURE_D32_FLOAT_S8_UINT
                                     ? rhi::opengl::kDepth32fStencil8
                                     : rhi::opengl::kDepthComponent32f;
    PreparedGraphicsDraw prepared;
    if (!prepareGraphicsDraw(invocation, plan, std::move(formats), depthFormat, state.rhiLayout, state.rhiValues,
                             prepared, invocationDiagnostic(*pipeline.context)))
        return VERNON_STATUS_INVALID_ARGUMENT;
    const VernonStatus variantStatus =
        ensureGraphicsVariant(state.rhiPipeline, prepared.variantKey, state.rhiGraphicsVariant);
    if (variantStatus != VERNON_STATUS_OK) {
        const VernonStringView providerError =
            vernonRuntimeRhiAdapterGetLastError(openGLState(*pipeline.context).adapter);
        return fail(*pipeline.context,
                    providerError.data ? std::string(providerError.data, providerError.size)
                                       : "failed to prepare OpenGL RHI graphics variant",
                    variantStatus);
    }
    const VernonStatus status = vernonRuntimeCoreEncodeGraphicsVariantDrawInvocation(
        state.rhiGraphicsVariant.handle, state.rhiBindings, &prepared.invocation);
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

VernonStatus invokeOpenGLComputePipeline(VernonStageExecutable &pipeline, const PlannedComputeLaunch &launch) {
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
        const ComputeBindingSource &source = state.rhiComputeBindingSources[index];
        if (source.kind != ComputeBindingSourceKind::Argument) {
            std::optional<int64_t> descriptor = computeBindingDescriptorValue(argument, source);
            if (!descriptor || *descriptor < INT32_MIN || *descriptor > INT32_MAX)
                return fail(*pipeline.context, "OpenGL TensorView descriptor exceeds the shader index range");
            state.rhiComputeDescriptorValues[index] = static_cast<int32_t>(*descriptor);
            value.payload.inline_value.data = &state.rhiComputeDescriptorValues[index];
            value.payload.inline_value.size = sizeof(int32_t);
            continue;
        }
        if (layout.kind == VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER) {
            if (layout.interface_kind == VERNON_RUNTIME_PROVIDER_INTERFACE_UNIFORM) {
                if (!bindComputeValueStorage(layout, argument, value))
                    return fail(*pipeline.context, "OpenGL prepared Value storage binding has invalid bytes");
                continue;
            }
            const auto *tensor = std::get_if<ComputeTensorArgument>(&argument);
            if (!tensor || !tensor->resource.resource.value)
                return fail(*pipeline.context, "OpenGL prepared storage binding requires an RHI Tensor");
            value.payload.buffer.resource = tensor->resource;
        } else if (layout.kind == VERNON_RUNTIME_PROVIDER_STORAGE_IMAGE ||
                   layout.kind == VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE) {
            const auto *image = std::get_if<ComputeImageArgument>(&argument);
            if (!image || !image->view.resource.value)
                return fail(*pipeline.context, "OpenGL prepared image binding requires an RHI Texture");
            value.payload.image.view = image->view;
        } else {
            const auto *scalar = std::get_if<ComputeScalarArgument>(&argument);
            if (!scalar || !scalar->data || !scalar->size)
                return fail(*pipeline.context, "OpenGL prepared inline binding requires host data");
            value.payload.inline_value.data = scalar->data;
            value.payload.inline_value.size = scalar->size;
        }
    }
    VernonStatus status =
        state.rhiBindings
            ? vernonRuntimeCoreUpdateBindings(state.rhiBindings, state.rhiValues.data(), state.rhiValues.size())
            : vernonRuntimeCoreCreateBindings(state.rhiPipeline, state.rhiValues.data(), state.rhiValues.size(),
                                              &state.rhiBindings);
    if (status != VERNON_STATUS_OK)
        return fail(*pipeline.context, "failed to prepare OpenGL provider bindings", status);
    const uint32_t groups[3]{launch.grid.x, launch.grid.y, launch.grid.z};
    status = vernonRuntimeCoreEncodeDispatch(state.rhiPipeline, state.rhiBindings, launch.commandEncoder, groups,
                                             nullptr, 0);
    return status == VERNON_STATUS_OK ? status
                                      : fail(*pipeline.context, "failed to encode OpenGL provider dispatch", status);
}

} // namespace vernon::runtime
