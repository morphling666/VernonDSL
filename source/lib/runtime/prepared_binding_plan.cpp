#include "prepared_binding_plan.h"

#include "runtime/autodiff/tape_allocator_abi.h"

#include <algorithm>
#include <cstring>
#include <optional>
#include <unordered_set>

namespace vernon::runtime {
namespace {

bool isOpenGLBackend(VernonRuntimeBackend backend) {
    return backend == VERNON_RUNTIME_OPENGL || backend == VERNON_RUNTIME_OPENGL_ES;
}

uint32_t accessMask(const Parameter &parameter) {
    return parameter.access == "read" ? 1u : parameter.access == "write" ? 2u : 3u;
}

bool appendBinding(PreparedComputeBindingPlan &plan, VernonRuntimeProviderBindingLayoutEntry layout,
                   ComputeBindingSource source, PreparedDescriptorWidth width, uint64_t resourceOffset,
                   std::string &error) {
    if (!layout.element_size) {
        error = "prepared compute binding has zero element size";
        return false;
    }
    plan.layouts.push_back(layout);
    plan.sources.push_back({source, width, resourceOffset});
    return true;
}

std::optional<VernonRuntimeProviderBindingKind> graphicsProviderKind(std::string_view transport) {
    if (transport == "storage_buffer")
        return VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER;
    if (transport == "uniform_buffer")
        return VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER;
    if (transport == "push_constant")
        return VERNON_RUNTIME_PROVIDER_INLINE_VALUE;
    return std::nullopt;
}

uint32_t graphicsStageMask(std::string_view stage) {
    if (stage == "vertex")
        return VERNON_RUNTIME_PROVIDER_STAGE_VERTEX;
    if (stage == "fragment")
        return VERNON_RUNTIME_PROVIDER_STAGE_FRAGMENT;
    return 0;
}

bool usesSequentialGraphicsSlots(VernonRuntimeBackend backend) { return backend != VERNON_RUNTIME_DIRECTX12; }

bool isBufferedUniform(VernonRuntimeProviderBindingKind kind) {
    return kind == VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER || kind == VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER;
}

} // namespace

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

bool buildPreparedComputeBindingPlan(const StageBindingPlan &stagePlan, const ReflectedEntry &reflection,
                                     VernonRuntimeBackend backend, PreparedComputeBindingPlan &output,
                                     std::string &error) {
    output = {};
    output.backend = backend;
    error.clear();
    if (backend == VERNON_RUNTIME_CPU) {
        output.packedArgumentSize = reflection.packedArguments ? reflection.packedArguments->size : 0;
        output.packedResultSize = reflection.packedResults ? reflection.packedResults->size : 0;
        for (const ReflectedArgument &argument : reflection.arguments) {
            const bool tapeBuiltin =
                argument.kind == "builtin" && (argument.builtin == VERNON_AD_TAPE_ALLOCATOR_BUILTIN ||
                                               argument.builtin == VERNON_AD_TAPE_ROOT_REGION_BUILTIN);
            if (argument.kind == "builtin" && !tapeBuiltin)
                continue;
            if (!tapeBuiltin && argument.index == UINT32_MAX) {
                error = "CPU compute reflection is missing a kernel argument index";
                return false;
            }
            VernonRuntimeProviderBindingLayoutEntry layout{};
            layout.slot = static_cast<uint32_t>(output.layouts.size());
            layout.set = argument.descriptorSet;
            layout.binding = argument.binding == UINT32_MAX ? layout.slot : argument.binding;
            layout.kind = argument.kind == "tensor" && !argument.tensorViewDescriptor
                              ? VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER
                              : VERNON_RUNTIME_PROVIDER_INLINE_VALUE;
            layout.stage_mask = VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE;
            layout.access = 3;
            layout.array_count = 1;
            layout.argument_index = tapeBuiltin ? UINT32_MAX : argument.index;
            layout.element_size =
                static_cast<uint32_t>(layout.kind == VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER ? argument.tensorElementSize
                                                                                            : argument.physical.size);
            if (!layout.element_size) {
                error = "CPU compute reflection contains a zero-sized argument";
                return false;
            }
            output.layouts.push_back(layout);
            output.sources.push_back(
                {{ComputeBindingSourceKind::Argument, layout.argument_index, 0}, PreparedDescriptorWidth::None, 0});
            output.cpuBindings.push_back(
                {tapeBuiltin ? argument.builtin : std::string(), argument.physical.offset, argument.physical.size,
                 argument.result,
                 argument.result && argument.autodiffRole == "gradient" ? argument.dtype : std::nullopt});
            if (tapeBuiltin && argument.builtin == VERNON_AD_TAPE_ALLOCATOR_BUILTIN)
                output.tapeAllocatorOffset = argument.physical.offset;
            if (tapeBuiltin && argument.builtin == VERNON_AD_TAPE_ROOT_REGION_BUILTIN)
                output.tapeRootOffset = argument.physical.offset;
        }
        return validatePreparedComputeBindingPlan(output, error);
    }

    uint32_t internalSlot = 0;
    for (const Parameter &parameter : stagePlan.parameters)
        internalSlot = std::max(internalSlot, parameter.slot);

    std::vector<uint32_t> argumentBindings(reflection.arguments.size());
    uint32_t flattenedBinding = 0;
    for (size_t index = 0; index < reflection.arguments.size(); ++index) {
        argumentBindings[index] = flattenedBinding;
        if (reflection.arguments[index].kind != "builtin")
            flattenedBinding +=
                static_cast<uint32_t>(std::max(reflection.arguments[index].storageLeaves.size(), size_t{1}));
    }

    std::vector<uint32_t> cudaDescriptorBindings(reflection.arguments.size(), UINT32_MAX);
    if (backend == VERNON_RUNTIME_CUDA)
        for (size_t index = 0; index < reflection.arguments.size(); ++index)
            if (reflection.arguments[index].tensorViewDescriptor) {
                cudaDescriptorBindings[index] = flattenedBinding;
                flattenedBinding += 1 + 2 * reflection.arguments[index].tensorViewDescriptor->rank;
            }

    for (const Parameter &parameter : stagePlan.parameters) {
        if (isOpenGLBackend(backend) && parameter.kind == "image" && parameter.bindingRole == "sampled") {
            error = "OpenGL compute pipelines do not support sampled textures in the current language contract";
            return false;
        }
        for (const ParameterUse &use : parameter.uses) {
            if (use.stage != "compute" && use.stage != stagePlan.compute)
                continue;
            if (use.index >= reflection.arguments.size()) {
                error = "prepared compute parameter use exceeds the reflected argument table";
                return false;
            }
            const ReflectedArgument &argument = reflection.arguments[use.index];
            if (isOpenGLBackend(backend) && use.binding == UINT32_MAX) {
                error = "OpenGL compute parameter reflection is incomplete";
                return false;
            }

            const size_t leafCount = std::max(argument.storageLeaves.size(), size_t{1});
            for (size_t leafIndex = 0; leafIndex < leafCount; ++leafIndex) {
                VernonRuntimeProviderBindingLayoutEntry layout{};
                layout.slot = leafIndex == 0 ? parameter.slot : ++internalSlot;
                layout.set = backend == VERNON_RUNTIME_CUDA || isOpenGLBackend(backend) ? 0 : argument.descriptorSet;
                if (backend == VERNON_RUNTIME_CUDA || backend == VERNON_RUNTIME_DIRECTX12)
                    layout.binding = argumentBindings[use.index] + static_cast<uint32_t>(leafIndex);
                else if (isOpenGLBackend(backend))
                    layout.binding =
                        argument.storageLeaves.empty() ? use.binding : argument.storageLeaves[leafIndex].binding;
                else if (backend == VERNON_RUNTIME_VULKAN)
                    layout.binding = argument.storageLeaves.size() <= 1
                                         ? argumentBindings[use.index] + static_cast<uint32_t>(leafIndex)
                                         : argument.storageLeaves[leafIndex].binding;
                else
                    layout.binding = argument.storageLeaves.empty()
                                         ? argumentBindings[use.index] + static_cast<uint32_t>(leafIndex)
                                         : argument.storageLeaves[leafIndex].binding;

                if (backend == VERNON_RUNTIME_CUDA)
                    layout.kind =
                        use.interfaceKind == "storage" || (use.transport == "storage_buffer" && !use.shape.empty())
                            ? VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER
                            : VERNON_RUNTIME_PROVIDER_INLINE_VALUE;
                else
                    layout.kind = parameter.kind == "image"
                                      ? (parameter.bindingRole == "sampled" ? VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE
                                                                            : VERNON_RUNTIME_PROVIDER_STORAGE_IMAGE)
                                  : argument.kind == "tensor" ? VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER
                                                              : VERNON_RUNTIME_PROVIDER_INLINE_VALUE;
                if (layout.kind == VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER && use.interfaceKind == "value" &&
                    !use.tensorViewDescriptor)
                    layout.interface_kind = VERNON_RUNTIME_PROVIDER_INTERFACE_UNIFORM;
                layout.stage_mask = VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE;
                layout.access = accessMask(parameter);
                layout.array_count = 1;
                layout.argument_index = use.index;
                layout.element_size = static_cast<uint32_t>(
                    argument.storageLeaves.empty()
                        ? (parameter.kind == "image"                               ? 1
                           : layout.kind == VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER ? argument.tensorElementSize
                                                                                   : argument.physical.size)
                        : argument.storageLeaves[leafIndex].elementSize);
                if (parameter.kind == "image" && !configureImageBindingLayout(parameter, layout)) {
                    error = "prepared compute image layout is incomplete";
                    return false;
                }
                if (layout.binding == UINT32_MAX) {
                    error = "prepared compute native binding is incomplete";
                    return false;
                }
                if (!appendBinding(output, layout, {ComputeBindingSourceKind::Argument, use.index, 0},
                                   PreparedDescriptorWidth::None, 0, error))
                    return false;
            }

            if (!use.tensorViewDescriptor)
                continue;
            uint32_t cudaBinding = backend == VERNON_RUNTIME_CUDA ? cudaDescriptorBindings[use.index] : UINT32_MAX;
            const PreparedDescriptorWidth width =
                backend == VERNON_RUNTIME_CUDA ? PreparedDescriptorWidth::I64 : PreparedDescriptorWidth::I32;
            const uint32_t elementSize = backend == VERNON_RUNTIME_CUDA ? 8u : 4u;
            const auto appendDescriptor = [&](ComputeBindingSourceKind kind, uint32_t dimension,
                                              uint32_t reflectedBinding) {
                VernonRuntimeProviderBindingLayoutEntry layout{};
                layout.slot = ++internalSlot;
                layout.set = backend == VERNON_RUNTIME_CUDA || isOpenGLBackend(backend) ? 0 : argument.descriptorSet;
                layout.binding = backend == VERNON_RUNTIME_CUDA ? cudaBinding++ : reflectedBinding;
                layout.kind = VERNON_RUNTIME_PROVIDER_INLINE_VALUE;
                layout.stage_mask = VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE;
                layout.access = 1;
                layout.array_count = 1;
                layout.argument_index = use.index;
                layout.element_size = elementSize;
                return appendBinding(output, layout, {kind, use.index, dimension}, width, 0, error);
            };
            if (!appendDescriptor(ComputeBindingSourceKind::TensorOffset, 0, use.tensorViewDescriptor->offsetBinding))
                return false;
            for (uint32_t dimension = 0; dimension < use.tensorViewDescriptor->rank; ++dimension)
                if (!appendDescriptor(ComputeBindingSourceKind::TensorExtent, dimension,
                                      use.tensorViewDescriptor->extentBindings[dimension]))
                    return false;
            for (uint32_t dimension = 0; dimension < use.tensorViewDescriptor->rank; ++dimension)
                if (!appendDescriptor(ComputeBindingSourceKind::TensorStride, dimension,
                                      use.tensorViewDescriptor->strideBindings[dimension]))
                    return false;
        }
    }

    std::vector<size_t> order(output.layouts.size());
    for (size_t index = 0; index < order.size(); ++index)
        order[index] = index;
    std::sort(order.begin(), order.end(),
              [&](size_t left, size_t right) { return output.layouts[left].slot < output.layouts[right].slot; });
    PreparedComputeBindingPlan sorted;
    sorted.backend = backend;
    sorted.layouts.reserve(order.size());
    sorted.sources.reserve(order.size());
    for (size_t index : order) {
        sorted.layouts.push_back(output.layouts[index]);
        sorted.sources.push_back(output.sources[index]);
    }
    output = std::move(sorted);
    return validatePreparedComputeBindingPlan(output, error);
}

bool validatePreparedComputeBindingPlan(const PreparedComputeBindingPlan &plan, std::string &error) {
    if (plan.layouts.size() != plan.sources.size()) {
        error = "prepared compute binding arrays have different lengths";
        return false;
    }
    if (plan.backend == VERNON_RUNTIME_CPU && plan.cpuBindings.size() != plan.layouts.size()) {
        error = "prepared CPU binding metadata does not match the physical binding sequence";
        return false;
    }
    if (plan.backend != VERNON_RUNTIME_CPU && !plan.cpuBindings.empty()) {
        error = "GPU prepared binding plan contains CPU call-frame metadata";
        return false;
    }
    std::unordered_set<uint32_t> slots;
    for (size_t index = 0; index < plan.layouts.size(); ++index) {
        const auto &layout = plan.layouts[index];
        const auto &source = plan.sources[index];
        if (!slots.insert(layout.slot).second) {
            error = "prepared compute binding contains a duplicate provider slot";
            return false;
        }
        const uint32_t expectedSize = source.descriptorWidth == PreparedDescriptorWidth::I32   ? 4u
                                      : source.descriptorWidth == PreparedDescriptorWidth::I64 ? 8u
                                                                                               : layout.element_size;
        if (!layout.element_size || layout.element_size != expectedSize) {
            error = "prepared compute descriptor width disagrees with its provider layout";
            return false;
        }
        if (source.source.kind != ComputeBindingSourceKind::Argument &&
            source.descriptorWidth == PreparedDescriptorWidth::None) {
            error = "prepared compute descriptor binding has no scalar width";
            return false;
        }
    }
    return true;
}

bool buildPreparedGraphicsBindingPlan(const StageBindingPlan &stagePlan, const PreparedGraphicsBindingOptions &options,
                                      PreparedGraphicsBindingPlan &output, std::string &error) {
    output = {};
    output.backend = options.backend;
    error.clear();
    if (stagePlan.vertex.empty() || stagePlan.fragment.empty() || !stagePlan.compute.empty()) {
        error = "prepared graphics binding plan requires vertex and fragment stages";
        return false;
    }
    const bool openGL = isOpenGLBackend(options.backend);
    const bool directX12 = options.backend == VERNON_RUNTIME_DIRECTX12;
    const bool metal = options.backend == VERNON_RUNTIME_METAL;
    if (metal && !options.resolveNativeBinding) {
        error = "Metal graphics binding plan has no native resource resolver";
        return false;
    }

    uint32_t nextProviderSlot = 0;
    uint32_t maximumExternalSlot = 0;
    uint32_t vertexBinding = 0;
    for (const Parameter &parameter : stagePlan.parameters)
        maximumExternalSlot = std::max(maximumExternalSlot, parameter.slot);
    uint32_t nextInternalSlot = maximumExternalSlot;

    const auto append = [&](VernonRuntimeProviderBindingLayoutEntry layout, PreparedGraphicsBindingSource source,
                            std::vector<VernonRuntimeProviderVertexAttribute> attributes) {
        output.layouts.push_back(layout);
        output.sources.push_back(std::move(source));
        output.vertexAttributes.insert(output.vertexAttributes.end(), attributes.begin(), attributes.end());
    };

    const auto addUse = [&](const Parameter &parameter, const ParameterUse &use, bool internal) {
        const uint32_t stageMask = graphicsStageMask(use.stage);
        if (!stageMask) {
            error = "graphics parameter references an unsupported stage";
            return false;
        }
        if ((openGL || directX12) &&
            ((use.binding != UINT32_MAX && use.descriptorSet != 0) ||
             std::any_of(use.sampledImageBindings.begin(), use.sampledImageBindings.end(),
                         [](const SampledImageBinding &binding) { return binding.descriptorSet != 0; }))) {
            error = openGL ? "OpenGL supports reflected resources only in descriptor set 0"
                           : "D3D12 graphics resources must use descriptor set 0";
            return false;
        }

        VernonRuntimeProviderBindingLayoutEntry layout{};
        if (usesSequentialGraphicsSlots(options.backend)) {
            if (nextProviderSlot == UINT32_MAX) {
                error = "graphics binding layout exceeds the provider slot range";
                return false;
            }
            layout.slot = nextProviderSlot++;
        } else {
            if (internal) {
                if (nextInternalSlot == UINT32_MAX) {
                    error = "D3D12 graphics binding layout exceeds the provider slot range";
                    return false;
                }
                layout.slot = ++nextInternalSlot;
            } else {
                layout.slot = parameter.slot;
            }
        }
        layout.argument_index = use.index;
        layout.stage_mask = stageMask;
        layout.array_count = 1;
        PreparedGraphicsBindingSource source;
        source.externalSlot = parameter.slot;
        source.descriptorSet = use.descriptorSet;
        source.descriptorBinding = use.binding;

        if (parameter.kind == "sampler") {
            if (use.sampledImageBindings.empty() || (directX12 && use.sampledImageBindings.size() != 1)) {
                error = "graphics sampler has no reflected image binding";
                return false;
            }
            for (size_t index = 0; index < use.sampledImageBindings.size(); ++index) {
                const SampledImageBinding &sampledBinding = use.sampledImageBindings[index];
                if (sampledBinding.binding == UINT32_MAX) {
                    error = "graphics sampler reflection is incomplete";
                    return false;
                }
                auto samplerLayout = layout;
                auto samplerSource = source;
                if (index != 0) {
                    if (!usesSequentialGraphicsSlots(options.backend) || nextProviderSlot == UINT32_MAX) {
                        error = "graphics sampler expansion exceeds the provider slot range";
                        return false;
                    }
                    samplerLayout.slot = nextProviderSlot++;
                }
                samplerLayout.kind = VERNON_RUNTIME_PROVIDER_SAMPLER;
                samplerLayout.interface_kind = VERNON_RUNTIME_PROVIDER_INTERFACE_RESOURCE;
                samplerLayout.set = sampledBinding.descriptorSet;
                samplerLayout.binding = sampledBinding.binding;
                samplerSource.kind = internal ? PreparedGraphicsBindingSourceKind::ImplicitSampler
                                              : PreparedGraphicsBindingSourceKind::ExternalSampler;
                samplerSource.descriptorSet = sampledBinding.descriptorSet;
                samplerSource.descriptorBinding = sampledBinding.binding;
                if (metal && !options.resolveNativeBinding(options.nativeBindingUserData, use.stage, "sampler",
                                                           sampledBinding.descriptorSet, sampledBinding.binding,
                                                           nullptr, samplerLayout, error))
                    return false;
                append(samplerLayout, std::move(samplerSource), {});
            }
            return true;
        }

        if (internal && parameter.source == StageParameterSource::Resolution && parameter.kind == "tensor" &&
            use.interfaceKind == "uniform" && (openGL || directX12)) {
            source.kind = PreparedGraphicsBindingSourceKind::Resolution;
            if (openGL) {
                if (use.uniformName.empty()) {
                    error = "OpenGL resolution uniform has no native name";
                    return false;
                }
                layout.kind = VERNON_RUNTIME_PROVIDER_INLINE_VALUE;
                layout.element_size = 2 * sizeof(float);
                layout.interface_kind = VERNON_RUNTIME_PROVIDER_INTERFACE_UNIFORM;
                layout.element_count = 2;
                layout.vector_count = 1;
                layout.numeric_type = VERNON_RUNTIME_PROVIDER_F32;
                source.nativeName = use.uniformName;
            } else {
                const auto &shape = use.shape.empty() ? parameter.shape : use.shape;
                const auto dtype = pipelineDataType(use.dtype);
                if (!use.interfacePlan || !use.interfacePlan->root || shape != std::vector<uint64_t>{2} ||
                    dtype != VERNON_DATA_F32 || use.interfacePlan->root->size != sizeof(float) * 2 ||
                    !use.interfacePlan->root->alignment || use.interfacePlan->root->alignment > UINT32_MAX ||
                    (use.transport != "push_constant" && use.transport != "uniform_buffer") ||
                    (use.transport == "uniform_buffer" && use.binding == UINT32_MAX)) {
                    error = "D3D12 resolution uniform layout is unsupported";
                    return false;
                }
                layout.kind = use.transport == "uniform_buffer" ? VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER
                                                                : VERNON_RUNTIME_PROVIDER_INLINE_VALUE;
                layout.element_size = static_cast<uint32_t>(use.interfacePlan->root->size);
                layout.element_alignment = static_cast<uint32_t>(use.interfacePlan->root->alignment);
                layout.interface_kind = VERNON_RUNTIME_PROVIDER_INTERFACE_UNIFORM;
                layout.element_count = 2;
                layout.vector_count = 1;
                layout.set = use.descriptorSet;
                layout.binding = use.binding;
            }
            source.storage.resize(layout.element_size);
            append(layout, std::move(source), {});
            return true;
        }

        if (parameter.kind == "tensor" && use.interfaceKind == "uniform") {
            if (!use.interfacePlan || !use.interfacePlan->root || !use.valueLayout) {
                error = "graphics uniform has no complete interface plan";
                return false;
            }
            const auto providerKind = graphicsProviderKind(use.transport);
            const auto dtype = pipelineDataType(use.dtype);
            const auto &shape = use.shape.empty() ? parameter.shape : use.shape;
            const uint64_t physicalSize = use.interfacePlan->root->size;
            const uint64_t physicalAlignment = use.interfacePlan->root->alignment;
            if (!providerKind || !dtype || !physicalSize || physicalSize > UINT32_MAX || !physicalAlignment ||
                physicalAlignment > UINT32_MAX || (isBufferedUniform(*providerKind) && use.binding == UINT32_MAX)) {
                error = "graphics uniform reflection is incomplete";
                return false;
            }
            layout.kind = *providerKind;
            layout.element_size = static_cast<uint32_t>(physicalSize);
            layout.element_alignment = static_cast<uint32_t>(physicalAlignment);
            layout.interface_kind = VERNON_RUNTIME_PROVIDER_INTERFACE_UNIFORM;
            layout.set = use.descriptorSet;
            layout.binding = use.binding;
            source.kind = internal ? PreparedGraphicsBindingSourceKind::Resolution
                                   : PreparedGraphicsBindingSourceKind::ExternalUniform;
            source.nativeName = use.uniformName;
            if (openGL) {
                const bool buffered = isBufferedUniform(layout.kind);
                OpenGLNativeUniformShape nativeShape;
                if ((!buffered &&
                     (use.uniformName.empty() || !resolveOpenGLNativeUniformShape(use.dtype, shape, nativeShape))) ||
                    (buffered && use.binding == UINT32_MAX)) {
                    error = "OpenGL uniform layout is unsupported";
                    return false;
                }
                layout.element_count = buffered ? 0 : nativeShape.scalarCount;
                layout.vector_count = buffered ? 0 : nativeShape.matrixColumns;
                layout.numeric_type = static_cast<uint32_t>(*dtype);
            }
            const bool wholeValue = use.tensorPacking == TensorRepresentation::WholeValue;
            const ValueLayout &canonical = *use.valueLayout;
            auto packing =
                wholeValue
                    ? compileWholeValueCopyPlan(pipelineValueLayout(canonical), *use.interfacePlan->root)
                    : compileElementStreamCopyPlan(pipelineValueLayout(canonical), shape, *use.interfacePlan->root);
            if (packing.isErr() || packing.value().elementSize != canonical.byteSize) {
                error = "graphics interface plan does not match the canonical layout";
                return false;
            }
            source.packing = std::move(packing).value();
            source.storage.resize(layout.element_size);
            if (metal) {
                const char *kind = layout.kind == VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER   ? "uniform_buffer"
                                   : layout.kind == VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER ? "storage_buffer"
                                                                                           : "inline_constant";
                const uint32_t set =
                    layout.kind == VERNON_RUNTIME_PROVIDER_INLINE_VALUE ? UINT32_MAX : use.descriptorSet;
                const uint32_t binding = layout.kind == VERNON_RUNTIME_PROVIDER_INLINE_VALUE ? UINT32_MAX : use.binding;
                const std::string inlineName = use.uniformName.empty() ? parameter.name : use.uniformName;
                const std::string *name = layout.kind == VERNON_RUNTIME_PROVIDER_INLINE_VALUE ? &inlineName : nullptr;
                if (!options.resolveNativeBinding(options.nativeBindingUserData, use.stage, kind, set, binding, name,
                                                  layout, error))
                    return false;
            }
            append(layout, std::move(source), {});
            return true;
        }

        if (!internal && parameter.kind == "tensor" && use.interfaceKind == "input" && use.stage == "vertex" &&
            use.location != UINT32_MAX && !use.attributeLeaves.empty()) {
            layout.kind = VERNON_RUNTIME_PROVIDER_VERTEX_BUFFER;
            layout.element_size = parameter.elementLayout.byteSize;
            layout.interface_kind = VERNON_RUNTIME_PROVIDER_INTERFACE_VERTEX_INPUT;
            if (!layout.element_size) {
                error = "graphics vertex element size is zero";
                return false;
            }
            if (openGL)
                layout.binding = parameter.slot;
            else if (metal) {
                if (vertexBinding >= 15) {
                    error = "Metal graphics vertex binding limit is exceeded";
                    return false;
                }
                layout.binding = 16 + vertexBinding++;
            } else
                layout.binding = vertexBinding++;
            layout.divisor = use.divisor;
            std::vector<VernonRuntimeProviderVertexAttribute> attributes;
            for (const AttributeLeaf &leaf : use.attributeLeaves) {
                const auto dtype = pipelineDataType(leaf.dtype);
                if (!dtype) {
                    error = "graphics vertex attribute dtype is unsupported";
                    return false;
                }
                attributes.push_back({layout.binding, use.location + leaf.locationOffset, static_cast<uint32_t>(*dtype),
                                      leaf.componentCount, leaf.byteOffset});
            }
            source.kind = PreparedGraphicsBindingSourceKind::ExternalVertex;
            append(layout, std::move(source), std::move(attributes));
            return true;
        }

        if (!internal && parameter.kind == "image" && use.interfaceKind == "resource" && use.binding != UINT32_MAX) {
            layout.kind = parameter.bindingRole == "sampled" ? VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE
                                                             : VERNON_RUNTIME_PROVIDER_STORAGE_IMAGE;
            if (!configureImageBindingLayout(parameter, layout)) {
                error = "graphics image parameter layout is incomplete";
                return false;
            }
            layout.interface_kind = VERNON_RUNTIME_PROVIDER_INTERFACE_RESOURCE;
            layout.set = use.descriptorSet;
            layout.binding = use.binding;
            source.kind = PreparedGraphicsBindingSourceKind::ExternalImage;
            source.nativeName =
                openGL ? (use.uniformName.empty() ? "main_arg_" + std::to_string(use.index) : use.uniformName)
                       : std::string{};
            if (metal) {
                const char *kind =
                    layout.kind == VERNON_RUNTIME_PROVIDER_STORAGE_IMAGE ? "storage_image" : "sampled_image";
                if (!options.resolveNativeBinding(options.nativeBindingUserData, use.stage, kind, use.descriptorSet,
                                                  use.binding, nullptr, layout, error))
                    return false;
            }
            append(layout, std::move(source), {});
            return true;
        }

        error = "graphics parameter layout is unsupported";
        return false;
    };

    for (const Parameter &parameter : stagePlan.parameters) {
        if (directX12 && parameter.uses.size() != 1) {
            error = "D3D12 graphics parameters must have exactly one stage use";
            return false;
        }
        for (const ParameterUse &use : parameter.uses)
            if (!addUse(parameter, use, false))
                return false;
    }
    for (const Parameter &parameter : stagePlan.runtimeParameters) {
        if (directX12 && parameter.uses.size() != 1) {
            error = "D3D12 internal graphics parameters must have exactly one stage use";
            return false;
        }
        for (const ParameterUse &use : parameter.uses) {
            if (parameter.source == StageParameterSource::ImplicitSampler) {
                if (parameter.kind != "sampler" || !addUse(parameter, use, true))
                    return false;
            } else if (parameter.source == StageParameterSource::Resolution) {
                if (parameter.kind != "tensor" || !addUse(parameter, use, true))
                    return false;
            } else {
                error = "graphics internal parameter source is unsupported";
                return false;
            }
        }
    }

    std::vector<size_t> order(output.layouts.size());
    for (size_t index = 0; index < order.size(); ++index)
        order[index] = index;
    std::sort(order.begin(), order.end(),
              [&](size_t left, size_t right) { return output.layouts[left].slot < output.layouts[right].slot; });
    PreparedGraphicsBindingPlan sorted;
    sorted.backend = options.backend;
    sorted.vertexAttributes = std::move(output.vertexAttributes);
    sorted.layouts.reserve(order.size());
    sorted.sources.reserve(order.size());
    for (size_t index : order) {
        sorted.layouts.push_back(output.layouts[index]);
        sorted.sources.push_back(std::move(output.sources[index]));
    }
    output = std::move(sorted);
    for (size_t index = 0; index < output.size(); ++index) {
        const std::string &name = output.sources[index].nativeName;
        output.layouts[index].name = {name.data(), name.size()};
    }
    return validatePreparedGraphicsBindingPlan(output, error);
}

bool validatePreparedGraphicsBindingPlan(const PreparedGraphicsBindingPlan &plan, std::string &error) {
    if (plan.layouts.size() != plan.sources.size()) {
        error = "prepared graphics binding arrays have different lengths";
        return false;
    }
    std::unordered_set<uint32_t> slots;
    for (size_t index = 0; index < plan.size(); ++index) {
        const auto &layout = plan.layouts[index];
        const auto &source = plan.sources[index];
        if (!slots.insert(layout.slot).second) {
            error = "prepared graphics binding contains a duplicate provider slot";
            return false;
        }
        if (!layout.stage_mask || !layout.array_count) {
            error = "prepared graphics binding has an incomplete provider layout";
            return false;
        }
        if ((source.kind == PreparedGraphicsBindingSourceKind::ExternalUniform ||
             source.kind == PreparedGraphicsBindingSourceKind::Resolution) &&
            (!layout.element_size || source.storage.size() != layout.element_size)) {
            error = "prepared graphics inline storage disagrees with its provider layout";
            return false;
        }
        if ((source.kind == PreparedGraphicsBindingSourceKind::ImplicitSampler ||
             source.kind == PreparedGraphicsBindingSourceKind::ExternalSampler ||
             source.kind == PreparedGraphicsBindingSourceKind::ExternalImage) &&
            (source.descriptorSet == UINT32_MAX || source.descriptorBinding == UINT32_MAX)) {
            error = "prepared graphics resource has no canonical descriptor identity";
            return false;
        }
    }
    return true;
}

bool fillPreparedGraphicsBindingValues(PreparedGraphicsBindingPlan &prepared,
                                       const PlannedGraphicsInvocation &invocation,
                                       std::vector<VernonRuntimeProviderBindingValue> &values, std::string &error) {
    if (values.size() != prepared.size()) {
        error = "prepared graphics value array has the wrong length";
        return false;
    }
    for (size_t index = 0; index < prepared.size(); ++index) {
        const auto &layout = prepared.layouts[index];
        auto &source = prepared.sources[index];
        auto &value = values[index];
        value = {};
        value.slot = layout.slot;
        value.kind = layout.kind;
        if (source.kind == PreparedGraphicsBindingSourceKind::ExternalVertex) {
            const auto found = invocation.arguments.find(source.externalSlot);
            if (found == invocation.arguments.end() || found->second->kind != VERNON_PROGRAM_TENSOR ||
                found->second->tensor.storage != VERNON_TENSOR_RHI_RESOURCE ||
                !found->second->tensor.resource.resource.value || !found->second->tensor.byte_strides ||
                found->second->tensor.byte_strides[0] <= 0 ||
                static_cast<uint64_t>(found->second->tensor.byte_strides[0]) > UINT32_MAX) {
                error = "graphics vertex argument is missing or invalid";
                return false;
            }
            const VernonTensorView &tensor = found->second->tensor;
            value.payload.buffer.resource = tensor.resource;
            value.payload.buffer.resource.offset += tensor.byte_offset;
            value.payload.buffer.stride = static_cast<uint32_t>(tensor.byte_strides[0]);
        } else if (source.kind == PreparedGraphicsBindingSourceKind::ExternalImage) {
            const auto found = invocation.arguments.find(source.externalSlot);
            if (found == invocation.arguments.end() || found->second->kind != VERNON_PROGRAM_IMAGE ||
                !found->second->image.view.resource.value) {
                error = "graphics image argument is missing";
                return false;
            }
            value.payload.image.view = found->second->image.view;
        } else if (source.kind == PreparedGraphicsBindingSourceKind::ExternalSampler) {
            const auto found = invocation.arguments.find(source.externalSlot);
            if (found == invocation.arguments.end() || found->second->kind != VERNON_PROGRAM_SAMPLER ||
                !found->second->resource.resource.value) {
                error = "graphics sampler argument is missing";
                return false;
            }
            value.payload.sampler.resource = found->second->resource;
        } else if (source.kind == PreparedGraphicsBindingSourceKind::ImplicitSampler) {
            const auto sampled = invocation.sampledResources.find({source.descriptorSet, source.descriptorBinding});
            if (sampled == invocation.sampledResources.end()) {
                error = "graphics implicit sampler binding is missing";
                return false;
            }
            if (sampled->second.samplerResource.resource.value)
                value.payload.sampler.resource = sampled->second.samplerResource;
            else
                value.flags = VERNON_RUNTIME_PROVIDER_BINDING_DEFAULT_RESOURCE;
        } else if (source.kind == PreparedGraphicsBindingSourceKind::Resolution) {
            if (source.storage.size() < sizeof(invocation.resolution)) {
                error = "graphics resolution storage is too small";
                return false;
            }
            std::memcpy(source.storage.data(), invocation.resolution.data(), sizeof(invocation.resolution));
            value.payload.inline_value.data = source.storage.data();
            value.payload.inline_value.size = source.storage.size();
        } else if (source.kind == PreparedGraphicsBindingSourceKind::ExternalUniform) {
            const auto found = invocation.arguments.find(source.externalSlot);
            if (found == invocation.arguments.end() || found->second->kind != VERNON_PROGRAM_TENSOR) {
                error = "graphics uniform argument is missing";
                return false;
            }
            auto packed = packTensor(found->second->tensor, source.packing);
            if (packed.isErr() || packed.value().size() != source.storage.size()) {
                error = "graphics uniform Tensor does not satisfy its prepared packing plan";
                return false;
            }
            source.storage = std::move(packed).value();
            value.payload.inline_value.data = source.storage.data();
            value.payload.inline_value.size = source.storage.size();
        } else {
            error = "prepared graphics binding source is invalid";
            return false;
        }
    }
    return true;
}

} // namespace vernon::runtime
