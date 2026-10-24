#include "prepared_binding_plan.h"

#include "runtime/autodiff/tape_allocator_abi.h"
#include "runtime/interface_names.h"

#include <algorithm>
#include <cstring>
#include <optional>
#include <unordered_set>

namespace vernon::runtime {
namespace {

constexpr uint64_t kCudaRankOneMemRefParameterCount = 5;

bool isOpenGLBackend(VernonRuntimeBackend backend) {
    return backend == VERNON_RUNTIME_OPENGL || backend == VERNON_RUNTIME_OPENGL_ES;
}

uint32_t accessMask(const std::string &access) { return access == "read" ? 1u : access == "write" ? 2u : 3u; }

bool appendBinding(PreparedComputeBindingPlan &plan, VernonRuntimeProviderBindingLayoutEntry layout,
                   uint32_t argumentIndex, uint32_t parameter, std::string &error) {
    if (!layout.element_size) {
        error = "prepared compute binding has zero element size";
        return false;
    }
    plan.layouts.push_back(layout);
    plan.sources.push_back({argumentIndex, 0, false, parameter});
    return true;
}

uint32_t parameterIndex(const StageBindingPlan &stagePlan, const Parameter &parameter) {
    return static_cast<uint32_t>(&parameter - stagePlan.parameters.data());
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

bool setMetadataLayoutSize(const MetadataCarrier &carrier, VernonRuntimeProviderBindingLayoutEntry &layout,
                           std::string &error) {
    if (!carrier.size || carrier.size > UINT32_MAX || !carrier.alignment || carrier.alignment > UINT32_MAX) {
        error = "metadata carrier size or alignment exceeds the provider binding ABI";
        return false;
    }
    layout.element_size = static_cast<uint32_t>(carrier.size);
    layout.element_alignment = static_cast<uint32_t>(carrier.alignment);
    return true;
}

bool hostSize(uint64_t value, size_t &result) {
    if constexpr (sizeof(size_t) < sizeof(uint64_t))
        if (value > std::numeric_limits<size_t>::max())
            return false;
    result = static_cast<size_t>(value);
    return true;
}

} // namespace

bool resolveOpenGLNativeUniformShape(std::string_view dtype, const std::vector<uint64_t> &shape,
                                     OpenGLNativeUniformShape &result) {
    result = {};
    if (!isNativeUniformShape(shape.data(), shape.size(), dtype == "f32"))
        return false;
    if (shape.size() == 2) {
        result.scalarCount = static_cast<uint32_t>(shape[0] * shape[1]);
        result.matrixColumns = static_cast<uint32_t>(shape[1]);
        return true;
    }
    result.scalarCount = shape.empty() ? 1 : static_cast<uint32_t>(shape[0]);
    return dtype == "f32" || dtype == "i32" || dtype == "u32";
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
            layout.kind = argument.kind == "tensor" ? VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER
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
            output.sources.push_back({layout.argument_index, 0, false});
            output.cpuBindings.push_back(
                {tapeBuiltin ? argument.builtin : std::string(), argument.physical.offset, argument.physical.size,
                 argument.result,
                 argument.result && argument.autodiffRole == "gradient" ? argument.dtype : std::nullopt});
            if (tapeBuiltin && argument.builtin == VERNON_AD_TAPE_ALLOCATOR_BUILTIN)
                output.tapeAllocatorOffset = argument.physical.offset;
            if (tapeBuiltin && argument.builtin == VERNON_AD_TAPE_ROOT_REGION_BUILTIN)
                output.tapeRootOffset = argument.physical.offset;
        }
        if (reflection.metadataCarrier) {
            const MetadataCarrier &carrier = *reflection.metadataCarrier;
            size_t frameOffset = 0;
            size_t frameSize = 0;
            if (!hostSize(carrier.interfacePlan.frameOffset, frameOffset) || !hostSize(carrier.size, frameSize)) {
                error = "CPU metadata carrier exceeds the host address space";
                return false;
            }
            VernonRuntimeProviderBindingLayoutEntry layout{};
            layout.slot = static_cast<uint32_t>(output.layouts.size());
            layout.binding = layout.slot;
            layout.kind = VERNON_RUNTIME_PROVIDER_INLINE_VALUE;
            layout.stage_mask = VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE;
            layout.access = 1;
            layout.array_count = 1;
            layout.argument_index = UINT32_MAX;
            if (!setMetadataLayoutSize(carrier, layout, error))
                return false;
            output.metadataCarrier = PreparedMetadataCarrier{carrier, layout};
            output.layouts.push_back(layout);
            output.sources.push_back({UINT32_MAX, 0, true});
            output.cpuBindings.push_back({"", frameOffset, frameSize, false, std::nullopt});
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

    for (const Parameter &parameter : stagePlan.parameters) {
        if (parameter.kind == "sampler")
            continue;
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
                        use.interfaceKind == "storage" || (use.transport == "storage_buffer" && use.bufferValue)
                            ? VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER
                            : VERNON_RUNTIME_PROVIDER_INLINE_VALUE;
                else
                    layout.kind = parameter.kind == "image"
                                      ? (parameter.bindingRole == "sampled" ? VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE
                                                                            : VERNON_RUNTIME_PROVIDER_STORAGE_IMAGE)
                                  : argument.kind == "tensor" ? VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER
                                                              : VERNON_RUNTIME_PROVIDER_INLINE_VALUE;
                if (layout.kind == VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER && use.interfaceKind == "value")
                    layout.interface_kind = VERNON_RUNTIME_PROVIDER_INTERFACE_UNIFORM;
                layout.stage_mask = VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE;
                layout.access = accessMask(argument.access);
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
                if (!appendBinding(output, layout, use.index, parameterIndex(stagePlan, parameter), error))
                    return false;
            }
        }
    }

    // A sampler has no descriptor of its own: it binds at the native descriptor of each image it samples, where the
    // provider merges both into one combined image-sampler, so image layouts must exist first.
    for (const Parameter &parameter : stagePlan.parameters) {
        if (parameter.kind != "sampler")
            continue;
        bool firstLayout = true;
        for (const ParameterUse &use : parameter.uses) {
            if (use.stage != "compute" && use.stage != stagePlan.compute)
                continue;
            for (const SampledImageBinding &sampled : use.sampledImageBindings) {
                const auto image = std::find_if(output.layouts.begin(), output.layouts.end(), [&](const auto &layout) {
                    if (layout.kind != VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE ||
                        layout.argument_index >= reflection.arguments.size())
                        return false;
                    const ReflectedArgument &argument = reflection.arguments[layout.argument_index];
                    return argument.descriptorSet == sampled.descriptorSet && argument.binding == sampled.binding;
                });
                if (image == output.layouts.end()) {
                    error = "prepared compute sampler pairs with no prepared sampled image";
                    return false;
                }
                VernonRuntimeProviderBindingLayoutEntry layout{};
                layout.slot = firstLayout ? parameter.slot : ++internalSlot;
                layout.set = image->set;
                layout.binding = image->binding;
                layout.kind = VERNON_RUNTIME_PROVIDER_SAMPLER;
                layout.interface_kind = VERNON_RUNTIME_PROVIDER_INTERFACE_RESOURCE;
                layout.stage_mask = VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE;
                layout.access = 1;
                layout.array_count = 1;
                layout.argument_index = use.index;
                layout.element_size = 1;
                if (!appendBinding(output, layout, use.index, parameterIndex(stagePlan, parameter), error))
                    return false;
                firstLayout = false;
            }
        }
    }

    if (const MetadataCarrier *computeCarrier = stageMetadataCarrier(stagePlan, "compute")) {
        const MetadataCarrier &carrier = *computeCarrier;
        if (carrier.size > UINT32_MAX || carrier.alignment > UINT32_MAX || output.layouts.size() > UINT32_MAX) {
            error = "prepared metadata carrier exceeds provider layout limits";
            return false;
        }
        VernonRuntimeProviderBindingLayoutEntry layout{};
        const bool parameterCarrier = backend == VERNON_RUNTIME_CUDA;
        if (parameterCarrier) {
            uint64_t nativeOrdinal = 0;
            for (const VernonRuntimeProviderBindingLayoutEntry &existing : output.layouts)
                nativeOrdinal +=
                    existing.kind == VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER ? kCudaRankOneMemRefParameterCount : 1;
            if (nativeOrdinal != carrier.parameterOrdinal) {
                error = "CUDA metadata carrier parameter ordinal " + std::to_string(carrier.parameterOrdinal) +
                        " does not match the native kernel ABI ordinal " + std::to_string(nativeOrdinal);
                return false;
            }
        }
        layout.slot = ++internalSlot;
        layout.set = carrier.descriptorSet == UINT32_MAX ? 0 : carrier.descriptorSet;
        layout.binding = parameterCarrier ? static_cast<uint32_t>(output.layouts.size()) : carrier.binding;
        layout.kind = parameterCarrier ? VERNON_RUNTIME_PROVIDER_INLINE_VALUE : VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER;
        layout.interface_kind = VERNON_RUNTIME_PROVIDER_INTERFACE_UNIFORM;
        layout.stage_mask = VERNON_RUNTIME_PROVIDER_STAGE_COMPUTE;
        layout.access = 1;
        layout.array_count = 1;
        layout.argument_index = UINT32_MAX;
        if (!setMetadataLayoutSize(carrier, layout, error))
            return false;
        output.metadataCarrier = PreparedMetadataCarrier{carrier, layout};
        output.layouts.push_back(layout);
        output.sources.push_back({UINT32_MAX, 0, true});
    }

    std::vector<size_t> order(output.layouts.size());
    for (size_t index = 0; index < order.size(); ++index)
        order[index] = index;
    std::sort(order.begin(), order.end(),
              [&](size_t left, size_t right) { return output.layouts[left].slot < output.layouts[right].slot; });
    std::vector<VernonRuntimeProviderBindingLayoutEntry> layouts = std::move(output.layouts);
    std::vector<PreparedBindingSource> sources = std::move(output.sources);
    output.layouts.reserve(order.size());
    output.sources.reserve(order.size());
    for (size_t index : order) {
        output.layouts.push_back(layouts[index]);
        output.sources.push_back(sources[index]);
    }
    return validatePreparedComputeBindingPlan(output, error);
}

bool bindComputeMetadataCarrier(const VernonRuntimeProviderBindingLayoutEntry &layout,
                                const PreparedBindingSource &source, const PlannedComputeLaunch &launch,
                                VernonRuntimeProviderBindingValue &value) {
    if (!source.metadataCarrier || launch.metadataPayload.empty() ||
        launch.metadataPayload.size() != layout.element_size || launch.metadataAlignment != layout.element_alignment)
        return false;
    value.payload.inline_value.data = launch.metadataPayload.data();
    value.payload.inline_value.size = launch.metadataPayload.size();
    return true;
}

bool fillPreparedComputeBindingValue(const PreparedComputeBindingPlan &prepared, const StageBindingPlan &projection,
                                     const PlannedComputeLaunch &launch, size_t slot,
                                     VernonRuntimeProviderBindingValue &value, std::string &error) {
    const VernonRuntimeProviderBindingLayoutEntry &layout = prepared.layouts[slot];
    const PreparedBindingSource &source = prepared.sources[slot];
    value = {};
    value.slot = layout.slot;
    value.kind = layout.kind;
    if (source.metadataCarrier) {
        if (!bindComputeMetadataCarrier(layout, source, launch, value)) {
            error = "metadata carrier payload does not match its compiled ABI";
            return false;
        }
        return true;
    }
    if (source.argumentIndex >= launch.arguments.size()) {
        error = "prepared argument index is invalid";
        return false;
    }
    const ComputeLaunchArgument &argument = launch.arguments[source.argumentIndex];
    if (layout.kind == VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER &&
        layout.interface_kind != VERNON_RUNTIME_PROVIDER_INTERFACE_UNIFORM) {
        const auto *tensor = std::get_if<ComputeTensorArgument>(&argument);
        if (!tensor || !tensor->resource.resource.value) {
            error = "prepared storage binding requires an RHI Tensor";
            return false;
        }
        value.payload.buffer.resource = tensor->resource;
        VernonRuntimeProviderResourceReference &resource = value.payload.buffer.resource;
        if (source.resourceOffset >= resource.size) {
            error = "aggregate storage leaf exceeds its Tensor resource";
            return false;
        }
        resource.offset += source.resourceOffset;
        resource.size -= source.resourceOffset;
    } else if (layout.kind == VERNON_RUNTIME_PROVIDER_STORAGE_IMAGE ||
               layout.kind == VERNON_RUNTIME_PROVIDER_SAMPLED_IMAGE) {
        const auto *image = std::get_if<ComputeImageArgument>(&argument);
        if (!image || !image->view.resource.value) {
            error = "prepared image binding requires an RHI Texture";
            return false;
        }
        value.payload.image.view = image->view;
    } else if (layout.kind == VERNON_RUNTIME_PROVIDER_SAMPLER) {
        const auto *sampler = std::get_if<ComputeSamplerArgument>(&argument);
        if (!sampler || !sampler->resource.resource.value) {
            error = "prepared sampler binding requires an RHI Sampler";
            return false;
        }
        value.payload.sampler.resource = sampler->resource;
    } else {
        // A Value-storage buffer carries its packed bytes inline, exactly like an inline value.
        const bool valueStorage = layout.kind == VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER;
        const auto *scalar = std::get_if<ComputeScalarArgument>(&argument);
        if (!scalar || !scalar->data || !scalar->size || (valueStorage && scalar->size != layout.element_size)) {
            error = "prepared inline binding requires host data at argument " + std::to_string(layout.argument_index) +
                    " (kind " + std::to_string(argument.index()) + ", size " +
                    std::to_string(scalar ? scalar->size : 0) + ")";
            for (const Parameter &parameter : projection.parameters)
                for (const ParameterUse &use : parameter.uses)
                    if (use.index == layout.argument_index)
                        error += " for parameter '" + parameter.name + "' (interface " + use.interfaceKind +
                                 ", transport " + use.transport + ")";
            return false;
        }
        value.payload.inline_value.data = scalar->data;
        value.payload.inline_value.size = scalar->size;
    }
    return true;
}

std::vector<PipelineGroups::SlotDependencies>
preparedComputeSlotDependencies(const PreparedComputeBindingPlan &prepared, const StageBindingPlan &stagePlan) {
    std::vector<PipelineGroups::SlotDependencies> result(prepared.size());
    for (size_t index = 0; index < prepared.size(); ++index) {
        const PreparedBindingSource &source = prepared.sources[index];
        PipelineGroups::SlotDependencies &dependencies = result[index];
        if (source.parameter != UINT32_MAX) {
            dependencies.parameters.push_back(source.parameter);
        } else if (source.metadataCarrier && prepared.metadataCarrier) {
            for (const MetadataFieldIdentity &field : prepared.metadataCarrier->plan.fields) {
                const size_t known = dependencies.parameters.size();
                for (size_t parameter = 0; parameter < stagePlan.parameters.size(); ++parameter)
                    for (const ParameterUse &use : stagePlan.parameters[parameter].uses)
                        if ((use.stage == "compute" || use.stage == stagePlan.compute) && use.index == field.argument)
                            dependencies.parameters.push_back(static_cast<uint32_t>(parameter));
                dependencies.command = dependencies.command || dependencies.parameters.size() == known;
            }
        } else {
            dependencies.command = true;
        }
    }
    return result;
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
    size_t metadataCarrierCount = 0;
    for (size_t index = 0; index < plan.layouts.size(); ++index) {
        const auto &layout = plan.layouts[index];
        const auto &source = plan.sources[index];
        if (!slots.insert(layout.slot).second) {
            error = "prepared compute binding contains a duplicate provider slot";
            return false;
        }
        if (!layout.element_size) {
            error = "prepared compute binding has zero element size";
            return false;
        }
        const bool carrier = source.metadataCarrier;
        metadataCarrierCount += carrier ? 1u : 0u;
        if (carrier != (plan.metadataCarrier && layout.slot == plan.metadataCarrier->layout.slot)) {
            error = "prepared compute metadata carrier is not represented exactly once";
            return false;
        }
    }
    if (metadataCarrierCount != (plan.metadataCarrier ? 1u : 0u)) {
        error = "prepared compute plan does not contain exactly one declared metadata carrier";
        return false;
    }
    if (plan.metadataCarrier) {
        const MetadataCarrier &carrier = plan.metadataCarrier->plan;
        const auto &layout = plan.metadataCarrier->layout;
        const bool shader = carrier.profile == "portable_shader_metadata_i32";
        const bool cuda = carrier.profile == "cuda_kernel_metadata_i64";
        const bool host = carrier.profile == "host_metadata";
        if ((!shader && !cuda && !host) || (shader && plan.backend == VERNON_RUNTIME_CUDA) ||
            (cuda != (plan.backend == VERNON_RUNTIME_CUDA)) || (host != (plan.backend == VERNON_RUNTIME_CPU)) ||
            layout.element_size != carrier.size || layout.element_alignment != carrier.alignment ||
            carrier.fields.size() != carrier.members.size()) {
            error = "prepared metadata carrier profile disagrees with its target or layout";
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

    // D3D12 places external parameters at their parameter slot, so every additional provider slot must come after
    // the highest parameter slot; sequential backends number every layout entry in order.
    const auto allocateSlot = [&](uint32_t &slot) {
        uint32_t &next = usesSequentialGraphicsSlots(options.backend) ? nextProviderSlot : nextInternalSlot;
        if (next == UINT32_MAX) {
            error = "graphics binding layout exceeds the provider slot range";
            return false;
        }
        slot = usesSequentialGraphicsSlots(options.backend) ? next++ : ++next;
        return true;
    };
    const auto append = [&](VernonRuntimeProviderBindingLayoutEntry layout, PreparedGraphicsBindingSource source,
                            std::vector<VernonRuntimeProviderVertexAttribute> attributes) {
        // D3D12 builds one root signature per pipeline, so a descriptor that both stages use is one range visible
        // to both. The link gives equal (kind, set, binding) only to the same Program Value. Inline root constants
        // are per-stage root parameters and stay separate.
        if (directX12 && layout.kind != VERNON_RUNTIME_PROVIDER_INLINE_VALUE &&
            layout.kind != VERNON_RUNTIME_PROVIDER_VERTEX_BUFFER) {
            const auto shared = std::find_if(output.layouts.begin(), output.layouts.end(), [&](const auto &existing) {
                return existing.kind == layout.kind && existing.set == layout.set && existing.binding == layout.binding;
            });
            if (shared != output.layouts.end()) {
                shared->stage_mask |= layout.stage_mask;
                return true;
            }
        }
        // OpenGL ignores the set, so a binding point names one descriptor across every set of the pipeline.
        if (openGL && layout.kind != VERNON_RUNTIME_PROVIDER_INLINE_VALUE &&
            layout.kind != VERNON_RUNTIME_PROVIDER_VERTEX_BUFFER &&
            std::any_of(output.layouts.begin(), output.layouts.end(), [&](const auto &existing) {
                return existing.kind == layout.kind && existing.binding == layout.binding && existing.set != layout.set;
            })) {
            error = "OpenGL binding " + std::to_string(layout.binding) + " is used by several descriptor sets";
            return false;
        }
        if (std::any_of(output.layouts.begin(), output.layouts.end(),
                        [&](const auto &existing) { return existing.slot == layout.slot; }) &&
            !allocateSlot(layout.slot))
            return false;
        output.layouts.push_back(layout);
        output.sources.push_back(std::move(source));
        output.vertexAttributes.insert(output.vertexAttributes.end(), attributes.begin(), attributes.end());
        return true;
    };

    const auto addUse = [&](const Parameter &parameter, const ParameterUse &use, bool internal) {
        // An explicit sampler binds through the images it samples; a stage that never samples with it has no slot.
        if (!internal && parameter.kind == "sampler" && use.sampledImageBindings.empty())
            return true;
        const uint32_t stageMask = graphicsStageMask(use.stage);
        if (!stageMask) {
            error = "graphics parameter references an unsupported stage";
            return false;
        }
        VernonRuntimeProviderBindingLayoutEntry layout{};
        if (usesSequentialGraphicsSlots(options.backend) || internal) {
            if (!allocateSlot(layout.slot))
                return false;
        } else {
            layout.slot = parameter.slot;
        }
        layout.argument_index = use.index;
        layout.stage_mask = stageMask;
        layout.array_count = 1;
        PreparedGraphicsBindingSource source;
        source.parameter = internal ? UINT32_MAX : parameterIndex(stagePlan, parameter);
        source.descriptorSet = use.descriptorSet;
        source.descriptorBinding = use.binding;

        if (parameter.kind == "sampler") {
            if (use.sampledImageBindings.empty()) {
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
                if (index != 0 && !allocateSlot(samplerLayout.slot))
                    return false;
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
                if (!append(samplerLayout, std::move(samplerSource), {}))
                    return false;
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
            return append(layout, std::move(source), {});
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
            return append(layout, std::move(source), {});
        }

        if (!internal && parameter.kind == "tensor" && use.interfaceKind == "input" && use.stage == "vertex" &&
            use.location != UINT32_MAX && !use.attributeLeaves.empty()) {
            layout.kind = VERNON_RUNTIME_PROVIDER_VERTEX_BUFFER;
            layout.element_size = parameter.elementLayout.byteSize;
            layout.interface_kind = VERNON_RUNTIME_PROVIDER_INTERFACE_VERTEX_INPUT;
            layout.set = use.descriptorSet;
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
            return append(layout, std::move(source), std::move(attributes));
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
            return append(layout, std::move(source), {});
        }

        if (!internal && parameter.kind == "tensor" && use.interfaceKind == "resource" && !use.storageLeaves.empty()) {
            for (size_t leafIndex = 0; leafIndex < use.storageLeaves.size(); ++leafIndex) {
                const ReflectedStorageLeaf &leaf = use.storageLeaves[leafIndex];
                if (leaf.binding == UINT32_MAX || !leaf.elementSize || leaf.elementSize > UINT32_MAX) {
                    error = "graphics TensorView storage leaf reflection is incomplete";
                    return false;
                }
                auto leafLayout = layout;
                if (leafIndex != 0 && !allocateSlot(leafLayout.slot))
                    return false;
                leafLayout.kind = VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER;
                leafLayout.interface_kind = VERNON_RUNTIME_PROVIDER_INTERFACE_RESOURCE;
                leafLayout.set = use.descriptorSet;
                leafLayout.binding = leaf.binding;
                leafLayout.element_size = static_cast<uint32_t>(leaf.elementSize);
                leafLayout.access = 1;
                auto leafSource = source;
                leafSource.kind = PreparedGraphicsBindingSourceKind::ExternalStorage;
                leafSource.descriptorBinding = leaf.binding;
                if (!append(leafLayout, std::move(leafSource), {}))
                    return false;
            }
            return true;
        }

        error = "graphics parameter layout is unsupported";
        return false;
    };

    for (const Parameter &parameter : stagePlan.parameters)
        for (const ParameterUse &use : parameter.uses)
            if (!addUse(parameter, use, false))
                return false;
    for (const Parameter &parameter : stagePlan.runtimeParameters) {
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
    for (const auto &[stage, carrier] : stagePlan.metadataCarriers) {
        const uint32_t stageMask = graphicsStageMask(stage);
        if (!stageMask || output.metadataCarriers.size() >= UINT32_MAX) {
            error = "graphics module metadata carrier cannot be assigned a provider slot";
            return false;
        }
        PreparedGraphicsMetadataCarrier prepared;
        prepared.plan = carrier;
        for (const MetadataFieldIdentity &field : carrier.fields) {
            uint32_t found = UINT32_MAX;
            for (const Parameter &parameter : stagePlan.parameters)
                for (const ParameterUse &use : parameter.uses)
                    if (parameter.kind == "tensor" && use.stage == stage && use.index == field.argument)
                        found = parameterIndex(stagePlan, parameter);
            if (found == UINT32_MAX) {
                error = "graphics module metadata field does not name a Tensor use of its module";
                return false;
            }
            if (prepared.argumentParameters.size() <= field.argument)
                prepared.argumentParameters.resize(static_cast<size_t>(field.argument) + 1, UINT32_MAX);
            prepared.argumentParameters[field.argument] = found;
        }
        VernonRuntimeProviderBindingLayoutEntry layout{};
        if (!allocateSlot(layout.slot))
            return false;
        layout.kind = VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER;
        layout.interface_kind = VERNON_RUNTIME_PROVIDER_INTERFACE_UNIFORM;
        layout.stage_mask = stageMask;
        layout.set = carrier.descriptorSet;
        layout.binding = carrier.binding;
        layout.access = 1;
        layout.array_count = 1;
        layout.argument_index = UINT32_MAX;
        if (!setMetadataLayoutSize(carrier, layout, error))
            return false;
        PreparedGraphicsBindingSource source;
        source.kind = PreparedGraphicsBindingSourceKind::ModuleMetadata;
        source.metadataCarrier = static_cast<uint32_t>(output.metadataCarriers.size());
        source.descriptorSet = carrier.descriptorSet;
        source.descriptorBinding = carrier.binding;
        output.metadataCarriers.push_back(std::move(prepared));
        if (!append(layout, std::move(source), {}))
            return false;
    }

    std::vector<size_t> order(output.layouts.size());
    for (size_t index = 0; index < order.size(); ++index)
        order[index] = index;
    std::sort(order.begin(), order.end(),
              [&](size_t left, size_t right) { return output.layouts[left].slot < output.layouts[right].slot; });
    PreparedGraphicsBindingPlan sorted;
    sorted.backend = options.backend;
    sorted.vertexAttributes = std::move(output.vertexAttributes);
    sorted.metadataCarriers = std::move(output.metadataCarriers);
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
            !layout.element_size) {
            error = "prepared graphics inline storage disagrees with its provider layout";
            return false;
        }
        const bool external = source.kind != PreparedGraphicsBindingSourceKind::ImplicitSampler &&
                              source.kind != PreparedGraphicsBindingSourceKind::Resolution &&
                              source.kind != PreparedGraphicsBindingSourceKind::ModuleMetadata;
        if (external != (source.parameter != UINT32_MAX)) {
            error = "prepared graphics binding source has no Stage parameter";
            return false;
        }
        if ((source.kind == PreparedGraphicsBindingSourceKind::ImplicitSampler ||
             source.kind == PreparedGraphicsBindingSourceKind::ExternalSampler ||
             source.kind == PreparedGraphicsBindingSourceKind::ExternalImage) &&
            (source.descriptorSet == UINT32_MAX || source.descriptorBinding == UINT32_MAX)) {
            error = "prepared graphics resource has no canonical descriptor identity";
            return false;
        }
        if (source.kind == PreparedGraphicsBindingSourceKind::ExternalStorage &&
            (layout.kind != VERNON_RUNTIME_PROVIDER_STORAGE_BUFFER || !layout.element_size ||
             source.descriptorBinding != layout.binding)) {
            error = "prepared graphics storage leaf disagrees with its provider layout";
            return false;
        }
        if (source.kind == PreparedGraphicsBindingSourceKind::ModuleMetadata &&
            (layout.kind != VERNON_RUNTIME_PROVIDER_UNIFORM_BUFFER ||
             source.metadataCarrier >= plan.metadataCarriers.size() || !layout.element_size ||
             plan.metadataCarriers[source.metadataCarrier].plan.size != layout.element_size)) {
            error = "prepared graphics metadata carrier disagrees with its provider layout";
            return false;
        }
    }
    return true;
}

bool fillPreparedGraphicsBindingValue(const PreparedGraphicsBindingPlan &prepared, const GraphicsSlotInputs &inputs,
                                      size_t slot, VernonRuntimeProviderBindingValue &value,
                                      std::vector<uint8_t> &storage, std::string &error) {
    const auto &layout = prepared.layouts[slot];
    const auto &source = prepared.sources[slot];
    value = {};
    value.slot = layout.slot;
    value.kind = layout.kind;
    const VernonProgramArgument *argument =
        source.parameter == UINT32_MAX ? nullptr : inputs.arguments[source.parameter];
    const auto setInline = [&]() {
        value.payload.inline_value.data = storage.data();
        value.payload.inline_value.size = storage.size();
    };
    switch (source.kind) {
    case PreparedGraphicsBindingSourceKind::ExternalVertex: {
        if (!argument || argument->kind != VERNON_PROGRAM_TENSOR ||
            argument->tensor.storage != VERNON_TENSOR_RHI_RESOURCE || !argument->tensor.resource.resource.value ||
            !argument->tensor.byte_strides || argument->tensor.byte_strides[0] <= 0 ||
            static_cast<uint64_t>(argument->tensor.byte_strides[0]) > UINT32_MAX ||
            argument->tensor.byte_offset >= argument->tensor.resource.size) {
            error = "graphics vertex argument is missing or invalid";
            return false;
        }
        const VernonTensorView &tensor = argument->tensor;
        value.payload.buffer.resource = tensor.resource;
        value.payload.buffer.resource.offset += tensor.byte_offset;
        value.payload.buffer.resource.size -= tensor.byte_offset;
        value.payload.buffer.stride = static_cast<uint32_t>(tensor.byte_strides[0]);
        return true;
    }
    case PreparedGraphicsBindingSourceKind::ExternalImage:
        if (!argument || argument->kind != VERNON_PROGRAM_IMAGE || !argument->image.view.resource.value) {
            error = "graphics image argument is missing";
            return false;
        }
        value.payload.image.view = argument->image.view;
        return true;
    case PreparedGraphicsBindingSourceKind::ExternalSampler:
        if (!argument || argument->kind != VERNON_PROGRAM_SAMPLER || !argument->resource.resource.value) {
            error = "graphics sampler argument is missing";
            return false;
        }
        value.payload.sampler.resource = argument->resource;
        return true;
    // An implicit sampler never shares its image with an explicit one, so it always binds the default sampler.
    case PreparedGraphicsBindingSourceKind::ImplicitSampler:
        value.flags = VERNON_RUNTIME_PROVIDER_BINDING_DEFAULT_RESOURCE;
        return true;
    case PreparedGraphicsBindingSourceKind::Resolution:
        storage.assign(layout.element_size, 0);
        if (storage.size() < sizeof(inputs.resolution)) {
            error = "graphics resolution storage is too small";
            return false;
        }
        std::memcpy(storage.data(), inputs.resolution.data(), sizeof(inputs.resolution));
        setInline();
        return true;
    case PreparedGraphicsBindingSourceKind::ExternalUniform: {
        if (!argument || argument->kind != VERNON_PROGRAM_TENSOR) {
            error = "graphics uniform argument is missing";
            return false;
        }
        auto packed = packTensor(argument->tensor, source.packing);
        if (packed.isErr() || packed.value().size() != layout.element_size) {
            error = "graphics uniform Tensor does not satisfy its prepared packing plan";
            return false;
        }
        storage = std::move(packed).value();
        setInline();
        return true;
    }
    case PreparedGraphicsBindingSourceKind::ExternalStorage:
        if (!argument || argument->kind != VERNON_PROGRAM_TENSOR ||
            argument->tensor.storage != VERNON_TENSOR_RHI_RESOURCE || !argument->tensor.resource.resource.value ||
            argument->tensor.byte_size > argument->tensor.resource.size) {
            error = "graphics TensorView storage argument must be an RHI Tensor within its resource";
            return false;
        }
        // The view offset travels in the metadata carrier, so every leaf
        // binds the base of the record buffer.
        value.payload.buffer.resource = argument->tensor.resource;
        return true;
    case PreparedGraphicsBindingSourceKind::ModuleMetadata: {
        const PreparedGraphicsMetadataCarrier &carrier = prepared.metadataCarriers[source.metadataCarrier];
        std::vector<const VernonTensorView *> views(carrier.argumentParameters.size());
        for (size_t endpoint = 0; endpoint < carrier.argumentParameters.size(); ++endpoint) {
            if (carrier.argumentParameters[endpoint] == UINT32_MAX)
                continue;
            const VernonProgramArgument *viewed = inputs.arguments[carrier.argumentParameters[endpoint]];
            if (!viewed || viewed->kind != VERNON_PROGRAM_TENSOR) {
                error = "graphics metadata carrier source is not a TensorView argument";
                return false;
            }
            views[endpoint] = &viewed->tensor;
        }
        storage.resize(layout.element_size);
        if (!materializeTensorViewMetadata(carrier.plan, views, storage, error))
            return false;
        setInline();
        return true;
    }
    }
    error = "prepared graphics binding source is invalid";
    return false;
}

std::vector<PipelineGroups::SlotDependencies>
preparedGraphicsSlotDependencies(const PreparedGraphicsBindingPlan &prepared) {
    std::vector<PipelineGroups::SlotDependencies> result(prepared.size());
    for (size_t index = 0; index < prepared.size(); ++index) {
        const PreparedGraphicsBindingSource &source = prepared.sources[index];
        PipelineGroups::SlotDependencies &dependencies = result[index];
        if (source.parameter != UINT32_MAX)
            dependencies.parameters.push_back(source.parameter);
        else if (source.kind == PreparedGraphicsBindingSourceKind::ModuleMetadata)
            for (uint32_t parameter : prepared.metadataCarriers[source.metadataCarrier].argumentParameters)
                if (parameter != UINT32_MAX)
                    dependencies.parameters.push_back(parameter);
        // The resolution follows the command's viewport.
        dependencies.command = source.kind == PreparedGraphicsBindingSourceKind::Resolution;
    }
    return result;
}

} // namespace vernon::runtime
