#ifndef VERNON_TESTS_SUPPORT_RUNTIME_RHI_TEST_UTILS_H
#define VERNON_TESTS_SUPPORT_RUNTIME_RHI_TEST_UTILS_H

#include "VernonExecutionGraph.h"
#include "VernonRuntime.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace vernon::tests {

struct DerivativeLeafFixture {
    uint32_t struct_size{};
    VernonStringView path{};
    VernonDataType dtype{};
    void *data{};
    size_t size{};
    uint32_t rank{};
    const uint64_t *shape{};
};

struct DerivativeLeafSetFixture {
    uint32_t struct_size{};
    DerivativeLeafFixture *values{};
    size_t value_count{};
    uint32_t reserved[4]{};
};

struct RhiRuntime {
    VernonRhiDevice device{static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
    VernonRuntimeContext *runtime{};
};

struct RhiBuffer {
    VernonRhiBuffer handle{static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
    VernonRuntimeProviderResourceReference reference{};
};

struct RhiImage {
    VernonRhiImage handle{static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
};

struct RhiImageView {
    VernonRhiImageView handle{static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
    VernonRuntimeProviderResourceReference reference{};
};

struct RhiSampler {
    VernonRhiSampler handle{static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
    VernonRuntimeProviderResourceReference reference{};
};

struct GraphicsInvocationControls {
    VernonGraphicsState state{};
    VernonRenderPass renderPass{};
    VernonDrawCommand draw{};
    VernonDynamicState dynamic{};
    std::vector<VernonColorBlendState> blends;

    GraphicsInvocationControls(const VernonColorAttachment *colors, size_t colorCount, uint32_t vertexCount = 0,
                               uint32_t instanceCount = 1,
                               VernonPrimitiveTopology topology = VERNON_TOPOLOGY_TRIANGLE_LIST) {
        state.struct_size = sizeof(state);
        state.topology = topology;
        blends.resize(colorCount);
        for (VernonColorBlendState &blend : blends)
            blend.write_mask = VERNON_RHI_COLOR_WRITE_ALL;
        state.color_blends = blends.data();
        state.color_blend_count = blends.size();
        renderPass.struct_size = sizeof(renderPass);
        renderPass.color_attachments = colors;
        renderPass.color_attachment_count = colorCount;
        draw.struct_size = sizeof(draw);
        draw.vertex_count = vertexCount;
        draw.instance_count = instanceCount;
        dynamic.struct_size = sizeof(dynamic);
    }

    void bind(VernonStageInvocationDescriptor &invocation) {
        invocation.graphics_state = &state;
        invocation.render_pass = &renderPass;
        invocation.draw_command = &draw;
        invocation.dynamic_state = &dynamic;
    }
};

/*
 * The canonical counterpart of GraphicsInvocationControls.
 *
 * A canonical Program reads its render pass, draw, and dynamic state as Program controls off an invocation, so a
 * cooked `vd.pipeline(...)` asset is driven through this rather than through the submit descriptor. There is no
 * graphics state to bind: topology, rasterization, and blending are pipeline state the asset declares at cook time,
 * which is why `vd.pipeline` takes `state=` and `targets=`.
 */
struct CanonicalGraphicsControls {
    VernonRenderPass renderPass{};
    VernonDrawCommand draw{};
    VernonDynamicState dynamic{};

    CanonicalGraphicsControls(const VernonColorAttachment *colors, size_t colorCount, uint32_t vertexCount = 0,
                              uint32_t instanceCount = 1) {
        renderPass.struct_size = sizeof(renderPass);
        renderPass.color_attachments = colors;
        renderPass.color_attachment_count = colorCount;
        draw.struct_size = sizeof(draw);
        draw.vertex_count = vertexCount;
        draw.instance_count = instanceCount;
        dynamic.struct_size = sizeof(dynamic);
    }

    VernonStatus bind(VernonProgramInvocation *invocation, const VernonProgramGraphicsControlsView &controls) const {
        const auto token = [](const char *value) {
            return VernonProgramBindingToken{sizeof(VernonProgramBindingToken), value, std::strlen(value)};
        };
        const VernonProgramBindingToken renderPassToken = token("render-pass");
        const VernonProgramBindingToken drawToken = token("draw-command");
        const VernonProgramBindingToken dynamicToken = token("dynamic-state");
        VernonStatus status = vernonRuntimeProgramInvocationBindRenderPass(invocation, controls.render_pass_control,
                                                                           &renderPassToken, &renderPass, nullptr, 0);
        if (status == VERNON_STATUS_OK)
            status = vernonRuntimeProgramInvocationBindDrawCommand(invocation, controls.draw_command_control,
                                                                   &drawToken, &draw, nullptr);
        if (status == VERNON_STATUS_OK)
            status = vernonRuntimeProgramInvocationBindDynamicState(invocation, controls.dynamic_state_control,
                                                                    &dynamicToken, &dynamic);
        return status;
    }
};

/*
 * Run one canonical graphics invocation to completion: bind every argument and control, then forward.
 *
 * Forward executes and waits, so this is the canonical equivalent of completeSubmission for a Program, and the
 * arguments carry their own slots exactly as they do in a submit descriptor.
 */
inline VernonStatus completeCanonicalInvocation(VernonProgramExecutable *pipeline,
                                                const VernonProgramArgument *arguments, size_t argumentCount,
                                                const CanonicalGraphicsControls &controls) {
    VernonProgramInstance *instance = vernonRuntimeProgramInstanceCreate(pipeline);
    if (!instance)
        return VERNON_STATUS_INVALID_ARGUMENT;
    VernonProgramInvocation *invocation = vernonRuntimeProgramInstanceBeginInvocation(instance);
    if (!invocation) {
        vernonRuntimeProgramInstanceDestroy(instance);
        return VERNON_STATUS_INVALID_ARGUMENT;
    }
    VernonStatus status = VERNON_STATUS_OK;
    for (size_t index = 0; index < argumentCount && status == VERNON_STATUS_OK; ++index) {
        const std::string name = "argument-" + std::to_string(index);
        const VernonProgramBindingToken argumentToken{sizeof(VernonProgramBindingToken), name.data(), name.size()};
        status = vernonRuntimeProgramInvocationBind(invocation, &argumentToken, &arguments[index], nullptr, 0, 0);
    }
    VernonProgramGraphicsControlsView controlSlots{};
    if (status == VERNON_STATUS_OK)
        status = vernonRuntimeProgramExecutableGetGraphicsControlsByIndex(pipeline, 0, &controlSlots);
    if (status == VERNON_STATUS_OK)
        status = controls.bind(invocation, controlSlots);
    if (status == VERNON_STATUS_OK)
        status = vernonRuntimeProgramInvocationForward(invocation, nullptr);
    else
        vernonRuntimeProgramInvocationRollback(invocation);
    vernonRuntimeProgramInvocationDestroy(invocation);
    vernonRuntimeProgramInstanceDestroy(instance);
    return status;
}

inline VernonStatus completeCanonicalComputeInvocation(VernonProgramExecutable *pipeline,
                                                       const VernonProgramArgument *arguments, size_t argumentCount,
                                                       VernonLaunchSize grid, VernonPullback **pullback = nullptr) {
    std::vector<VernonProgramArgument> bindings(arguments, arguments + argumentCount);
    const size_t parameterCount = vernonRuntimeProgramExecutableGetParameterCount(pipeline);
    std::vector<VernonProgramParameterView> parameters(parameterCount);
    for (size_t index = 0; index < parameterCount; ++index)
        if (vernonRuntimeProgramExecutableGetParameterByIndex(pipeline, index, &parameters[index]) != VERNON_STATUS_OK)
            return VERNON_STATUS_INVALID_ARGUMENT;
    bindings.reserve(parameterCount);
    const uint32_t gridAxes[3]{grid.x, grid.y, grid.z};
    for (const VernonProgramParameterView &parameter : parameters) {
        if (std::any_of(bindings.begin(), bindings.end(),
                        [&](const VernonProgramArgument &argument) { return argument.slot == parameter.slot; }))
            continue;
        std::optional<size_t> gridAxis;
        for (size_t axis = 0; axis < 3; ++axis) {
            const std::string name = "__grid_" + std::string(1, "xyz"[axis]);
            if (parameter.name.size == name.size() && std::memcmp(parameter.name.data, name.data(), name.size()) == 0) {
                gridAxis = axis;
                break;
            }
        }
        if (gridAxis) {
            VernonProgramArgument control{};
            control.slot = parameter.slot;
            control.kind = VERNON_PROGRAM_TENSOR;
            control.tensor.struct_size = sizeof(VernonTensorView);
            control.tensor.storage = VERNON_TENSOR_HOST;
            control.tensor.host_data = &gridAxes[*gridAxis];
            control.tensor.element_layout = vernonRuntimeGetScalarValueLayout(VERNON_DATA_U32);
            control.tensor.access = VERNON_ACCESS_READ;
            control.tensor.byte_size = sizeof(uint32_t);
            bindings.push_back(control);
            continue;
        }
        const auto source = std::find_if(bindings.begin(), bindings.end(), [&](const VernonProgramArgument &argument) {
            const auto sourceParameter =
                std::find_if(parameters.begin(), parameters.end(), [&](const VernonProgramParameterView &candidate) {
                    return candidate.slot == argument.slot;
                });
            return sourceParameter != parameters.end() && sourceParameter->name.size == parameter.name.size &&
                   std::memcmp(sourceParameter->name.data, parameter.name.data, parameter.name.size) == 0;
        });
        if (source != bindings.end()) {
            VernonProgramArgument alias = *source;
            alias.slot = parameter.slot;
            bindings.push_back(alias);
        }
    }
    VernonProgramInstance *instance = vernonRuntimeProgramInstanceCreate(pipeline);
    if (!instance)
        return VERNON_STATUS_INVALID_ARGUMENT;
    VernonProgramInvocation *invocation = vernonRuntimeProgramInstanceBeginInvocation(instance);
    if (!invocation) {
        vernonRuntimeProgramInstanceDestroy(instance);
        return VERNON_STATUS_INVALID_ARGUMENT;
    }
    VernonStatus status = VERNON_STATUS_OK;
    for (size_t index = 0; index < bindings.size() && status == VERNON_STATUS_OK; ++index) {
        const std::string token = "argument-" + std::to_string(index);
        const VernonProgramBindingToken bindingToken{sizeof(VernonProgramBindingToken), token.data(), token.size()};
        status = vernonRuntimeProgramInvocationBind(invocation, &bindingToken, &bindings[index], nullptr, 0, 0);
    }
    if (status == VERNON_STATUS_OK)
        status = vernonRuntimeProgramInvocationForward(invocation, pullback);
    else
        vernonRuntimeProgramInvocationRollback(invocation);
    vernonRuntimeProgramInvocationDestroy(invocation);
    vernonRuntimeProgramInstanceDestroy(instance);
    return status;
}

inline VernonStatus completeCanonicalAutodiffInvocation(VernonProgramExecutable *pipeline, VernonLaunchSize grid,
                                                        const DerivativeLeafSetFixture &inputs,
                                                        const DerivativeLeafSetFixture &outputs,
                                                        VernonPullback **pullback) {
    const size_t valueCount = inputs.value_count + outputs.value_count;
    std::vector<VernonProgramArgument> arguments;
    std::vector<std::vector<int64_t>> strides;
    arguments.reserve(valueCount);
    strides.reserve(valueCount);

    auto appendValues = [&](const DerivativeLeafSetFixture &values) -> VernonStatus {
        for (size_t index = 0; index < values.value_count; ++index) {
            const DerivativeLeafFixture &value = values.values[index];
            VernonProgramParameterView parameter{};
            if (vernonRuntimeProgramExecutableFindParameter(pipeline, value.path, &parameter) != VERNON_STATUS_OK ||
                parameter.kind != VERNON_PROGRAM_TENSOR)
                return VERNON_STATUS_INVALID_ARGUMENT;

            strides.emplace_back(value.rank);
            int64_t stride = static_cast<int64_t>(vernonRuntimeGetScalarValueLayout(value.dtype).byte_size);
            for (size_t axis = value.rank; axis-- > 0;) {
                strides.back()[axis] = stride;
                stride *= static_cast<int64_t>(value.shape[axis]);
            }

            VernonProgramArgument argument{};
            argument.slot = parameter.slot;
            argument.kind = VERNON_PROGRAM_TENSOR;
            argument.tensor.struct_size = sizeof(VernonTensorView);
            argument.tensor.storage = VERNON_TENSOR_HOST;
            argument.tensor.host_data = value.data;
            argument.tensor.element_layout = vernonRuntimeGetScalarValueLayout(value.dtype);
            argument.tensor.access = parameter.access;
            argument.tensor.rank = value.rank;
            argument.tensor.shape = value.shape;
            argument.tensor.byte_strides = strides.back().empty() ? nullptr : strides.back().data();
            argument.tensor.byte_size = value.size;
            arguments.push_back(argument);
        }
        return VERNON_STATUS_OK;
    };

    VernonStatus status = appendValues(inputs);
    if (status == VERNON_STATUS_OK)
        status = appendValues(outputs);
    if (status != VERNON_STATUS_OK)
        return status;
    return completeCanonicalComputeInvocation(pipeline, arguments.data(), arguments.size(), grid, pullback);
}

inline VernonStatus applyCanonicalPullback(VernonProgramExecutable *pipeline, VernonPullback *pullback,
                                           const DerivativeLeafSetFixture *cotangents,
                                           DerivativeLeafSetFixture *gradients,
                                           const VernonPullbackApplyOptions *options = nullptr) {
    if (!pipeline || !pullback || !gradients)
        return VERNON_STATUS_INVALID_ARGUMENT;
    std::vector<VernonProgramArgument> arguments;
    std::vector<std::vector<int64_t>> strides;
    const auto append = [&](const DerivativeLeafSetFixture &values, VernonProgramBoundaryRole role,
                            VernonValueAccess access) -> VernonStatus {
        for (size_t index = 0; index < values.value_count; ++index) {
            const DerivativeLeafFixture &value = values.values[index];
            VernonProgramParameterView boundary{};
            if (vernonRuntimeProgramExecutableFindBoundary(pipeline, role, value.path, &boundary) != VERNON_STATUS_OK ||
                boundary.kind != VERNON_PROGRAM_TENSOR)
                return VERNON_STATUS_INVALID_ARGUMENT;
            strides.emplace_back(value.rank);
            int64_t stride = static_cast<int64_t>(boundary.element_layout.byte_size);
            for (size_t axis = value.rank; axis-- > 0;) {
                strides.back()[axis] = stride;
                stride *= static_cast<int64_t>(value.shape[axis]);
            }
            VernonProgramArgument argument{};
            argument.slot = boundary.slot;
            argument.kind = VERNON_PROGRAM_TENSOR;
            argument.tensor.struct_size = sizeof(VernonTensorView);
            argument.tensor.storage = VERNON_TENSOR_HOST;
            argument.tensor.host_data = value.data;
            argument.tensor.element_layout = boundary.element_layout;
            argument.tensor.access = access;
            argument.tensor.rank = value.rank;
            argument.tensor.shape = value.shape;
            argument.tensor.byte_strides = strides.back().empty() ? nullptr : strides.back().data();
            argument.tensor.byte_size = value.size;
            arguments.push_back(argument);
        }
        return VERNON_STATUS_OK;
    };
    if ((cotangents &&
         append(*cotangents, VERNON_PROGRAM_BOUNDARY_COTANGENT, VERNON_ACCESS_READ) != VERNON_STATUS_OK) ||
        append(*gradients, VERNON_PROGRAM_BOUNDARY_GRADIENT, VERNON_ACCESS_WRITE) != VERNON_STATUS_OK)
        return VERNON_STATUS_INVALID_ARGUMENT;
    return options ? vernonProgramPullbackApplyWithOptions(pullback, arguments.data(), arguments.size(), options)
                   : vernonProgramPullbackApply(pullback, arguments.data(), arguments.size());
}

inline VernonStatus completeSubmission(VernonStageExecutable *pipeline,
                                       const VernonStageInvocationDescriptor *invocation) {
    VernonSubmission *submission{};
    const VernonStatus submitStatus = vernonRuntimeStageSubmit(pipeline, invocation, &submission);
    if (submitStatus != VERNON_STATUS_OK)
        return submitStatus;
    const VernonStatus completionStatus = vernonSubmissionWait(submission);
    vernonSubmissionDestroy(submission);
    return completionStatus;
}

inline VernonRhiStatus completeSubmission(VernonRhiDevice device, VernonRhiCommandEncoder encoder,
                                          VernonRhiCommandEncoderStats *stats = nullptr) {
    VernonRhiCompletion completion{};
    VernonRhiStatus status = vernonRhiDeviceSubmit(device, encoder, &completion);
    if (status != VERNON_RHI_STATUS_OK)
        return status;
    status = vernonRhiCompletionWait(device, completion);
    if (status == VERNON_RHI_STATUS_OK && stats)
        status = vernonRhiCompletionGetCommandStats(device, completion, stats);
    const VernonRhiStatus destroyStatus = vernonRhiDeviceDestroyCompletion(device, completion);
    return status == VERNON_RHI_STATUS_OK ? destroyStatus : status;
}

inline VernonRhiBackend rhiBackend(VernonRuntimeBackend backend) {
    switch (backend) {
    case VERNON_RUNTIME_CUDA:
        return VERNON_RHI_BACKEND_CUDA;
    case VERNON_RUNTIME_VULKAN:
        return VERNON_RHI_BACKEND_VULKAN;
    case VERNON_RUNTIME_DIRECTX12:
        return VERNON_RHI_BACKEND_DIRECTX12;
    case VERNON_RUNTIME_OPENGL:
        return VERNON_RHI_BACKEND_OPENGL;
    case VERNON_RUNTIME_OPENGL_ES:
        return VERNON_RHI_BACKEND_OPENGL_ES;
    case VERNON_RUNTIME_METAL:
        return VERNON_RHI_BACKEND_METAL;
    default:
        return VERNON_RHI_BACKEND_CUDA;
    }
}

inline RhiRuntime createRhiRuntime(VernonRuntimeBackend backend,
                                   const VernonOpenGLContextCallbacks *openglCallbacks = nullptr,
                                   bool forceSoftware = false) {
    RhiRuntime result;
    if (backend == VERNON_RUNTIME_OPENGL || backend == VERNON_RUNTIME_OPENGL_ES)
        result.device = vernonRhiCreateOpenGLDevice(openglCallbacks, backend == VERNON_RUNTIME_OPENGL_ES);
    else {
        VernonRhiOwnedDeviceDescriptor descriptor{};
        descriptor.struct_size = sizeof(descriptor);
        descriptor.backend = rhiBackend(backend);
        descriptor.flags = forceSoftware ? VERNON_RHI_OWNED_DEVICE_FORCE_SOFTWARE : 0;
        result.device = vernonRhiCreateDevice(&descriptor);
    }
    if (result.device.index != VERNON_RHI_INVALID_HANDLE_INDEX)
        result.runtime = vernonRuntimeCreateForRhiDevice(backend, result.device);
    return result;
}

inline void destroyRhiRuntime(RhiRuntime &context) {
    if (context.runtime)
        vernonRuntimeDestroy(context.runtime);
    if (context.device.index != VERNON_RHI_INVALID_HANDLE_INDEX)
        vernonRhiDestroyDevice(context.device);
    context = {};
}

inline RhiBuffer createBuffer(RhiRuntime &context, uint64_t size, uint64_t alignment, uint32_t usage,
                              const void *initialData = nullptr) {
    VernonRhiBufferDescriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.size = size;
    descriptor.alignment = alignment;
    descriptor.usage = usage | VERNON_RHI_BUFFER_TRANSFER_SOURCE | VERNON_RHI_BUFFER_TRANSFER_DESTINATION;
    descriptor.memory_class = VERNON_RHI_MEMORY_DEVICE;
    RhiBuffer result;
    if (vernonRhiDeviceCreateBuffer(context.device, &descriptor, &result.handle) != VERNON_RHI_STATUS_OK)
        return result;
    if (initialData &&
        vernonRhiDeviceUploadBuffer(context.device, result.handle, 0, initialData, size) != VERNON_RHI_STATUS_OK)
        return result;
    if (vernonRuntimeReferenceRhiBuffer(context.runtime, result.handle, 0, size, &result.reference) != VERNON_STATUS_OK)
        return result;
    return result;
}

inline RhiImage createImage(RhiRuntime &context, VernonRhiImageDimension dimension, VernonRhiFormat format,
                            uint32_t width, uint32_t height, uint32_t depth, uint32_t usage, uint32_t arrayLayers = 1) {
    VernonRhiImageDescriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.dimension = dimension;
    descriptor.format = format;
    descriptor.width = width;
    descriptor.height = height;
    descriptor.depth = depth;
    descriptor.mip_levels = 1;
    descriptor.array_layers = arrayLayers;
    descriptor.sample_count = 1;
    descriptor.usage = usage | VERNON_RHI_IMAGE_TRANSFER_SOURCE | VERNON_RHI_IMAGE_TRANSFER_DESTINATION;
    RhiImage result;
    if (vernonRhiDeviceCreateImage(context.device, &descriptor, &result.handle) != VERNON_RHI_STATUS_OK)
        return result;
    return result;
}

inline RhiImageView createImageView(RhiRuntime &context, const RhiImage &image, VernonRhiImageDimension dimension,
                                    VernonRhiFormat format, uint32_t mipLevels = 1, uint32_t arrayLayers = 1,
                                    uint32_t aspects = VERNON_RHI_IMAGE_ASPECT_COLOR) {
    VernonRhiImageViewDescriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.image = image.handle;
    descriptor.dimension = dimension;
    descriptor.format = format;
    descriptor.mip_level_count = mipLevels;
    descriptor.array_layer_count = arrayLayers;
    descriptor.aspects = aspects;
    RhiImageView result;
    if (vernonRhiDeviceCreateImageView(context.device, &descriptor, &result.handle) != VERNON_RHI_STATUS_OK)
        return result;
    if (vernonRuntimeReferenceRhiImageView(context.runtime, result.handle, &result.reference) != VERNON_STATUS_OK)
        return {};
    return result;
}

inline RhiSampler createSampler(RhiRuntime &context, uint32_t filter = VERNON_RHI_FILTER_NEAREST,
                                uint32_t address = VERNON_RHI_ADDRESS_CLAMP_TO_EDGE) {
    VernonRhiSamplerDescriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.min_filter = filter;
    descriptor.mag_filter = filter;
    descriptor.mip_filter = filter;
    descriptor.address_u = address;
    descriptor.address_v = address;
    descriptor.address_w = address;
    RhiSampler result;
    if (vernonRhiDeviceCreateSampler(context.device, &descriptor, &result.handle) != VERNON_RHI_STATUS_OK)
        return result;
    if (vernonRuntimeReferenceRhiSampler(context.runtime, result.handle, &result.reference) != VERNON_STATUS_OK)
        return result;
    return result;
}

} // namespace vernon::tests

#endif
