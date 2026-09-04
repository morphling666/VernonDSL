#ifndef VERNON_TESTS_SUPPORT_RUNTIME_RHI_TEST_UTILS_H
#define VERNON_TESTS_SUPPORT_RUNTIME_RHI_TEST_UTILS_H

#include "VernonExecutionGraph.h"
#include "VernonRuntime.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace vernon::tests {

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

    void bind(VernonProgramSubmitDescriptor &invocation) {
        invocation.graphics_state = &state;
        invocation.render_pass = &renderPass;
        invocation.draw_command = &draw;
        invocation.dynamic_state = &dynamic;
    }
};

inline VernonStatus completeSubmission(VernonProgramExecutable *pipeline,
                                       const VernonProgramSubmitDescriptor *invocation) {
    VernonSubmission *submission{};
    const VernonStatus submitStatus = vernonRuntimeProgramSubmit(pipeline, invocation, &submission);
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

class RuntimeGraphRenderPass final : public execution::RenderPass {
public:
    RuntimeGraphRenderPass(std::string name, execution::GraphImage target, VernonRuntimeContext *runtime,
                           VernonProgramExecutable *pipeline, const VernonProgramSubmitDescriptor *invocation,
                           VernonRhiLoadOperation load = VERNON_RHI_LOAD_CLEAR,
                           VernonRhiStoreOperation store = VERNON_RHI_STORE_PRESERVE)
        : RenderPass(std::move(name)), target_(target), runtime_(runtime), pipeline_(pipeline), invocation_(invocation),
          load_(load), store_(store) {}

    void declare() override {
        execution::ColorAttachmentUse attachment{};
        attachment.image = target_;
        attachment.load = load_;
        attachment.store = store_;
        color(0, attachment);
        renderArea(0, 0, target_.width, target_.height);
    }

    VernonRhiStatus execute(execution::GraphicsEncoder &encoder, const execution::ExecutionResources &) override {
        VernonRuntimeProviderObject providerEncoder{};
        if (vernonRuntimeReferenceRhiCommandEncoder(runtime_, encoder.native(), &providerEncoder) != VERNON_STATUS_OK)
            return VERNON_RHI_STATUS_INTERNAL_ERROR;
        return vernonRuntimeProgramEncode(providerEncoder, pipeline_, invocation_) == VERNON_STATUS_OK
                   ? VERNON_RHI_STATUS_OK
                   : VERNON_RHI_STATUS_INTERNAL_ERROR;
    }

private:
    execution::GraphImage target_;
    VernonRuntimeContext *runtime_{};
    VernonProgramExecutable *pipeline_{};
    const VernonProgramSubmitDescriptor *invocation_{};
    VernonRhiLoadOperation load_{};
    VernonRhiStoreOperation store_{};
};

class RuntimeGraphComputePass final : public execution::ComputePass {
public:
    RuntimeGraphComputePass(std::string name, execution::GraphBuffer buffer, VernonRuntimeContext *runtime,
                            VernonProgramExecutable *pipeline, const VernonProgramSubmitDescriptor *invocation)
        : ComputePass(std::move(name)), buffer_(buffer), runtime_(runtime), pipeline_(pipeline),
          invocation_(invocation) {}

    void declare() override { readWrite(buffer_, VERNON_RHI_STATE_SHADER_WRITE, VERNON_RHI_STAGE_COMPUTE); }

    VernonRhiStatus execute(execution::ComputeEncoder &encoder, const execution::ExecutionResources &) override {
        VernonRuntimeProviderObject providerEncoder{};
        if (vernonRuntimeReferenceRhiCommandEncoder(runtime_, encoder.native(), &providerEncoder) != VERNON_STATUS_OK)
            return VERNON_RHI_STATUS_INTERNAL_ERROR;
        return vernonRuntimeProgramEncode(providerEncoder, pipeline_, invocation_) == VERNON_STATUS_OK
                   ? VERNON_RHI_STATUS_OK
                   : VERNON_RHI_STATUS_INTERNAL_ERROR;
    }

private:
    execution::GraphBuffer buffer_;
    VernonRuntimeContext *runtime_{};
    VernonProgramExecutable *pipeline_{};
    const VernonProgramSubmitDescriptor *invocation_{};
};

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
