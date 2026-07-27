#include "VernonRHI.h"

#include "rhi_internal.h"

#include <cstring>
#include <mutex>
#include <vector>

namespace {

struct EncoderSlot {
    VernonRhiDevice device{};
    VernonRhiCommandEncoderStats stats{};
    VernonRhiGraphicsPipeline graphicsPipeline{static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
    VernonRhiComputePipeline computePipeline{static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
    VernonRhiViewport viewport{};
    VernonRhiRectangle scissor{};
    uint32_t generation{1};
    bool occupied{};
    bool rendering{};
    bool finished{};
    bool submitted{};
    bool hasViewport{};
    bool hasScissor{};
};

std::mutex encoderMutex;
std::vector<EncoderSlot> encoders;

template <typename Handle> bool validHandle(Handle handle) {
    return handle.index != VERNON_RHI_INVALID_HANDLE_INDEX && handle.generation != 0;
}

bool sameDevice(VernonRhiDevice left, VernonRhiDevice right) {
    return left.index == right.index && left.generation == right.generation;
}

EncoderSlot *lookup(VernonRhiDevice device, VernonRhiCommandEncoder encoder) {
    if (!vernon::rhi::deviceExists(device) || encoder.index >= encoders.size())
        return nullptr;
    EncoderSlot &slot = encoders[encoder.index];
    return slot.occupied && slot.generation == encoder.generation && sameDevice(slot.device, device) ? &slot : nullptr;
}

bool validLoad(VernonRhiLoadOperation operation) {
    return operation >= VERNON_RHI_LOAD_CLEAR && operation <= VERNON_RHI_LOAD_DISCARD;
}

bool validStore(VernonRhiStoreOperation operation) {
    return operation >= VERNON_RHI_STORE_PRESERVE && operation <= VERNON_RHI_STORE_DISCARD;
}

bool validRendering(const VernonRhiRenderingDescriptor &descriptor) {
    if (!descriptor.width || !descriptor.height || !descriptor.layers ||
        (descriptor.color_attachment_count && !descriptor.color_attachments))
        return false;
    for (size_t index = 0; index < descriptor.color_attachment_count; ++index) {
        const auto &attachment = descriptor.color_attachments[index];
        if (!validHandle(attachment.view) || !validLoad(attachment.load_operation) ||
            !validStore(attachment.store_operation))
            return false;
    }
    if (descriptor.depth_stencil_attachment) {
        const auto &attachment = *descriptor.depth_stencil_attachment;
        if (!validHandle(attachment.view) || !validLoad(attachment.depth_load_operation) ||
            !validStore(attachment.depth_store_operation) || !validLoad(attachment.stencil_load_operation) ||
            !validStore(attachment.stencil_store_operation) || attachment.clear_depth < 0.0f ||
            attachment.clear_depth > 1.0f)
            return false;
    }
    return descriptor.color_attachment_count != 0 || descriptor.depth_stencil_attachment;
}

} // namespace

extern "C" VernonRhiStatus vernonRhiDeviceCreateCommandEncoder(VernonRhiDevice device,
                                                               const VernonRhiCommandEncoderDescriptor *descriptor,
                                                               VernonRhiCommandEncoder *output) {
    if (output)
        *output = {static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
    if (!descriptor || !output || descriptor->struct_size < sizeof(*descriptor) || !vernon::rhi::deviceExists(device))
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(encoderMutex);
    uint32_t index = 0;
    while (index < encoders.size() && encoders[index].occupied)
        ++index;
    if (index == encoders.size())
        encoders.emplace_back();
    EncoderSlot &slot = encoders[index];
    const uint32_t generation = slot.generation;
    slot = {};
    slot.generation = generation ? generation : 1;
    slot.occupied = true;
    slot.device = device;
    *output = {index, slot.generation};
    return VERNON_RHI_STATUS_OK;
}

extern "C" VernonRhiStatus vernonRhiDeviceDestroyCommandEncoder(VernonRhiDevice device,
                                                                VernonRhiCommandEncoder encoder) {
    std::lock_guard<std::mutex> guard(encoderMutex);
    EncoderSlot *slot = lookup(device, encoder);
    if (!slot || slot->rendering)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    slot->occupied = false;
    if (++slot->generation == 0)
        slot->generation = 1;
    return VERNON_RHI_STATUS_OK;
}

extern "C" VernonRhiStatus vernonRhiCommandEncoderBarrier(VernonRhiDevice device, VernonRhiCommandEncoder encoder,
                                                          const VernonRhiBarrier *barriers, size_t barrierCount) {
    std::lock_guard<std::mutex> guard(encoderMutex);
    EncoderSlot *slot = lookup(device, encoder);
    if (!slot || slot->rendering || slot->finished || (barrierCount && !barriers))
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    for (size_t index = 0; index < barrierCount; ++index)
        if (barriers[index].struct_size < sizeof(VernonRhiBarrier) ||
            barriers[index].old_state > VERNON_RHI_STATE_PRESENT ||
            barriers[index].new_state > VERNON_RHI_STATE_PRESENT)
            return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    slot->stats.barrier_count += static_cast<uint32_t>(barrierCount);
    return VERNON_RHI_STATUS_OK;
}

extern "C" VernonRhiStatus vernonRhiCommandEncoderBeginRendering(VernonRhiDevice device,
                                                                 VernonRhiCommandEncoder encoder,
                                                                 const VernonRhiRenderingDescriptor *descriptor) {
    std::lock_guard<std::mutex> guard(encoderMutex);
    EncoderSlot *slot = lookup(device, encoder);
    if (!slot || slot->rendering || slot->finished || !descriptor || descriptor->struct_size < sizeof(*descriptor) ||
        !validRendering(*descriptor))
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    slot->rendering = true;
    ++slot->stats.rendering_scope_count;
    return VERNON_RHI_STATUS_OK;
}

extern "C" VernonRhiStatus vernonRhiCommandEncoderEndRendering(VernonRhiDevice device,
                                                               VernonRhiCommandEncoder encoder) {
    std::lock_guard<std::mutex> guard(encoderMutex);
    EncoderSlot *slot = lookup(device, encoder);
    if (!slot || !slot->rendering || slot->finished)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    slot->rendering = false;
    return VERNON_RHI_STATUS_OK;
}

extern "C" VernonRhiStatus vernonRhiCommandEncoderBindGraphicsPipeline(VernonRhiDevice device,
                                                                       VernonRhiCommandEncoder encoder,
                                                                       VernonRhiGraphicsPipeline pipeline) {
    std::lock_guard<std::mutex> guard(encoderMutex);
    EncoderSlot *slot = lookup(device, encoder);
    if (!slot || !slot->rendering || slot->finished || !validHandle(pipeline))
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    if (slot->graphicsPipeline.index != pipeline.index || slot->graphicsPipeline.generation != pipeline.generation) {
        slot->graphicsPipeline = pipeline;
        ++slot->stats.graphics_pipeline_bind_count;
    }
    return VERNON_RHI_STATUS_OK;
}

extern "C" VernonRhiStatus vernonRhiCommandEncoderBindComputePipeline(VernonRhiDevice device,
                                                                      VernonRhiCommandEncoder encoder,
                                                                      VernonRhiComputePipeline pipeline) {
    std::lock_guard<std::mutex> guard(encoderMutex);
    EncoderSlot *slot = lookup(device, encoder);
    if (!slot || slot->rendering || slot->finished || !validHandle(pipeline))
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    if (slot->computePipeline.index != pipeline.index || slot->computePipeline.generation != pipeline.generation) {
        slot->computePipeline = pipeline;
        ++slot->stats.compute_pipeline_bind_count;
    }
    return VERNON_RHI_STATUS_OK;
}

extern "C" VernonRhiStatus vernonRhiCommandEncoderSetViewport(VernonRhiDevice device, VernonRhiCommandEncoder encoder,
                                                              const VernonRhiViewport *viewport) {
    std::lock_guard<std::mutex> guard(encoderMutex);
    EncoderSlot *slot = lookup(device, encoder);
    if (!slot || !slot->rendering || slot->finished || !viewport || viewport->width <= 0.0f ||
        viewport->height <= 0.0f || viewport->minimum_depth < 0.0f || viewport->maximum_depth > 1.0f ||
        viewport->minimum_depth > viewport->maximum_depth)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    if (!slot->hasViewport || std::memcmp(&slot->viewport, viewport, sizeof(*viewport)) != 0) {
        slot->viewport = *viewport;
        slot->hasViewport = true;
        ++slot->stats.viewport_change_count;
    }
    return VERNON_RHI_STATUS_OK;
}

extern "C" VernonRhiStatus vernonRhiCommandEncoderSetScissor(VernonRhiDevice device, VernonRhiCommandEncoder encoder,
                                                             const VernonRhiRectangle *scissor) {
    std::lock_guard<std::mutex> guard(encoderMutex);
    EncoderSlot *slot = lookup(device, encoder);
    if (!slot || !slot->rendering || slot->finished || !scissor || !scissor->width || !scissor->height)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    if (!slot->hasScissor || std::memcmp(&slot->scissor, scissor, sizeof(*scissor)) != 0) {
        slot->scissor = *scissor;
        slot->hasScissor = true;
        ++slot->stats.scissor_change_count;
    }
    return VERNON_RHI_STATUS_OK;
}

extern "C" VernonRhiStatus vernonRhiCommandEncoderClearColorAttachment(VernonRhiDevice device,
                                                                       VernonRhiCommandEncoder encoder,
                                                                       uint32_t location, const float clearColor[4]) {
    std::lock_guard<std::mutex> guard(encoderMutex);
    EncoderSlot *slot = lookup(device, encoder);
    if (!slot || !slot->rendering || slot->finished || location >= 8 || !clearColor)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    ++slot->stats.clear_count;
    return VERNON_RHI_STATUS_OK;
}

extern "C" VernonRhiStatus vernonRhiCommandEncoderClearDepthStencilAttachment(VernonRhiDevice device,
                                                                              VernonRhiCommandEncoder encoder,
                                                                              float clearDepth, uint32_t,
                                                                              uint32_t aspects) {
    std::lock_guard<std::mutex> guard(encoderMutex);
    EncoderSlot *slot = lookup(device, encoder);
    if (!slot || !slot->rendering || slot->finished || clearDepth < 0.0f || clearDepth > 1.0f || !aspects ||
        (aspects & ~(VERNON_RHI_ATTACHMENT_DEPTH | VERNON_RHI_ATTACHMENT_STENCIL)))
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    ++slot->stats.clear_count;
    return VERNON_RHI_STATUS_OK;
}

extern "C" VernonRhiStatus vernonRhiCommandEncoderDraw(VernonRhiDevice device, VernonRhiCommandEncoder encoder,
                                                       const VernonRhiDrawDescriptor *descriptor) {
    std::lock_guard<std::mutex> guard(encoderMutex);
    EncoderSlot *slot = lookup(device, encoder);
    if (!slot || !slot->rendering || slot->finished || !validHandle(slot->graphicsPipeline) || !descriptor ||
        !descriptor->vertex_count || !descriptor->instance_count)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    ++slot->stats.draw_count;
    return VERNON_RHI_STATUS_OK;
}

extern "C" VernonRhiStatus vernonRhiCommandEncoderDrawIndexed(VernonRhiDevice device, VernonRhiCommandEncoder encoder,
                                                              const VernonRhiDrawIndexedDescriptor *descriptor) {
    std::lock_guard<std::mutex> guard(encoderMutex);
    EncoderSlot *slot = lookup(device, encoder);
    if (!slot || !slot->rendering || slot->finished || !validHandle(slot->graphicsPipeline) || !descriptor ||
        !descriptor->index_count || !descriptor->instance_count)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    ++slot->stats.draw_count;
    return VERNON_RHI_STATUS_OK;
}

extern "C" VernonRhiStatus vernonRhiCommandEncoderDispatch(VernonRhiDevice device, VernonRhiCommandEncoder encoder,
                                                           uint32_t x, uint32_t y, uint32_t z) {
    std::lock_guard<std::mutex> guard(encoderMutex);
    EncoderSlot *slot = lookup(device, encoder);
    if (!slot || slot->rendering || slot->finished || !validHandle(slot->computePipeline) || !x || !y || !z)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    ++slot->stats.dispatch_count;
    return VERNON_RHI_STATUS_OK;
}

extern "C" VernonRhiStatus vernonRhiCommandEncoderFinish(VernonRhiDevice device, VernonRhiCommandEncoder encoder) {
    std::lock_guard<std::mutex> guard(encoderMutex);
    EncoderSlot *slot = lookup(device, encoder);
    if (!slot || slot->rendering || slot->finished)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    slot->finished = true;
    return VERNON_RHI_STATUS_OK;
}

extern "C" VernonRhiStatus vernonRhiDeviceSubmit(VernonRhiDevice device, VernonRhiCommandEncoder encoder) {
    std::lock_guard<std::mutex> guard(encoderMutex);
    EncoderSlot *slot = lookup(device, encoder);
    if (!slot || !slot->finished || slot->submitted)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    slot->submitted = true;
    return VERNON_RHI_STATUS_OK;
}

extern "C" VernonRhiStatus vernonRhiCommandEncoderGetStats(VernonRhiDevice device, VernonRhiCommandEncoder encoder,
                                                           VernonRhiCommandEncoderStats *output) {
    std::lock_guard<std::mutex> guard(encoderMutex);
    EncoderSlot *slot = lookup(device, encoder);
    if (!slot || !output)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    *output = slot->stats;
    return VERNON_RHI_STATUS_OK;
}
