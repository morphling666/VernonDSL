#include "prepared_graphics_draw.h"

#include <algorithm>

namespace vernon::runtime {

bool prepareGraphicsDraw(const VernonProgramSubmitDescriptor &source, const PlannedGraphicsInvocation &plan,
                         std::vector<uint32_t> attachmentFormats, uint32_t depthFormat,
                         const std::vector<VernonRuntimeProviderBindingLayoutEntry> &layout,
                         const std::vector<VernonRuntimeProviderBindingValue> &values, PreparedGraphicsDraw &prepared,
                         std::string &error) {
    if ((!plan.depthAttachment && plan.attachments.empty()) ||
        plan.attachments.size() > VERNON_RUNTIME_PROVIDER_MAX_COLOR_ATTACHMENTS || layout.size() != values.size()) {
        error = "graphics draw has invalid attachments or target bindings";
        return false;
    }
    prepared = {};
    for (size_t index = 0; index < plan.attachments.size(); ++index) {
        const VernonColorAttachment &attachment = *plan.attachments[index];
        auto &target = prepared.attachments[index];
        target.location = attachment.location;
        target.view = attachment.view;
        target.load_operation = attachment.load_operation;
        target.store_operation = attachment.store_operation;
        std::copy(std::begin(attachment.clear_color), std::end(attachment.clear_color), target.clear_color);
    }
    if (plan.depthAttachment)
        prepared.depthAttachment = plan.depthAttachment->view;
    const bool hasStencil = plan.depthAttachment && plan.depthFormat == VERNON_TEXTURE_D32_FLOAT_S8_UINT;
    if (!planGraphicsState(source, attachmentFormats.size(), plan.depthAttachment != nullptr, hasStencil,
                           prepared.graphicsState, error))
        return false;
    for (size_t index = 0; index < layout.size(); ++index) {
        if (layout[index].kind != VERNON_RUNTIME_PROVIDER_VERTEX_BUFFER)
            continue;
        const uint32_t binding = layout[index].binding;
        if (prepared.vertexStrides.size() <= binding)
            prepared.vertexStrides.resize(binding + 1);
        prepared.vertexStrides[binding] = values[index].payload.buffer.stride;
    }
    prepared.variantKey = {static_cast<uint32_t>(plan.topology),
                           std::move(attachmentFormats),
                           depthFormat,
                           1,
                           prepared.vertexStrides,
                           prepared.graphicsState.rasterization,
                           prepared.graphicsState.depthStencil,
                           prepared.graphicsState.colorBlends};

    auto &draw = prepared.invocation;
    draw.struct_size = sizeof(draw);
    draw.command_encoder = source.command_encoder;
    draw.vertex_count = plan.vertexCount;
    draw.instance_count = plan.instanceCount;
    draw.color_attachments = prepared.attachments.data();
    draw.color_attachment_count = plan.attachments.size();
    draw.depth_stencil_view = prepared.depthAttachment;
    draw.depth_load_operation =
        plan.depthAttachment ? plan.depthAttachment->load_operation : VERNON_RUNTIME_PROVIDER_LOAD_DISCARD;
    draw.depth_store_operation =
        plan.depthAttachment ? plan.depthAttachment->store_operation : VERNON_RUNTIME_PROVIDER_STORE_DISCARD;
    draw.clear_depth = plan.depthAttachment ? plan.depthAttachment->clear_depth : 1.0f;
    draw.stencil_load_operation =
        hasStencil ? plan.depthAttachment->stencil_load_operation : VERNON_RUNTIME_PROVIDER_LOAD_DISCARD;
    draw.stencil_store_operation =
        hasStencil ? plan.depthAttachment->stencil_store_operation : VERNON_RUNTIME_PROVIDER_STORE_DISCARD;
    draw.clear_stencil = hasStencil ? plan.depthAttachment->clear_stencil : 0;
    draw.stencil_reference = prepared.graphicsState.stencilReference;
    std::copy(std::begin(plan.viewport), std::end(plan.viewport), draw.viewport);
    std::copy(std::begin(plan.scissor), std::end(plan.scissor), draw.scissor);
    draw.topology = plan.topology;
    if (plan.indexBinding) {
        draw.index_buffer = plan.indexBinding->resource;
        draw.index_buffer.offset += plan.indexBinding->offset;
        draw.index_count = plan.indexBinding->index_count;
        draw.index_type = plan.indexBinding->type;
    }
    return true;
}

} // namespace vernon::runtime
