#include "prepared_graphics_draw.h"

#include "VernonFormats.hpp"

#include <algorithm>

namespace vernon::runtime {

bool prepareGraphicsDraw(const VernonStageInvocationDescriptor &source, const PlannedGraphicsInvocation &plan,
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
    const bool hasStencil = plan.depthAttachment && vernon::formats::hasStencil(plan.depthFormat);
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
    draw.first_instance = plan.firstInstance;
    draw.stencil_reference = prepared.graphicsState.stencilReference;
    std::copy(std::begin(plan.viewport), std::end(plan.viewport), draw.viewport);
    std::copy(std::begin(plan.scissor), std::end(plan.scissor), draw.scissor);
    draw.topology = plan.topology;
    if (plan.indexBinding) {
        const VernonIndexBinding &index = *plan.indexBinding;
        const uint64_t indexBytes = static_cast<uint64_t>(index.index_count) * sizeof(uint32_t);
        if (index.offset > index.resource.size || indexBytes > index.resource.size - index.offset ||
            index.resource.offset > UINT64_MAX - index.offset) {
            error = "index binding exceeds its index buffer";
            return false;
        }
        draw.index_buffer = index.resource;
        draw.index_buffer.offset += index.offset;
        draw.index_buffer.size -= index.offset;
        draw.index_count = index.index_count;
        draw.index_type = index.type;
        draw.base_vertex = index.base_vertex;
    }
    return true;
}

} // namespace vernon::runtime
