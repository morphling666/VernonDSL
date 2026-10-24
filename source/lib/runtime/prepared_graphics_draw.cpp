#include "prepared_graphics_draw.h"

#include "VernonFormats.hpp"

#include <algorithm>

namespace vernon::runtime {

bool prepareGraphicsVariantKey(const VernonGraphicsState &graphicsState, const PlannedRenderTargets &targets,
                               std::vector<uint32_t> colorFormats, uint32_t depthFormat,
                               const std::vector<uint32_t> &vertexStrides, GraphicsVariantKey &key,
                               std::string &error) {
    if ((!targets.depthAttachment && targets.attachments.empty()) ||
        targets.attachments.size() > VERNON_RUNTIME_PROVIDER_MAX_COLOR_ATTACHMENTS) {
        error = "graphics draw has invalid attachments or target bindings";
        return false;
    }
    const bool hasStencil = targets.depthAttachment && vernon::formats::hasStencil(targets.depthFormat);
    PlannedGraphicsState state;
    if (!planGraphicsState(&graphicsState, colorFormats.size(), targets.depthAttachment != nullptr, hasStencil, state,
                           error))
        return false;
    key = {static_cast<uint32_t>(state.topology),
           std::move(colorFormats),
           depthFormat,
           1,
           vertexStrides,
           state.rasterization,
           state.depthStencil,
           std::move(state.colorBlends)};
    return true;
}

bool prepareGraphicsDrawInvocation(const PlannedGraphicsDraw &plan, VernonPrimitiveTopology topology,
                                   VernonRuntimeCoreDrawInvocation &draw, std::string &error) {
    draw = {};
    draw.struct_size = sizeof(draw);
    draw.vertex_count = plan.vertexCount;
    draw.instance_count = plan.instanceCount;
    draw.first_instance = plan.firstInstance;
    draw.stencil_reference = plan.stencilReference;
    std::copy(std::begin(plan.viewport), std::end(plan.viewport), draw.viewport);
    std::copy(std::begin(plan.scissor), std::end(plan.scissor), draw.scissor);
    draw.topology = topology;
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
