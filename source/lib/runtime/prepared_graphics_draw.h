#ifndef VERNON_RUNTIME_PREPARED_GRAPHICS_DRAW_H
#define VERNON_RUNTIME_PREPARED_GRAPHICS_DRAW_H

#include "graphics_invocation_planner.h"
#include "pipeline_metadata.h"

#include <string>
#include <vector>

namespace vernon::runtime {

// One draw a device records, its counts and dynamic state already validated.
struct DrawInvocation {
    uint32_t vertexCount{};
    uint32_t instanceCount{};
    uint32_t firstInstance{};
    // Indexed draws only: added to each index before vertex fetch.
    int32_t baseVertex{};
    uint32_t viewport[4]{};
    uint32_t scissor[4]{};
    uint32_t topology{};
    // The whole 32-bit index buffer; the draw reads indexCount indices from firstIndex on. A zero indexCount marks a
    // non-indexed draw.
    VernonRuntimeProviderResourceReference indexBuffer{};
    uint32_t firstIndex{};
    uint32_t indexCount{};
    uint32_t indexType{};
    uint32_t stencilReference{};
};

// `colorFormats` and `depthFormat` are the provider's native formats of `targets`; `vertexStrides` is indexed by
// vertex buffer binding.
bool prepareGraphicsVariantKey(const VernonGraphicsState &graphicsState, const PlannedRenderTargets &targets,
                               std::vector<uint32_t> colorFormats, uint32_t depthFormat,
                               const std::vector<uint32_t> &vertexStrides, GraphicsVariantKey &key, std::string &error);
bool prepareGraphicsDrawInvocation(const PlannedGraphicsDraw &draw, VernonPrimitiveTopology topology,
                                   DrawInvocation &invocation, std::string &error);

} // namespace vernon::runtime

#endif
