#ifndef VERNON_RUNTIME_PREPARED_GRAPHICS_DRAW_H
#define VERNON_RUNTIME_PREPARED_GRAPHICS_DRAW_H

#include "graphics_invocation_planner.h"
#include "pipeline_metadata.h"

#include <string>
#include <vector>

namespace vernon::runtime {

// `colorFormats` and `depthFormat` are the provider's native formats of `targets`; `vertexStrides` is indexed by
// vertex buffer binding.
bool prepareGraphicsVariantKey(const VernonGraphicsState &graphicsState, const PlannedRenderTargets &targets,
                               std::vector<uint32_t> colorFormats, uint32_t depthFormat,
                               const std::vector<uint32_t> &vertexStrides, GraphicsVariantKey &key, std::string &error);
bool prepareGraphicsDrawInvocation(const PlannedGraphicsDraw &draw, VernonPrimitiveTopology topology,
                                   VernonRuntimeCoreDrawInvocation &invocation, std::string &error);

} // namespace vernon::runtime

#endif
