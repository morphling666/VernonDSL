#ifndef VERNON_RUNTIME_PREPARED_GRAPHICS_DRAW_H
#define VERNON_RUNTIME_PREPARED_GRAPHICS_DRAW_H

#include "graphics_invocation_planner.h"
#include "pipeline_metadata.h"

#include <string>
#include <vector>

namespace vernon::runtime {

struct PreparedGraphicsDraw {
    PlannedGraphicsState graphicsState;
    std::vector<uint32_t> vertexStrides;
    GraphicsVariantKey variantKey;
    VernonRuntimeCoreDrawInvocation invocation{};
};

bool prepareGraphicsDraw(const VernonStageInvocationDescriptor &invocation, const PlannedGraphicsInvocation &plan,
                         std::vector<uint32_t> attachmentFormats, uint32_t depthFormat,
                         const std::vector<VernonRuntimeProviderBindingLayoutEntry> &layout,
                         const std::vector<VernonRuntimeProviderBindingValue> &values, PreparedGraphicsDraw &prepared,
                         std::string &error);

} // namespace vernon::runtime

#endif
