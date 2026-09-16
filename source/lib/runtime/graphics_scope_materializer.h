#ifndef VERNON_RUNTIME_GRAPHICS_SCOPE_MATERIALIZER_H
#define VERNON_RUNTIME_GRAPHICS_SCOPE_MATERIALIZER_H

#include "graphics_invocation_planner.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace vernon::runtime {

enum class GraphicsScopeMaterialization {
    Begin,
    Fuse,
    Split,
};

class GraphicsScopeMaterializer {
public:
    GraphicsScopeMaterialization materialize(uint32_t candidateRegion, const PlannedGraphicsInvocation &invocation,
                                             bool interrupted = false);
    void reset();

private:
    struct ColorState {
        uint32_t location{};
        VernonRuntimeProviderResourceReference view{};
        VernonRuntimeProviderStoreOperation store{};
    };

    struct DepthState {
        VernonRuntimeProviderResourceReference view{};
        VernonRuntimeProviderStoreOperation depthStore{};
        VernonRuntimeProviderStoreOperation stencilStore{};
    };

    bool canFuse(const PlannedGraphicsInvocation &next) const;
    void capture(const PlannedGraphicsInvocation &invocation);

    std::optional<uint32_t> candidateRegion_;
    std::vector<ColorState> colors_;
    std::vector<VernonTextureFormat> colorFormats_;
    uint32_t attachmentWidth_{};
    uint32_t attachmentHeight_{};
    VernonTextureFormat depthFormat_{};
    std::optional<DepthState> depth_;
};

} // namespace vernon::runtime

#endif
