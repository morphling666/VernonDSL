#ifndef VERNON_RUNTIME_GRAPHICS_SCOPE_MATERIALIZER_H
#define VERNON_RUNTIME_GRAPHICS_SCOPE_MATERIALIZER_H

#include "graphics_invocation_planner.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace vernon::runtime {

// The distinct images the draws of a render scope sample, with the subresources each covers at the same index.
struct GraphicsScopeReads {
    std::vector<VernonRuntimeProviderResourceReference> images;
    std::vector<ProviderImageSubresources> subresources;
};

void addGraphicsScopeReads(GraphicsScopeReads &reads, const PlannedGraphicsInvocation &invocation);
// Whether a read covers a subresource that `targets` render to; a scope cannot sample its own attachments.
bool graphicsScopeReadsOverlap(const GraphicsScopeReads &reads, const PlannedRenderTargets &targets);

// The attachments and reads of one render scope. A fused scope takes its load operations from its first node, its
// store operations from its last node, and the reads of every node. A depth-only attachment has no stencil contents,
// so its stencil operations discard.
struct GraphicsRenderScope {
    std::vector<VernonRuntimeProviderColorAttachment> colors;
    std::vector<VernonTextureFormat> colorFormats;
    std::optional<VernonRuntimeProviderDepthStencilAttachment> depth;
    VernonTextureFormat depthFormat{};
    std::vector<ProviderImageSubresources> attachmentSubresources;
    GraphicsScopeReads reads;
    uint32_t width{};
    uint32_t height{};
};

GraphicsRenderScope graphicsRenderScope(const PlannedRenderTargets &targets, GraphicsScopeReads reads);

// Points into `scope`, which must outlive the descriptor.
VernonRuntimeProviderRenderingDescriptor graphicsRenderingDescriptor(const GraphicsRenderScope &scope);

enum class GraphicsScopeMaterialization {
    Begin,
    Fuse,
    Split,
};

class GraphicsScopeMaterializer {
public:
    // Materializes a node rendering to `targets` whose draws sample `reads`, which must not overlap `targets`.
    GraphicsScopeMaterialization materialize(uint32_t candidateRegion, const PlannedRenderTargets &targets,
                                             const GraphicsScopeReads &reads, bool interrupted = false);
    void reset();
    // The scope of the last materialized node, extended by every node fused into it.
    const GraphicsRenderScope &scope() const { return scope_; }

private:
    bool canFuse(const PlannedRenderTargets &next) const;

    std::optional<uint32_t> candidateRegion_;
    GraphicsRenderScope scope_;
};

} // namespace vernon::runtime

#endif
