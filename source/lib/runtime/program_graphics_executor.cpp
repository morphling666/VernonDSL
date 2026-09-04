#include "program_graphics_executor.h"

#include <algorithm>

namespace vernon::runtime {

bool bindProgramGraphicsControlResources(const program::Program &program, const program::Graph &graph,
                                         const program::ResolvedGraph &resolved, ad::ProgramInvocationFrame &invocation,
                                         const ResolveProgramRenderPass &resolveRenderPass, std::string &error) {
    for (const program::ResolvedGraph::ControlResourceProjection &projection : resolved.controlResources) {
        if (projection.node >= graph.nodes.size())
            return error = "resolved control resource references an unknown graph node", false;
        const program::GraphicsOperation &graphics = program::graphicsOperation(graph.nodes[projection.node]);
        const VernonRenderPass *renderPass = resolveRenderPass(projection.control);
        if (!renderPass)
            return error = "managed graphics node has no bound RenderPass control", false;
        if (renderPass->color_attachment_count != graphics.colorAttachments.size())
            return error = "managed graphics fragment outputs must exactly match the color attachments", false;
        VernonRuntimeProviderResourceReference view{};
        if (projection.aspects & VERNON_IMAGE_ASPECT_COLOR) {
            const auto attachment = std::find_if(graphics.colorAttachments.begin(), graphics.colorAttachments.end(),
                                                 [&](const program::GraphicsAttachmentSignature &candidate) {
                                                     return candidate.location == projection.location;
                                                 });
            if (attachment == graphics.colorAttachments.end())
                return error = "resolved color control resource has no attachment signature", false;
            const size_t index = static_cast<size_t>(attachment - graphics.colorAttachments.begin());
            view = renderPass->color_attachments[index].view;
        } else {
            if (!renderPass->depth_attachment)
                return error = "managed graphics node requires a depth-stencil attachment", false;
            view = renderPass->depth_attachment->view;
        }
        if (!invocation.bindControlImageStorage(program, projection.storage, view, error))
            return false;
    }
    return true;
}

} // namespace vernon::runtime
