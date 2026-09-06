#include "program_graphics_executor.h"

#include <algorithm>

namespace vernon::runtime {
namespace {

const char *textureFormatName(VernonTextureFormat format) {
    switch (format) {
    case VERNON_TEXTURE_RGBA8_UNORM:
        return "rgba8_unorm";
    case VERNON_TEXTURE_RGBA8_SRGB:
        return "rgba8_srgb";
    case VERNON_TEXTURE_RGBA16_FLOAT:
        return "rgba16_float";
    case VERNON_TEXTURE_RGBA32_FLOAT:
        return "rgba32_float";
    case VERNON_TEXTURE_R8_UNORM:
        return "r8_unorm";
    case VERNON_TEXTURE_R16_FLOAT:
        return "r16_float";
    case VERNON_TEXTURE_R32_FLOAT:
        return "r32_float";
    case VERNON_TEXTURE_RG8_UNORM:
        return "rg8_unorm";
    case VERNON_TEXTURE_RGB8_UNORM:
        return "rgb8_unorm";
    case VERNON_TEXTURE_R11G11B10_FLOAT:
        return "r11g11b10_float";
    case VERNON_TEXTURE_D32_FLOAT:
        return "d32_float";
    case VERNON_TEXTURE_D32_FLOAT_S8_UINT:
        return "d32_float_s8_uint";
    }
    return "unknown";
}

std::string formatList(const std::vector<VernonTextureFormat> &formats) {
    std::string text;
    for (const VernonTextureFormat format : formats) {
        if (!text.empty())
            text += ", ";
        text += textureFormatName(format);
    }
    return text;
}

bool checkAttachmentFormat(const program::GraphicsAttachmentSignature &signature, VernonTextureFormat bound,
                           const char *aspect, std::string &error) {
    if (signature.formats.empty())
        return error = std::string("compiled ") + aspect + " attachment signature declares no format", false;
    if (std::find(signature.formats.begin(), signature.formats.end(), bound) != signature.formats.end())
        return true;
    error = std::string("bound ") + aspect + " attachment format " + textureFormatName(bound) +
            " is not one this Program was compiled for (" + formatList(signature.formats) + ")";
    return false;
}

} // namespace

bool bindProgramGraphicsControlResources(const program::Program &program, const program::Graph &graph,
                                         const program::ResolvedExecutionPlan &resolved,
                                         ad::LogicalValueFrame &invocation,
                                         const ResolveProgramRenderPass &resolveRenderPass, std::string &error) {
    const std::optional<program::GraphDirection> direction = program::graphDirection(graph.direction);
    if (!direction)
        return error = "resolved graphics graph has an invalid direction", false;
    for (const program::Node &node : graph.nodes) {
        if (program::executionKind(node) != program::ExecutionKind::Graphics)
            continue;
        const program::ResolvedNodePlan *nodePlan = resolved.node(*direction, node.id);
        const auto *controls = nodePlan ? std::get_if<program::ResolvedGraphicsControls>(&nodePlan->controls) : nullptr;
        if (!controls)
            return error = "graphics Node has no resolved operation controls", false;
        const program::GraphicsOperation &graphics = program::graphicsOperation(node);
        const VernonRenderPass *renderPass = resolveRenderPass(controls->renderPassControl);
        if (!renderPass)
            return error = "managed graphics node has no bound RenderPass control", false;
        if (renderPass->color_attachment_count != graphics.colorAttachments.size())
            return error = "managed graphics fragment outputs must exactly match the color attachments", false;
        const auto bind = [&](const program::ResolvedGraphicsAttachment &attachment) {
            VernonRuntimeProviderResourceReference view{};
            if (attachment.aspects & VERNON_IMAGE_ASPECT_COLOR) {
                const auto signature = std::find_if(graphics.colorAttachments.begin(), graphics.colorAttachments.end(),
                                                    [&](const program::GraphicsAttachmentSignature &candidate) {
                                                        return candidate.location == attachment.location;
                                                    });
                if (signature == graphics.colorAttachments.end())
                    return error = "resolved color control resource has no attachment signature", false;
                const size_t index = static_cast<size_t>(signature - graphics.colorAttachments.begin());
                view = renderPass->color_attachments[index].view;
            } else {
                if (!renderPass->depth_attachment)
                    return error = "managed graphics node requires a depth-stencil attachment", false;
                view = renderPass->depth_attachment->view;
            }
            return invocation.bindControlImageStorage(program, attachment.storage, view, error);
        };
        for (const program::ResolvedGraphicsAttachment &attachment : controls->colorAttachments)
            if (!bind(attachment))
                return false;
        if (controls->depthStencilAttachment && !bind(*controls->depthStencilAttachment))
            return false;
    }
    return true;
}

bool checkProgramGraphicsAttachmentSignature(const program::GraphicsOperation &graphics,
                                             const PlannedGraphicsInvocation &plan, std::string &error) {
    if (plan.attachmentFormats.size() != graphics.colorAttachments.size())
        return error = "bound color attachment count does not match the compiled Program", false;
    for (size_t index = 0; index < plan.attachmentFormats.size(); ++index) {
        // The planner has already sorted the attachments and required contiguous locations from zero.
        const auto signature = std::find_if(
            graphics.colorAttachments.begin(), graphics.colorAttachments.end(),
            [&](const program::GraphicsAttachmentSignature &candidate) { return candidate.location == index; });
        if (signature == graphics.colorAttachments.end())
            return error = "bound color attachment has no compiled attachment signature", false;
        if (!checkAttachmentFormat(*signature, plan.attachmentFormats[index], "color", error))
            return false;
    }
    if (!graphics.depthStencilAttachment.has_value())
        return true;
    if (!plan.depthAttachment)
        return error = "compiled Program requires a depth-stencil attachment", false;
    return checkAttachmentFormat(*graphics.depthStencilAttachment, plan.depthFormat, "depth", error);
}

} // namespace vernon::runtime
