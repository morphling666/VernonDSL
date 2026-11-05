#include "program_graphics_executor.h"

#include "VernonFormats.hpp"

#include <algorithm>

namespace vernon::runtime {
namespace {

std::string textureFormatName(VernonTextureFormat format) {
    const vernon::formats::Format *entry = vernon::formats::find(format);
    return entry ? std::string(entry->name) : "unknown";
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

bool checkProgramGraphicsAttachmentSignature(const program::GraphicsOperation &graphics,
                                             const PlannedRenderTargets &plan, std::string &error) {
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

void ProgramGraphicsState::describe(const program::GraphicsOperation &operation) {
    const program::GraphicsPipelineState &source = operation.pipelineState;
    colorBlends.clear();
    colorBlends.reserve(source.colorBlends.size());
    for (size_t location = 0; location < source.colorBlends.size(); ++location)
        colorBlends.push_back(source.colorBlends.at(static_cast<uint32_t>(location)));
    state = {};
    state.struct_size = sizeof(state);
    state.topology = source.topology;
    state.rasterization = source.rasterization;
    state.depth_stencil = source.depthStencil;
    state.color_blends = colorBlends.data();
    state.color_blend_count = colorBlends.size();
}

} // namespace vernon::runtime
