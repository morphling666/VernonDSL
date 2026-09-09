#include "graphics_scope_materializer.h"

#include <algorithm>

namespace vernon::runtime {
namespace {

bool sameResource(VernonRuntimeProviderResourceReference left, VernonRuntimeProviderResourceReference right) {
    return left.identity == right.identity && left.resource.value == right.resource.value &&
           left.offset == right.offset && left.size == right.size;
}

bool samplesAttachment(const PlannedGraphicsInvocation &invocation, VernonRuntimeProviderResourceReference attachment) {
    return std::any_of(invocation.sampledResources.begin(), invocation.sampledResources.end(),
                       [&](const auto &entry) { return sameResource(entry.second.imageView, attachment); });
}

} // namespace

bool GraphicsScopeMaterializer::canFuse(const PlannedGraphicsInvocation &next) const {
    if (!candidateRegion_)
        return false;
    if (colors_.size() != next.attachments.size() || colorFormats_ != next.attachmentFormats ||
        attachmentWidth_ != next.attachmentWidth || attachmentHeight_ != next.attachmentHeight ||
        depthFormat_ != next.depthFormat || depth_.has_value() != bool(next.depthAttachment))
        return false;
    for (size_t index = 0; index < colors_.size(); ++index) {
        const ColorState &previous = colors_[index];
        const VernonColorAttachment &following = *next.attachments[index];
        if (previous.location != following.location || !sameResource(previous.view, following.view) ||
            previous.store != VERNON_RUNTIME_PROVIDER_STORE_PRESERVE ||
            following.load_operation != VERNON_RUNTIME_PROVIDER_LOAD_PRESERVE || samplesAttachment(next, previous.view))
            return false;
    }
    if (depth_) {
        const DepthState &previous = *depth_;
        const VernonDepthAttachment &following = *next.depthAttachment;
        if (!sameResource(previous.view, following.view) ||
            previous.depthStore != VERNON_RUNTIME_PROVIDER_STORE_PRESERVE ||
            following.load_operation != VERNON_RUNTIME_PROVIDER_LOAD_PRESERVE ||
            previous.stencilStore != VERNON_RUNTIME_PROVIDER_STORE_PRESERVE ||
            following.stencil_load_operation != VERNON_RUNTIME_PROVIDER_LOAD_PRESERVE ||
            samplesAttachment(next, previous.view))
            return false;
    }
    return true;
}

GraphicsScopeMaterialization GraphicsScopeMaterializer::materialize(uint32_t candidateRegion,
                                                                    const PlannedGraphicsInvocation &invocation,
                                                                    bool interrupted) {
    if (interrupted)
        reset();
    const GraphicsScopeMaterialization result = !candidateRegion_ || candidateRegion_ != candidateRegion
                                                    ? GraphicsScopeMaterialization::Begin
                                                : canFuse(invocation) ? GraphicsScopeMaterialization::Fuse
                                                                      : GraphicsScopeMaterialization::Split;
    candidateRegion_ = candidateRegion;
    capture(invocation);
    return result;
}

void GraphicsScopeMaterializer::capture(const PlannedGraphicsInvocation &invocation) {
    colors_.clear();
    colors_.reserve(invocation.attachments.size());
    for (const VernonColorAttachment *attachment : invocation.attachments)
        colors_.push_back({attachment->location, attachment->view, attachment->store_operation});
    colorFormats_ = invocation.attachmentFormats;
    attachmentWidth_ = invocation.attachmentWidth;
    attachmentHeight_ = invocation.attachmentHeight;
    depthFormat_ = invocation.depthFormat;
    if (invocation.depthAttachment)
        depth_ = DepthState{invocation.depthAttachment->view, invocation.depthAttachment->store_operation,
                            invocation.depthAttachment->stencil_store_operation};
    else
        depth_.reset();
}

void GraphicsScopeMaterializer::reset() {
    candidateRegion_.reset();
    colors_.clear();
    colorFormats_.clear();
    attachmentWidth_ = 0;
    attachmentHeight_ = 0;
    depthFormat_ = {};
    depth_.reset();
}

} // namespace vernon::runtime
