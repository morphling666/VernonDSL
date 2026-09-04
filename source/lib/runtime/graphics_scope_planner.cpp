#include "graphics_scope_planner.h"

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

bool GraphicsScopePlanner::canAppend(const PlannedGraphicsInvocation &next) const {
    if (!current_)
        return false;
    const PlannedGraphicsInvocation &current = *current_;
    if (current.attachments.size() != next.attachments.size() || current.attachmentFormats != next.attachmentFormats ||
        current.attachmentWidth != next.attachmentWidth || current.attachmentHeight != next.attachmentHeight ||
        current.depthFormat != next.depthFormat || bool(current.depthAttachment) != bool(next.depthAttachment))
        return false;
    for (size_t index = 0; index < current.attachments.size(); ++index) {
        const VernonColorAttachment &previous = *current.attachments[index];
        const VernonColorAttachment &following = *next.attachments[index];
        if (previous.location != following.location || !sameResource(previous.view, following.view) ||
            previous.store_operation != VERNON_RUNTIME_PROVIDER_STORE_PRESERVE ||
            following.load_operation != VERNON_RUNTIME_PROVIDER_LOAD_PRESERVE || samplesAttachment(next, previous.view))
            return false;
    }
    if (current.depthAttachment) {
        const VernonDepthAttachment &previous = *current.depthAttachment;
        const VernonDepthAttachment &following = *next.depthAttachment;
        if (!sameResource(previous.view, following.view) ||
            previous.store_operation != VERNON_RUNTIME_PROVIDER_STORE_PRESERVE ||
            following.load_operation != VERNON_RUNTIME_PROVIDER_LOAD_PRESERVE ||
            previous.stencil_store_operation != VERNON_RUNTIME_PROVIDER_STORE_PRESERVE ||
            following.stencil_load_operation != VERNON_RUNTIME_PROVIDER_LOAD_PRESERVE ||
            samplesAttachment(next, previous.view))
            return false;
    }
    return true;
}

void GraphicsScopePlanner::append(const PlannedGraphicsInvocation &invocation) { current_ = &invocation; }

void GraphicsScopePlanner::reset() { current_ = nullptr; }

} // namespace vernon::runtime
