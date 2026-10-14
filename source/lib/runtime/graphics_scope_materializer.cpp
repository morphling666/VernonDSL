#include "graphics_scope_materializer.h"

#include "VernonFormats.hpp"

#include <algorithm>
#include <iterator>
#include <utility>

namespace vernon::runtime {
namespace {

bool sameResource(VernonRuntimeProviderResourceReference left, VernonRuntimeProviderResourceReference right) {
    return left.identity == right.identity && left.resource.value == right.resource.value &&
           left.offset == right.offset && left.size == right.size;
}

void storeLast(GraphicsRenderScope &scope, const PlannedRenderTargets &last) {
    for (size_t index = 0; index < scope.colors.size(); ++index)
        scope.colors[index].store_operation = last.attachments[index]->store_operation;
    if (scope.depth) {
        scope.depth->depth_store_operation = last.depthAttachment->store_operation;
        if (vernon::formats::hasStencil(scope.depthFormat))
            scope.depth->stencil_store_operation = last.depthAttachment->stencil_store_operation;
    }
}

void addRead(GraphicsScopeReads &reads, VernonRuntimeProviderResourceReference image,
             const ProviderImageSubresources &subresources) {
    if (std::any_of(reads.images.begin(), reads.images.end(),
                    [&](VernonRuntimeProviderResourceReference read) { return sameResource(read, image); }))
        return;
    reads.images.push_back(image);
    reads.subresources.push_back(subresources);
}

} // namespace

void addGraphicsScopeReads(GraphicsScopeReads &reads, const PlannedGraphicsInvocation &invocation) {
    for (const auto &[binding, resource] : invocation.sampledResources)
        addRead(reads, resource.imageView, resource.subresources);
}

bool graphicsScopeReadsOverlap(const GraphicsScopeReads &reads, const PlannedRenderTargets &targets) {
    return std::any_of(
        reads.subresources.begin(), reads.subresources.end(), [&](const ProviderImageSubresources &read) {
            return std::any_of(targets.attachmentSubresources.begin(), targets.attachmentSubresources.end(),
                               [&](const ProviderImageSubresources &attachment) {
                                   return providerImageSubresourcesOverlap(read, attachment);
                               });
        });
}

GraphicsRenderScope graphicsRenderScope(const PlannedRenderTargets &targets, GraphicsScopeReads reads) {
    GraphicsRenderScope scope;
    scope.colors.reserve(targets.attachments.size());
    for (const VernonColorAttachment *source : targets.attachments) {
        VernonRuntimeProviderColorAttachment target{};
        target.location = source->location;
        target.view = source->view;
        target.load_operation = source->load_operation;
        target.store_operation = source->store_operation;
        target.clear_color = source->clear_color;
        scope.colors.push_back(target);
    }
    scope.colorFormats = targets.attachmentFormats;
    scope.depthFormat = targets.depthFormat;
    if (const VernonDepthAttachment *source = targets.depthAttachment) {
        const bool stencil = vernon::formats::hasStencil(targets.depthFormat);
        scope.depth = VernonRuntimeProviderDepthStencilAttachment{
            source->view,
            source->load_operation,
            source->store_operation,
            source->clear_depth,
            stencil ? source->stencil_load_operation : VERNON_RUNTIME_PROVIDER_LOAD_DISCARD,
            stencil ? source->stencil_store_operation : VERNON_RUNTIME_PROVIDER_STORE_DISCARD,
            stencil ? source->clear_stencil : 0};
    }
    scope.attachmentSubresources = targets.attachmentSubresources;
    scope.reads = std::move(reads);
    scope.width = targets.attachmentWidth;
    scope.height = targets.attachmentHeight;
    return scope;
}

VernonRuntimeProviderRenderingDescriptor graphicsRenderingDescriptor(const GraphicsRenderScope &scope) {
    VernonRuntimeProviderRenderingDescriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.color_attachments = scope.colors.data();
    descriptor.color_attachment_count = scope.colors.size();
    descriptor.depth_stencil_attachment = scope.depth ? &*scope.depth : nullptr;
    descriptor.render_area[2] = scope.width;
    descriptor.render_area[3] = scope.height;
    descriptor.layers = 1;
    descriptor.sampled_images = scope.reads.images.data();
    descriptor.sampled_image_count = scope.reads.images.size();
    return descriptor;
}

// A node whose reads overlap its own attachments never reaches the materializer, so a node fusing into the scope,
// which renders to the scope's attachments, cannot sample them either.
bool GraphicsScopeMaterializer::canFuse(const PlannedRenderTargets &next) const {
    if (!candidateRegion_)
        return false;
    if (scope_.colors.size() != next.attachments.size() || scope_.colorFormats != next.attachmentFormats ||
        scope_.width != next.attachmentWidth || scope_.height != next.attachmentHeight ||
        scope_.depthFormat != next.depthFormat || scope_.depth.has_value() != bool(next.depthAttachment))
        return false;
    for (size_t index = 0; index < scope_.colors.size(); ++index) {
        const VernonRuntimeProviderColorAttachment &previous = scope_.colors[index];
        const VernonColorAttachment &following = *next.attachments[index];
        if (previous.location != following.location || !sameResource(previous.view, following.view) ||
            previous.store_operation != VERNON_RUNTIME_PROVIDER_STORE_PRESERVE ||
            following.load_operation != VERNON_RUNTIME_PROVIDER_LOAD_PRESERVE)
            return false;
    }
    if (scope_.depth) {
        const VernonRuntimeProviderDepthStencilAttachment &previous = *scope_.depth;
        const VernonDepthAttachment &following = *next.depthAttachment;
        const bool stencil = vernon::formats::hasStencil(scope_.depthFormat);
        if (!sameResource(previous.view, following.view) ||
            previous.depth_store_operation != VERNON_RUNTIME_PROVIDER_STORE_PRESERVE ||
            following.load_operation != VERNON_RUNTIME_PROVIDER_LOAD_PRESERVE ||
            (stencil && (previous.stencil_store_operation != VERNON_RUNTIME_PROVIDER_STORE_PRESERVE ||
                         following.stencil_load_operation != VERNON_RUNTIME_PROVIDER_LOAD_PRESERVE)))
            return false;
    }
    return true;
}

GraphicsScopeMaterialization GraphicsScopeMaterializer::materialize(uint32_t candidateRegion,
                                                                    const PlannedRenderTargets &targets,
                                                                    const GraphicsScopeReads &reads, bool interrupted) {
    if (interrupted)
        reset();
    const GraphicsScopeMaterialization result = !candidateRegion_ || candidateRegion_ != candidateRegion
                                                    ? GraphicsScopeMaterialization::Begin
                                                : canFuse(targets) ? GraphicsScopeMaterialization::Fuse
                                                                   : GraphicsScopeMaterialization::Split;
    candidateRegion_ = candidateRegion;
    if (result == GraphicsScopeMaterialization::Fuse) {
        storeLast(scope_, targets);
        for (size_t index = 0; index < reads.images.size(); ++index)
            addRead(scope_.reads, reads.images[index], reads.subresources[index]);
    } else {
        scope_ = graphicsRenderScope(targets, reads);
    }
    return result;
}

void GraphicsScopeMaterializer::reset() {
    candidateRegion_.reset();
    scope_ = {};
}

} // namespace vernon::runtime
