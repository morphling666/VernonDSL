#include "VernonExecutionGraph.h"

#include "rhi/logical_resource_record.h"
#include "rhi/rhi_internal.h"

#include <algorithm>
#include <unordered_set>

namespace vernon::execution {
namespace {

bool writes(AccessMode access) { return access != AccessMode::Read; }

} // namespace

bool ExecutionGraph::validateDeclarations(std::string &error) const {
    for (size_t index = 0; index < resourceRecords_.size(); ++index)
        if (!resourceRecords_[index].resourceKey) {
            error = "execution graph resource " + std::to_string(index) + " is stale";
            return false;
        }
    const auto validateResource = [&](const std::string &passName, const GraphResource &resource) {
        if (resource.graphIdentity != graphIdentity_ || resource.id >= resources_.size() ||
            resources_[resource.id].graphIdentity != graphIdentity_ || resources_[resource.id].kind != resource.kind) {
            error = "pass '" + passName + "' references resource " + std::to_string(resource.id) +
                    " from another execution graph";
            return false;
        }
        return true;
    };
    const auto validLoad = [](VernonRhiLoadOperation operation) {
        return operation >= VERNON_RHI_LOAD_CLEAR && operation <= VERNON_RHI_LOAD_DISCARD;
    };
    const auto validStore = [](VernonRhiStoreOperation operation) {
        return operation >= VERNON_RHI_STORE_PRESERVE && operation <= VERNON_RHI_STORE_DISCARD;
    };
    const auto validateImage = [&](const std::string &passName, const GraphImage &image) {
        if (!validateResource(passName, image) || image.kind != ResourceKind::Image)
            return false;
        const detail::ExecutionResourceRecord &record = resourceRecords_[image.id];
        VernonRhiImageViewDescriptor viewDescriptor{};
        VernonRhiImageDescriptor imageDescriptor{};
        uint64_t parentKey = 0;
        bool retainedView = true;
        if (!record.imageViewKeys.empty()) {
            const uint64_t viewKey = vernon::rhi::encodeResourceKey(image.view);
            retainedView =
                viewKey &&
                std::find(record.imageViewKeys.begin(), record.imageViewKeys.end(), viewKey) !=
                    record.imageViewKeys.end() &&
                vernon::rhi::describeImageViewResource(device_, viewKey, viewDescriptor, imageDescriptor, parentKey) &&
                parentKey == record.resourceKey;
        } else {
            viewDescriptor.format = image.format;
            viewDescriptor.base_mip_level = image.subresources.base_mip_level;
            viewDescriptor.mip_level_count = image.subresources.mip_level_count;
            viewDescriptor.base_array_layer = image.subresources.base_array_layer;
            viewDescriptor.array_layer_count = image.subresources.array_layer_count;
            viewDescriptor.aspects = image.subresources.aspects;
            imageDescriptor.width = image.width << image.subresources.base_mip_level;
            imageDescriptor.height = image.height << image.subresources.base_mip_level;
            imageDescriptor.sample_count = image.samples;
        }
        if (record.image.index != image.handle.index || record.image.generation != image.handle.generation ||
            !retainedView || image.format != viewDescriptor.format ||
            image.width != std::max(imageDescriptor.width >> viewDescriptor.base_mip_level, 1u) ||
            image.height != std::max(imageDescriptor.height >> viewDescriptor.base_mip_level, 1u) ||
            image.layers != viewDescriptor.array_layer_count || image.samples != imageDescriptor.sample_count ||
            image.subresources.base_mip_level != viewDescriptor.base_mip_level ||
            image.subresources.mip_level_count != viewDescriptor.mip_level_count ||
            image.subresources.base_array_layer != viewDescriptor.base_array_layer ||
            image.subresources.array_layer_count != viewDescriptor.array_layer_count ||
            image.subresources.aspects != viewDescriptor.aspects) {
            error = "pass '" + passName + "' contains an invalid image resource " + std::to_string(image.id);
            return false;
        }
        return true;
    };
    std::unordered_set<const ExecutionPass *> owned;
    std::unordered_set<std::string> names;
    for (const auto &pass : passes_) {
        if (!pass || pass->name().empty() || !names.insert(pass->name()).second) {
            error = "execution graph pass names must be non-empty and unique";
            return false;
        }
        owned.insert(pass.get());
        for (const ResourceUse &use : pass->uses_) {
            if (!validateResource(pass->name(), use.resource))
                return false;
            if (use.resource.kind == ResourceKind::Image) {
                if (!use.image || use.image->id != use.resource.id ||
                    use.image->graphIdentity != use.resource.graphIdentity) {
                    error = "pass '" + pass->name() + "' discards image view metadata for resource " +
                            std::to_string(use.resource.id);
                    return false;
                }
                if (!validateImage(pass->name(), *use.image))
                    return false;
            }
            const uint32_t validStages = VERNON_RHI_STAGE_COMPUTE | VERNON_RHI_STAGE_VERTEX | VERNON_RHI_STAGE_FRAGMENT;
            if ((use.stageMask & ~validStages) != 0 || use.state <= VERNON_RHI_STATE_UNDEFINED ||
                use.state > VERNON_RHI_STATE_PRESENT ||
                (use.stageMask != 0 && use.state != VERNON_RHI_STATE_SHADER_READ &&
                 use.state != VERNON_RHI_STATE_SHADER_WRITE) ||
                (use.state == VERNON_RHI_STATE_TRANSFER_SOURCE && use.access != AccessMode::Read) ||
                (use.state == VERNON_RHI_STATE_TRANSFER_DESTINATION && use.access == AccessMode::Read) ||
                (use.state == VERNON_RHI_STATE_SHADER_READ && writes(use.access)) ||
                (use.state == VERNON_RHI_STATE_SHADER_WRITE && !writes(use.access)) ||
                (use.state == VERNON_RHI_STATE_PRESENT && use.access != AccessMode::Read) ||
                (use.imageRole != ImageUseRole::None &&
                 (!use.imageSubresources.mip_level_count || !use.imageSubresources.array_layer_count ||
                  !use.imageSubresources.aspects)) ||
                ((use.state == VERNON_RHI_STATE_COLOR_ATTACHMENT ||
                  use.state == VERNON_RHI_STATE_DEPTH_STENCIL_ATTACHMENT || use.state == VERNON_RHI_STATE_PRESENT) &&
                 use.resource.kind != ResourceKind::Image)) {
                error = "pass '" + pass->name() + "' declares an incompatible state/access for resource " +
                        std::to_string(use.resource.id);
                return false;
            }
        }
        const auto *render = dynamic_cast<const RenderPass *>(pass.get());
        if (!render)
            continue;
        if (provider_ == detail::ExecutionProvider::Cpu) {
            error = "CPU execution graphs support compute passes only";
            return false;
        }
        if ((render->colors_.empty() && !render->depth_) || render->colors_.size() > 8) {
            error = "render pass '" + pass->name() + "' must declare between one and eight attachments";
            return false;
        }
        uint32_t width = 0;
        uint32_t height = 0;
        uint32_t layers = 0;
        uint32_t samples = 0;
        std::unordered_set<uint32_t> locations;
        std::unordered_set<uint32_t> attachmentResources;
        for (const ColorAttachmentUse &color : render->colors_) {
            if (!validateImage(pass->name(), color.image))
                return false;
            if (color.image.format == VERNON_RHI_FORMAT_D32_FLOAT ||
                color.image.format == VERNON_RHI_FORMAT_D32_FLOAT_S8_UINT || color.location >= 8 ||
                !locations.insert(color.location).second || !attachmentResources.insert(color.image.id).second ||
                !validLoad(color.load) || !validStore(color.store)) {
                error = "render pass '" + pass->name() + "' contains invalid color attachment resource " +
                        std::to_string(color.image.id);
                return false;
            }
            if (!width) {
                width = color.image.width;
                height = color.image.height;
                layers = color.image.layers;
                samples = color.image.samples;
            } else if (width != color.image.width || height != color.image.height || layers != color.image.layers ||
                       samples != color.image.samples) {
                error = "render pass '" + pass->name() + "' color attachment resource " +
                        std::to_string(color.image.id) + " has an incompatible extent or sample count";
                return false;
            }
        }
        if (render->depth_) {
            const DepthStencilAttachmentUse &depth = *render->depth_;
            if (!validateImage(pass->name(), depth.image))
                return false;
            if ((depth.image.format != VERNON_RHI_FORMAT_D32_FLOAT &&
                 depth.image.format != VERNON_RHI_FORMAT_D32_FLOAT_S8_UINT) ||
                !attachmentResources.insert(depth.image.id).second || !validLoad(depth.depthLoad) ||
                !validStore(depth.depthStore) || !validLoad(depth.stencilLoad) || !validStore(depth.stencilStore) ||
                depth.clearDepth < 0.0f || depth.clearDepth > 1.0f ||
                (depth.image.format == VERNON_RHI_FORMAT_D32_FLOAT &&
                 (depth.stencilLoad != VERNON_RHI_LOAD_DISCARD || depth.stencilStore != VERNON_RHI_STORE_DISCARD ||
                  depth.clearStencil != 0)) ||
                (depth.readOnlyDepth &&
                 (depth.depthLoad == VERNON_RHI_LOAD_CLEAR || depth.depthStore == VERNON_RHI_STORE_DISCARD)) ||
                (depth.readOnlyStencil &&
                 (depth.stencilLoad == VERNON_RHI_LOAD_CLEAR || depth.stencilStore == VERNON_RHI_STORE_DISCARD))) {
                error = "render pass '" + pass->name() + "' contains invalid depth attachment resource " +
                        std::to_string(depth.image.id);
                return false;
            }
            if (!width) {
                width = depth.image.width;
                height = depth.image.height;
                layers = depth.image.layers;
                samples = depth.image.samples;
            } else if (width != depth.image.width || height != depth.image.height || layers != depth.image.layers ||
                       samples != depth.image.samples) {
                error = "render pass '" + pass->name() + "' depth attachment resource " +
                        std::to_string(depth.image.id) + " is incompatible with its colors";
                return false;
            }
        }
        if (!render->renderArea_[2] || !render->renderArea_[3] || render->renderArea_[0] > width ||
            render->renderArea_[1] > height || render->renderArea_[2] > width - render->renderArea_[0] ||
            render->renderArea_[3] > height - render->renderArea_[1]) {
            const uint32_t resourceId =
                !render->colors_.empty() ? render->colors_.front().image.id : render->depth_->image.id;
            error = "render pass '" + pass->name() + "' render area exceeds attachment resource " +
                    std::to_string(resourceId);
            return false;
        }
    }
    for (const auto &pass : passes_)
        for (const ExecutionPass *dependency : pass->dependencies_)
            if (!owned.count(dependency)) {
                error = "pass '" + pass->name() + "' has an external dependency";
                return false;
            }
    for (const CompiledScope &scope : scopes_)
        if (scope.passIndices.empty()) {
            error = "compiled execution scope is empty";
            return false;
        }
    return true;
}

} // namespace vernon::execution
