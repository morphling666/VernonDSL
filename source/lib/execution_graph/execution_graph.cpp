#include "VernonExecutionGraph.h"

#include "../rhi/rhi_internal.h"

#include <algorithm>
#include <atomic>
#include <queue>
#include <unordered_map>
#include <unordered_set>

namespace vernon::execution {
namespace {

bool writes(AccessMode access) { return access != AccessMode::Read; }

uint64_t handleKey(uint32_t index, uint32_t generation) { return (static_cast<uint64_t>(generation) << 32u) | index; }

std::atomic<uint64_t> nextGraphIdentity{1};

uint32_t accessBits(const ResourceUse &use) {
    const bool reads = use.access != AccessMode::Write;
    const bool writesResource = use.access != AccessMode::Read;
    if (use.state == VERNON_RHI_STATE_COLOR_ATTACHMENT)
        return (reads ? VERNON_RHI_ACCESS_COLOR_READ : 0) | (writesResource ? VERNON_RHI_ACCESS_COLOR_WRITE : 0);
    if (use.state == VERNON_RHI_STATE_DEPTH_STENCIL_ATTACHMENT)
        return (reads ? VERNON_RHI_ACCESS_DEPTH_STENCIL_READ : 0) |
               (writesResource ? VERNON_RHI_ACCESS_DEPTH_STENCIL_WRITE : 0);
    if (use.state == VERNON_RHI_STATE_TRANSFER_SOURCE)
        return VERNON_RHI_ACCESS_TRANSFER_READ;
    if (use.state == VERNON_RHI_STATE_TRANSFER_DESTINATION)
        return VERNON_RHI_ACCESS_TRANSFER_WRITE;
    if (use.state == VERNON_RHI_STATE_SHADER_READ || use.state == VERNON_RHI_STATE_SHADER_WRITE)
        return (reads ? VERNON_RHI_ACCESS_SHADER_READ : 0) | (writesResource ? VERNON_RHI_ACCESS_SHADER_WRITE : 0);
    return VERNON_RHI_ACCESS_NONE;
}

bool sameView(const GraphImage &left, const GraphImage &right) {
    return left.graphIdentity == right.graphIdentity && left.id == right.id &&
           left.handle.index == right.handle.index && left.handle.generation == right.handle.generation &&
           left.view.index == right.view.index && left.view.generation == right.view.generation &&
           left.width == right.width && left.height == right.height && left.layers == right.layers &&
           left.samples == right.samples && left.format == right.format;
}

bool preservesBoundary(VernonRhiStoreOperation previousStore, VernonRhiLoadOperation nextLoad) {
    return nextLoad == VERNON_RHI_LOAD_CLEAR ||
           (nextLoad == VERNON_RHI_LOAD_PRESERVE && previousStore == VERNON_RHI_STORE_PRESERVE);
}

bool attachmentUse(const ResourceUse &use) {
    return use.state == VERNON_RHI_STATE_COLOR_ATTACHMENT || use.state == VERNON_RHI_STATE_DEPTH_STENCIL_ATTACHMENT;
}

bool hasUnrepresentableHazard(const RenderPass &left, const RenderPass &right) {
    for (const ResourceUse &first : left.uses())
        for (const ResourceUse &second : right.uses())
            if (first.resource.id == second.resource.id && (writes(first.access) || writes(second.access)) &&
                !(attachmentUse(first) && attachmentUse(second) && first.state == second.state))
                return true;
    return false;
}

bool compatible(const RenderPass &left, const RenderPass &right) {
    if ((left.flags() & PassNoMerge) || (right.flags() & PassNoMerge) ||
        left.colors().size() != right.colors().size() ||
        static_cast<bool>(left.depthAttachment()) != static_cast<bool>(right.depthAttachment()))
        return false;
    for (size_t index = 0; index < left.colors().size(); ++index)
        if (left.colors()[index].location != right.colors()[index].location ||
            !sameView(left.colors()[index].image, right.colors()[index].image) ||
            !preservesBoundary(left.colors()[index].store, right.colors()[index].load))
            return false;
    if (left.depthAttachment()) {
        const auto &previous = *left.depthAttachment();
        const auto &next = *right.depthAttachment();
        if (!sameView(previous.image, next.image) || previous.readOnlyDepth != next.readOnlyDepth ||
            previous.readOnlyStencil != next.readOnlyStencil || !preservesBoundary(previous.depthStore, next.depthLoad))
            return false;
    }
    return std::equal(left.renderAreaData(), left.renderAreaData() + 4, right.renderAreaData());
}

} // namespace

ExecutionPass::ExecutionPass(std::string name) : name_(std::move(name)) {}

void ExecutionPass::dependsOn(ExecutionPass &dependency) {
    if (std::find(dependencies_.begin(), dependencies_.end(), &dependency) == dependencies_.end()) {
        dependencies_.push_back(&dependency);
        if (owner_)
            owner_->dirty_ = true;
    }
}

void ExecutionPass::setFlags(uint32_t flags) {
    if (flags_ == flags)
        return;
    flags_ = flags;
    if (owner_)
        owner_->dirty_ = true;
}

void ExecutionPass::read(GraphResource resource, VernonRhiResourceState state, uint32_t stageMask) {
    uses_.push_back({resource, AccessMode::Read, state, stageMask});
}

void ExecutionPass::write(GraphResource resource, VernonRhiResourceState state, uint32_t stageMask) {
    uses_.push_back({resource, AccessMode::Write, state, stageMask});
}

void ExecutionPass::readWrite(GraphResource resource, VernonRhiResourceState state, uint32_t stageMask) {
    uses_.push_back({resource, AccessMode::ReadWrite, state, stageMask});
}

void ExecutionPass::resetDeclaration() { uses_.clear(); }

void RenderPass::color(uint32_t location, const ColorAttachmentUse &attachment) {
    auto value = attachment;
    value.location = location;
    colors_.push_back(value);
    if (value.load == VERNON_RHI_LOAD_PRESERVE)
        readWrite(value.image, VERNON_RHI_STATE_COLOR_ATTACHMENT);
    else
        write(value.image, VERNON_RHI_STATE_COLOR_ATTACHMENT);
}

void RenderPass::depth(const DepthStencilAttachmentUse &attachment) {
    depth_ = std::make_unique<DepthStencilAttachmentUse>(attachment);
    if ((attachment.image.format == VERNON_RHI_FORMAT_D32_FLOAT && attachment.readOnlyDepth) ||
        (attachment.readOnlyDepth && attachment.readOnlyStencil))
        read(attachment.image, VERNON_RHI_STATE_DEPTH_STENCIL_ATTACHMENT);
    else
        readWrite(attachment.image, VERNON_RHI_STATE_DEPTH_STENCIL_ATTACHMENT);
}

void RenderPass::renderArea(uint32_t x, uint32_t y, uint32_t width, uint32_t height) {
    renderArea_[0] = x;
    renderArea_[1] = y;
    renderArea_[2] = width;
    renderArea_[3] = height;
}

ExecutionGraph::ExecutionGraph(VernonRhiDevice device)
    : device_(device), graphIdentity_(nextGraphIdentity.fetch_add(1, std::memory_order_relaxed)) {}
ExecutionGraph::~ExecutionGraph() {
    for (const ResourceRecord &record : resourceRecords_)
        if (record.resourceKey && record.resourceKey != UINT64_MAX)
            vernon::rhi::releaseResource(device_,
                                         record.resource.kind == ResourceKind::Buffer
                                             ? vernon::rhi::ResourceKind::Buffer
                                             : vernon::rhi::ResourceKind::Image,
                                         record.resourceKey);
}

GraphBuffer ExecutionGraph::importBuffer(VernonRhiBuffer buffer, bool exported) {
    const uint64_t key = handleKey(buffer.index, buffer.generation);
    if (const auto found = importedBuffers_.find(key); found != importedBuffers_.end()) {
        ResourceRecord &record = resourceRecords_[found->second];
        record.exported = record.exported || exported;
        GraphBuffer result;
        result.id = found->second;
        result.kind = ResourceKind::Buffer;
        result.graphIdentity = graphIdentity_;
        result.handle = buffer;
        return result;
    }
    GraphBuffer result;
    result.id = static_cast<uint32_t>(resources_.size());
    result.kind = ResourceKind::Buffer;
    result.graphIdentity = graphIdentity_;
    result.handle = buffer;
    resources_.push_back(result);
    resourceRecords_.push_back({result, exported});
    importedBuffers_.emplace(key, result.id);
    resourceRecords_.back().buffer = buffer;
    if (!vernon::rhi::deviceExists(device_)) {
        resourceRecords_.back().resourceKey = UINT64_MAX;
        dirty_ = true;
        return result;
    }
    resourceRecords_.back().resourceKey = vernon::rhi::bufferResource(device_, buffer);
    if (!resourceRecords_.back().resourceKey ||
        !vernon::rhi::retainResource(device_, vernon::rhi::ResourceKind::Buffer, resourceRecords_.back().resourceKey))
        resourceRecords_.back().resourceKey = 0;
    dirty_ = true;
    return result;
}

GraphImage ExecutionGraph::importImage(VernonRhiImage image, VernonRhiImageView view, VernonRhiFormat format,
                                       uint32_t width, uint32_t height, uint32_t layers, uint32_t samples,
                                       bool exported) {
    const uint64_t key = handleKey(image.index, image.generation);
    if (const auto found = importedImages_.find(key); found != importedImages_.end()) {
        ResourceRecord &record = resourceRecords_[found->second];
        record.exported = record.exported || exported;
        GraphImage result;
        result.id = found->second;
        result.kind = ResourceKind::Image;
        result.graphIdentity = graphIdentity_;
        result.handle = image;
        result.view = view;
        result.format = format;
        result.width = width;
        result.height = height;
        result.layers = layers;
        result.samples = samples;
        return result;
    }
    GraphImage result;
    result.id = static_cast<uint32_t>(resources_.size());
    result.kind = ResourceKind::Image;
    result.graphIdentity = graphIdentity_;
    result.handle = image;
    result.view = view;
    result.format = format;
    result.width = width;
    result.height = height;
    result.layers = layers;
    result.samples = samples;
    resources_.push_back(result);
    resourceRecords_.push_back({result, exported});
    importedImages_.emplace(key, result.id);
    resourceRecords_.back().image = image;
    if (!vernon::rhi::deviceExists(device_)) {
        resourceRecords_.back().resourceKey = UINT64_MAX;
        dirty_ = true;
        return result;
    }
    resourceRecords_.back().resourceKey = vernon::rhi::imageResource(device_, image);
    if (!resourceRecords_.back().resourceKey ||
        !vernon::rhi::retainResource(device_, vernon::rhi::ResourceKind::Image, resourceRecords_.back().resourceKey))
        resourceRecords_.back().resourceKey = 0;
    dirty_ = true;
    return result;
}

bool ExecutionGraph::compile(std::string &error) {
    error.clear();
    schedule_.clear();
    scopes_.clear();
    bool compiled = false;
    struct FailedCompileCleanup {
        std::vector<uint32_t> &schedule;
        std::vector<CompiledScope> &scopes;
        bool &compiled;
        ~FailedCompileCleanup() {
            if (!compiled) {
                schedule.clear();
                scopes.clear();
            }
        }
    } cleanup{schedule_, scopes_, compiled};
    const size_t count = passes_.size();
    std::unordered_map<ExecutionPass *, uint32_t> indices;
    for (uint32_t index = 0; index < count; ++index) {
        indices.emplace(passes_[index].get(), index);
        passes_[index]->resetDeclaration();
        if (auto *render = dynamic_cast<RenderPass *>(passes_[index].get())) {
            render->colors_.clear();
            render->depth_.reset();
            std::fill(std::begin(render->renderArea_), std::end(render->renderArea_), 0);
        }
        passes_[index]->declare();
        if (auto *render = dynamic_cast<RenderPass *>(passes_[index].get())) {
            std::sort(render->colors_.begin(), render->colors_.end(),
                      [](const auto &left, const auto &right) { return left.location < right.location; });
            if (render->renderArea_[2] == 0 && (!render->colors_.empty() || render->depth_)) {
                const GraphImage &image =
                    !render->colors_.empty() ? render->colors_.front().image : render->depth_->image;
                render->renderArea_[2] = image.width;
                render->renderArea_[3] = image.height;
            }
        }
    }
    if (!validate(error))
        return false;

    std::vector<std::vector<bool>> edges(count, std::vector<bool>(count));
    for (uint32_t index = 0; index < count; ++index) {
        for (ExecutionPass *dependency : passes_[index]->dependencies_) {
            auto found = indices.find(dependency);
            if (found == indices.end()) {
                error = "pass '" + passes_[index]->name() + "' depends on a pass outside this graph";
                return false;
            }
            if (found->second == index) {
                error = "pass '" + passes_[index]->name() + "' depends on itself";
                return false;
            }
            edges[found->second][index] = true;
        }
    }
    for (uint32_t left = 0; left < count; ++left)
        for (uint32_t right = left + 1; right < count; ++right)
            for (const ResourceUse &first : passes_[left]->uses_)
                for (const ResourceUse &second : passes_[right]->uses_)
                    if (first.resource.id == second.resource.id && (writes(first.access) || writes(second.access)))
                        edges[left][right] = true;

    std::vector<bool> live(count);
    std::vector<uint32_t> work;
    for (uint32_t index = 0; index < count; ++index) {
        bool root = (passes_[index]->flags() & (PassNeverCull | PassSideEffect)) != 0;
        for (const ResourceUse &use : passes_[index]->uses_)
            if (writes(use.access) && use.resource.id < resourceRecords_.size() &&
                resourceRecords_[use.resource.id].exported)
                root = true;
        if (root) {
            live[index] = true;
            work.push_back(index);
        }
    }
    while (!work.empty()) {
        const uint32_t current = work.back();
        work.pop_back();
        for (uint32_t dependency = 0; dependency < count; ++dependency)
            if (edges[dependency][current] && !live[dependency]) {
                live[dependency] = true;
                work.push_back(dependency);
            }
    }

    std::vector<uint32_t> indegree(count);
    for (uint32_t left = 0; left < count; ++left)
        if (live[left])
            for (uint32_t right = 0; right < count; ++right)
                if (live[right] && edges[left][right])
                    ++indegree[right];
    std::priority_queue<uint32_t, std::vector<uint32_t>, std::greater<uint32_t>> ready;
    for (uint32_t index = 0; index < count; ++index)
        if (live[index] && indegree[index] == 0)
            ready.push(index);
    while (!ready.empty()) {
        const uint32_t current = ready.top();
        ready.pop();
        schedule_.push_back(current);
        for (uint32_t next = 0; next < count; ++next)
            if (live[next] && edges[current][next] && --indegree[next] == 0)
                ready.push(next);
    }
    if (schedule_.size() != static_cast<size_t>(std::count(live.begin(), live.end(), true))) {
        error = "execution graph contains a dependency cycle";
        return false;
    }

    for (uint32_t passIndex : schedule_) {
        auto *render = dynamic_cast<RenderPass *>(passes_[passIndex].get());
        if (render && !scopes_.empty() && scopes_.back().rendering) {
            auto *previous = dynamic_cast<RenderPass *>(passes_[scopes_.back().passIndices.back()].get());
            bool canMerge = previous && compatible(*previous, *render);
            for (uint32_t previousIndex : scopes_.back().passIndices) {
                const auto *scopePass = static_cast<const RenderPass *>(passes_[previousIndex].get());
                if (hasUnrepresentableHazard(*scopePass, *render)) {
                    canMerge = false;
                    break;
                }
            }
            if (canMerge) {
                scopes_.back().passIndices.push_back(passIndex);
                continue;
            }
        }
        scopes_.push_back({render != nullptr, {passIndex}});
    }
    struct LastUse {
        ResourceUse use;
        bool present{};
    };
    std::vector<LastUse> lastUses(resources_.size());
    for (CompiledScope &scope : scopes_) {
        std::vector<ResourceUse> firstUses(resources_.size());
        std::vector<ResourceUse> finalUses(resources_.size());
        std::vector<bool> firstUsePresent(resources_.size());
        std::vector<bool> finalUsePresent(resources_.size());
        for (uint32_t passIndex : scope.passIndices) {
            const uint32_t stageMask = dynamic_cast<ComputePass *>(passes_[passIndex].get())
                                           ? VERNON_RHI_STAGE_COMPUTE
                                           : VERNON_RHI_STAGE_VERTEX | VERNON_RHI_STAGE_FRAGMENT;
            for (const ResourceUse &use : passes_[passIndex]->uses_) {
                ResourceUse effective = use;
                if (!effective.stageMask && (effective.state == VERNON_RHI_STATE_SHADER_READ ||
                                             effective.state == VERNON_RHI_STATE_SHADER_WRITE))
                    effective.stageMask = stageMask;
                if (!firstUsePresent[use.resource.id]) {
                    firstUses[use.resource.id] = effective;
                    firstUsePresent[use.resource.id] = true;
                }
                finalUses[use.resource.id] = effective;
                finalUsePresent[use.resource.id] = true;
            }
        }
        for (uint32_t resourceId = 0; resourceId < firstUses.size(); ++resourceId) {
            if (!firstUsePresent[resourceId])
                continue;
            const ResourceUse &use = firstUses[resourceId];
            const LastUse &previous = lastUses[resourceId];
            if (!previous.present ||
                (previous.use.state == use.state && !writes(previous.use.access) && !writes(use.access)))
                continue;
            VernonRhiBarrier barrier{};
            barrier.struct_size = sizeof(barrier);
            barrier.source_stage_mask = previous.use.stageMask;
            barrier.destination_stage_mask = use.stageMask;
            barrier.source_access = accessBits(previous.use);
            barrier.destination_access = accessBits(use);
            barrier.old_state = previous.use.state;
            barrier.new_state = use.state;
            const ResourceRecord &record = resourceRecords_[resourceId];
            barrier.is_image = record.resource.kind == ResourceKind::Image;
            if (barrier.is_image)
                barrier.image = record.image;
            else
                barrier.buffer = record.buffer;
            scope.barriers.push_back(barrier);
        }
        for (uint32_t resourceId = 0; resourceId < finalUses.size(); ++resourceId)
            if (finalUsePresent[resourceId])
                lastUses[resourceId] = {finalUses[resourceId], true};
    }
    if (!validate(error))
        return false;
    dirty_ = false;
    compiled = true;
    return true;
}

bool ExecutionGraph::validate(std::string &error) const {
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
        const ResourceRecord &record = resourceRecords_[image.id];
        if (record.image.index != image.handle.index || record.image.generation != image.handle.generation ||
            image.view.index == static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX) || !image.width ||
            !image.height || !image.layers || !image.samples || image.format == VERNON_RHI_FORMAT_UNDEFINED) {
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
            if (color.image.format == VERNON_RHI_FORMAT_D32_FLOAT || color.location >= 8 ||
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
            if (depth.image.format != VERNON_RHI_FORMAT_D32_FLOAT ||
                !attachmentResources.insert(depth.image.id).second || !validLoad(depth.depthLoad) ||
                !validStore(depth.depthStore) || !validLoad(depth.stencilLoad) || !validStore(depth.stencilStore) ||
                depth.clearDepth < 0.0f || depth.clearDepth > 1.0f ||
                (depth.readOnlyDepth &&
                 (depth.depthLoad == VERNON_RHI_LOAD_CLEAR || depth.depthStore == VERNON_RHI_STORE_DISCARD))) {
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

VernonRhiStatus ExecutionGraph::execute() {
    lastStats_ = {};
    std::string error;
    if ((dirty_ && !compile(error)) || !validate(error))
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    VernonRhiCommandEncoderDescriptor encoderDescriptor{};
    encoderDescriptor.struct_size = sizeof(encoderDescriptor);
    encoderDescriptor.required_capabilities = VERNON_RHI_QUEUE_COMPUTE | VERNON_RHI_QUEUE_GRAPHICS;
    VernonRhiCommandEncoder native{static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
    VernonRhiStatus status = vernonRhiDeviceCreateCommandEncoder(device_, &encoderDescriptor, &native);
    if (status != VERNON_RHI_STATUS_OK)
        return status;
    ExecutionResources resources(resources_);
    for (const CompiledScope &scope : scopes_) {
        if (!scope.barriers.empty()) {
            status = vernonRhiCommandEncoderBarrier(device_, native, scope.barriers.data(), scope.barriers.size());
            if (status != VERNON_RHI_STATUS_OK)
                break;
        }
        if (scope.rendering) {
            auto *first = static_cast<RenderPass *>(passes_[scope.passIndices.front()].get());
            auto *last = static_cast<RenderPass *>(passes_[scope.passIndices.back()].get());
            std::vector<VernonRhiColorAttachment> colors;
            colors.reserve(first->colors_.size());
            for (size_t index = 0; index < first->colors_.size(); ++index) {
                const auto &begin = first->colors_[index];
                const auto &finish = last->colors_[index];
                VernonRhiColorAttachment attachment{};
                attachment.view = begin.image.view;
                attachment.location = begin.location;
                attachment.initial_state = VERNON_RHI_STATE_COLOR_ATTACHMENT;
                attachment.final_state = VERNON_RHI_STATE_COLOR_ATTACHMENT;
                attachment.load_operation = begin.load;
                attachment.store_operation = finish.store;
                std::copy(std::begin(begin.clear), std::end(begin.clear), attachment.clear_color);
                colors.push_back(attachment);
            }
            VernonRhiDepthStencilAttachment depth{};
            if (first->depth_) {
                depth.view = first->depth_->image.view;
                depth.initial_state = VERNON_RHI_STATE_DEPTH_STENCIL_ATTACHMENT;
                depth.final_state = VERNON_RHI_STATE_DEPTH_STENCIL_ATTACHMENT;
                depth.depth_load_operation = first->depth_->depthLoad;
                depth.depth_store_operation = last->depth_->depthStore;
                depth.clear_depth = first->depth_->clearDepth;
                depth.stencil_load_operation = first->depth_->stencilLoad;
                depth.stencil_store_operation = last->depth_->stencilStore;
                depth.clear_stencil = first->depth_->clearStencil;
                depth.read_only_depth = first->depth_->readOnlyDepth;
                depth.read_only_stencil = first->depth_->readOnlyStencil;
            }
            VernonRhiRenderingDescriptor descriptor{};
            descriptor.struct_size = sizeof(descriptor);
            descriptor.color_attachments = colors.data();
            descriptor.color_attachment_count = colors.size();
            descriptor.depth_stencil_attachment = first->depth_ ? &depth : nullptr;
            descriptor.offset_x = first->renderArea_[0];
            descriptor.offset_y = first->renderArea_[1];
            descriptor.width = first->renderArea_[2];
            descriptor.height = first->renderArea_[3];
            descriptor.layers =
                !first->colors_.empty() ? first->colors_.front().image.layers : first->depth_->image.layers;
            status = vernonRhiCommandEncoderBeginRendering(device_, native, &descriptor);
            if (status != VERNON_RHI_STATUS_OK)
                break;
            GraphicsEncoder encoder(device_, native);
            for (size_t scopeIndex = 0; scopeIndex < scope.passIndices.size(); ++scopeIndex) {
                auto *renderPass = static_cast<RenderPass *>(passes_[scope.passIndices[scopeIndex]].get());
                if (scopeIndex) {
                    for (const auto &color : renderPass->colors_)
                        if (color.load == VERNON_RHI_LOAD_CLEAR) {
                            status = vernonRhiCommandEncoderClearColorAttachment(device_, native, color.location,
                                                                                 color.clear);
                            if (status != VERNON_RHI_STATUS_OK)
                                break;
                        }
                    if (status == VERNON_RHI_STATUS_OK && renderPass->depth_) {
                        uint32_t aspects = 0;
                        if (renderPass->depth_->depthLoad == VERNON_RHI_LOAD_CLEAR)
                            aspects |= VERNON_RHI_ATTACHMENT_DEPTH;
                        if (renderPass->depth_->stencilLoad == VERNON_RHI_LOAD_CLEAR)
                            aspects |= VERNON_RHI_ATTACHMENT_STENCIL;
                        if (aspects)
                            status = vernonRhiCommandEncoderClearDepthStencilAttachment(
                                device_, native, renderPass->depth_->clearDepth, renderPass->depth_->clearStencil,
                                aspects);
                    }
                }
                if (status == VERNON_RHI_STATUS_OK)
                    status = renderPass->execute(encoder, resources);
                if (status != VERNON_RHI_STATUS_OK)
                    break;
            }
            const VernonRhiStatus endStatus = vernonRhiCommandEncoderEndRendering(device_, native);
            if (status == VERNON_RHI_STATUS_OK)
                status = endStatus;
        } else {
            ComputeEncoder encoder(device_, native);
            status = static_cast<ComputePass *>(passes_[scope.passIndices.front()].get())->execute(encoder, resources);
        }
        if (status != VERNON_RHI_STATUS_OK)
            break;
    }
    if (status == VERNON_RHI_STATUS_OK)
        status = vernonRhiCommandEncoderFinish(device_, native);
    if (status == VERNON_RHI_STATUS_OK)
        status = vernonRhiDeviceSubmit(device_, native);
    if (status == VERNON_RHI_STATUS_OK)
        status = vernonRhiCommandEncoderGetStats(device_, native, &lastStats_);
    const VernonRhiStatus destroyStatus = vernonRhiDeviceDestroyCommandEncoder(device_, native);
    return status == VERNON_RHI_STATUS_OK ? destroyStatus : status;
}

} // namespace vernon::execution
