#include "VernonExecutionGraph.h"

#include "rhi/rhi_internal.h"

#include <algorithm>
#include <atomic>
#include <memory>
#include <mutex>
#include <queue>
#include <unordered_map>
#include <unordered_set>

namespace vernon::execution {
namespace {

bool writes(AccessMode access) { return access != AccessMode::Read; }

ImageUseRole imageRole(VernonRhiResourceState state) {
    switch (state) {
    case VERNON_RHI_STATE_SHADER_READ:
        return ImageUseRole::Sampled;
    case VERNON_RHI_STATE_SHADER_WRITE:
        return ImageUseRole::Storage;
    case VERNON_RHI_STATE_COLOR_ATTACHMENT:
        return ImageUseRole::ColorAttachment;
    case VERNON_RHI_STATE_DEPTH_STENCIL_ATTACHMENT:
        return ImageUseRole::DepthAttachment;
    case VERNON_RHI_STATE_TRANSFER_SOURCE:
    case VERNON_RHI_STATE_TRANSFER_DESTINATION:
        return ImageUseRole::Transfer;
    default:
        return ImageUseRole::None;
    }
}

bool intervalsOverlap(uint32_t leftBase, uint32_t leftCount, uint32_t rightBase, uint32_t rightCount) {
    const uint64_t leftEnd = leftCount == UINT32_MAX ? UINT64_MAX : uint64_t{leftBase} + leftCount;
    const uint64_t rightEnd = rightCount == UINT32_MAX ? UINT64_MAX : uint64_t{rightBase} + rightCount;
    return uint64_t{leftBase} < rightEnd && uint64_t{rightBase} < leftEnd;
}

bool usesOverlap(const ResourceUse &left, const ResourceUse &right) {
    if (left.resource.id != right.resource.id)
        return false;
    if (left.resource.kind != ResourceKind::Image || left.imageRole == ImageUseRole::None ||
        right.imageRole == ImageUseRole::None)
        return true;
    const auto &a = left.imageSubresources;
    const auto &b = right.imageSubresources;
    return (a.aspects & b.aspects) != 0 &&
           intervalsOverlap(a.base_mip_level, a.mip_level_count, b.base_mip_level, b.mip_level_count) &&
           intervalsOverlap(a.base_array_layer, a.array_layer_count, b.base_array_layer, b.array_layer_count);
}

VernonRhiImageSubresourceRange intersection(const VernonRhiImageSubresourceRange &left,
                                            const VernonRhiImageSubresourceRange &right) {
    const auto interval = [](uint32_t leftBase, uint32_t leftCount, uint32_t rightBase, uint32_t rightCount) {
        const uint64_t begin = std::max(leftBase, rightBase);
        const uint64_t leftEnd = leftCount == UINT32_MAX ? UINT64_MAX : uint64_t{leftBase} + leftCount;
        const uint64_t rightEnd = rightCount == UINT32_MAX ? UINT64_MAX : uint64_t{rightBase} + rightCount;
        const uint64_t end = std::min(leftEnd, rightEnd);
        return std::pair{static_cast<uint32_t>(begin),
                         end == UINT64_MAX ? UINT32_MAX : static_cast<uint32_t>(end - begin)};
    };
    const auto [baseMip, mipCount] =
        interval(left.base_mip_level, left.mip_level_count, right.base_mip_level, right.mip_level_count);
    const auto [baseLayer, layerCount] =
        interval(left.base_array_layer, left.array_layer_count, right.base_array_layer, right.array_layer_count);
    return {baseMip, mipCount, baseLayer, layerCount, left.aspects & right.aspects};
}

uint64_t handleKey(uint32_t index, uint32_t generation) { return (static_cast<uint64_t>(generation) << 32u) | index; }

std::atomic<uint64_t> nextGraphIdentity{1};
std::mutex executionSessionRegistryMutex;
std::unordered_map<uint64_t, std::weak_ptr<std::recursive_mutex>> executionSessions;

std::shared_ptr<std::recursive_mutex> executionSession(VernonRhiDevice device) {
    const uint64_t key = handleKey(device.index, device.generation);
    std::lock_guard<std::mutex> lock(executionSessionRegistryMutex);
    if (const auto found = executionSessions.find(key); found != executionSessions.end())
        if (auto existing = found->second.lock())
            return existing;
    auto created = std::make_shared<std::recursive_mutex>();
    executionSessions[key] = created;
    return created;
}

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
            if (usesOverlap(first, second) && (writes(first.access) || writes(second.access)) &&
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

class DeviceExecutionSession::Impl {
public:
    explicit Impl(VernonRhiDevice device) : mutex(executionSession(device)), lock(*mutex) {}

private:
    std::shared_ptr<std::recursive_mutex> mutex;
    std::unique_lock<std::recursive_mutex> lock;
};

DeviceExecutionSession::DeviceExecutionSession(VernonRhiDevice device) : impl_(std::make_unique<Impl>(device)) {}
DeviceExecutionSession::~DeviceExecutionSession() = default;
DeviceExecutionSession::DeviceExecutionSession(DeviceExecutionSession &&) noexcept = default;
DeviceExecutionSession &DeviceExecutionSession::operator=(DeviceExecutionSession &&) noexcept = default;

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

void ExecutionPass::read(GraphImage resource, VernonRhiResourceState state, uint32_t stageMask) {
    uses_.push_back({resource, AccessMode::Read, state, stageMask, imageRole(state), resource.subresources});
}

void ExecutionPass::write(GraphResource resource, VernonRhiResourceState state, uint32_t stageMask) {
    uses_.push_back({resource, AccessMode::Write, state, stageMask});
}

void ExecutionPass::write(GraphImage resource, VernonRhiResourceState state, uint32_t stageMask) {
    uses_.push_back({resource, AccessMode::Write, state, stageMask, imageRole(state), resource.subresources});
}

void ExecutionPass::readWrite(GraphResource resource, VernonRhiResourceState state, uint32_t stageMask) {
    uses_.push_back({resource, AccessMode::ReadWrite, state, stageMask});
}

void ExecutionPass::readWrite(GraphImage resource, VernonRhiResourceState state, uint32_t stageMask) {
    uses_.push_back({resource, AccessMode::ReadWrite, state, stageMask, imageRole(state), resource.subresources});
}
void ExecutionPass::resetDeclaration() { uses_.clear(); }

const std::shared_ptr<const ExecutionBindingValue> &ExecutionBindings::at(ExecutionParameter parameter) const {
    if (parameter.graphIdentity != graphIdentity_ || parameter.id >= values_.size() || !values_[parameter.id])
        throw std::invalid_argument("execution parameter does not belong to this binding snapshot");
    return values_[parameter.id];
}

ExecutionBindingsBuilder::ExecutionBindingsBuilder(uint64_t graphIdentity, size_t parameterCount)
    : graphIdentity_(graphIdentity), values_(parameterCount) {}

void ExecutionBindingsBuilder::set(ExecutionParameter parameter, std::shared_ptr<const ExecutionBindingValue> value) {
    if (parameter.graphIdentity != graphIdentity_ || parameter.id >= values_.size())
        throw std::invalid_argument("execution parameter does not belong to this binding builder");
    if (!value)
        throw std::invalid_argument("execution parameter binding value must not be null");
    if (values_[parameter.id] == value)
        return;
    values_[parameter.id] = std::move(value);
    cached_.reset();
}

std::shared_ptr<const ExecutionBindings> ExecutionBindingsBuilder::snapshot() const {
    if (std::any_of(values_.begin(), values_.end(), [](const auto &value) { return !value; }))
        throw std::invalid_argument("execution parameter bindings are incomplete");
    if (!cached_)
        cached_ = std::shared_ptr<const ExecutionBindings>(new ExecutionBindings(graphIdentity_, values_));
    return cached_;
}

bool ExecutionResources::buffer(GraphBuffer resource, VernonRhiBuffer &output) const {
    output = {static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
    if (resource.kind != ResourceKind::Buffer || resource.id >= resources_.size() || resource.id >= buffers_.size())
        return false;
    const GraphResource &resolved = resources_[resource.id];
    if (resolved.graphIdentity != resource.graphIdentity || resolved.kind != ResourceKind::Buffer ||
        buffers_[resource.id].index == VERNON_RHI_INVALID_HANDLE_INDEX)
        return false;
    output = buffers_[resource.id];
    return true;
}

const std::shared_ptr<const ExecutionBindingValue> &ExecutionResources::binding(ExecutionParameter parameter) const {
    if (!bindings_)
        throw std::invalid_argument("execution submission has no parameter bindings");
    return bindings_->at(parameter);
}

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
    if (((attachment.image.format == VERNON_RHI_FORMAT_D32_FLOAT ||
          attachment.image.format == VERNON_RHI_FORMAT_D32_FLOAT_S8_UINT) &&
         attachment.readOnlyDepth) ||
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

ExecutionGraph::ExecutionGraph() : graphIdentity_(nextGraphIdentity.fetch_add(1, std::memory_order_relaxed)) {}
ExecutionGraph::ExecutionGraph(VernonRhiDevice device)
    : provider_(detail::ExecutionProvider::Rhi), device_(device),
      graphIdentity_(nextGraphIdentity.fetch_add(1, std::memory_order_relaxed)) {}
ExecutionGraph::~ExecutionGraph() {
    for (const detail::ExecutionResourceRecord &record : resourceRecords_) {
        if (provider_ == detail::ExecutionProvider::Rhi && record.graphOwned)
            vernonRhiDeviceDestroyBuffer(device_, record.buffer);
        if (provider_ == detail::ExecutionProvider::Rhi)
            for (uint64_t viewKey : record.imageViewKeys)
                vernon::rhi::releaseResource(device_, vernon::rhi::ResourceKind::ImageView, viewKey);
        if (provider_ == detail::ExecutionProvider::Rhi && record.resourceKey)
            vernon::rhi::releaseResource(device_,
                                         record.resource.kind == ResourceKind::Buffer
                                             ? vernon::rhi::ResourceKind::Buffer
                                             : vernon::rhi::ResourceKind::Image,
                                         record.resourceKey);
    }
}

ExecutionParameter ExecutionGraph::parameter(std::string name) {
    if (compiled_)
        throw std::logic_error("cannot mutate a compiled execution graph builder");
    if (name.empty())
        throw std::invalid_argument("execution parameter name must not be empty");
    if (parameterIds_.find(name) != parameterIds_.end())
        throw std::invalid_argument("execution parameter name must be unique");
    const uint32_t id = static_cast<uint32_t>(parameterNames_.size());
    parameterIds_.emplace(name, id);
    parameterNames_.push_back(std::move(name));
    dirty_ = true;
    return {id, graphIdentity_};
}

GraphBuffer ExecutionGraph::importHostBuffer(uint64_t identity, bool exported) {
    if (compiled_ || !identity || provider_ != detail::ExecutionProvider::Cpu)
        return {};
    if (const auto found = importedHostBuffers_.find(identity); found != importedHostBuffers_.end()) {
        detail::ExecutionResourceRecord &record = resourceRecords_[found->second];
        record.exported = record.exported || exported;
        GraphBuffer result;
        result.id = found->second;
        result.kind = ResourceKind::Buffer;
        result.graphIdentity = graphIdentity_;
        return result;
    }
    GraphBuffer result;
    result.id = static_cast<uint32_t>(resources_.size());
    result.kind = ResourceKind::Buffer;
    result.graphIdentity = graphIdentity_;
    resources_.push_back(result);
    resourceRecords_.push_back({result, exported});
    resourceRecords_.back().resourceKey = identity;
    importedHostBuffers_.emplace(identity, result.id);
    dirty_ = true;
    return result;
}

VernonRhiStatus ExecutionGraph::createBuffer(const VernonRhiBufferDescriptor &descriptor, GraphBuffer &output,
                                             bool exported) {
    output = {};
    if (compiled_ || provider_ != detail::ExecutionProvider::Rhi)
        return VERNON_RHI_STATUS_UNSUPPORTED;
    VernonRhiBuffer buffer{};
    const VernonRhiStatus status = vernonRhiDeviceCreateBuffer(device_, &descriptor, &buffer);
    if (status != VERNON_RHI_STATUS_OK)
        return status;
    output = importBuffer(buffer, exported);
    resourceRecords_[output.id].graphOwned = true;
    return VERNON_RHI_STATUS_OK;
}

GraphBuffer ExecutionGraph::importBuffer(VernonRhiBuffer buffer, bool exported) {
    if (compiled_ || provider_ != detail::ExecutionProvider::Rhi || !vernon::rhi::deviceExists(device_))
        return {};
    const uint64_t key = handleKey(buffer.index, buffer.generation);
    if (const auto found = importedBuffers_.find(key); found != importedBuffers_.end()) {
        detail::ExecutionResourceRecord &record = resourceRecords_[found->second];
        record.exported = record.exported || exported;
        GraphBuffer result;
        result.id = found->second;
        result.kind = ResourceKind::Buffer;
        result.graphIdentity = graphIdentity_;
        result.handle = buffer;
        return result;
    }
    const uint64_t resourceKey = vernon::rhi::bufferResource(device_, buffer);
    if (!resourceKey || !vernon::rhi::retainResource(device_, vernon::rhi::ResourceKind::Buffer, resourceKey))
        return {};
    GraphBuffer result;
    result.id = static_cast<uint32_t>(resources_.size());
    result.kind = ResourceKind::Buffer;
    result.graphIdentity = graphIdentity_;
    result.handle = buffer;
    try {
        resources_.reserve(resources_.size() + 1);
        resourceRecords_.reserve(resourceRecords_.size() + 1);
        importedBuffers_.reserve(importedBuffers_.size() + 1);
        if (!importedBuffers_.emplace(key, result.id).second) {
            vernon::rhi::releaseResource(device_, vernon::rhi::ResourceKind::Buffer, resourceKey);
            return {};
        }
    } catch (const std::bad_alloc &) {
        vernon::rhi::releaseResource(device_, vernon::rhi::ResourceKind::Buffer, resourceKey);
        return {};
    }
    resources_.push_back(result);
    resourceRecords_.push_back({result, exported});
    resourceRecords_.back().buffer = buffer;
    resourceRecords_.back().resourceKey = resourceKey;
    dirty_ = true;
    return result;
}

GraphImage ExecutionGraph::importImage(VernonRhiImage image, VernonRhiImageView view, bool exported) {
    if (compiled_ || provider_ != detail::ExecutionProvider::Rhi || !vernon::rhi::deviceExists(device_))
        return {};
    const uint64_t imageKey = vernon::rhi::imageResource(device_, image);
    const uint64_t viewKey = vernon::rhi::imageViewResource(device_, view);
    VernonRhiImageDescriptor imageDescriptor{};
    VernonRhiImageViewDescriptor viewDescriptor{};
    uint64_t parentKey{};
    if (!imageKey || !viewKey ||
        !vernon::rhi::describeImageViewResource(device_, viewKey, viewDescriptor, imageDescriptor, parentKey) ||
        parentKey != imageKey)
        return {};
    GraphImage result;
    result.kind = ResourceKind::Image;
    result.graphIdentity = graphIdentity_;
    result.handle = image;
    result.view = view;
    result.format = viewDescriptor.format;
    result.width = std::max(imageDescriptor.width >> viewDescriptor.base_mip_level, 1u);
    result.height = std::max(imageDescriptor.height >> viewDescriptor.base_mip_level, 1u);
    result.layers = viewDescriptor.array_layer_count;
    result.samples = imageDescriptor.sample_count;
    result.subresources = {viewDescriptor.base_mip_level, viewDescriptor.mip_level_count,
                           viewDescriptor.base_array_layer, viewDescriptor.array_layer_count, viewDescriptor.aspects};
    const uint64_t key = handleKey(image.index, image.generation);
    if (const auto found = importedImages_.find(key); found != importedImages_.end()) {
        detail::ExecutionResourceRecord &record = resourceRecords_[found->second];
        result.id = found->second;
        if (std::find(record.imageViewKeys.begin(), record.imageViewKeys.end(), viewKey) ==
            record.imageViewKeys.end()) {
            try {
                record.imageViewKeys.reserve(record.imageViewKeys.size() + 1);
            } catch (const std::bad_alloc &) {
                return {};
            }
            if (!vernon::rhi::retainResource(device_, vernon::rhi::ResourceKind::ImageView, viewKey))
                return {};
            record.imageViewKeys.push_back(viewKey);
        }
        record.exported |= exported;
        return result;
    }
    try {
        resources_.reserve(resources_.size() + 1);
        resourceRecords_.reserve(resourceRecords_.size() + 1);
        importedImages_.reserve(importedImages_.size() + 1);
    } catch (const std::bad_alloc &) {
        return {};
    }
    result.id = static_cast<uint32_t>(resources_.size());
    detail::ExecutionResourceRecord record{result, exported};
    record.image = image;
    record.resourceKey = imageKey;
    try {
        record.imageViewKeys.push_back(viewKey);
        if (!importedImages_.emplace(key, result.id).second)
            return {};
    } catch (const std::bad_alloc &) {
        return {};
    }
    if (!vernon::rhi::retainResource(device_, vernon::rhi::ResourceKind::Image, imageKey)) {
        importedImages_.erase(key);
        return {};
    }
    if (!vernon::rhi::retainResource(device_, vernon::rhi::ResourceKind::ImageView, viewKey)) {
        vernon::rhi::releaseResource(device_, vernon::rhi::ResourceKind::Image, imageKey);
        importedImages_.erase(key);
        return {};
    }
    resources_.push_back(result);
    resourceRecords_.push_back(std::move(record));
    dirty_ = true;
    return result;
}

bool ExecutionGraph::buildPlan(std::string &error) {
    error.clear();
    if (compiled_) {
        error = "execution graph builder was already compiled";
        return false;
    }
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
                    if (usesOverlap(first, second) && (writes(first.access) || writes(second.access)))
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
    std::vector<std::vector<ResourceUse>> lastUses(resources_.size());
    for (CompiledScope &scope : scopes_) {
        std::vector<std::vector<ResourceUse>> firstUses(resources_.size());
        std::vector<std::vector<ResourceUse>> finalUses(resources_.size());
        for (uint32_t passIndex : scope.passIndices) {
            const uint32_t stageMask = dynamic_cast<ComputePass *>(passes_[passIndex].get())
                                           ? VERNON_RHI_STAGE_COMPUTE
                                           : VERNON_RHI_STAGE_VERTEX | VERNON_RHI_STAGE_FRAGMENT;
            for (const ResourceUse &use : passes_[passIndex]->uses_) {
                ResourceUse effective = use;
                if (!effective.stageMask && (effective.state == VERNON_RHI_STATE_SHADER_READ ||
                                             effective.state == VERNON_RHI_STATE_SHADER_WRITE))
                    effective.stageMask = stageMask;
                auto &current = finalUses[use.resource.id];
                if (std::none_of(current.begin(), current.end(),
                                 [&](const ResourceUse &prior) { return usesOverlap(prior, effective); }))
                    firstUses[use.resource.id].push_back(effective);
                current.erase(std::remove_if(current.begin(), current.end(),
                                             [&](const ResourceUse &prior) { return usesOverlap(prior, effective); }),
                              current.end());
                current.push_back(effective);
            }
        }
        for (uint32_t resourceId = 0; resourceId < firstUses.size(); ++resourceId) {
            for (const ResourceUse &use : firstUses[resourceId])
                for (const ResourceUse &previous : lastUses[resourceId]) {
                    if (!usesOverlap(previous, use) ||
                        (previous.state == use.state && !writes(previous.access) && !writes(use.access)))
                        continue;
                    VernonRhiBarrier barrier{};
                    barrier.struct_size = sizeof(barrier);
                    barrier.source_stage_mask = previous.stageMask;
                    barrier.destination_stage_mask = use.stageMask;
                    barrier.source_access = accessBits(previous);
                    barrier.destination_access = accessBits(use);
                    barrier.old_state = previous.state;
                    barrier.new_state = use.state;
                    const detail::ExecutionResourceRecord &record = resourceRecords_[resourceId];
                    barrier.is_image = record.resource.kind == ResourceKind::Image;
                    if (barrier.is_image) {
                        barrier.image = record.image;
                        barrier.image_subresources = intersection(previous.imageSubresources, use.imageSubresources);
                    } else {
                        barrier.buffer = record.buffer;
                    }
                    scope.barriers.push_back(barrier);
                }
        }
        for (uint32_t resourceId = 0; resourceId < finalUses.size(); ++resourceId)
            for (const ResourceUse &use : finalUses[resourceId]) {
                auto &previous = lastUses[resourceId];
                previous.erase(std::remove_if(previous.begin(), previous.end(),
                                              [&](const ResourceUse &prior) { return usesOverlap(prior, use); }),
                               previous.end());
                previous.push_back(use);
            }
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
        const detail::ExecutionResourceRecord &record = resourceRecords_[image.id];
        if (record.image.index != image.handle.index || record.image.generation != image.handle.generation ||
            image.view.index == static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX) || !image.width ||
            !image.height || !image.layers || !image.samples || image.format == VERNON_RHI_FORMAT_UNDEFINED ||
            !image.subresources.mip_level_count || !image.subresources.array_layer_count ||
            !image.subresources.aspects ||
            (image.subresources.aspects &
             ~(VERNON_RHI_IMAGE_ASPECT_COLOR | VERNON_RHI_IMAGE_ASPECT_DEPTH | VERNON_RHI_IMAGE_ASPECT_STENCIL))) {
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

struct CompiledExecutionGraph::State {
    detail::ExecutionProvider provider{detail::ExecutionProvider::Cpu};
    VernonRhiDevice device{};
    uint64_t graphIdentity{};
    std::vector<std::unique_ptr<ExecutionPass>> passes;
    std::vector<GraphResource> resources;
    std::vector<detail::ExecutionResourceRecord> resourceRecords;
    std::vector<std::string> parameterNames;
    std::vector<uint32_t> schedule;
    std::vector<CompiledScope> scopes;

    ~State() {
        for (const detail::ExecutionResourceRecord &record : resourceRecords) {
            if (provider == detail::ExecutionProvider::Rhi && record.graphOwned)
                vernonRhiDeviceDestroyBuffer(device, record.buffer);
            if (provider == detail::ExecutionProvider::Rhi)
                for (uint64_t viewKey : record.imageViewKeys)
                    vernon::rhi::releaseResource(device, vernon::rhi::ResourceKind::ImageView, viewKey);
            if (provider == detail::ExecutionProvider::Rhi && record.resourceKey)
                vernon::rhi::releaseResource(device,
                                             record.resource.kind == ResourceKind::Buffer
                                                 ? vernon::rhi::ResourceKind::Buffer
                                                 : vernon::rhi::ResourceKind::Image,
                                             record.resourceKey);
        }
    }
};

class ExecutionSubmission::Impl {
public:
    Impl(std::shared_ptr<CompiledExecutionGraph::State> retainedPlan,
         std::shared_ptr<const ExecutionBindings> retainedBindings)
        : plan(std::move(retainedPlan)), bindings(std::move(retainedBindings)) {}
    ~Impl() {
        if (completion.index != VERNON_RHI_INVALID_HANDLE_INDEX)
            (void)vernonRhiDeviceDestroyCompletion(plan->device, completion);
    }

    std::shared_ptr<CompiledExecutionGraph::State> plan;
    std::shared_ptr<const ExecutionBindings> bindings;
    State state{State::Pending};
    VernonRhiStatus status{VERNON_RHI_STATUS_OK};
    VernonRhiCompletion completion{static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
    VernonRhiCommandEncoderStats stats{};
};

std::shared_ptr<CompiledExecutionGraph> ExecutionGraph::compile(std::string &error) {
    if (!buildPlan(error))
        return {};
    std::shared_ptr<CompiledExecutionGraph::State> state;
    try {
        state = std::make_shared<CompiledExecutionGraph::State>();
    } catch (const std::bad_alloc &) {
        error = "cannot allocate compiled execution graph";
        return {};
    }
    state->provider = provider_;
    state->device = device_;
    state->graphIdentity = graphIdentity_;
    state->passes = std::move(passes_);
    state->resources = std::move(resources_);
    state->resourceRecords = std::move(resourceRecords_);
    state->parameterNames = std::move(parameterNames_);
    state->schedule = std::move(schedule_);
    state->scopes = std::move(scopes_);
    for (const auto &pass : state->passes)
        pass->owner_ = nullptr;
    importedBuffers_.clear();
    importedHostBuffers_.clear();
    importedImages_.clear();
    parameterIds_.clear();
    compiled_ = true;
    return std::shared_ptr<CompiledExecutionGraph>(new CompiledExecutionGraph(std::move(state)));
}

ExecutionSubmission::ExecutionSubmission() = default;
ExecutionSubmission::ExecutionSubmission(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
ExecutionSubmission::~ExecutionSubmission() = default;
ExecutionSubmission::ExecutionSubmission(ExecutionSubmission &&) noexcept = default;
ExecutionSubmission &ExecutionSubmission::operator=(ExecutionSubmission &&) noexcept = default;

ExecutionSubmission::State ExecutionSubmission::state() const {
    if (!impl_)
        return State::Failed;
    if (impl_->completion.index == VERNON_RHI_INVALID_HANDLE_INDEX)
        return impl_->state;
    VernonRhiCompletionState state{};
    if (vernonRhiCompletionGetState(impl_->plan->device, impl_->completion, &state) != VERNON_RHI_STATUS_OK)
        return State::Failed;
    if (state == VERNON_RHI_COMPLETION_PENDING)
        return State::Pending;
    return state == VERNON_RHI_COMPLETION_SUCCEEDED ? State::Succeeded : State::Failed;
}

VernonRhiStatus ExecutionSubmission::wait() {
    if (!impl_)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    if (impl_->completion.index != VERNON_RHI_INVALID_HANDLE_INDEX) {
        impl_->status = vernonRhiCompletionWait(impl_->plan->device, impl_->completion);
        impl_->state = impl_->status == VERNON_RHI_STATUS_OK ? State::Succeeded : State::Failed;
        const VernonRhiStatus statsStatus =
            vernonRhiCompletionGetCommandStats(impl_->plan->device, impl_->completion, &impl_->stats);
        if (statsStatus != VERNON_RHI_STATUS_OK) {
            impl_->status = statsStatus;
            impl_->state = State::Failed;
        }
    }
    return impl_->status;
}

VernonRhiStatus ExecutionSubmission::signal(VernonRhiStatus result) {
    if (!impl_ || impl_->completion.index == VERNON_RHI_INVALID_HANDLE_INDEX)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    return vernonRhiCompletionSignal(impl_->plan->device, impl_->completion, result);
}

VernonRhiStatus ExecutionSubmission::status() const {
    return impl_ ? impl_->status : VERNON_RHI_STATUS_INVALID_ARGUMENT;
}

const VernonRhiCommandEncoderStats &ExecutionSubmission::commandStats() const {
    static const VernonRhiCommandEncoderStats empty{};
    return impl_ ? impl_->stats : empty;
}

CompiledExecutionGraph::CompiledExecutionGraph(std::shared_ptr<State> state) : state_(std::move(state)) {}
CompiledExecutionGraph::~CompiledExecutionGraph() = default;

const std::vector<uint32_t> &CompiledExecutionGraph::schedule() const { return state_->schedule; }
const std::vector<CompiledScope> &CompiledExecutionGraph::scopes() const { return state_->scopes; }

ExecutionBindingsBuilder CompiledExecutionGraph::createBindings(const std::vector<ExecutionBinding> &initial) const {
    ExecutionBindingsBuilder builder(state_->graphIdentity, state_->parameterNames.size());
    std::vector<bool> seen(state_->parameterNames.size());
    for (const ExecutionBinding &binding : initial) {
        if (binding.parameter.graphIdentity != state_->graphIdentity ||
            binding.parameter.id >= state_->parameterNames.size())
            throw std::invalid_argument("execution parameter does not belong to this compiled graph");
        if (seen[binding.parameter.id])
            throw std::invalid_argument("execution parameter was bound more than once");
        seen[binding.parameter.id] = true;
        builder.set(binding.parameter, binding.value);
    }
    (void)builder.snapshot();
    return builder;
}

ExecutionSubmission CompiledExecutionGraph::submit(std::shared_ptr<const ExecutionBindings> bindings) const {
    if (!bindings) {
        if (!state_->parameterNames.empty())
            throw std::invalid_argument("compiled execution graph requires parameter bindings");
    } else if (bindings->graphIdentity_ != state_->graphIdentity ||
               bindings->values_.size() != state_->parameterNames.size()) {
        throw std::invalid_argument("execution bindings do not belong to this compiled graph");
    }
    auto submission = std::make_unique<ExecutionSubmission::Impl>(state_, std::move(bindings));
    if (state_->provider == detail::ExecutionProvider::Cpu) {
        std::vector<VernonRhiBuffer> buffers(
            state_->resources.size(), VernonRhiBuffer{static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0});
        ExecutionResources resources(state_->resources, buffers, submission->bindings);
        ComputeEncoder encoder(state_->device, {static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0});
        for (const CompiledScope &scope : state_->scopes) {
            if (scope.rendering) {
                submission->status = VERNON_RHI_STATUS_UNSUPPORTED;
                break;
            }
            for (uint32_t passIndex : scope.passIndices) {
                submission->status =
                    static_cast<ComputePass *>(state_->passes[passIndex].get())->execute(encoder, resources);
                if (submission->status != VERNON_RHI_STATUS_OK)
                    break;
            }
            if (submission->status != VERNON_RHI_STATUS_OK)
                break;
        }
        submission->state = submission->status == VERNON_RHI_STATUS_OK ? ExecutionSubmission::State::Succeeded
                                                                       : ExecutionSubmission::State::Failed;
        return ExecutionSubmission(std::move(submission));
    }

    DeviceExecutionSession session(state_->device);
    VernonRhiCommandEncoderDescriptor encoderDescriptor{};
    encoderDescriptor.struct_size = sizeof(encoderDescriptor);
    encoderDescriptor.required_capabilities = VERNON_RHI_QUEUE_COMPUTE | VERNON_RHI_QUEUE_GRAPHICS;
    VernonRhiCommandEncoder native{static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
    VernonRhiStatus status = vernonRhiDeviceCreateCommandEncoder(state_->device, &encoderDescriptor, &native);
    if (status != VERNON_RHI_STATUS_OK) {
        submission->status = status;
        submission->state = ExecutionSubmission::State::Failed;
        return ExecutionSubmission(std::move(submission));
    }
    std::vector<VernonRhiBuffer> buffers(state_->resources.size(),
                                         VernonRhiBuffer{static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0});
    for (size_t index = 0; index < state_->resourceRecords.size(); ++index)
        if (state_->resourceRecords[index].resource.kind == ResourceKind::Buffer)
            buffers[index] = state_->resourceRecords[index].buffer;
    ExecutionResources resources(state_->resources, buffers, submission->bindings);
    for (const CompiledScope &scope : state_->scopes) {
        if (!scope.barriers.empty()) {
            status =
                vernonRhiCommandEncoderBarrier(state_->device, native, scope.barriers.data(), scope.barriers.size());
            if (status != VERNON_RHI_STATUS_OK)
                break;
        }
        if (scope.rendering) {
            auto *first = static_cast<RenderPass *>(state_->passes[scope.passIndices.front()].get());
            auto *last = static_cast<RenderPass *>(state_->passes[scope.passIndices.back()].get());
            std::vector<VernonRhiColorAttachment> colors;
            colors.reserve(first->colors().size());
            for (size_t index = 0; index < first->colors().size(); ++index) {
                const auto &begin = first->colors()[index];
                const auto &finish = last->colors()[index];
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
            if (first->depthAttachment()) {
                const DepthStencilAttachmentUse &firstDepth = *first->depthAttachment();
                const DepthStencilAttachmentUse &lastDepth = *last->depthAttachment();
                depth.view = firstDepth.image.view;
                depth.initial_state = VERNON_RHI_STATE_DEPTH_STENCIL_ATTACHMENT;
                depth.final_state = VERNON_RHI_STATE_DEPTH_STENCIL_ATTACHMENT;
                depth.depth_load_operation = firstDepth.depthLoad;
                depth.depth_store_operation = lastDepth.depthStore;
                depth.clear_depth = firstDepth.clearDepth;
                depth.stencil_load_operation = firstDepth.stencilLoad;
                depth.stencil_store_operation = lastDepth.stencilStore;
                depth.clear_stencil = firstDepth.clearStencil;
                depth.read_only_depth = firstDepth.readOnlyDepth;
                depth.read_only_stencil = firstDepth.readOnlyStencil;
            }
            VernonRhiRenderingDescriptor descriptor{};
            descriptor.struct_size = sizeof(descriptor);
            descriptor.color_attachments = colors.data();
            descriptor.color_attachment_count = colors.size();
            descriptor.depth_stencil_attachment = first->depthAttachment() ? &depth : nullptr;
            descriptor.offset_x = first->renderAreaData()[0];
            descriptor.offset_y = first->renderAreaData()[1];
            descriptor.width = first->renderAreaData()[2];
            descriptor.height = first->renderAreaData()[3];
            descriptor.layers = !first->colors().empty() ? first->colors().front().image.layers
                                                         : first->depthAttachment()->image.layers;
            status = vernonRhiCommandEncoderBeginRendering(state_->device, native, &descriptor);
            if (status != VERNON_RHI_STATUS_OK)
                break;
            GraphicsEncoder encoder(state_->device, native);
            for (size_t scopeIndex = 0; scopeIndex < scope.passIndices.size(); ++scopeIndex) {
                auto *renderPass = static_cast<RenderPass *>(state_->passes[scope.passIndices[scopeIndex]].get());
                if (scopeIndex) {
                    for (const auto &color : renderPass->colors())
                        if (color.load == VERNON_RHI_LOAD_CLEAR) {
                            status = vernonRhiCommandEncoderClearColorAttachment(state_->device, native, color.location,
                                                                                 color.clear);
                            if (status != VERNON_RHI_STATUS_OK)
                                break;
                        }
                    if (status == VERNON_RHI_STATUS_OK && renderPass->depthAttachment()) {
                        const DepthStencilAttachmentUse &depthUse = *renderPass->depthAttachment();
                        uint32_t aspects = 0;
                        if (depthUse.depthLoad == VERNON_RHI_LOAD_CLEAR)
                            aspects |= VERNON_RHI_ATTACHMENT_DEPTH;
                        if (depthUse.stencilLoad == VERNON_RHI_LOAD_CLEAR)
                            aspects |= VERNON_RHI_ATTACHMENT_STENCIL;
                        if (aspects)
                            status = vernonRhiCommandEncoderClearDepthStencilAttachment(
                                state_->device, native, depthUse.clearDepth, depthUse.clearStencil, aspects);
                    }
                }
                if (status == VERNON_RHI_STATUS_OK)
                    status = renderPass->execute(encoder, resources);
                if (status != VERNON_RHI_STATUS_OK)
                    break;
            }
            const VernonRhiStatus endStatus = vernonRhiCommandEncoderEndRendering(state_->device, native);
            if (status == VERNON_RHI_STATUS_OK)
                status = endStatus;
        } else {
            ComputeEncoder encoder(state_->device, native);
            status = static_cast<ComputePass *>(state_->passes[scope.passIndices.front()].get())
                         ->execute(encoder, resources);
        }
        if (status != VERNON_RHI_STATUS_OK)
            break;
    }
    if (status == VERNON_RHI_STATUS_OK)
        status = vernonRhiCommandEncoderFinish(state_->device, native);
    if (status == VERNON_RHI_STATUS_OK)
        status = vernonRhiDeviceSubmit(state_->device, native, &submission->completion);
    if (status != VERNON_RHI_STATUS_OK)
        (void)vernonRhiDeviceDestroyCommandEncoder(state_->device, native);
    submission->status = status;
    submission->state =
        status == VERNON_RHI_STATUS_OK ? ExecutionSubmission::State::Pending : ExecutionSubmission::State::Failed;
    if (status == VERNON_RHI_STATUS_OK) {
        VernonRhiCompletionState completionState{};
        if (vernonRhiCompletionGetState(state_->device, submission->completion, &completionState) ==
                VERNON_RHI_STATUS_OK &&
            completionState == VERNON_RHI_COMPLETION_SUCCEEDED)
            submission->state = ExecutionSubmission::State::Succeeded;
        (void)vernonRhiCompletionGetCommandStats(state_->device, submission->completion, &submission->stats);
    }
    return ExecutionSubmission(std::move(submission));
}

} // namespace vernon::execution
