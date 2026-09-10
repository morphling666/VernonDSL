#include "execution_graph/command_graph.h"

#include "rhi/rhi_internal.h"

#include <algorithm>
#include <atomic>
#include <memory>
#include <mutex>
#include <queue>
#include <unordered_map>

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
            previous.readOnlyStencil != next.readOnlyStencil ||
            !preservesBoundary(previous.depthStore, next.depthLoad) ||
            !preservesBoundary(previous.stencilStore, next.stencilLoad))
            return false;
    }
    return std::equal(left.renderAreaData(), left.renderAreaData() + 4, right.renderAreaData());
}

} // namespace

class DeviceExecutionSession::Impl {
public:
    explicit Impl(VernonRhiDevice device) {
        if (vernon::rhi::deviceCommandCapabilities(device) & vernon::rhi::BackendCommandIndependentRecording)
            return;
        mutex = executionSession(device);
        lock = std::unique_lock<std::recursive_mutex>(*mutex);
    }

private:
    std::shared_ptr<std::recursive_mutex> mutex;
    std::unique_lock<std::recursive_mutex> lock;
};

DeviceExecutionSession::DeviceExecutionSession(VernonRhiDevice device) : impl_(std::make_unique<Impl>(device)) {}
DeviceExecutionSession::~DeviceExecutionSession() = default;
DeviceExecutionSession::DeviceExecutionSession(DeviceExecutionSession &&) noexcept = default;
DeviceExecutionSession &DeviceExecutionSession::operator=(DeviceExecutionSession &&) noexcept = default;

ExecutionPass::ExecutionPass(std::string name) : name_(std::move(name)) {}

void ExecutionPass::ensureMutable() const {
    if (frozen_)
        throw std::logic_error("cannot mutate a pass owned by a compiled execution graph");
}

void ExecutionPass::dependsOn(ExecutionPass &dependency) {
    ensureMutable();
    if (std::find(dependencies_.begin(), dependencies_.end(), &dependency) == dependencies_.end()) {
        dependencies_.push_back(&dependency);
        if (owner_)
            owner_->dirty_ = true;
    }
}

void ExecutionPass::setFlags(uint32_t flags) {
    ensureMutable();
    if (declaring_) {
        flags_ = flags;
        return;
    }
    if (configuredFlags_ == flags && flags_ == flags)
        return;
    configuredFlags_ = flags;
    flags_ = flags;
    if (owner_)
        owner_->dirty_ = true;
}

void ExecutionPass::read(GraphResource resource, VernonRhiResourceState state, uint32_t stageMask) {
    ensureMutable();
    uses_.push_back({resource, AccessMode::Read, state, stageMask});
}

void ExecutionPass::read(GraphImage resource, VernonRhiResourceState state, uint32_t stageMask) {
    ensureMutable();
    uses_.push_back({resource, AccessMode::Read, state, stageMask, imageRole(state), resource.subresources, resource});
}

void ExecutionPass::write(GraphResource resource, VernonRhiResourceState state, uint32_t stageMask) {
    ensureMutable();
    uses_.push_back({resource, AccessMode::Write, state, stageMask});
}

void ExecutionPass::write(GraphImage resource, VernonRhiResourceState state, uint32_t stageMask) {
    ensureMutable();
    uses_.push_back({resource, AccessMode::Write, state, stageMask, imageRole(state), resource.subresources, resource});
}

void ExecutionPass::readWrite(GraphResource resource, VernonRhiResourceState state, uint32_t stageMask) {
    ensureMutable();
    uses_.push_back({resource, AccessMode::ReadWrite, state, stageMask});
}

void ExecutionPass::readWrite(GraphImage resource, VernonRhiResourceState state, uint32_t stageMask) {
    ensureMutable();
    uses_.push_back(
        {resource, AccessMode::ReadWrite, state, stageMask, imageRole(state), resource.subresources, resource});
}
void ExecutionPass::resetDeclaration() {
    uses_.clear();
    flags_ = configuredFlags_;
    declaring_ = true;
}
void ExecutionPass::finishDeclaration() { declaring_ = false; }

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
    ensureMutable();
    auto value = attachment;
    value.location = location;
    colors_.push_back(value);
    if (value.load == VERNON_RHI_LOAD_PRESERVE)
        readWrite(value.image, VERNON_RHI_STATE_COLOR_ATTACHMENT);
    else
        write(value.image, VERNON_RHI_STATE_COLOR_ATTACHMENT);
}

void RenderPass::depth(const DepthStencilAttachmentUse &attachment) {
    ensureMutable();
    depth_ = std::make_unique<DepthStencilAttachmentUse>(attachment);
    const bool hasStencil = attachment.image.format == VERNON_RHI_FORMAT_D32_FLOAT_S8_UINT;
    if (attachment.readOnlyDepth && (!hasStencil || attachment.readOnlyStencil))
        read(attachment.image, VERNON_RHI_STATE_DEPTH_STENCIL_ATTACHMENT);
    else
        readWrite(attachment.image, VERNON_RHI_STATE_DEPTH_STENCIL_ATTACHMENT);
}

void RenderPass::renderArea(uint32_t x, uint32_t y, uint32_t width, uint32_t height) {
    ensureMutable();
    renderArea_[0] = x;
    renderArea_[1] = y;
    renderArea_[2] = width;
    renderArea_[3] = height;
}

CommandGraph::CommandGraph() : graphIdentity_(nextGraphIdentity.fetch_add(1, std::memory_order_relaxed)) {}
CommandGraph::CommandGraph(VernonRhiDevice device)
    : provider_(detail::ExecutionProvider::Rhi), device_(device),
      graphIdentity_(nextGraphIdentity.fetch_add(1, std::memory_order_relaxed)) {}

bool CommandGraph::validate(std::string &error) { return buildPlan(error); }

CommandGraph::~CommandGraph() {
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

ExecutionParameter CommandGraph::parameter(std::string name) {
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

GraphBuffer CommandGraph::importHostBuffer(uint64_t identity, bool exported) {
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

VernonRhiStatus CommandGraph::createBuffer(const VernonRhiBufferDescriptor &descriptor, GraphBuffer &output,
                                           bool exported) {
    output = {};
    if (compiled_ || provider_ != detail::ExecutionProvider::Rhi)
        return VERNON_RHI_STATUS_UNSUPPORTED;
    VernonRhiBuffer buffer{};
    const VernonRhiStatus status = vernonRhiDeviceCreateBuffer(device_, &descriptor, &buffer);
    if (status != VERNON_RHI_STATUS_OK)
        return status;
    output = importBuffer(buffer, exported);
    if (output.id >= resourceRecords_.size()) {
        (void)vernonRhiDeviceDestroyBuffer(device_, buffer);
        output = {};
        return VERNON_RHI_STATUS_INTERNAL_ERROR;
    }
    resourceRecords_[output.id].graphOwned = true;
    return VERNON_RHI_STATUS_OK;
}

GraphBuffer CommandGraph::importBuffer(VernonRhiBuffer buffer, bool exported) {
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

GraphImage CommandGraph::importImage(VernonRhiImage image, VernonRhiImageView view, bool exported) {
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

bool CommandGraph::buildPlan(std::string &error) {
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
        try {
            passes_[index]->declare();
        } catch (...) {
            passes_[index]->finishDeclaration();
            throw;
        }
        passes_[index]->finishDeclaration();
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
    if (!validateDeclarations(error))
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
    std::vector<uint32_t> passScopes(passes_.size(), UINT32_MAX);
    for (uint32_t scopeIndex = 0; scopeIndex < scopes_.size(); ++scopeIndex)
        for (uint32_t passIndex : scopes_[scopeIndex].passIndices)
            passScopes[passIndex] = scopeIndex;
    for (uint32_t predecessor = 0; predecessor < passes_.size(); ++predecessor)
        for (uint32_t dependent = 0; dependent < passes_.size(); ++dependent) {
            if (!edges[predecessor][dependent])
                continue;
            const uint32_t predecessorScope = passScopes[predecessor];
            const uint32_t dependentScope = passScopes[dependent];
            if (predecessorScope == UINT32_MAX || dependentScope == UINT32_MAX || predecessorScope == dependentScope)
                continue;
            scopes_[dependentScope].predecessors.push_back(predecessorScope);
        }
    for (CompiledScope &scope : scopes_) {
        std::sort(scope.predecessors.begin(), scope.predecessors.end());
        scope.predecessors.erase(std::unique(scope.predecessors.begin(), scope.predecessors.end()),
                                 scope.predecessors.end());
    }
    if (!validateDeclarations(error))
        return false;
    dirty_ = false;
    compiled = true;
    return true;
}

} // namespace vernon::execution
