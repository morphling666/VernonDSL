#include "VernonExecutionGraph.h"

#include "execution_graph_internal.h"
#include "rhi/rhi_internal.h"

#include <algorithm>
#include <chrono>
#include <memory>

namespace vernon::execution {

CompiledExecutionGraph::State::~State() {
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

ExecutionSubmission::Impl::~Impl() {
    if (completion.index != VERNON_RHI_INVALID_HANDLE_INDEX)
        (void)vernonRhiDeviceDestroyCompletion(plan->device, completion);
}

std::shared_ptr<CompiledExecutionGraph> ExecutionGraph::compile(std::string &error) {
    if (!buildPlan(error))
        return {};
    detail::CommandDag commandDag;
    if (!detail::buildCommandDag(resourceRecords_, passes_, scopes_, commandDag, error))
        return {};
    std::vector<std::vector<VernonRhiBarrier>> commandBarriers;
    if (!detail::buildCommandBarriers(resourceRecords_, commandDag, commandBarriers, error))
        return {};
    for (uint32_t index = 0; index < scopes_.size(); ++index)
        scopes_[index].barriers = std::move(commandBarriers[index]);
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
    state->commandDag = std::move(commandDag);
    state->autodiffCheckpointPlan = std::move(autodiffCheckpointPlan_);
    state->autodiffInitialResources = std::move(autodiffInitialResources_);
    state->autodiffInitialRanges = std::move(autodiffInitialRanges_);
    state->autodiffTransactionResources = std::move(autodiffTransactionResources_);
    state->autodiffTransactionRanges = std::move(autodiffTransactionRanges_);
    state->autodiffRestorationResources = std::move(autodiffRestorationResources_);
    state->autodiffRestorationRanges = std::move(autodiffRestorationRanges_);
    state->hasAutodiffCheckpointPlan = hasAutodiffSchedule_;
    state->differentiableInputs = std::move(differentiableInputs_);
    state->objectives = std::move(objectives_);
    for (const auto &pass : state->passes) {
        pass->owner_ = nullptr;
        pass->frozen_ = true;
    }
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
    if (impl_->completion.index == VERNON_RHI_INVALID_HANDLE_INDEX) {
        std::lock_guard lock(impl_->stateMutex);
        return impl_->state;
    }
    VernonRhiCompletionState state{};
    const VernonRhiStatus queryStatus = vernonRhiCompletionGetState(impl_->plan->device, impl_->completion, &state);
    if (queryStatus != VERNON_RHI_STATUS_OK) {
        std::lock_guard lock(impl_->stateMutex);
        impl_->status = queryStatus;
        impl_->state = State::Failed;
        return State::Failed;
    }
    if (state == VERNON_RHI_COMPLETION_PENDING)
        return State::Pending;
    const VernonRhiStatus status = vernonRhiCompletionWait(impl_->plan->device, impl_->completion);
    std::lock_guard lock(impl_->stateMutex);
    impl_->status = status;
    impl_->state = impl_->status == VERNON_RHI_STATUS_OK ? State::Succeeded : State::Failed;
    return impl_->state;
}

VernonRhiStatus ExecutionSubmission::wait() {
    if (!impl_)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    if (impl_->completion.index != VERNON_RHI_INVALID_HANDLE_INDEX) {
        const VernonRhiStatus status = vernonRhiCompletionWait(impl_->plan->device, impl_->completion);
        std::lock_guard lock(impl_->stateMutex);
        impl_->status = status;
        impl_->state = impl_->status == VERNON_RHI_STATUS_OK ? State::Succeeded : State::Failed;
        return impl_->status;
    }
    std::lock_guard lock(impl_->stateMutex);
    return impl_->status;
}

VernonRhiStatus ExecutionSubmission::signal(VernonRhiStatus result) {
    if (!impl_ || impl_->completion.index == VERNON_RHI_INVALID_HANDLE_INDEX)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    const VernonRhiStatus signalStatus = vernonRhiCompletionSignal(impl_->plan->device, impl_->completion, result);
    if (signalStatus == VERNON_RHI_STATUS_OK) {
        std::lock_guard lock(impl_->stateMutex);
        impl_->status = result;
        impl_->state = result == VERNON_RHI_STATUS_OK ? State::Succeeded : State::Failed;
    }
    return signalStatus;
}

VernonRhiStatus ExecutionSubmission::status() const {
    if (!impl_)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    if (impl_->completion.index == VERNON_RHI_INVALID_HANDLE_INDEX) {
        std::lock_guard lock(impl_->stateMutex);
        return impl_->status;
    }
    VernonRhiCompletionState completionState{};
    const VernonRhiStatus queryStatus =
        vernonRhiCompletionGetState(impl_->plan->device, impl_->completion, &completionState);
    if (queryStatus != VERNON_RHI_STATUS_OK) {
        std::lock_guard lock(impl_->stateMutex);
        impl_->status = queryStatus;
        impl_->state = State::Failed;
    } else if (completionState != VERNON_RHI_COMPLETION_PENDING) {
        const VernonRhiStatus status = vernonRhiCompletionWait(impl_->plan->device, impl_->completion);
        std::lock_guard lock(impl_->stateMutex);
        impl_->status = status;
        impl_->state = impl_->status == VERNON_RHI_STATUS_OK ? State::Succeeded : State::Failed;
    }
    std::lock_guard lock(impl_->stateMutex);
    return impl_->status;
}

const VernonRhiCommandEncoderStats &ExecutionSubmission::commandStats() const {
    static const VernonRhiCommandEncoderStats empty{};
    return impl_ ? impl_->stats : empty;
}

CompiledExecutionGraph::CompiledExecutionGraph(std::shared_ptr<State> state) : state_(std::move(state)) {}
CompiledExecutionGraph::~CompiledExecutionGraph() = default;

const std::vector<uint32_t> &CompiledExecutionGraph::schedule() const { return state_->schedule; }
const std::vector<CompiledScope> &CompiledExecutionGraph::scopes() const { return state_->scopes; }
const AutodiffDagCheckpointPlan *CompiledExecutionGraph::autodiffCheckpointPlan() const {
    return state_->hasAutodiffCheckpointPlan ? &state_->autodiffCheckpointPlan : nullptr;
}

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

VernonRhiStatus detail::executeCpuScheduleRange(VernonRhiDevice device, const std::vector<GraphResource> &resources,
                                                const std::vector<std::unique_ptr<ExecutionPass>> &passes,
                                                const std::vector<uint32_t> &schedule,
                                                std::shared_ptr<const ExecutionBindings> bindings, uint32_t beginOffset,
                                                uint32_t endOffset, CpuPassExecutor executor, void *context,
                                                const std::vector<VernonRhiBuffer> *resourceBuffers) {
    if (beginOffset > endOffset || endOffset > schedule.size())
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::vector<VernonRhiBuffer> buffers(resources.size(),
                                         VernonRhiBuffer{static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0});
    if (resourceBuffers) {
        if (resourceBuffers->size() != buffers.size())
            return VERNON_RHI_STATUS_INVALID_ARGUMENT;
        buffers = *resourceBuffers;
    }
    ExecutionResources executionResources(resources, buffers, std::move(bindings));
    ComputeEncoder encoder(device, {static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0});
    for (uint32_t offset = beginOffset; offset < endOffset; ++offset) {
        auto *pass = dynamic_cast<ComputePass *>(passes[schedule[offset]].get());
        if (!pass)
            return VERNON_RHI_STATUS_UNSUPPORTED;
        const VernonRhiStatus status = executor ? executor(context, offset, *pass, encoder, executionResources)
                                                : pass->execute(encoder, executionResources);
        if (status != VERNON_RHI_STATUS_OK)
            return status;
    }
    return VERNON_RHI_STATUS_OK;
}

static uint32_t commandAccessBits(const detail::CommandResourceAccess &access) {
    const bool reads = access.access != AccessMode::Write;
    const bool writesResource = access.access != AccessMode::Read;
    if (access.state == VERNON_RHI_STATE_COLOR_ATTACHMENT)
        return (reads ? VERNON_RHI_ACCESS_COLOR_READ : 0) | (writesResource ? VERNON_RHI_ACCESS_COLOR_WRITE : 0);
    if (access.state == VERNON_RHI_STATE_DEPTH_STENCIL_ATTACHMENT)
        return (reads ? VERNON_RHI_ACCESS_DEPTH_STENCIL_READ : 0) |
               (writesResource ? VERNON_RHI_ACCESS_DEPTH_STENCIL_WRITE : 0);
    if (access.state == VERNON_RHI_STATE_TRANSFER_SOURCE)
        return VERNON_RHI_ACCESS_TRANSFER_READ;
    if (access.state == VERNON_RHI_STATE_TRANSFER_DESTINATION)
        return VERNON_RHI_ACCESS_TRANSFER_WRITE;
    if (access.state == VERNON_RHI_STATE_SHADER_READ || access.state == VERNON_RHI_STATE_SHADER_WRITE)
        return (reads ? VERNON_RHI_ACCESS_SHADER_READ : 0) | (writesResource ? VERNON_RHI_ACCESS_SHADER_WRITE : 0);
    return VERNON_RHI_ACCESS_NONE;
}

static VernonRhiImageSubresourceRange commandImageIntersection(const VernonRhiImageSubresourceRange &left,
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

static VernonRhiStatus
buildRhiCommandBarriers(const detail::CommandDag &dag,
                        const std::vector<detail::RhiCommandResourceBinding> &resourceBindings,
                        std::vector<std::vector<VernonRhiBarrier>> &barriers,
                        const std::vector<detail::CommandResourceAccess> *initialAccesses = nullptr) {
    barriers.assign(dag.nodes.size(), {});
    std::vector<detail::CommandResourceAccess> lastAccesses =
        initialAccesses ? *initialAccesses : std::vector<detail::CommandResourceAccess>{};
    for (uint32_t nodeIndex = 0; nodeIndex < dag.nodes.size(); ++nodeIndex) {
        std::vector<detail::CommandResourceAccess> firstAccesses;
        std::vector<detail::CommandResourceAccess> finalAccesses;
        for (const detail::CommandResourceAccess &access : dag.nodes[nodeIndex].accesses) {
            if (std::none_of(finalAccesses.begin(), finalAccesses.end(),
                             [&](const detail::CommandResourceAccess &current) {
                                 return detail::commandAccessesOverlap(current, access);
                             }))
                firstAccesses.push_back(access);
            finalAccesses.erase(std::remove_if(finalAccesses.begin(), finalAccesses.end(),
                                               [&](const detail::CommandResourceAccess &current) {
                                                   return detail::commandAccessesOverlap(current, access);
                                               }),
                                finalAccesses.end());
            finalAccesses.push_back(access);
        }
        for (const detail::CommandResourceAccess &access : firstAccesses) {
            const auto binding =
                std::find_if(resourceBindings.begin(), resourceBindings.end(),
                             [&](const detail::RhiCommandResourceBinding &candidate) {
                                 return candidate.aliasDomain == access.aliasDomain && candidate.kind == access.kind;
                             });
            if (binding == resourceBindings.end())
                return VERNON_RHI_STATUS_INVALID_ARGUMENT;
            for (const detail::CommandResourceAccess &previous : lastAccesses) {
                if (!detail::commandAccessesOverlap(previous, access) ||
                    (previous.state == access.state && previous.access == AccessMode::Read &&
                     access.access == AccessMode::Read))
                    continue;
                VernonRhiBarrier barrier{};
                barrier.struct_size = sizeof(barrier);
                barrier.source_stage_mask = previous.stageMask;
                barrier.destination_stage_mask = access.stageMask;
                barrier.source_access = commandAccessBits(previous);
                barrier.destination_access = commandAccessBits(access);
                barrier.old_state = previous.state;
                barrier.new_state = access.state;
                barrier.is_image = access.kind == ResourceKind::Image;
                if (barrier.is_image) {
                    barrier.image = binding->image;
                    barrier.image_subresources =
                        commandImageIntersection(previous.imageSubresources, access.imageSubresources);
                } else {
                    barrier.buffer = binding->buffer;
                }
                barriers[nodeIndex].push_back(barrier);
            }
        }
        for (const detail::CommandResourceAccess &access : finalAccesses) {
            lastAccesses.erase(std::remove_if(lastAccesses.begin(), lastAccesses.end(),
                                              [&](const detail::CommandResourceAccess &previous) {
                                                  return detail::commandAccessesOverlap(previous, access);
                                              }),
                               lastAccesses.end());
            lastAccesses.push_back(access);
        }
    }
    return VERNON_RHI_STATUS_OK;
}

static void updateCommandFinalAccesses(std::vector<detail::CommandResourceAccess> &lastAccesses,
                                       const detail::CommandNode &node) {
    for (const detail::CommandResourceAccess &access : node.accesses) {
        lastAccesses.erase(std::remove_if(lastAccesses.begin(), lastAccesses.end(),
                                          [&](const detail::CommandResourceAccess &previous) {
                                              return detail::commandAccessesOverlap(previous, access);
                                          }),
                           lastAccesses.end());
        lastAccesses.push_back(access);
    }
}

static VernonRhiStatus submitRhiCommandDagWithInitialAccesses(
    VernonRhiDevice device, uint32_t requiredCapabilities, const detail::CommandDag &dag,
    const std::vector<detail::RhiCommandNodeEncoder> &encoders, VernonRhiCompletion *outputCompletion,
    const std::vector<detail::RhiCommandResourceBinding> *resourceBindings,
    const std::vector<detail::CommandResourceAccess> *initialAccesses) {
    if (!outputCompletion || dag.nodes.empty() || encoders.size() != dag.nodes.size())
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::string error;
    if (!validateCommandDag(dag, error))
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    for (size_t index = 0; index < encoders.size(); ++index)
        if (!encoders[index].encode || encoders[index].complete ||
            dag.nodes[index].kind == detail::CommandNodeKind::Status)
            return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::vector<std::vector<VernonRhiBarrier>> nodeBarriers;
    if (resourceBindings) {
        const VernonRhiStatus barrierStatus =
            buildRhiCommandBarriers(dag, *resourceBindings, nodeBarriers, initialAccesses);
        if (barrierStatus != VERNON_RHI_STATUS_OK)
            return barrierStatus;
    } else {
        nodeBarriers.resize(dag.nodes.size());
    }
    *outputCompletion = {static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
    DeviceExecutionSession session(device);
    VernonRhiCommandEncoderDescriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.required_capabilities = requiredCapabilities;
    VernonRhiCommandEncoder encoder{static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
    VernonRhiStatus status = vernonRhiDeviceCreateCommandEncoder(device, &descriptor, &encoder);
    if (status == VERNON_RHI_STATUS_OK) {
        for (uint32_t index = 0; index < encoders.size(); ++index) {
            if (!nodeBarriers[index].empty())
                status = vernonRhiCommandEncoderBarrier(device, encoder, nodeBarriers[index].data(),
                                                        nodeBarriers[index].size());
            if (status == VERNON_RHI_STATUS_OK)
                status = encoders[index].encode(encoders[index].context, encoder);
            if (status != VERNON_RHI_STATUS_OK)
                break;
        }
    }
    if (status == VERNON_RHI_STATUS_OK)
        status = vernonRhiCommandEncoderFinish(device, encoder);
    if (status == VERNON_RHI_STATUS_OK)
        status = vernonRhiDeviceSubmit(device, encoder, outputCompletion);
    else if (encoder.index != VERNON_RHI_INVALID_HANDLE_INDEX)
        (void)vernonRhiDeviceDestroyCommandEncoder(device, encoder);
    return status;
}

VernonRhiStatus detail::submitRhiCommandDag(VernonRhiDevice device, uint32_t requiredCapabilities,
                                            const CommandDag &dag, const std::vector<RhiCommandNodeEncoder> &encoders,
                                            VernonRhiCompletion *outputCompletion,
                                            const std::vector<RhiCommandResourceBinding> *resourceBindings) {
    return submitRhiCommandDagWithInitialAccesses(device, requiredCapabilities, dag, encoders, outputCompletion,
                                                  resourceBindings, nullptr);
}

VernonRhiStatus detail::executeRhiCommandDagAndWait(VernonRhiDevice device, uint32_t requiredCapabilities,
                                                    const CommandDag &dag,
                                                    const std::vector<RhiCommandNodeEncoder> &encoders,
                                                    RhiCommandDagExecutionStats *stats,
                                                    const std::vector<RhiCommandResourceBinding> *resourceBindings,
                                                    const std::vector<CommandResourceAccess> *initialAccesses) {
    if (dag.nodes.empty() || encoders.size() != dag.nodes.size())
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::string error;
    if (!validateCommandDag(dag, error))
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    if (stats)
        *stats = {};
    std::vector<CommandResourceAccess> carriedAccesses =
        initialAccesses ? *initialAccesses : std::vector<CommandResourceAccess>{};
    uint32_t offset = 0;
    while (offset < dag.nodes.size()) {
        const uint32_t begin = offset;
        while (offset < dag.nodes.size() && dag.nodes[offset].kind != CommandNodeKind::Status) {
            if (!encoders[offset].encode || encoders[offset].complete)
                return VERNON_RHI_STATUS_INVALID_ARGUMENT;
            ++offset;
        }
        if (begin != offset) {
            CommandDag segment;
            std::vector<RhiCommandNodeEncoder> segmentEncoders;
            const bool finalState = offset < dag.nodes.size() && !dag.nodes[offset].accesses.empty();
            segment.nodes.reserve(offset - begin + (finalState ? 1 : 0));
            segmentEncoders.reserve(offset - begin + (finalState ? 1 : 0));
            for (uint32_t index = begin; index < offset; ++index) {
                CommandNode node = dag.nodes[index];
                std::vector<uint32_t> predecessors;
                for (uint32_t predecessor : node.predecessors)
                    if (predecessor >= begin)
                        predecessors.push_back(predecessor - begin);
                node.predecessors = std::move(predecessors);
                segment.nodes.push_back(std::move(node));
                segmentEncoders.push_back(encoders[index]);
            }
            if (finalState) {
                CommandNode node;
                node.kind = CommandNodeKind::Transfer;
                node.queue = CommandQueueClass::Ordered;
                node.accesses = dag.nodes[offset].accesses;
                for (uint32_t predecessor : dag.nodes[offset].predecessors)
                    if (predecessor >= begin && predecessor < offset)
                        node.predecessors.push_back(predecessor - begin);
                segment.nodes.push_back(std::move(node));
                segmentEncoders.push_back(
                    {[](void *, VernonRhiCommandEncoder) { return VERNON_RHI_STATUS_OK; }, nullptr});
            }
            VernonRhiCompletion completion{static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
            VernonRhiStatus status =
                submitRhiCommandDagWithInitialAccesses(device, requiredCapabilities, segment, segmentEncoders,
                                                       &completion, resourceBindings, &carriedAccesses);
            if (status != VERNON_RHI_STATUS_OK)
                return status;
            if (stats)
                ++stats->submissions;
            const auto waitStarted = std::chrono::steady_clock::now();
            status = vernonRhiCompletionWait(device, completion);
            if (stats) {
                ++stats->waits;
                stats->deviceWaitNanoseconds += static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - waitStarted)
                        .count());
            }
            (void)vernonRhiDeviceDestroyCompletion(device, completion);
            if (status != VERNON_RHI_STATUS_OK)
                return status;
            for (uint32_t index = begin; index < offset; ++index)
                updateCommandFinalAccesses(carriedAccesses, dag.nodes[index]);
            if (finalState)
                updateCommandFinalAccesses(carriedAccesses, dag.nodes[offset]);
        }
        if (offset < dag.nodes.size()) {
            const RhiCommandNodeEncoder &status = encoders[offset];
            if (status.encode || !status.complete)
                return VERNON_RHI_STATUS_INVALID_ARGUMENT;
            const VernonRhiStatus completed = status.complete(status.context);
            if (completed != VERNON_RHI_STATUS_OK)
                return completed;
            ++offset;
        }
    }
    return VERNON_RHI_STATUS_OK;
}

VernonRhiStatus detail::submitRhiCommandList(VernonRhiDevice device, uint32_t requiredCapabilities,
                                             RhiCommandListEncoder encode, void *context,
                                             VernonRhiCompletion *outputCompletion) {
    CommandDag dag;
    CommandNode node;
    node.kind = CommandNodeKind::Compute;
    node.queue = CommandQueueClass::Compute;
    node.scopeIndices.push_back(0);
    dag.nodes.push_back(std::move(node));
    return submitRhiCommandDag(device, requiredCapabilities, dag, {{encode, context}}, outputCompletion);
}

VernonRhiStatus detail::submitRhiComputeCommandRange(
    VernonRhiDevice device, const std::vector<GraphResource> &resources,
    const std::vector<ExecutionResourceRecord> &resourceRecords,
    const std::vector<std::unique_ptr<ExecutionPass>> &passes, const std::vector<uint32_t> &schedule,
    const std::vector<CompiledScope> &scopes, const CommandDag &commandDag,
    std::shared_ptr<const ExecutionBindings> bindings, uint32_t beginOffset, uint32_t endOffset,
    CpuPassExecutor executor, void *context, VernonRhiCompletion *outputCompletion, RhiCommandPlanExtension extension,
    void *extensionContext, RhiPassCommandPlanner planner, void *plannerContext, RhiCommandExecutionPlan *outputPlan) {
    if (beginOffset > endOffset || endOffset > schedule.size() || endOffset > scopes.size() ||
        endOffset > commandDag.nodes.size() || resourceRecords.size() > resources.size())
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    if (beginOffset == endOffset)
        return outputCompletion ? VERNON_RHI_STATUS_INVALID_ARGUMENT : VERNON_RHI_STATUS_OK;
    std::vector<VernonRhiBuffer> buffers(resources.size(),
                                         VernonRhiBuffer{static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0});
    for (size_t index = 0; index < resourceRecords.size(); ++index)
        if (resourceRecords[index].resource.kind == ResourceKind::Buffer)
            buffers[index] = resourceRecords[index].buffer;
    struct ResourceContext {
        ResourceContext(const std::vector<GraphResource> &graphResources, std::vector<VernonRhiBuffer> values,
                        std::shared_ptr<const ExecutionBindings> bindings)
            : buffers(std::move(values)), resources(graphResources, buffers, std::move(bindings)) {}
        std::vector<VernonRhiBuffer> buffers;
        ExecutionResources resources;
    };
    auto resourceContext = std::make_shared<ResourceContext>(resources, std::move(buffers), std::move(bindings));
    const ExecutionResources &executionResources = resourceContext->resources;
    struct EncodeContext {
        VernonRhiDevice device;
        const std::vector<std::unique_ptr<ExecutionPass>> &passes;
        const std::vector<uint32_t> &schedule;
        const std::vector<CompiledScope> &scopes;
        const CommandDag &commandDag;
        uint32_t offset;
        CpuPassExecutor executor;
        void *executorContext;
        const ExecutionResources &resources;
    };
    const auto encode = [](void *opaque, VernonRhiCommandEncoder native) {
        auto &state = *static_cast<EncodeContext *>(opaque);
        ComputeEncoder encoder(state.device, native);
        const CommandNode &node = state.commandDag.nodes[state.offset];
        const CompiledScope &scope = state.scopes[state.offset];
        auto *compute = dynamic_cast<ComputePass *>(state.passes[state.schedule[state.offset]].get());
        if ((node.kind != CommandNodeKind::Compute && node.kind != CommandNodeKind::Derivative) ||
            node.scopeIndices.size() != 1 || node.scopeIndices.front() != state.offset || scope.rendering ||
            scope.passIndices.size() != 1 || scope.passIndices.front() != state.schedule[state.offset] || !compute)
            return VERNON_RHI_STATUS_UNSUPPORTED;
        return state.executor ? state.executor(state.executorContext, state.offset, *compute, encoder, state.resources)
                              : compute->execute(encoder, state.resources);
    };
    RhiCommandExecutionPlan rangePlan;
    std::vector<uint32_t> completionNodes;
    rangePlan.commands.nodes.reserve(endOffset - beginOffset);
    rangePlan.encoders.reserve(endOffset - beginOffset);
    completionNodes.reserve(endOffset - beginOffset);
    rangePlan.retainedContexts.push_back(resourceContext);
    for (uint32_t offset = beginOffset; offset < endOffset; ++offset) {
        auto *compute = dynamic_cast<ComputePass *>(passes[schedule[offset]].get());
        if (!compute)
            return VERNON_RHI_STATUS_UNSUPPORTED;
        if (planner) {
            RhiCommandExecutionPlan planned;
            const VernonRhiStatus plannedStatus =
                planner(plannerContext, offset, *compute, executionResources, planned);
            if (plannedStatus == VERNON_RHI_STATUS_OK && !planned.commands.nodes.empty()) {
                std::string error;
                if (!appendRhiCommandExecutionPlan(rangePlan, std::move(planned), true, error))
                    return VERNON_RHI_STATUS_INVALID_ARGUMENT;
                auto encodeContext = std::make_shared<EncodeContext>(EncodeContext{
                    device, passes, schedule, scopes, commandDag, offset, executor, context, executionResources});
                CommandNode finalize;
                finalize.kind = CommandNodeKind::Derivative;
                finalize.queue = CommandQueueClass::Ordered;
                finalize.predecessors = {static_cast<uint32_t>(rangePlan.commands.nodes.size() - 1)};
                rangePlan.commands.nodes.push_back(std::move(finalize));
                rangePlan.encoders.push_back({encode, encodeContext.get()});
                rangePlan.retainedContexts.push_back(std::move(encodeContext));
                const uint32_t finalizer = static_cast<uint32_t>(rangePlan.commands.nodes.size() - 1);
                if (extension && extension(extensionContext, offset, finalizer, rangePlan) != VERNON_RHI_STATUS_OK)
                    return VERNON_RHI_STATUS_INTERNAL_ERROR;
                completionNodes.push_back(static_cast<uint32_t>(rangePlan.commands.nodes.size() - 1));
                continue;
            }
            if (plannedStatus != VERNON_RHI_STATUS_UNSUPPORTED)
                return plannedStatus;
        }
        CommandNode node = commandDag.nodes[offset];
        std::vector<uint32_t> predecessors;
        for (uint32_t predecessor : node.predecessors)
            if (predecessor >= beginOffset) {
                const uint32_t relative = predecessor - beginOffset;
                if (relative >= completionNodes.size())
                    return VERNON_RHI_STATUS_INVALID_ARGUMENT;
                predecessors.push_back(completionNodes[relative]);
            }
        node.predecessors = std::move(predecessors);
        rangePlan.commands.nodes.push_back(std::move(node));
        for (const CommandResourceAccess &access : rangePlan.commands.nodes.back().accesses) {
            if (std::any_of(rangePlan.bindings.begin(), rangePlan.bindings.end(), [&](const auto &binding) {
                    return binding.aliasDomain == access.aliasDomain && binding.kind == access.kind;
                }))
                continue;
            if (access.resource >= resourceRecords.size())
                return VERNON_RHI_STATUS_INVALID_ARGUMENT;
            const ExecutionResourceRecord &record = resourceRecords[access.resource];
            rangePlan.bindings.push_back({access.aliasDomain, access.kind, record.buffer, record.image});
        }
        auto encodeContext = std::make_shared<EncodeContext>(
            EncodeContext{device, passes, schedule, scopes, commandDag, offset, executor, context, executionResources});
        rangePlan.encoders.push_back({encode, encodeContext.get()});
        rangePlan.retainedContexts.push_back(std::move(encodeContext));
        const uint32_t computeNode = static_cast<uint32_t>(rangePlan.commands.nodes.size() - 1);
        if (extension && extension(extensionContext, offset, computeNode, rangePlan) != VERNON_RHI_STATUS_OK)
            return VERNON_RHI_STATUS_INTERNAL_ERROR;
        completionNodes.push_back(static_cast<uint32_t>(rangePlan.commands.nodes.size() - 1));
    }
    for (const CommandNode &node : rangePlan.commands.nodes)
        for (const CommandResourceAccess &access : node.accesses) {
            if (std::any_of(rangePlan.bindings.begin(), rangePlan.bindings.end(), [&](const auto &binding) {
                    return binding.aliasDomain == access.aliasDomain && binding.kind == access.kind;
                }))
                continue;
            if (access.resource >= resourceRecords.size())
                return VERNON_RHI_STATUS_INVALID_ARGUMENT;
            const ExecutionResourceRecord &record = resourceRecords[access.resource];
            rangePlan.bindings.push_back({access.aliasDomain, access.kind, record.buffer, record.image});
        }
    if (outputPlan) {
        if (outputCompletion)
            return VERNON_RHI_STATUS_INVALID_ARGUMENT;
        *outputPlan = std::move(rangePlan);
        return VERNON_RHI_STATUS_OK;
    }
    VernonRhiCompletion completion{static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
    VernonRhiStatus status = submitRhiCommandDag(device, VERNON_RHI_QUEUE_COMPUTE, rangePlan.commands,
                                                 rangePlan.encoders, &completion, &rangePlan.bindings);
    if (status != VERNON_RHI_STATUS_OK)
        return status;
    if (outputCompletion) {
        *outputCompletion = completion;
        return VERNON_RHI_STATUS_OK;
    }
    status = vernonRhiCompletionWait(device, completion);
    (void)vernonRhiDeviceDestroyCompletion(device, completion);
    return status;
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
        submission->status =
            detail::executeCpuScheduleRange(state_->device, state_->resources, state_->passes, state_->schedule,
                                            submission->bindings, 0, static_cast<uint32_t>(state_->schedule.size()));
        submission->state = submission->status == VERNON_RHI_STATUS_OK ? ExecutionSubmission::State::Succeeded
                                                                       : ExecutionSubmission::State::Failed;
        return ExecutionSubmission(std::move(submission));
    }

    const bool computeOnly = std::all_of(
        state_->commandDag.nodes.begin(), state_->commandDag.nodes.end(), [](const detail::CommandNode &node) {
            return node.kind == detail::CommandNodeKind::Compute || node.kind == detail::CommandNodeKind::Derivative;
        });
    if (computeOnly) {
        const VernonRhiStatus status = detail::submitRhiComputeCommandRange(
            state_->device, state_->resources, state_->resourceRecords, state_->passes, state_->schedule,
            state_->scopes, state_->commandDag, submission->bindings, 0, static_cast<uint32_t>(state_->schedule.size()),
            nullptr, nullptr, &submission->completion);
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

    std::vector<VernonRhiBuffer> buffers(state_->resources.size(),
                                         VernonRhiBuffer{static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0});
    for (size_t index = 0; index < state_->resourceRecords.size(); ++index)
        if (state_->resourceRecords[index].resource.kind == ResourceKind::Buffer)
            buffers[index] = state_->resourceRecords[index].buffer;
    ExecutionResources resources(state_->resources, buffers, submission->bindings);
    struct NodeContext {
        const CompiledExecutionGraph::State &state;
        const detail::CommandNode &node;
        const ExecutionResources &resources;
    };
    const auto encodeNode = [](void *opaque, VernonRhiCommandEncoder native) {
        const auto &context = *static_cast<NodeContext *>(opaque);
        VernonRhiStatus status = VERNON_RHI_STATUS_OK;
        for (uint32_t scopeIndex : context.node.scopeIndices) {
            if (scopeIndex >= context.state.scopes.size())
                return VERNON_RHI_STATUS_INTERNAL_ERROR;
            const CompiledScope &scope = context.state.scopes[scopeIndex];
            if (scope.rendering) {
                auto *first = static_cast<RenderPass *>(context.state.passes[scope.passIndices.front()].get());
                auto *last = static_cast<RenderPass *>(context.state.passes[scope.passIndices.back()].get());
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
                status = vernonRhiCommandEncoderBeginRendering(context.state.device, native, &descriptor);
                if (status != VERNON_RHI_STATUS_OK)
                    break;
                GraphicsEncoder encoder(context.state.device, native);
                for (size_t scopeIndex = 0; scopeIndex < scope.passIndices.size(); ++scopeIndex) {
                    auto *renderPass =
                        static_cast<RenderPass *>(context.state.passes[scope.passIndices[scopeIndex]].get());
                    if (scopeIndex) {
                        for (const auto &color : renderPass->colors())
                            if (color.load == VERNON_RHI_LOAD_CLEAR) {
                                status = vernonRhiCommandEncoderClearColorAttachment(context.state.device, native,
                                                                                     color.location, color.clear);
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
                                    context.state.device, native, depthUse.clearDepth, depthUse.clearStencil, aspects);
                        }
                    }
                    if (status == VERNON_RHI_STATUS_OK)
                        status = renderPass->execute(encoder, context.resources);
                    if (status != VERNON_RHI_STATUS_OK)
                        break;
                }
                const VernonRhiStatus endStatus = vernonRhiCommandEncoderEndRendering(context.state.device, native);
                if (status == VERNON_RHI_STATUS_OK)
                    status = endStatus;
            } else {
                ComputeEncoder encoder(context.state.device, native);
                status = static_cast<ComputePass *>(context.state.passes[scope.passIndices.front()].get())
                             ->execute(encoder, context.resources);
            }
            if (status != VERNON_RHI_STATUS_OK)
                break;
        }
        return status;
    };
    std::vector<NodeContext> contexts;
    std::vector<detail::RhiCommandNodeEncoder> encoders;
    contexts.reserve(state_->commandDag.nodes.size());
    encoders.reserve(state_->commandDag.nodes.size());
    for (const detail::CommandNode &node : state_->commandDag.nodes) {
        contexts.push_back({*state_, node, resources});
        encoders.push_back({encodeNode, &contexts.back()});
    }
    std::vector<detail::RhiCommandResourceBinding> resourceBindings;
    for (const detail::CommandNode &node : state_->commandDag.nodes)
        for (const detail::CommandResourceAccess &access : node.accesses) {
            if (access.resource >= state_->resourceRecords.size())
                continue;
            const auto &record = state_->resourceRecords[access.resource];
            if (std::none_of(resourceBindings.begin(), resourceBindings.end(), [&](const auto &binding) {
                    return binding.aliasDomain == access.aliasDomain && binding.kind == access.kind;
                }))
                resourceBindings.push_back({access.aliasDomain, access.kind, record.buffer, record.image});
        }
    const VernonRhiStatus status =
        detail::submitRhiCommandDag(state_->device, VERNON_RHI_QUEUE_COMPUTE | VERNON_RHI_QUEUE_GRAPHICS,
                                    state_->commandDag, encoders, &submission->completion, &resourceBindings);
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
