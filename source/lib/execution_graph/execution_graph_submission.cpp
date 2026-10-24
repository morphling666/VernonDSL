#include "execution_graph/command_graph.h"

#include "execution_graph_internal.h"

#include <algorithm>
#include <memory>
#include <unordered_set>

namespace vernon::execution {

CompiledCommandGraph::State::~State() { detail::releaseExecutionResourceRecords(provider, device, resourceRecords); }

ExecutionSubmission::Impl::~Impl() {
    if (completion.index != VERNON_RHI_INVALID_HANDLE_INDEX)
        (void)vernonRhiDeviceDestroyCompletion(plan->device, completion);
}

std::shared_ptr<CompiledCommandGraph> CommandGraph::compile(std::string &error) {
    if (!buildPlan(error))
        return {};
    detail::CommandDag commandDag;
    if (!detail::buildCommandDag(resourceRecords_, passes_, scopes_, commandDag, error))
        return {};
    const auto recordTarget = [](const void *context, const detail::CommandResourceAccess &access,
                                 VernonRhiBarrier &barrier, std::string &targetError) {
        const auto &records = *static_cast<const std::vector<detail::ExecutionResourceRecord> *>(context);
        if (access.resource >= records.size())
            return targetError = "command barrier access refers to an invalid resource", false;
        const detail::ExecutionResourceRecord &record = records[access.resource];
        if (record.resource.kind != access.kind)
            return targetError = "command barrier access kind does not match its resource", false;
        if (barrier.is_image)
            barrier.image = record.image;
        else
            barrier.buffer = record.buffer;
        return true;
    };
    std::vector<std::vector<VernonRhiBarrier>> commandBarriers;
    if (!detail::buildCommandBarriers(commandDag, {}, recordTarget, &resourceRecords_, commandBarriers, error))
        return {};
    for (uint32_t index = 0; index < scopes_.size(); ++index)
        scopes_[index].barriers = std::move(commandBarriers[index]);
    std::shared_ptr<CompiledCommandGraph::State> state;
    try {
        state = std::make_shared<CompiledCommandGraph::State>();
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
    for (const auto &pass : state->passes) {
        pass->owner_ = nullptr;
        pass->frozen_ = true;
    }
    importedBuffers_.clear();
    importedHostBuffers_.clear();
    importedImages_.clear();
    parameterIds_.clear();
    compiled_ = true;
    return std::shared_ptr<CompiledCommandGraph>(new CompiledCommandGraph(std::move(state)));
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

CompiledCommandGraph::CompiledCommandGraph(std::shared_ptr<State> state) : state_(std::move(state)) {}
CompiledCommandGraph::~CompiledCommandGraph() = default;

const std::vector<uint32_t> &CompiledCommandGraph::schedule() const { return state_->schedule; }
const std::vector<CompiledScope> &CompiledCommandGraph::scopes() const { return state_->scopes; }

ExecutionBindingsBuilder CompiledCommandGraph::createBindings(const std::vector<ExecutionBinding> &initial) const {
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

VernonRhiStatus detail::submitRhiCommandPlan(VernonRhiDevice device, uint32_t requiredCapabilities,
                                             const ValidatedRhiCommandPlan &validated,
                                             VernonRhiCompletion &outputCompletion, TimedCommandScopes *timedScopes) {
    const RhiCommandExecutionPlan &plan = validated.plan();
    const std::vector<RhiCommandNodeEncoder> &encoders = plan.encoders;
    if (plan.commands.nodes.empty())
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    const auto planBinding = [](const void *context, const CommandResourceAccess &access, VernonRhiBarrier &barrier,
                                std::string &error) {
        const RhiCommandResourceBinding *binding =
            static_cast<const ValidatedRhiCommandPlan *>(context)->binding(commandAccessDomain(access));
        if (!binding)
            return error = "RHI command plan has an unbound resource access", false;
        if (barrier.is_image)
            barrier.image = binding->image;
        else
            barrier.buffer = binding->buffer;
        return true;
    };
    std::vector<std::vector<VernonRhiBarrier>> nodeBarriers;
    std::string error;
    if (!buildCommandBarriers(plan.commands, plan.initialAccesses, planBinding, &validated, nodeBarriers, error))
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    outputCompletion = {static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
    DeviceExecutionSession session(device);
    VernonRhiCommandEncoderDescriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.required_capabilities = requiredCapabilities;
    VernonRhiCommandEncoder encoder{static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
    VernonRhiStatus status = vernonRhiDeviceCreateCommandEncoder(device, &descriptor, &encoder);
    if (status == VERNON_RHI_STATUS_OK) {
        const detail::CommandScope *openScope = nullptr;
        uint32_t openTimer{};
        const auto closeScope = [&] {
            VernonRhiStatus closed = VERNON_RHI_STATUS_OK;
            if (timedScopes)
                closed = vernonRhiCommandEncoderEndTimer(device, encoder, openTimer);
            return closed == VERNON_RHI_STATUS_OK ? vernonRhiCommandEncoderPopDebugGroup(device, encoder) : closed;
        };
        for (uint32_t index = 0; index < encoders.size(); ++index) {
            const std::shared_ptr<const detail::CommandScope> &scope = encoders[index].scope;
            if (scope.get() != openScope) {
                if (openScope)
                    status = closeScope();
                if (status == VERNON_RHI_STATUS_OK && scope)
                    status = vernonRhiCommandEncoderPushDebugGroup(device, encoder,
                                                                   {scope->label.data(), scope->label.size()});
                if (status == VERNON_RHI_STATUS_OK && scope && timedScopes) {
                    status = vernonRhiCommandEncoderBeginTimer(device, encoder, &openTimer);
                    if (status == VERNON_RHI_STATUS_OK)
                        timedScopes->push_back(scope);
                }
                openScope = scope.get();
            }
            if (status == VERNON_RHI_STATUS_OK && !nodeBarriers[index].empty())
                status = vernonRhiCommandEncoderBarrier(device, encoder, nodeBarriers[index].data(),
                                                        nodeBarriers[index].size());
            if (status == VERNON_RHI_STATUS_OK)
                status = encoders[index].encode(encoders[index].context, encoder);
            if (status != VERNON_RHI_STATUS_OK)
                break;
        }
        if (status == VERNON_RHI_STATUS_OK && openScope)
            status = closeScope();
        for (const RhiCommandFinisher &finisher : plan.finishers)
            if (const VernonRhiStatus finished = finisher.finish(finisher.context, encoder);
                status == VERNON_RHI_STATUS_OK)
                status = finished;
    }
    if (status == VERNON_RHI_STATUS_OK)
        status = vernonRhiCommandEncoderFinish(device, encoder);
    if (status == VERNON_RHI_STATUS_OK)
        status = vernonRhiDeviceSubmit(device, encoder, &outputCompletion);
    if (status != VERNON_RHI_STATUS_OK && encoder.index != VERNON_RHI_INVALID_HANDLE_INDEX)
        (void)vernonRhiDeviceDestroyCommandEncoder(device, encoder);
    return status;
}

ExecutionSubmission CompiledCommandGraph::submit(std::shared_ptr<const ExecutionBindings> bindings) const {
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
    std::vector<VernonRhiBuffer> buffers(state_->resources.size(),
                                         VernonRhiBuffer{static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0});
    for (size_t index = 0; index < state_->resourceRecords.size(); ++index)
        if (state_->resourceRecords[index].resource.kind == ResourceKind::Buffer)
            buffers[index] = state_->resourceRecords[index].buffer;
    ExecutionResources resources(state_->resources, buffers, submission->bindings);
    struct NodeContext {
        const CompiledCommandGraph::State &state;
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
                    attachment.clear_color = begin.clear;
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
                                                                                     color.location, &color.clear);
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
    detail::RhiCommandExecutionPlan fragment;
    fragment.commands = state_->commandDag;
    contexts.reserve(state_->commandDag.nodes.size());
    fragment.encoders.reserve(state_->commandDag.nodes.size());
    for (const detail::CommandNode &node : state_->commandDag.nodes) {
        contexts.push_back({*state_, node, resources});
        fragment.encoders.push_back({encodeNode, &contexts.back()});
    }
    std::unordered_set<detail::CommandAccessDomain, detail::CommandAccessDomainHash> bound;
    for (const detail::CommandNode &node : state_->commandDag.nodes)
        for (const detail::CommandResourceAccess &access : node.accesses) {
            if (access.resource >= state_->resourceRecords.size())
                continue;
            const auto &record = state_->resourceRecords[access.resource];
            if (bound.insert(detail::commandAccessDomain(access)).second)
                fragment.bindings.push_back({access.aliasDomain, access.kind, record.buffer, record.image});
        }
    detail::ValidatedRhiCommandPlan plan;
    std::string error;
    const VernonRhiStatus status =
        plan.append(std::move(fragment), error)
            ? detail::submitRhiCommandPlan(state_->device,
                                           computeOnly ? VERNON_RHI_QUEUE_COMPUTE
                                                       : VERNON_RHI_QUEUE_COMPUTE | VERNON_RHI_QUEUE_GRAPHICS,
                                           plan, submission->completion)
            : VERNON_RHI_STATUS_INVALID_ARGUMENT;
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
