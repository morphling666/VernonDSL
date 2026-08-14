#include "VernonExecutionGraph.h"

#include "execution_graph_internal.h"
#include "rhi/rhi_internal.h"

#include <algorithm>
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
    state->autodiffCheckpointPlan = std::move(autodiffCheckpointPlan_);
    state->autodiffInitialResources = std::move(autodiffInitialResources_);
    state->autodiffRestorationResources = std::move(autodiffRestorationResources_);
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
                                                uint32_t endOffset, CpuPassExecutor executor, void *context) {
    if (beginOffset > endOffset || endOffset > schedule.size())
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::vector<VernonRhiBuffer> buffers(resources.size(),
                                         VernonRhiBuffer{static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0});
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
