#include "VernonRuntime.h"
#include "runtime/capi/boundary.h"
#include "runtime/capi/graphics_recording.h"
#include "runtime/capi/program_state.h"
#include "runtime/program_execution/invocation_outcome.h"
#include "runtime/program_execution/program_boundary_contract.h"
#include "runtime/program_execution/program_forward.h"
#include "runtime/program_execution/submission_queue.h"
#include "runtime/program_execution/submission_record.h"
#include "runtime/program_invocation_context.h"

#include <algorithm>
#include <iterator>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace vernon::runtime;
using namespace vernon::runtime::capi;

namespace {
using ProgramInstanceHandleResult = vernon::Result<VernonProgramInstance *, vernon::RuntimeError>;

ProgramInstanceHandleResult createProgramInstanceOperation(VernonProgramExecutable *pipeline) {
    VernonRuntimeContext *context = pipeline ? pipeline->context : nullptr;
    if (!pipeline || !pipeline->context) {
        fail(context, "invalid Program instance creation");
        return ProgramInstanceHandleResult{vernon::err(vernon::RuntimeError{
            vernon::RuntimeErrorCode::InvalidArgument, {"invalid Program instance creation", 0, 0}})};
    }
    auto pipelinePin = pipeline->lifecycle.pin();
    if (pipelinePin.isErr()) {
        fail(context, "Program executable is closing", vernon::toVernonStatus(pipelinePin.error()));
        return ProgramInstanceHandleResult{vernon::err(std::move(pipelinePin).error())};
    }
    auto child = RuntimeChildLifecycle::reserve(pipeline->instanceOwner);
    if (child.isErr()) {
        fail(context, "Program executable cannot admit an instance", vernon::toVernonStatus(child.error()));
        return ProgramInstanceHandleResult{vernon::err(std::move(child).error())};
    }
    auto invocationOwner = vernon::OwnerControlBlock::create();
    if (invocationOwner.isErr()) {
        fail(context, "cannot create Program instance invocation owner",
             vernon::toVernonStatus(invocationOwner.error()));
        return ProgramInstanceHandleResult{vernon::err(vernon::toRuntimeError(invocationOwner.error()))};
    }
    auto state = std::make_shared<RuntimeProgramInstanceState>(*pipeline, std::move(child).value(),
                                                               std::move(invocationOwner).value());
    auto instance = std::make_unique<VernonProgramInstance>(state);
    auto published = state->lifecycle.publish();
    if (published.isErr()) {
        fail(context, "cannot publish Program instance", vernon::toVernonStatus(published.error()));
        return ProgramInstanceHandleResult{vernon::err(std::move(published).error())};
    }
    return ProgramInstanceHandleResult{vernon::ok(instance.release())};
}
} // namespace

VernonRuntimeOperationStatus vernonRuntimeProgramInstanceCreateResult(VernonProgramExecutable *pipeline,
                                                                      VernonProgramInstance **output) {
    VernonRuntimeContext *context = pipeline ? pipeline->context : nullptr;
    return runtimeHandleResultBoundary(context, output, [&] { return createProgramInstanceOperation(pipeline); });
}

VernonProgramInstance *vernonRuntimeProgramInstanceCreate(VernonProgramExecutable *pipeline) {
    VernonProgramInstance *output = nullptr;
    (void)vernonRuntimeProgramInstanceCreateResult(pipeline, &output);
    return output;
}

void vernonRuntimeProgramInstanceDestroy(VernonProgramInstance *instance) {
    VernonRuntimeContext *context = instanceContext(instance);
    runtimeVoidBoundary(context, [&] { delete instance; });
}

namespace {
using ProgramInvocationHandleResult = vernon::Result<VernonProgramInvocation *, vernon::RuntimeError>;

ProgramInvocationHandleResult beginProgramInvocationOperation(VernonProgramInstance *instance) {
    VernonRuntimeContext *context = instanceContext(instance);
    if (!instance || !instance->state || !instance->state->pipeline) {
        fail(context, "invalid Program invocation begin");
        return ProgramInvocationHandleResult{vernon::err(vernon::RuntimeError{
            vernon::RuntimeErrorCode::InvalidArgument, {"invalid Program invocation begin", 0, 0}})};
    }
    auto instancePin = instance->state->lifecycle.pin();
    if (instancePin.isErr()) {
        fail(context, "Program instance is closing", vernon::toVernonStatus(instancePin.error()));
        return ProgramInvocationHandleResult{vernon::err(std::move(instancePin).error())};
    }
    auto child = RuntimeChildLifecycle::reserve(instance->state->owner);
    if (child.isErr()) {
        fail(context, "Program instance cannot admit an invocation", vernon::toVernonStatus(child.error()));
        return ProgramInvocationHandleResult{vernon::err(std::move(child).error())};
    }
    auto transaction = instance->state->bindings.beginInvocation();
    if (transaction.isErr()) {
        fail(context, "cannot begin Program binding transaction", VERNON_STATUS_INTERNAL_ERROR);
        return ProgramInvocationHandleResult{vernon::err(vernon::RuntimeError{
            vernon::RuntimeErrorCode::InternalFailure, {"cannot begin Program binding transaction", 0, 0}})};
    }
    auto invocation =
        std::make_unique<VernonProgramInvocation>(*instance, std::move(child).value(), std::move(transaction).value());
    invocation->executionWorkspace = instance->state->workspaces->acquire();
    if (!invocation->executionWorkspace) {
        fail(context, "cannot acquire Program execution workspace", VERNON_STATUS_INTERNAL_ERROR);
        return ProgramInvocationHandleResult{vernon::err(vernon::RuntimeError{
            vernon::RuntimeErrorCode::ResourceExhausted, {"cannot acquire Program execution workspace", 0, 0}})};
    }
    auto published = invocation->lifecycle.publish();
    if (published.isErr()) {
        fail(context, "cannot publish Program invocation", vernon::toVernonStatus(published.error()));
        return ProgramInvocationHandleResult{vernon::err(std::move(published).error())};
    }
    return ProgramInvocationHandleResult{vernon::ok(invocation.release())};
}
} // namespace

VernonRuntimeOperationStatus vernonRuntimeProgramInstanceBeginInvocationResult(VernonProgramInstance *instance,
                                                                               VernonProgramInvocation **output) {
    VernonRuntimeContext *context = instanceContext(instance);
    return runtimeHandleResultBoundary(context, output, [&] { return beginProgramInvocationOperation(instance); });
}

VernonProgramInvocation *vernonRuntimeProgramInstanceBeginInvocation(VernonProgramInstance *instance) {
    VernonProgramInvocation *output = nullptr;
    (void)vernonRuntimeProgramInstanceBeginInvocationResult(instance, &output);
    return output;
}

VernonStatus vernonRuntimeProgramInvocationSetAutodiffOptions(VernonProgramInvocation *invocation,
                                                              const VernonProgramAutodiffInvocationOptions *options) {
    VernonRuntimeContext *context = invocationContext(invocation);
    return runtimeStatusBoundary(context, [&] {
        const bool reservedBytesSet =
            options && std::any_of(std::begin(options->reserved_bytes), std::end(options->reserved_bytes),
                                   [](uint8_t value) { return value != 0; });
        const bool reservedWordsSet = options && std::any_of(std::begin(options->reserved), std::end(options->reserved),
                                                             [](uint32_t value) { return value != 0; });
        if (!invocation || invocation->executed || invocation->finished || !options ||
            options->struct_size < sizeof(*options) ||
            options->abi_version != VERNON_PROGRAM_AUTODIFF_INVOCATION_OPTIONS_VERSION ||
            options->has_checkpoint_memory_budget > 1 || reservedBytesSet || reservedWordsSet ||
            (options->checkpoint_policy.size && !options->checkpoint_policy.data))
            return fail(context, "invalid Program autodiff invocation options");
        auto invocationPin = invocation->lifecycle.pin();
        if (invocationPin.isErr())
            return fail(context, "Program invocation is closing", vernon::toVernonStatus(invocationPin.error()));
        std::string policy;
        if (options->checkpoint_policy.size)
            policy.assign(options->checkpoint_policy.data, options->checkpoint_policy.size);
        if (!policy.empty() && policy != "min_memory" && policy != "balanced" && policy != "min_runtime")
            return fail(context, "Program autodiff invocation checkpoint policy is invalid");
        invocation->checkpointMemoryBudget = options->has_checkpoint_memory_budget
                                                 ? std::optional<uint64_t>(options->checkpoint_memory_budget)
                                                 : std::nullopt;
        invocation->checkpointPolicy.swap(policy);
        return VERNON_STATUS_OK;
    });
}

VernonStatus vernonRuntimeProgramInvocationSetOptions(VernonProgramInvocation *invocation,
                                                      const VernonProgramInvocationOptions *options) {
    VernonRuntimeContext *context = invocationContext(invocation);
    return runtimeStatusBoundary(context, [&] {
        const bool reservedSet = options && std::any_of(std::begin(options->reserved), std::end(options->reserved),
                                                        [](uint32_t value) { return value != 0; });
        if (!invocation || invocation->executed || invocation->finished || !options ||
            options->struct_size < sizeof(*options) ||
            options->abi_version != VERNON_PROGRAM_INVOCATION_OPTIONS_VERSION || reservedSet ||
            (options->completion_mode != VERNON_COMPLETION_WAIT &&
             options->completion_mode != VERNON_COMPLETION_ASYNCHRONOUS) ||
            options->timings > 1)
            return fail(context, "invalid Program invocation options");
        auto invocationPin = invocation->lifecycle.pin();
        if (invocationPin.isErr())
            return fail(context, "Program invocation is closing", vernon::toVernonStatus(invocationPin.error()));
        invocation->completionMode = options->completion_mode == VERNON_COMPLETION_ASYNCHRONOUS
                                         ? vernon::runtime::program_execution::CompletionMode::Asynchronous
                                         : vernon::runtime::program_execution::CompletionMode::Wait;
        invocation->timings = options->timings != 0;
        return VERNON_STATUS_OK;
    });
}

namespace {
using ProgramBindingResult = vernon::Result<void, vernon::RuntimeError>;

ProgramBindingResult programBindingFailure(VernonRuntimeContext *context, vernon::RuntimeErrorCode code,
                                           const char *diagnostic) {
    (void)fail(context, diagnostic, vernon::toVernonStatus(vernon::RuntimeError{code, {}}));
    return ProgramBindingResult{vernon::err(vernon::RuntimeError{code, {diagnostic, 0, 0}})};
}

// A binding is reused only when every alias already holds `key`; the reuse then counts once per alias.
program::BindingResult<bool> reuseAliasBindings(program::BindingTransaction &transaction,
                                                const std::vector<uint32_t> &aliases, std::string_view key,
                                                uint64_t uploadBytes, uint64_t uploadRanges) {
    bool allMatch = true;
    for (uint32_t alias : aliases) {
        auto matches = transaction.matches(alias, key);
        if (matches.isErr())
            return matches;
        allMatch &= matches.value();
    }
    if (!allMatch)
        return program::BindingResult<bool>{vernon::ok(false)};
    auto reuses = transaction.observeReuses(aliases.size());
    if (reuses.isErr())
        return program::BindingResult<bool>{vernon::err(std::move(reuses).error())};
    auto uploads = transaction.observeUploads(uploadBytes, uploadRanges);
    if (uploads.isErr())
        return program::BindingResult<bool>{vernon::err(std::move(uploads).error())};
    return program::BindingResult<bool>{vernon::ok(true)};
}

ProgramBindingResult tryReuseProgramBindingOperation(VernonProgramInvocation *invocation, uint32_t slot,
                                                     const VernonProgramBindingToken *token, uint64_t uploadBytes,
                                                     uint64_t uploadRanges, uint8_t *reused) {
    VernonRuntimeContext *context = invocationContext(invocation);
    if (reused)
        *reused = 0;
    if (!invocation || invocation->executed || invocation->finished || !token || token->struct_size < sizeof(*token) ||
        !token->size || !token->data || !reused)
        return programBindingFailure(context, vernon::RuntimeErrorCode::InvalidArgument,
                                     "invalid persistent Program binding reuse query");
    auto invocationPin = invocation->lifecycle.pin();
    if (invocationPin.isErr())
        return programBindingFailure(context, vernon::RuntimeErrorCode::LifecycleFailure,
                                     "Program invocation is closing");
    if (!bindableParameterSlot(*invocation->instance->pipeline, slot))
        return programBindingFailure(context, vernon::RuntimeErrorCode::InvalidArgument,
                                     "Program binding reuse query does not reference an exported parameter");
    const std::string_view key(static_cast<const char *>(token->data), token->size);
    const std::vector<uint32_t> *aliases = bindingAliasSlots(*invocation->instance->pipeline, slot);
    if (!aliases)
        return programBindingFailure(context, vernon::RuntimeErrorCode::InternalFailure,
                                     "Program binding reuse query has no resolved aliases");
    auto reuse = reuseAliasBindings(*invocation->transaction, *aliases, key, uploadBytes, uploadRanges);
    if (reuse.isErr())
        return programBindingFailure(context, vernon::RuntimeErrorCode::LifecycleFailure,
                                     "Program binding reuse query is not in the staging phase");
    *reused = reuse.value() ? 1 : 0;
    return ProgramBindingResult{vernon::ok()};
}
} // namespace

VernonStatus vernonRuntimeProgramInvocationTryReuse(VernonProgramInvocation *invocation, uint32_t slot,
                                                    const VernonProgramBindingToken *token, uint64_t uploadBytes,
                                                    uint64_t uploadRanges, uint8_t *reused) {
    VernonRuntimeContext *context = invocationContext(invocation);
    return runtimeResultBoundary(context, [&] {
        return tryReuseProgramBindingOperation(invocation, slot, token, uploadBytes, uploadRanges, reused);
    });
}

namespace {
ProgramBindingResult bindProgramInvocationOperation(VernonProgramInvocation *invocation,
                                                    const VernonProgramBindingToken *token,
                                                    const VernonProgramArgument *argument,
                                                    const VernonProgramResourceLease *lease, uint64_t uploadBytes,
                                                    uint64_t uploadRanges) {
    VernonRuntimeContext *context = invocationContext(invocation);
    if (!invocation || invocation->executed || invocation->finished || !token || token->struct_size < sizeof(*token) ||
        !token->size || !token->data || !argument)
        return programBindingFailure(context, vernon::RuntimeErrorCode::InvalidArgument,
                                     "invalid persistent Program binding update");
    auto invocationPin = invocation->lifecycle.pin();
    if (invocationPin.isErr())
        return programBindingFailure(context, vernon::RuntimeErrorCode::LifecycleFailure,
                                     "Program invocation is closing");
    const program::Program &program = *executableProgram(*invocation->instance->pipeline);
    if (!bindableParameterSlot(*invocation->instance->pipeline, argument->slot))
        return programBindingFailure(context, vernon::RuntimeErrorCode::InvalidArgument,
                                     "Program binding does not reference an exported parameter");
    if (const char *diagnostic = programArgumentDiagnostic(program.abi.boundarySlots[argument->slot], *argument))
        return programBindingFailure(context, vernon::RuntimeErrorCode::InvalidArgument, diagnostic);
    const std::string_view key(static_cast<const char *>(token->data), token->size);
    const std::vector<uint32_t> *aliases = bindingAliasSlots(*invocation->instance->pipeline, argument->slot);
    if (!aliases)
        return programBindingFailure(context, vernon::RuntimeErrorCode::InternalFailure,
                                     "Program binding has no resolved aliases");
    for (uint32_t alias : *aliases) {
        if (alias >= program.abi.boundarySlots.size() || program.abi.boundarySlots[alias].id != alias)
            return programBindingFailure(context, vernon::RuntimeErrorCode::VerificationFailure,
                                         "Program binding alias does not reference a canonical boundary slot");
    }
    auto reuse = reuseAliasBindings(*invocation->transaction, *aliases, key, uploadBytes, uploadRanges);
    if (reuse.isErr())
        return programBindingFailure(context, vernon::RuntimeErrorCode::LifecycleFailure,
                                     "Program binding update is not in the staging phase");
    if (reuse.value())
        return ProgramBindingResult{vernon::ok()};
    auto made = makeProgramBinding(*argument, lease);
    if (made.isErr())
        return ProgramBindingResult{vernon::err(std::move(made).error())};
    const std::shared_ptr<RuntimeProgramBinding> binding = std::move(made).value();
    std::vector<std::pair<uint64_t, vernon::runtime::program::BindingEntry>> staged;
    staged.reserve(aliases->size());
    for (uint32_t alias : *aliases) {
        auto remapped = std::make_shared<RuntimeProgramBinding>(*binding);
        remapped->argument.slot = alias;
        if (remapped->argument.kind == VERNON_PROGRAM_TENSOR) {
            const program::BoundarySlot &boundary = program.abi.boundarySlots[alias];
            remapped->argument.tensor.access = vernon::runtime::program_execution::boundaryValueAccess(boundary.access);
        }
        remapped->refresh();
        staged.emplace_back(alias, vernon::runtime::program::BindingEntry{std::string(key), std::move(remapped)});
    }
    if (invocation->transaction->stageMany(std::move(staged), uploadBytes, uploadRanges).isErr())
        return programBindingFailure(context, vernon::RuntimeErrorCode::ResourceExhausted,
                                     "cannot stage persistent Program binding");
    return ProgramBindingResult{vernon::ok()};
}
} // namespace

VernonStatus vernonRuntimeProgramInvocationBind(VernonProgramInvocation *invocation,
                                                const VernonProgramBindingToken *token,
                                                const VernonProgramArgument *argument,
                                                const VernonProgramResourceLease *lease, uint64_t uploadBytes,
                                                uint64_t uploadRanges) {
    VernonRuntimeContext *context = invocationContext(invocation);
    return runtimeResultBoundary(context, [&] {
        return bindProgramInvocationOperation(invocation, token, argument, lease, uploadBytes, uploadRanges);
    });
}

namespace {
using ProgramControlResult = vernon::Result<void, vernon::RuntimeError>;

vernon::Result<std::string_view, vernon::RuntimeError>
programControlToken(const VernonProgramBindingToken *token) noexcept {
    if (!token || token->struct_size < sizeof(*token) || !token->data || !token->size)
        return vernon::Result<std::string_view, vernon::RuntimeError>{vernon::err(vernon::RuntimeError{
            vernon::RuntimeErrorCode::InvalidArgument, {"Program control binding token is invalid", 0, 0}})};
    return vernon::Result<std::string_view, vernon::RuntimeError>{
        vernon::ok(std::string_view{static_cast<const char *>(token->data), token->size})};
}

ProgramControlResult retainProgramControlLease(RuntimeProgramControl &control,
                                               const VernonProgramResourceLease &lease) {
    auto retained = retainProgramResourceLease(lease);
    if (retained.isErr())
        return ProgramControlResult{vernon::err(std::move(retained).error())};
    if (retained.value())
        control.leases.push_back(std::move(retained).value());
    return ProgramControlResult{vernon::ok()};
}

ProgramControlResult stageProgramControl(VernonProgramInvocation *invocation, uint32_t slot,
                                         const VernonProgramBindingToken *token,
                                         std::shared_ptr<RuntimeProgramControl> control) {
    if (!invocation || invocation->executed || invocation->finished || !control)
        return ProgramControlResult{vernon::err(vernon::RuntimeError{
            vernon::RuntimeErrorCode::InvalidArgument, {"invalid persistent Program control update", slot, 0}})};
    auto invocationPin = invocation->lifecycle.pin();
    if (invocationPin.isErr())
        return ProgramControlResult{vernon::err(vernon::RuntimeError{vernon::RuntimeErrorCode::LifecycleFailure,
                                                                     {"Program invocation is closing", slot, 0}})};
    auto keyResult = programControlToken(token);
    if (keyResult.isErr())
        return ProgramControlResult{vernon::err(std::move(keyResult).error())};
    const std::string_view key = std::move(keyResult).value();
    const uint64_t storageSlot = programControlKey(slot, control->kind);
    if (invocation->instance->pipeline->graphicsControlKeys.find(storageSlot) ==
        invocation->instance->pipeline->graphicsControlKeys.end())
        return ProgramControlResult{vernon::err(vernon::RuntimeError{
            vernon::RuntimeErrorCode::InvalidArgument, {"Program control slot or kind is not resolved", slot, 0}})};
    auto matches = invocation->transaction->matches(storageSlot, key);
    if (matches.isErr())
        return ProgramControlResult{
            vernon::err(vernon::RuntimeError{vernon::RuntimeErrorCode::LifecycleFailure,
                                             {"Program control update is not in the staging phase", slot, 0}})};
    if (matches.value()) {
        if (invocation->transaction->observeReuses(1).isErr())
            return ProgramControlResult{
                vernon::err(vernon::RuntimeError{vernon::RuntimeErrorCode::LifecycleFailure,
                                                 {"Program control reuse is not in the staging phase", slot, 0}})};
        return ProgramControlResult{vernon::ok()};
    }
    control->refresh();
    if (invocation->transaction->stage(storageSlot, std::string(key), std::move(control), 0, 0).isErr())
        return ProgramControlResult{vernon::err(vernon::RuntimeError{
            vernon::RuntimeErrorCode::ResourceExhausted, {"cannot stage persistent Program control", slot, 0}})};
    return ProgramControlResult{vernon::ok()};
}

ProgramControlResult bindProgramRenderPassOperation(VernonProgramInvocation *invocation, uint32_t controlSlot,
                                                    const VernonProgramBindingToken *token,
                                                    const VernonRenderPass *renderPass,
                                                    const VernonProgramResourceLease *leases, size_t leaseCount) {
    if (!renderPass || renderPass->struct_size < sizeof(*renderPass) ||
        (renderPass->color_attachment_count && !renderPass->color_attachments) || (leaseCount && !leases))
        return ProgramControlResult{
            vernon::err(vernon::RuntimeError{vernon::RuntimeErrorCode::InvalidArgument,
                                             {"invalid persistent Program RenderPass control", controlSlot, 0}})};
    auto control = std::make_shared<RuntimeProgramControl>();
    control->kind = RuntimeProgramControl::RenderPass;
    control->renderPass = *renderPass;
    control->colors.assign(renderPass->color_attachments,
                           renderPass->color_attachments + renderPass->color_attachment_count);
    if (renderPass->depth_attachment)
        control->depth = *renderPass->depth_attachment;
    for (size_t index = 0; index < leaseCount; ++index) {
        auto retained = retainProgramControlLease(*control, leases[index]);
        if (retained.isErr())
            return retained;
    }
    return stageProgramControl(invocation, controlSlot, token, std::move(control));
}

ProgramControlResult bindProgramDrawCommandOperation(VernonProgramInvocation *invocation, uint32_t controlSlot,
                                                     const VernonProgramBindingToken *token,
                                                     const VernonDrawCommand *draw,
                                                     const VernonProgramResourceLease *lease) {
    std::shared_ptr<RuntimeProgramControl> control = makeDrawCommandControl(draw);
    if (!control)
        return ProgramControlResult{
            vernon::err(vernon::RuntimeError{vernon::RuntimeErrorCode::InvalidArgument,
                                             {"invalid persistent Program DrawCommand control", controlSlot, 0}})};
    if (lease) {
        auto retained = retainProgramControlLease(*control, *lease);
        if (retained.isErr())
            return retained;
    }
    return stageProgramControl(invocation, controlSlot, token, std::move(control));
}

ProgramControlResult bindProgramDynamicStateOperation(VernonProgramInvocation *invocation, uint32_t controlSlot,
                                                      const VernonProgramBindingToken *token,
                                                      const VernonDynamicState *dynamicState) {
    if (!dynamicState || dynamicState->struct_size < sizeof(*dynamicState) || dynamicState->stencil_reference > 0xff)
        return ProgramControlResult{
            vernon::err(vernon::RuntimeError{vernon::RuntimeErrorCode::InvalidArgument,
                                             {"invalid persistent Program DynamicState control", controlSlot, 0}})};
    auto control = std::make_shared<RuntimeProgramControl>();
    control->kind = RuntimeProgramControl::DynamicState;
    control->dynamic = *dynamicState;
    return stageProgramControl(invocation, controlSlot, token, std::move(control));
}
} // namespace

VernonStatus vernonRuntimeProgramInvocationBindRenderPass(VernonProgramInvocation *invocation, uint32_t controlSlot,
                                                          const VernonProgramBindingToken *token,
                                                          const VernonRenderPass *renderPass,
                                                          const VernonProgramResourceLease *leases, size_t leaseCount) {
    VernonRuntimeContext *context = invocationContext(invocation);
    return runtimeResultBoundary(context, [&] {
        return bindProgramRenderPassOperation(invocation, controlSlot, token, renderPass, leases, leaseCount);
    });
}

VernonStatus vernonRuntimeProgramInvocationBindDrawCommand(VernonProgramInvocation *invocation, uint32_t controlSlot,
                                                           const VernonProgramBindingToken *token,
                                                           const VernonDrawCommand *draw,
                                                           const VernonProgramResourceLease *lease) {
    VernonRuntimeContext *context = invocationContext(invocation);
    return runtimeResultBoundary(
        context, [&] { return bindProgramDrawCommandOperation(invocation, controlSlot, token, draw, lease); });
}

VernonStatus vernonRuntimeProgramInvocationBindDynamicState(VernonProgramInvocation *invocation, uint32_t controlSlot,
                                                            const VernonProgramBindingToken *token,
                                                            const VernonDynamicState *dynamicState) {
    VernonRuntimeContext *context = invocationContext(invocation);
    return runtimeResultBoundary(
        context, [&] { return bindProgramDynamicStateOperation(invocation, controlSlot, token, dynamicState); });
}

VernonStatus vernonRuntimeProgramInvocationExecute(VernonProgramInvocation *invocation, uint8_t retainPullback,
                                                   VernonInvocationMutationOutcome *publicOutcome) {
    VernonProgramExecutable *pipeline = invocationExecutable(invocation);
    return runtimeStatusBoundary(pipeline ? pipeline->context : nullptr, [&] {
        if (!invocation || invocation->executed || invocation->finished || !pipeline)
            return fail(pipeline ? pipeline->context : nullptr, "invalid persistent Program invocation");
        if (!vernon::runtime::program_execution::preparePublicOutcome(publicOutcome, mutationCapacity(*pipeline)))
            return fail(pipeline->context, "invalid Program mutation outcome storage");
        auto invocationPin = invocation->lifecycle.pin();
        if (invocationPin.isErr())
            return fail(pipeline->context, "Program invocation is closing",
                        vernon::toVernonStatus(invocationPin.error()));
        auto submissionChild = vernon::runtime::RuntimeChildLifecycle::reserve(pipeline->context->owner);
        if (submissionChild.isErr())
            return fail(pipeline->context, "runtime context cannot admit a submission",
                        vernon::toVernonStatus(submissionChild.error()));
        auto submission = std::make_unique<VernonSubmission>(
            *pipeline->context, std::make_shared<vernon::runtime::program_execution::SubmissionRecord>(
                                    *pipeline->context, invocation->completionMode, invocation->timings));
        submission->lifecycle.emplace(std::move(submissionChild).value());
        invocation->executed = true;
        auto snapshot = invocation->transaction->freeze();
        if (snapshot.isErr())
            return fail(pipeline->context, "cannot freeze Program invocation bindings", VERNON_STATUS_INTERNAL_ERROR);
        invocation->snapshot = std::move(snapshot).value();
        std::vector<VernonProgramArgument> &arguments = invocation->executionWorkspace->boundaryArguments;
        arguments.clear();
        const program::Program &program = *executableProgram(*pipeline);
        const size_t count = publicBoundaryCount(program);
        if (arguments.capacity() < count)
            arguments.reserve(count);
        for (size_t index = 0; index < count; ++index) {
            const program::BoundarySlot *parameter = publicBoundaryAt(program, index);
            if (!parameter)
                return fail(pipeline->context, "cannot reflect persistent Program boundary slot");
            const std::shared_ptr<void> *payload = invocation->snapshot->find(parameter->id);
            if (!payload)
                return fail(pipeline->context,
                            "persistent Program invocation has an unbound boundary slot '" + parameter->path + "'");
            auto binding = std::static_pointer_cast<RuntimeProgramBinding>(*payload);
            arguments.push_back(binding->argument);
        }
        std::shared_ptr<vernon::runtime::RecordedGraphicsPasses> recordedGraphics;
        if (!invocation->recordings.empty()) {
            recordedGraphics = std::make_shared<vernon::runtime::RecordedGraphicsPasses>();
            if (const VernonStatus status = validateGraphicsRecordings(*invocation, *recordedGraphics);
                status != VERNON_STATUS_OK)
                return status;
        }
        const ProgramInvocationContext programContext{
            *invocation->snapshot,        invocation->checkpointMemoryBudget,
            invocation->checkpointPolicy, invocation->executionWorkspace.get(),
            recordedGraphics.get(),
        };
        VernonPullback *pullback = nullptr;
        vernon::runtime::program_execution::InvocationMutationOutcome outcome;
        vernon::runtime::program_execution::SubmissionRecord &record = *submission->record;
        pipeline->context->submissions.admit({invocation->instance->submissionTransientBytes.load(),
                                              invocation->instance->submissionRhiSubmissions.load()});
        vernon::runtime::program_execution::TransientAllocationMeter meter(*pipeline->context);
        const VernonStatus status = vernon::runtime::program_execution::forwardProgramInvocation(
            *pipeline, arguments.data(), arguments.size(), pullback, outcome, record, &programContext,
            pipeline->programGraphId && retainPullback ? &invocation->nodePullbacks : nullptr, retainPullback != 0);
        invocation->instance->submissionTransientBytes.store(meter.bytes());
        invocation->instance->submissionRhiSubmissions.store(record.submittedCount());
        vernon::runtime::program_execution::publishPublicOutcome(outcome, publicOutcome);
        if (recordedGraphics)
            for (const auto &[node, pass] : recordedGraphics->nodes)
                invocation->graphicsTelemetry[node] = pass.telemetry;
        if (outcome.submission != vernon::runtime::program_execution::SubmissionState::NotSubmitted) {
            const bool pending = outcome.submission == vernon::runtime::program_execution::SubmissionState::Submitted;
            record.retain(invocation->snapshot);
            record.retain(invocation->executionWorkspace);
            if (recordedGraphics) {
                record.retain(std::move(recordedGraphics));
                record.retain(std::make_shared<std::map<VernonProgramNodeId, RuntimeGraphicsRecording>>(
                    std::exchange(invocation->recordings, {})));
            }
            record.setOutcome(std::move(outcome));
            if (pending)
                pipeline->context->submissions.push(submission->record, meter.bytes());
            invocation->submission = std::move(submission);
        }
        if (status != VERNON_STATUS_OK) {
            if (pullback)
                vernonProgramPullbackDestroy(pullback);
            return status;
        }
        invocation->pendingPullback = pullback;
        invocation->succeeded = true;
        return VERNON_STATUS_OK;
    });
}

VernonStatus vernonRuntimeProgramInvocationCommit(VernonProgramInvocation *invocation,
                                                  VernonPullback **outputPullback) {
    VernonProgramExecutable *pipeline = invocationExecutable(invocation);
    return runtimeStatusBoundary(pipeline ? pipeline->context : nullptr, [&] {
        if (outputPullback)
            *outputPullback = nullptr;
        if (!invocation || !invocation->executed || !invocation->succeeded || invocation->finished || !pipeline)
            return fail(pipeline ? pipeline->context : nullptr, "invalid Program invocation commit");
        auto invocationPin = invocation->lifecycle.pin();
        if (invocationPin.isErr())
            return fail(pipeline->context, "Program invocation is closing",
                        vernon::toVernonStatus(invocationPin.error()));
        auto snapshot = invocation->transaction->commit();
        if (snapshot.isErr())
            return fail(pipeline->context, "cannot commit Program invocation bindings", VERNON_STATUS_INTERNAL_ERROR);
        invocation->snapshot = std::move(snapshot).value();
        invocation->finished = true;
        if (invocation->pendingPullback)
            vernon::runtime::program_execution::attachProgramSnapshot(*invocation->pendingPullback,
                                                                      invocation->snapshot);
        if (outputPullback)
            *outputPullback = std::exchange(invocation->pendingPullback, nullptr);
        else if (invocation->pendingPullback) {
            vernonProgramPullbackDestroy(invocation->pendingPullback);
            invocation->pendingPullback = nullptr;
        }
        return VERNON_STATUS_OK;
    });
}

VernonStatus vernonRuntimeProgramInvocationTakeSubmission(VernonProgramInvocation *invocation,
                                                          VernonSubmission **output) {
    VernonProgramExecutable *pipeline = invocationExecutable(invocation);
    return runtimeStatusBoundary(pipeline ? pipeline->context : nullptr, [&] {
        if (output)
            *output = nullptr;
        if (!invocation || !pipeline || !output)
            return fail(pipeline ? pipeline->context : nullptr, "invalid Program invocation submission retrieval");
        auto invocationPin = invocation->lifecycle.pin();
        if (invocationPin.isErr())
            return fail(pipeline->context, "Program invocation is closing",
                        vernon::toVernonStatus(invocationPin.error()));
        if (!invocation->submission)
            return fail(pipeline->context, "Program invocation has no submission to take");
        if (invocation->submission->lifecycle.value().publish().isErr())
            return fail(pipeline->context, "cannot publish Program invocation submission",
                        VERNON_STATUS_INTERNAL_ERROR);
        *output = invocation->submission.release();
        return VERNON_STATUS_OK;
    });
}

VernonStatus vernonRuntimeProgramInvocationGetNodePullback(VernonProgramInvocation *invocation,
                                                           VernonProgramNodeId node, VernonPullback **outputPullback) {
    VernonProgramExecutable *pipeline = invocationExecutable(invocation);
    return runtimeStatusBoundary(pipeline ? pipeline->context : nullptr, [&] {
        if (outputPullback)
            *outputPullback = nullptr;
        if (!invocation || !invocation->finished || !invocation->succeeded || !pipeline || !pipeline->programGraphId ||
            !outputPullback)
            return fail(pipeline ? pipeline->context : nullptr, "invalid ProgramGraph node pullback retrieval");
        auto invocationPin = invocation->lifecycle.pin();
        if (invocationPin.isErr())
            return fail(pipeline->context, "Program invocation is closing",
                        vernon::toVernonStatus(invocationPin.error()));
        const auto declared = pipeline->programGraphNodeAutodiff.find(node);
        if (declared == pipeline->programGraphNodeAutodiff.end())
            return fail(pipeline->context, "ProgramGraph node is not differentiable");
        const auto retained = invocation->nodePullbacks.find(node);
        if (retained == invocation->nodePullbacks.end() || !retained->second)
            return fail(pipeline->context, "ProgramGraph node pullback is unavailable");
        auto transfer = vernon::runtime::program_execution::makeRetainedProgramPullback(*pipeline, retained->second,
                                                                                        invocation->snapshot);
        if (transfer.isErr())
            return fail(pipeline->context, "cannot retain ProgramGraph node pullback", VERNON_STATUS_INTERNAL_ERROR);
        *outputPullback = std::move(transfer).value().release();
        invocation->nodePullbacks.erase(retained);
        return VERNON_STATUS_OK;
    });
}

void vernonRuntimeProgramInvocationRollback(VernonProgramInvocation *invocation) {
    VernonRuntimeContext *context = invocationContext(invocation);
    runtimeVoidBoundary(context, [&] {
        if (!invocation || invocation->finished)
            return;
        auto invocationPin = invocation->lifecycle.pin();
        if (invocationPin.isErr())
            return;
        invocation->transaction->rollback();
        if (invocation->pendingPullback) {
            vernonProgramPullbackDestroy(invocation->pendingPullback);
            invocation->pendingPullback = nullptr;
        }
        invocation->nodePullbacks.clear();
        invocation->succeeded = false;
        invocation->finished = true;
    });
}

void vernonRuntimeProgramInvocationDestroy(VernonProgramInvocation *invocation) {
    VernonRuntimeContext *context = invocationContext(invocation);
    runtimeVoidBoundary(context, [&] {
        if (!invocation)
            return;
        auto destruction = invocation->lifecycle.beginDestroy();
        if (destruction.isErr())
            return;
        if (destruction.value().commit().isErr())
            vernon::resultContractViolation();
        delete invocation;
    });
}

VernonStatus vernonRuntimeProgramInstanceGetTelemetry(const VernonProgramInstance *instance,
                                                      VernonProgramBindingTelemetry *output) {
    VernonRuntimeContext *context = instanceContext(instance);
    return runtimeStatusBoundary(context, [&] {
        if (!instance || !instance->state || !output || output->struct_size < sizeof(*output))
            return VERNON_STATUS_INVALID_ARGUMENT;
        auto instancePin = instance->state->lifecycle.pin();
        if (instancePin.isErr())
            return vernon::toVernonStatus(instancePin.error());
        const vernon::runtime::program::BindingTelemetry telemetry = instance->state->bindings.telemetry();
        *output = {sizeof(*output),         telemetry.prepareCount, telemetry.reuseCount,
                   telemetry.rollbackCount, telemetry.uploadBytes,  telemetry.uploadRanges};
        return VERNON_STATUS_OK;
    });
}

VernonStatus vernonSubmissionGetState(const VernonSubmission *submission, VernonSubmissionState *output) {
    return runtimeStatusBoundary(submission ? submission->context : nullptr, [&] {
        if (!submission || !output || !submission->lifecycle)
            return VERNON_STATUS_INVALID_ARGUMENT;
        auto pin = submission->lifecycle.value().pin();
        if (pin.isErr())
            return vernon::toVernonStatus(pin.error());
        return submission->record->poll(*output);
    });
}

VernonStatus vernonSubmissionWait(VernonSubmission *submission) {
    return runtimeStatusBoundary(submission ? submission->context : nullptr, [&] {
        if (!submission || !submission->lifecycle)
            return VERNON_STATUS_INVALID_ARGUMENT;
        auto pin = submission->lifecycle.value().pin();
        if (pin.isErr())
            return vernon::toVernonStatus(pin.error());
        return submission->record->wait();
    });
}

VernonStatus vernonSubmissionGetMutationOutcome(const VernonSubmission *submission,
                                                VernonInvocationMutationOutcome *outcome) {
    return runtimeStatusBoundary(submission ? submission->context : nullptr, [&] {
        if (!submission || !outcome || !submission->lifecycle)
            return VERNON_STATUS_INVALID_ARGUMENT;
        auto pin = submission->lifecycle.value().pin();
        if (pin.isErr())
            return vernon::toVernonStatus(pin.error());
        const vernon::runtime::program_execution::InvocationMutationOutcome *completed = nullptr;
        if (const VernonStatus status = submission->record->completionOutcome(completed); status != VERNON_STATUS_OK)
            return status;
        if (!completed)
            return VERNON_STATUS_NOT_READY;
        if (!vernon::runtime::program_execution::preparePublicOutcome(outcome, completed->boundaries.size()))
            return fail(submission->context, "invalid submission mutation outcome storage");
        vernon::runtime::program_execution::publishPublicOutcome(*completed, outcome);
        return VERNON_STATUS_OK;
    });
}

VernonStatus vernonSubmissionGetScopeTimings(const VernonSubmission *submission, VernonScopeTiming *scopes,
                                             size_t capacity, size_t *count) {
    return runtimeStatusBoundary(submission ? submission->context : nullptr, [&] {
        if (count)
            *count = 0;
        if (!submission || !count || !submission->lifecycle)
            return VERNON_STATUS_INVALID_ARGUMENT;
        auto pin = submission->lifecycle.value().pin();
        if (pin.isErr())
            return vernon::toVernonStatus(pin.error());
        return submission->record->scopeTimings(scopes, capacity, *count);
    });
}

void vernonSubmissionDestroy(VernonSubmission *submission) {
    runtimeVoidBoundary(submission ? submission->context : nullptr, [&] {
        if (!submission)
            return;
        if (!submission->lifecycle)
            vernon::resultContractViolation();
        auto destruction = submission->lifecycle.value().beginDestroy();
        if (destruction.isErr())
            return;
        // The context may destroy its device once the destruction commits.
        (void)submission->record->wait();
        if (destruction.value().commit().isErr())
            vernon::resultContractViolation();
        delete submission;
    });
}
