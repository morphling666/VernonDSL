#include "submission_record.h"

#include "execution_graph/execution_graph_internal.h"
#include "failure_injection.h"
#include "runtime/runtime_state.h"

#include <algorithm>
#include <chrono>
#include <iterator>
#include <string>
#include <utility>

namespace vernon::runtime::program_execution {

SubmissionRecord::~SubmissionRecord() {
    Released released;
    (void)retire(released);
}

bool SubmissionRecord::retire(Released &released) noexcept {
    bool succeeded = true;
    for (const Pending &pending : pending_) {
        const auto started = std::chrono::steady_clock::now();
        if (vernonRhiCompletionWait(context_->rhiDevice, pending.completion) != VERNON_RHI_STATUS_OK)
            succeeded = false;
        else if (injectFailure(FailureBoundary::Completion))
            succeeded = false, injectedCompletionFailure_ = true;
        deviceWaitNanoseconds_ += static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count());
        (void)vernonRhiDeviceDestroyCompletion(context_->rhiDevice, pending.completion);
    }
    std::move(pending_.begin(), pending_.end(), std::back_inserter(released.pending));
    pending_.clear();
    std::move(retained_.begin(), retained_.end(), std::back_inserter(released.retained));
    retained_.clear();
    if (!succeeded) {
        state_ = VERNON_SUBMISSION_FAILED;
        status_ = VERNON_STATUS_INTERNAL_ERROR;
    }
    return succeeded;
}

VernonStatus SubmissionRecord::fail(const char *fallback) {
    std::string &detail = invocationDiagnostic(*context_);
    if (std::exchange(injectedCompletionFailure_, false))
        detail = "injected GPU command DAG wait failure";
    if (detail.empty()) {
        const VernonStringView backendError = vernonRhiDeviceGetLastError(context_->rhiDevice);
        if (backendError.data && backendError.size)
            detail.assign(backendError.data, backendError.size);
        else
            detail = fallback;
    }
    return VERNON_STATUS_INTERNAL_ERROR;
}

VernonStatus SubmissionRecord::submit(execution::detail::RhiCommandExecutionPlan plan) {
    if (injectFailure(FailureBoundary::Submission)) {
        invocationDiagnostic(*context_) = "injected GPU command DAG encoding failure";
        return VERNON_STATUS_INTERNAL_ERROR;
    }
    const uint32_t requiredCapabilities =
        std::any_of(plan.commands.nodes.begin(), plan.commands.nodes.end(),
                    [](const auto &node) { return node.queue == execution::detail::CommandQueueClass::Graphics; })
            ? VERNON_RHI_QUEUE_GRAPHICS
            : VERNON_RHI_QUEUE_COMPUTE;
    pending_.reserve(pending_.size() + 1);
    VernonRhiCompletion completion{static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
    if (execution::detail::submitRhiCommandPlan(context_->rhiDevice, requiredCapabilities, plan, completion) !=
        VERNON_RHI_STATUS_OK)
        return fail("GPU command program failed");
    pending_.push_back({completion, std::move(plan.retainedContexts)});
    ++submitted_;
    return VERNON_STATUS_OK;
}

VernonRhiStatus SubmissionRecord::submit(VernonRhiCommandEncoder encoder) {
    pending_.reserve(pending_.size() + 1);
    VernonRhiCompletion completion{static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
    const VernonRhiStatus status = vernonRhiDeviceSubmit(context_->rhiDevice, encoder, &completion);
    if (status != VERNON_RHI_STATUS_OK) {
        (void)vernonRhiDeviceDestroyCommandEncoder(context_->rhiDevice, encoder);
        return status;
    }
    pending_.push_back({completion, {}});
    ++submitted_;
    return VERNON_RHI_STATUS_OK;
}

void SubmissionRecord::retain(std::shared_ptr<const void> object) {
    if (pending_.empty())
        return;
    retained_.push_back(std::move(object));
}

VernonStatus SubmissionRecord::wait() {
    Released released;
    std::lock_guard lock(mutex_);
    const bool succeeded = retire(released);
    return succeeded && status_ == VERNON_STATUS_OK ? VERNON_STATUS_OK : fail("GPU command program failed");
}

VernonStatus SubmissionRecord::execute(execution::detail::RhiCommandExecutionPlan plan, SubmissionState &step) {
    step = SubmissionState::NotSubmitted;
    if (const VernonStatus status = submit(std::move(plan)); status != VERNON_STATUS_OK)
        return status;
    if (asynchronous()) {
        step = SubmissionState::Submitted;
        return VERNON_STATUS_OK;
    }
    step = SubmissionState::Indeterminate;
    if (const VernonStatus status = wait(); status != VERNON_STATUS_OK)
        return status;
    step = SubmissionState::Completed;
    return VERNON_STATUS_OK;
}

SubmissionRecord::Progress SubmissionRecord::progress() const noexcept {
    for (const Pending &entry : pending_) {
        VernonRhiCompletionState state{};
        if (vernonRhiCompletionGetState(context_->rhiDevice, entry.completion, &state) != VERNON_RHI_STATUS_OK)
            return Progress::QueryFailed;
        if (state == VERNON_RHI_COMPLETION_PENDING)
            return Progress::Pending;
    }
    return Progress::Finished;
}

VernonStatus SubmissionRecord::observe(bool &finished, Released &released) {
    finished = false;
    switch (progress()) {
    case Progress::QueryFailed:
        return fail("cannot query a GPU command program completion");
    case Progress::Pending:
        return VERNON_STATUS_OK;
    case Progress::Finished:
        break;
    }
    finished = true;
    if (!retire(released))
        (void)fail("GPU command program failed");
    return VERNON_STATUS_OK;
}

VernonStatus SubmissionRecord::poll(VernonSubmissionState &output) {
    Released released;
    std::lock_guard lock(mutex_);
    bool finished = false;
    if (const VernonStatus status = observe(finished, released); status != VERNON_STATUS_OK)
        return status;
    output = finished ? state_ : VERNON_SUBMISSION_PENDING;
    return VERNON_STATUS_OK;
}

VernonStatus SubmissionRecord::completionOutcome(const InvocationMutationOutcome *&output) {
    Released released;
    std::lock_guard lock(mutex_);
    output = nullptr;
    bool finished = false;
    if (const VernonStatus status = observe(finished, released); status != VERNON_STATUS_OK || !finished)
        return status;
    completedOutcome_ = outcome_.completed(state_ == VERNON_SUBMISSION_SUCCEEDED);
    output = &completedOutcome_;
    return VERNON_STATUS_OK;
}

bool SubmissionRecord::settle(bool block) noexcept {
    Released released;
    std::lock_guard lock(mutex_);
    if (!block && progress() != Progress::Finished)
        return false;
    (void)retire(released);
    return true;
}

} // namespace vernon::runtime::program_execution
