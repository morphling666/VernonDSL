#ifndef VERNON_RUNTIME_PROGRAM_EXECUTION_SUBMISSION_RECORD_H
#define VERNON_RUNTIME_PROGRAM_EXECUTION_SUBMISSION_RECORD_H

#include "VernonRHI.h"
#include "VernonRuntime.h"
#include "invocation_outcome.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

struct VernonRuntimeContext;

namespace vernon::execution::detail {
struct RhiCommandExecutionPlan;
} // namespace vernon::execution::detail

namespace vernon::runtime::program_execution {

enum class CompletionMode : uint8_t { Wait, Asynchronous };

// The device work of one Execute, pullback apply or Stage submission. Each submitted plan keeps its retained contexts
// until its completion has finished, because the device reads the storage they own until then. The producing call
// alone submits, retains and sets the outcome; afterwards the handle owner and the context's submission queue may
// query, wait and settle from different threads.
class SubmissionRecord {
public:
    explicit SubmissionRecord(VernonRuntimeContext &context, CompletionMode mode = CompletionMode::Wait) noexcept
        : context_(&context), mode_(mode) {}
    SubmissionRecord(const SubmissionRecord &) = delete;
    SubmissionRecord &operator=(const SubmissionRecord &) = delete;
    ~SubmissionRecord();

    bool asynchronous() const noexcept { return mode_ == CompletionMode::Asynchronous; }

    VernonStatus submit(execution::detail::RhiCommandExecutionPlan plan);
    // Takes ownership of a finished encoder, and destroys it when the device rejects it.
    VernonRhiStatus submit(VernonRhiCommandEncoder encoder);
    // Keeps `object` alive until every completion of the record has finished.
    void retain(std::shared_ptr<const void> object);
    // The outcome the producing Execute reported; its submitted states resolve when the record completes.
    void setOutcome(InvocationMutationOutcome outcome) { outcome_ = std::move(outcome); }
    // Waits for everything submitted so far.
    VernonStatus wait();
    // Submits `plan`, and in WAIT mode waits for it. `step` reports how far the plan got: NotSubmitted when the device
    // never accepted it, Submitted when it was queued without a wait, Indeterminate when it was submitted but did not
    // complete successfully, Completed otherwise.
    VernonStatus execute(execution::detail::RhiCommandExecutionPlan plan, SubmissionState &step);
    // PENDING until every submitted completion has finished.
    VernonStatus poll(VernonSubmissionState &output);
    // `output` is null while any completion is pending.
    VernonStatus completionOutcome(const InvocationMutationOutcome *&output);
    // Retires finished work without reporting it, waiting for it when `block` is set; false while work is pending.
    bool settle(bool block) noexcept;

    uint64_t deviceWaitNanoseconds() const noexcept { return deviceWaitNanoseconds_; }
    // The RHI submissions the device accepted for this record; only the producing call may read it.
    uint32_t submittedCount() const noexcept { return submitted_; }

private:
    struct Pending {
        VernonRhiCompletion completion;
        std::vector<std::shared_ptr<void>> retained;
    };

    // What a retired record held. Releasing it can call back into Runtime, including into this record, so it is
    // destroyed only after `mutex_` is unlocked.
    struct Released {
        std::vector<Pending> pending;
        std::vector<std::shared_ptr<const void>> retained;
    };

    enum class Progress : uint8_t { Pending, Finished, QueryFailed };

    Progress progress() const noexcept;
    // Retires the pending completions once all of them have finished; `finished` is false while any is pending.
    VernonStatus observe(bool &finished, Released &released);
    // Waits for every pending completion, failed ones included, so that no retained storage is released while the
    // device may still read it, then moves everything the record retains into `released`. Completion-boundary
    // failure injection applies to each completion that succeeded.
    bool retire(Released &released) noexcept;
    VernonStatus fail(const char *fallback);

    std::mutex mutex_;
    VernonRuntimeContext *context_;
    CompletionMode mode_;
    std::vector<Pending> pending_;
    std::vector<std::shared_ptr<const void>> retained_;
    InvocationMutationOutcome outcome_;
    InvocationMutationOutcome completedOutcome_;
    VernonSubmissionState state_{VERNON_SUBMISSION_SUCCEEDED};
    VernonStatus status_{VERNON_STATUS_OK};
    bool injectedCompletionFailure_{};
    uint32_t submitted_{};
    uint64_t deviceWaitNanoseconds_{};
};

} // namespace vernon::runtime::program_execution

#endif
