#ifndef VERNON_RUNTIME_PROGRAM_EXECUTION_SUBMISSION_QUEUE_H
#define VERNON_RUNTIME_PROGRAM_EXECUTION_SUBMISSION_QUEUE_H

#include "VernonRHI.h"
#include "VernonRuntime.h"

#include <array>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>

struct VernonRuntimeContext;

namespace vernon::runtime::program_execution {

class SubmissionRecord;

struct SubmissionBounds {
    uint64_t transientBudgetBytes;
    uint32_t maxPendingSubmissions;
};

// Provisional defaults, sized so that two frames of the Engine FrameGraph fit; they are tuning values.
inline constexpr SubmissionBounds defaultSubmissionBounds{uint64_t{256} << 20, 16};

// What one more submission is expected to hold while it is pending.
struct SubmissionDemand {
    uint64_t transientBytes;
    uint32_t rhiSubmissions;
};

// The submissions of one context that were queued without a wait and are not retired yet, oldest first. A record is
// retired once its work has finished, which releases the storage it retains; the queue observes records but never
// keeps one alive, because the handle that owns a record waits for it before destroying it.
class SubmissionQueue {
public:
    // `device` is the RHI device the context submits to; an invalid handle means the context has no RHI quota.
    void configure(SubmissionBounds bounds, VernonRhiDevice device) noexcept {
        bounds_ = bounds;
        device_ = device;
    }

    // Retires finished submissions, then waits for the oldest pending one while admitting `demand` would exceed a
    // bound. The device's in-flight submission quota is read on every admission, since its owner may change it, and
    // only this queue's pending RHI submissions are counted against it. Waiting never changes an outcome.
    void admit(SubmissionDemand demand);
    // Tracks a record whose work was submitted without a wait.
    void push(const std::shared_ptr<SubmissionRecord> &record, uint64_t transientBytes);
    VernonSubmissionTelemetry telemetry();

private:
    enum class Bound : uint8_t { PendingLimit, TransientBudget, DeviceQuota, Count };

    struct Entry {
        std::weak_ptr<SubmissionRecord> record;
        uint64_t transientBytes;
        uint32_t rhiSubmissions;
    };

    struct Waits {
        uint64_t count;
        uint64_t nanoseconds;
    };

    // Retires the records whose work has finished and drops entries whose record was destroyed.
    void retireFinished();
    uint32_t deviceInFlightQuota() const noexcept;

    std::mutex mutex_;
    SubmissionBounds bounds_{defaultSubmissionBounds};
    VernonRhiDevice device_{static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
    std::deque<Entry> pending_;
    uint64_t pendingBytes_{};
    uint64_t pendingRhiSubmissions_{};
    std::array<Waits, static_cast<size_t>(Bound::Count)> waits_{};
};

// Charges the device storage Runtime allocates for `context` on this thread, while the meter is alive, to the
// submission being built. Allocation sites call `charge` without knowing which Execute they serve.
class TransientAllocationMeter {
public:
    explicit TransientAllocationMeter(const VernonRuntimeContext &context) noexcept;
    TransientAllocationMeter(const TransientAllocationMeter &) = delete;
    TransientAllocationMeter &operator=(const TransientAllocationMeter &) = delete;
    ~TransientAllocationMeter();

    uint64_t bytes() const noexcept { return bytes_; }
    static void charge(const VernonRuntimeContext &context, uint64_t bytes) noexcept;

private:
    const VernonRuntimeContext *context_;
    TransientAllocationMeter *outer_;
    uint64_t bytes_{};
};

} // namespace vernon::runtime::program_execution

#endif
