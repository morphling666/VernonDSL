#include "submission_queue.h"

#include "submission_record.h"

#include <algorithm>
#include <chrono>
#include <limits>
#include <utility>
#include <vector>

namespace vernon::runtime::program_execution {

namespace {
thread_local TransientAllocationMeter *currentMeter = nullptr;
} // namespace

void SubmissionQueue::retireFinished() {
    std::vector<std::shared_ptr<SubmissionRecord>> records;
    {
        std::lock_guard lock(mutex_);
        records.reserve(pending_.size());
        for (const Entry &entry : pending_)
            records.push_back(entry.record.lock());
    }
    // Retiring releases retained storage whose owners may call back into Runtime, so it runs without the lock.
    std::vector<const SubmissionRecord *> finished;
    finished.reserve(records.size());
    for (const std::shared_ptr<SubmissionRecord> &record : records)
        if (record && record->settle(false))
            finished.push_back(record.get());
    std::lock_guard lock(mutex_);
    pending_.erase(std::remove_if(pending_.begin(), pending_.end(),
                                  [&](const Entry &entry) {
                                      const std::shared_ptr<SubmissionRecord> record = entry.record.lock();
                                      if (record &&
                                          std::find(finished.begin(), finished.end(), record.get()) == finished.end())
                                          return false;
                                      pendingBytes_ -= entry.transientBytes;
                                      pendingRhiSubmissions_ -= entry.rhiSubmissions;
                                      return true;
                                  }),
                   pending_.end());
}

uint32_t SubmissionQueue::deviceInFlightQuota() const noexcept {
    VernonRhiCommandLimits limits{};
    limits.struct_size = sizeof(limits);
    if (device_.index == VERNON_RHI_INVALID_HANDLE_INDEX ||
        vernonRhiDeviceGetCommandLimits(device_, &limits) != VERNON_RHI_STATUS_OK)
        return (std::numeric_limits<uint32_t>::max)();
    return limits.max_in_flight_submissions;
}

void SubmissionQueue::admit(SubmissionDemand demand) {
    for (;;) {
        retireFinished();
        const uint32_t quota = deviceInFlightQuota();
        std::shared_ptr<SubmissionRecord> oldest;
        Bound bound{};
        {
            std::lock_guard lock(mutex_);
            if (pending_.empty())
                return;
            if (pending_.size() >= bounds_.maxPendingSubmissions)
                bound = Bound::PendingLimit;
            else if (pendingBytes_ > bounds_.transientBudgetBytes ||
                     demand.transientBytes > bounds_.transientBudgetBytes - pendingBytes_)
                bound = Bound::TransientBudget;
            else if (pendingRhiSubmissions_ + demand.rhiSubmissions > quota)
                bound = Bound::DeviceQuota;
            else
                return;
            oldest = pending_.front().record.lock();
        }
        if (!oldest)
            continue;
        const auto started = std::chrono::steady_clock::now();
        (void)oldest->settle(true);
        const auto waited = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count());
        std::lock_guard lock(mutex_);
        Waits &waits = waits_[static_cast<size_t>(bound)];
        ++waits.count;
        waits.nanoseconds += waited;
    }
}

void SubmissionQueue::push(const std::shared_ptr<SubmissionRecord> &record, uint64_t transientBytes) {
    const uint32_t rhiSubmissions = record->submittedCount();
    std::lock_guard lock(mutex_);
    pending_.push_back({record, transientBytes, rhiSubmissions});
    pendingBytes_ += transientBytes;
    pendingRhiSubmissions_ += rhiSubmissions;
}

VernonSubmissionTelemetry SubmissionQueue::telemetry() {
    std::lock_guard lock(mutex_);
    VernonSubmissionTelemetry result{};
    result.struct_size = sizeof(result);
    result.pending_submissions =
        static_cast<uint32_t>(std::min<size_t>(pending_.size(), (std::numeric_limits<uint32_t>::max)()));
    result.pending_transient_bytes = pendingBytes_;
    result.pending_rhi_submissions = pendingRhiSubmissions_;
    const Waits &pendingLimit = waits_[static_cast<size_t>(Bound::PendingLimit)];
    const Waits &transientBudget = waits_[static_cast<size_t>(Bound::TransientBudget)];
    const Waits &deviceQuota = waits_[static_cast<size_t>(Bound::DeviceQuota)];
    result.pending_limit_waits = pendingLimit.count;
    result.pending_limit_wait_nanoseconds = pendingLimit.nanoseconds;
    result.transient_budget_waits = transientBudget.count;
    result.transient_budget_wait_nanoseconds = transientBudget.nanoseconds;
    result.device_quota_waits = deviceQuota.count;
    result.device_quota_wait_nanoseconds = deviceQuota.nanoseconds;
    return result;
}

TransientAllocationMeter::TransientAllocationMeter(const VernonRuntimeContext &context) noexcept
    : context_(&context), outer_(std::exchange(currentMeter, this)) {}

TransientAllocationMeter::~TransientAllocationMeter() { currentMeter = outer_; }

void TransientAllocationMeter::charge(const VernonRuntimeContext &context, uint64_t bytes) noexcept {
    for (TransientAllocationMeter *meter = currentMeter; meter; meter = meter->outer_)
        if (meter->context_ == &context) {
            meter->bytes_ = bytes > (std::numeric_limits<uint64_t>::max)() - meter->bytes_
                                ? (std::numeric_limits<uint64_t>::max)()
                                : meter->bytes_ + bytes;
            return;
        }
}

} // namespace vernon::runtime::program_execution
