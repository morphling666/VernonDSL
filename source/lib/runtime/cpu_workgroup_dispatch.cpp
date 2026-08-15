#include "cpu_workgroup_dispatch.h"

#include "VernonRuntime.h"

#include <algorithm>
#include <atomic>
#include <charconv>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

namespace vernon::runtime {
namespace {

constexpr size_t kDefaultMaxWorkgroupVolume = 256;
constexpr size_t kImplementationMaxWorkgroupVolume = 1024;
constexpr size_t kImplementationMaxThreadBudget = 4096;
constexpr uint64_t kPullbackAllocationSiteBit = uint64_t{1} << 63;

thread_local std::string schedulerDiagnostic;

bool checkedVolume(const uint32_t dimensions[3], size_t &volume) {
    volume = 1;
    for (size_t dimension = 0; dimension < 3; ++dimension) {
        if (!dimensions[dimension] || volume > std::numeric_limits<size_t>::max() / dimensions[dimension])
            return false;
        volume *= dimensions[dimension];
    }
    return true;
}

size_t environmentSize(const char *name, size_t fallback) noexcept {
    const char *value = std::getenv(name);
    if (!value || !*value)
        return fallback;
    size_t parsed = 0;
    const char *end = value + std::strlen(value);
    const auto result = std::from_chars(value, end, parsed);
    return result.ec == std::errc{} && result.ptr == end && parsed ? parsed : fallback;
}

struct SharedAllocation {
    uint64_t site{};
    size_t offset{};
    size_t size{};
    size_t alignment{};
};

class CpuWorkgroupContext {
public:
    uint64_t address(uint64_t site, size_t size, size_t alignment, size_t offset) noexcept {
        if (site & kPullbackAllocationSiteBit)
            return pullback_.address(*this, site, size, alignment, offset);
        return primal_.address(*this, site, size, alignment, offset, "primal workgroup");
    }

    void seal(uint64_t site) noexcept {
        primal_.seal();
        if (site != std::numeric_limits<uint64_t>::max())
            pullback_.seal();
    }

    void fail(VernonStatus failureStatus, const char *message) noexcept {
        try {
            std::lock_guard lock(diagnosticMutex_);
            if (status_.load(std::memory_order_relaxed) == VERNON_STATUS_OK) {
                diagnostic_ = message;
                status_.store(failureStatus, std::memory_order_release);
            }
        } catch (...) {
            int expected = VERNON_STATUS_OK;
            status_.compare_exchange_strong(expected, failureStatus, std::memory_order_release,
                                            std::memory_order_relaxed);
        }
    }

    VernonStatus status() const noexcept { return static_cast<VernonStatus>(status_.load(std::memory_order_acquire)); }

    std::string diagnostic() const noexcept {
        try {
            std::lock_guard lock(diagnosticMutex_);
            if (!diagnostic_.empty())
                return diagnostic_;
        } catch (...) {
        }
        return {};
    }

private:
    static constexpr size_t kArenaSize = 16 * 1024;

    class Arena {
    public:
        uint64_t address(CpuWorkgroupContext &workgroup, uint64_t site, size_t size, size_t alignment, size_t offset,
                         const char *kind) noexcept {
            try {
                std::lock_guard lock(mutex);
                if (workgroup.status() != VERNON_STATUS_OK)
                    return 0;
                if (!size || !alignment || alignment > kArenaSize || (alignment & (alignment - 1)) || offset >= size) {
                    workgroup.fail(VERNON_STATUS_INTERNAL_ERROR, "invalid workgroup allocation request");
                    return 0;
                }
                const auto found =
                    std::find_if(allocations.begin(), allocations.end(),
                                 [&](const SharedAllocation &allocation) { return allocation.site == site; });
                if (found != allocations.end()) {
                    if (found->size != size || found->alignment != alignment) {
                        workgroup.fail(VERNON_STATUS_INTERNAL_ERROR, "inconsistent workgroup allocation site");
                        return 0;
                    }
                    return reinterpret_cast<uint64_t>(addressBase + found->offset + offset);
                }
                if (sealed) {
                    workgroup.fail(VERNON_STATUS_INTERNAL_ERROR,
                                   "workgroup allocation site was not established during preflight");
                    return 0;
                }
                if (!storage) {
                    storage.reset(new (std::nothrow) Storage);
                    if (!storage) {
                        workgroup.fail(VERNON_STATUS_INTERNAL_ERROR, "cannot allocate workgroup storage");
                        return 0;
                    }
                    addressBase = storage->bytes;
                    std::memset(addressBase, 0, kArenaSize);
                }
                if (nextOffset > kArenaSize - (alignment - 1)) {
                    workgroup.fail(VERNON_STATUS_INTERNAL_ERROR, "workgroup allocation alignment overflows");
                    return 0;
                }
                const size_t aligned = (nextOffset + alignment - 1) & ~(alignment - 1);
                if (size > kArenaSize - aligned) {
                    std::string diagnostic = std::string(kind) + " storage exceeds the 16 KiB budget";
                    workgroup.fail(VERNON_STATUS_INTERNAL_ERROR, diagnostic.c_str());
                    return 0;
                }
                allocations.push_back({site, aligned, size, alignment});
                nextOffset = aligned + size;
                return reinterpret_cast<uint64_t>(addressBase + aligned + offset);
            } catch (...) {
                workgroup.fail(VERNON_STATUS_INTERNAL_ERROR, "workgroup allocation bookkeeping failed");
                return 0;
            }
        }

        void seal() noexcept {
            try {
                std::lock_guard lock(mutex);
                sealed = true;
            } catch (...) {
            }
        }

    private:
        struct alignas(kArenaSize) Storage {
            unsigned char bytes[kArenaSize];
        };

        std::mutex mutex;
        std::unique_ptr<Storage> storage;
        unsigned char *addressBase{};
        size_t nextOffset{};
        std::vector<SharedAllocation> allocations;
        bool sealed{};
    };

    class DynamicArena {
    public:
        uint64_t address(CpuWorkgroupContext &workgroup, uint64_t site, size_t size, size_t alignment,
                         size_t offset) noexcept {
            try {
                std::lock_guard lock(mutex);
                if (workgroup.status() != VERNON_STATUS_OK)
                    return 0;
                if (!size || !alignment || (alignment & (alignment - 1)) || offset >= size ||
                    size > std::numeric_limits<size_t>::max() - (alignment - 1)) {
                    workgroup.fail(VERNON_STATUS_INTERNAL_ERROR, "invalid pullback shared-adjoint allocation request");
                    return 0;
                }
                auto found = std::find_if(allocations.begin(), allocations.end(),
                                          [&](const DynamicAllocation &allocation) { return allocation.site == site; });
                if (found != allocations.end()) {
                    if (found->size != size || found->alignment != alignment) {
                        workgroup.fail(VERNON_STATUS_INTERNAL_ERROR,
                                       "inconsistent pullback shared-adjoint allocation site");
                        return 0;
                    }
                    return reinterpret_cast<uint64_t>(found->address + offset);
                }
                if (sealed) {
                    workgroup.fail(VERNON_STATUS_INTERNAL_ERROR,
                                   "pullback shared-adjoint site was not established during preflight");
                    return 0;
                }
                DynamicAllocation allocation;
                allocation.site = site;
                allocation.size = size;
                allocation.alignment = alignment;
                allocation.storage.reset(new (std::nothrow) unsigned char[size + alignment - 1]);
                if (!allocation.storage) {
                    workgroup.fail(VERNON_STATUS_INTERNAL_ERROR, "cannot allocate pullback shared-adjoint storage");
                    return 0;
                }
                const uintptr_t raw = reinterpret_cast<uintptr_t>(allocation.storage.get());
                allocation.address = reinterpret_cast<unsigned char *>((raw + alignment - 1) & ~(alignment - 1));
                std::memset(allocation.address, 0, size);
                allocations.push_back(std::move(allocation));
                return reinterpret_cast<uint64_t>(allocations.back().address + offset);
            } catch (...) {
                workgroup.fail(VERNON_STATUS_INTERNAL_ERROR, "pullback shared-adjoint allocation bookkeeping failed");
                return 0;
            }
        }

        void seal() noexcept {
            try {
                std::lock_guard lock(mutex);
                sealed = true;
            } catch (...) {
            }
        }

    private:
        struct DynamicAllocation {
            uint64_t site{};
            size_t size{};
            size_t alignment{};
            std::unique_ptr<unsigned char[]> storage;
            unsigned char *address{};
        };

        std::mutex mutex;
        std::vector<DynamicAllocation> allocations;
        bool sealed{};
    };

    std::atomic<int> status_{VERNON_STATUS_OK};
    mutable std::mutex diagnosticMutex_;
    std::string diagnostic_;
    Arena primal_;
    DynamicArena pullback_;
};

struct LaneAllocation {
    uint64_t site{};
    size_t size{};
    size_t alignment{};
    size_t stride{};
    size_t blockStride{};
    std::unique_ptr<unsigned char[]> storage;
    unsigned char *address{};
};

class CpuLaneArena {
    static constexpr size_t kLaneBlockWidth = 8;
    static constexpr size_t kCacheAlignment = 64;

public:
    explicit CpuLaneArena(size_t laneCount) : laneCount_(laneCount) {}

    uint64_t address(CpuWorkgroupContext &workgroup, uint64_t site, size_t size, size_t alignment, size_t offset,
                     size_t lane) noexcept {
        try {
            std::lock_guard lock(mutex_);
            if (!alignment || (alignment & (alignment - 1)) || (size ? offset >= size : offset != 0) ||
                size > std::numeric_limits<size_t>::max() - (alignment - 1) || lane >= laneCount_) {
                workgroup.fail(VERNON_STATUS_INTERNAL_ERROR, "invalid lane-private allocation request");
                return 0;
            }
            if (!size)
                return reinterpret_cast<uint64_t>(&empty_);
            auto found = std::find_if(allocations_.begin(), allocations_.end(),
                                      [&](const LaneAllocation &allocation) { return allocation.site == site; });
            if (found == allocations_.end()) {
                const size_t stride = (size + alignment - 1) & ~(alignment - 1);
                const size_t blockAlignment = std::max(alignment, kCacheAlignment);
                if (stride > std::numeric_limits<size_t>::max() / kLaneBlockWidth) {
                    workgroup.fail(VERNON_STATUS_INTERNAL_ERROR, "lane-private allocation size overflows");
                    return 0;
                }
                const size_t blockPayload = stride * kLaneBlockWidth;
                if (blockPayload > std::numeric_limits<size_t>::max() - (blockAlignment - 1)) {
                    workgroup.fail(VERNON_STATUS_INTERNAL_ERROR, "lane-private allocation size overflows");
                    return 0;
                }
                const size_t blockStride = (blockPayload + blockAlignment - 1) & ~(blockAlignment - 1);
                allocations_.push_back({site, size, alignment, stride, blockStride});
                found = std::prev(allocations_.end());
            } else if (found->size != size || found->alignment != alignment) {
                workgroup.fail(VERNON_STATUS_INTERNAL_ERROR, "inconsistent lane-private allocation site");
                return 0;
            }
            if (!found->storage) {
                const size_t blockCount = (laneCount_ + kLaneBlockWidth - 1) / kLaneBlockWidth;
                if (blockCount > std::numeric_limits<size_t>::max() / found->blockStride) {
                    workgroup.fail(VERNON_STATUS_INTERNAL_ERROR, "lane-private allocation size overflows");
                    return 0;
                }
                const size_t allocationSize = blockCount * found->blockStride;
                const size_t storageAlignment = std::max(alignment, kCacheAlignment);
                if (allocationSize > std::numeric_limits<size_t>::max() - (storageAlignment - 1)) {
                    workgroup.fail(VERNON_STATUS_INTERNAL_ERROR, "lane-private allocation size overflows");
                    return 0;
                }
                found->storage.reset(new (std::nothrow) unsigned char[allocationSize + storageAlignment - 1]);
                if (!found->storage) {
                    workgroup.fail(VERNON_STATUS_INTERNAL_ERROR, "cannot allocate lane-private storage");
                    return 0;
                }
                const uintptr_t raw = reinterpret_cast<uintptr_t>(found->storage.get());
                found->address =
                    reinterpret_cast<unsigned char *>((raw + storageAlignment - 1) & ~(storageAlignment - 1));
                std::memset(found->address, 0, allocationSize);
            }
            const size_t block = lane / kLaneBlockWidth;
            const size_t blockLane = lane % kLaneBlockWidth;
            return reinterpret_cast<uint64_t>(found->address + block * found->blockStride + blockLane * found->stride +
                                              offset);
        } catch (...) {
            workgroup.fail(VERNON_STATUS_INTERNAL_ERROR, "lane-private allocation bookkeeping failed");
            return 0;
        }
    }

private:
    size_t laneCount_{};
    std::mutex mutex_;
    std::vector<LaneAllocation> allocations_;
    unsigned char empty_{};
};

thread_local CpuWorkgroupContext *activeWorkgroup;
thread_local VernonCpuRangeV1 *activeRange;
thread_local CpuLaneArena *activeLaneArena;

[[noreturn]] void failInactiveWorkgroupHelper(const char *helper) noexcept {
    std::fputs("Vernon CPU runtime fatal error: ", stderr);
    std::fputs(helper, stderr);
    std::fputs(" called outside an active CPU workgroup\n", stderr);
    std::fflush(stderr);
    std::abort();
}

class ActiveWorkgroupScope {
public:
    ActiveWorkgroupScope(CpuWorkgroupContext &context, VernonCpuRangeV1 &range, CpuLaneArena &laneArena) noexcept
        : previous_(std::exchange(activeWorkgroup, &context)), previousRange_(std::exchange(activeRange, &range)),
          previousLaneArena_(std::exchange(activeLaneArena, &laneArena)) {}
    ~ActiveWorkgroupScope() {
        activeWorkgroup = previous_;
        activeRange = previousRange_;
        activeLaneArena = previousLaneArena_;
    }

private:
    CpuWorkgroupContext *previous_{};
    VernonCpuRangeV1 *previousRange_{};
    CpuLaneArena *previousLaneArena_{};
};

class SchedulerJob {
public:
    virtual ~SchedulerJob() = default;
    virtual void run() noexcept = 0;
};

class CompletedJob : public SchedulerJob {
public:
    explicit CompletedJob(size_t taskCount) : remaining_(taskCount) {}

    void wait() noexcept {
        while (remaining_.load(std::memory_order_acquire) != 0) {
            try {
                std::unique_lock lock(completionMutex_);
                completion_.wait(lock, [&] { return remaining_.load(std::memory_order_acquire) == 0; });
            } catch (...) {
                std::this_thread::yield();
            }
        }
    }

    VernonStatus status() const noexcept { return static_cast<VernonStatus>(status_.load(std::memory_order_acquire)); }

    std::string diagnostic() const noexcept {
        try {
            std::lock_guard lock(diagnosticMutex_);
            return diagnostic_;
        } catch (...) {
            return {};
        }
    }

protected:
    class CompletionScope {
    public:
        explicit CompletionScope(CompletedJob &job) : job_(job) {}
        ~CompletionScope() { job_.completeTask(); }

    private:
        CompletedJob &job_;
    };

    bool recordFailure(VernonStatus failureStatus, const VernonCpuRangeV1 &range, const std::string &detail) noexcept {
        int expected = VERNON_STATUS_OK;
        if (!status_.compare_exchange_strong(expected, failureStatus, std::memory_order_acq_rel,
                                             std::memory_order_relaxed))
            return false;
        try {
            std::lock_guard lock(diagnosticMutex_);
            const size_t localLinear = std::min(range.active_lane, range.lane_end);
            const size_t localX = localLinear % range.workgroup[0];
            const size_t localY = (localLinear / range.workgroup[0]) % range.workgroup[1];
            const size_t localZ = localLinear / (static_cast<size_t>(range.workgroup[0]) * range.workgroup[1]);
            diagnostic_ = "CPU workgroup (" + std::to_string(range.group[0]) + "," + std::to_string(range.group[1]) +
                          "," + std::to_string(range.group[2]) + "), local lane (" + std::to_string(localX) + "," +
                          std::to_string(localY) + "," + std::to_string(localZ) + ") failed";
            if (!detail.empty())
                diagnostic_ += ": " + detail;
        } catch (...) {
            try {
                std::lock_guard lock(diagnosticMutex_);
                diagnostic_ = "CPU workgroup lane failed";
            } catch (...) {
            }
        }
        return true;
    }

private:
    void completeTask() noexcept {
        if (remaining_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            try {
                std::lock_guard lock(completionMutex_);
                completion_.notify_all();
            } catch (...) {
            }
        }
    }

    std::atomic<size_t> remaining_;
    std::atomic<int> status_{VERNON_STATUS_OK};
    mutable std::mutex diagnosticMutex_;
    std::string diagnostic_;
    std::mutex completionMutex_;
    std::condition_variable completion_;
};

class RangeJob final : public CompletedJob {
public:
    RangeJob(size_t runners, size_t groupBegin, size_t groupCount, size_t workgroupVolume, const uint32_t grid[3],
             const uint32_t workgroup[3], const CpuRangeCallback &callback)
        : CompletedJob(runners), groupCount_(groupCount), workgroupVolume_(workgroupVolume), callback_(callback),
          tasks_(groupCount < runners ? runners : 0), groupsRemaining_(groupCount), groupBegin_(groupBegin) {
        std::copy_n(grid, 3, grid_);
        std::copy_n(workgroup, 3, workgroup_);
        groupRangeMode_ = groupCount_ >= runners;
        if (groupRangeMode_) {
            nextGroupClaim_.store(runners, std::memory_order_relaxed);
            return;
        }
        const size_t activeGroups = std::min(groupCount_, runners);
        const size_t baseChunks = runners / activeGroups;
        const size_t extraChunks = runners % activeGroups;
        groups_.reserve(activeGroups);
        size_t runnerBegin = 0;
        for (size_t slot = 0; slot < activeGroups; ++slot) {
            const size_t chunks = std::min(workgroupVolume_, baseChunks + (slot < extraChunks ? 1 : 0));
            groups_.push_back(std::make_unique<GroupState>(workgroupVolume_, chunks, groupBegin_ + slot, runnerBegin));
            enqueuePhase(slot, 0);
            runnerBegin += chunks;
        }
    }

    void run() noexcept override {
        CompletionScope completion(*this);
        const size_t runnerOrdinal = nextRunnerOrdinal_.fetch_add(1, std::memory_order_relaxed);
        if (groupRangeMode_) {
            if (status() != VERNON_STATUS_OK)
                return;
            runGroupRanges(runnerOrdinal);
            return;
        }
        while (true) {
            Task task;
            {
                std::unique_lock lock(taskMutex_);
                taskAvailable_.wait(lock, [&] {
                    return status() != VERNON_STATUS_OK || groupsRemaining_ == 0 || !tasks_[runnerOrdinal].empty();
                });
                if (status() != VERNON_STATUS_OK || groupsRemaining_ == 0)
                    return;
                task = tasks_[runnerOrdinal].front();
                tasks_[runnerOrdinal].pop_front();
            }
            runRange(task);
            if (status() != VERNON_STATUS_OK) {
                taskAvailable_.notify_all();
                return;
            }
        }
    }

private:
    struct Task {
        size_t group{};
        size_t laneBegin{};
        size_t laneEnd{};
        uint64_t phase{};
    };

    struct GroupState {
        GroupState(size_t workgroupVolume, size_t chunkCount, size_t groupLinear, size_t runnerBegin)
            : laneArena(workgroupVolume), chunkCount(chunkCount), groupLinear(groupLinear), runnerBegin(runnerBegin) {}

        CpuWorkgroupContext context;
        CpuLaneArena laneArena;
        size_t chunkCount{};
        size_t groupLinear{};
        size_t runnerBegin{};
        uint64_t phase{};
        size_t outstanding{};
        bool hasOutcome{};
        uint32_t outcome{};
        uint64_t yieldedSite{};
    };

    void enqueuePhase(size_t slot, uint64_t phase) {
        GroupState &group = *groups_[slot];
        group.phase = phase;
        group.outstanding = group.chunkCount;
        group.hasOutcome = false;
        for (size_t chunk = 0; chunk < group.chunkCount; ++chunk) {
            const size_t begin = workgroupVolume_ * chunk / group.chunkCount;
            const size_t end = workgroupVolume_ * (chunk + 1) / group.chunkCount;
            tasks_[group.runnerBegin + chunk].push_back({slot, begin, end, phase});
        }
    }

    void failRange(GroupState &group, VernonCpuRangeV1 &range, VernonStatus failureStatus, const char *message) {
        group.context.fail(failureStatus, message);
        recordFailure(group.context.status(), range, group.context.diagnostic());
    }

    bool invokeRange(GroupState &groupState, size_t laneBegin, size_t laneEnd, uint64_t phase,
                     VernonCpuRangeV1 &range) noexcept {
        range.struct_size = sizeof(range);
        std::copy_n(grid_, 3, range.grid);
        std::copy_n(workgroup_, 3, range.workgroup);
        range.group[0] = static_cast<uint32_t>(groupState.groupLinear % grid_[0]);
        range.group[1] = static_cast<uint32_t>((groupState.groupLinear / grid_[0]) % grid_[1]);
        range.group[2] = static_cast<uint32_t>(groupState.groupLinear / (static_cast<size_t>(grid_[0]) * grid_[1]));
        range.lane_begin = laneBegin;
        range.lane_end = laneEnd;
        range.active_lane = laneBegin;
        range.phase = phase;
        range.outcome = VERNON_CPU_RANGE_COMPLETE_V1;
        VernonStatus rangeStatus = VERNON_STATUS_INTERNAL_ERROR;
        std::string exceptionDiagnostic;
        {
            ActiveWorkgroupScope scope(groupState.context, range, groupState.laneArena);
            try {
                rangeStatus = callback_(range);
            } catch (const std::exception &exception) {
                try {
                    exceptionDiagnostic = exception.what();
                } catch (...) {
                }
            } catch (...) {
                try {
                    exceptionDiagnostic = "unknown C++ exception";
                } catch (...) {
                }
            }
        }
        if (!exceptionDiagnostic.empty()) {
            failRange(groupState, range, VERNON_STATUS_INTERNAL_ERROR, exceptionDiagnostic.c_str());
            return false;
        }
        if (rangeStatus != VERNON_STATUS_OK) {
            failRange(groupState, range, rangeStatus, "CPU range returned an error");
            return false;
        }
        if (groupState.context.status() != VERNON_STATUS_OK) {
            recordFailure(groupState.context.status(), range, groupState.context.diagnostic());
            return false;
        }
        if (range.outcome != VERNON_CPU_RANGE_COMPLETE_V1 && range.outcome != VERNON_CPU_RANGE_YIELDED_V1) {
            failRange(groupState, range, VERNON_STATUS_INTERNAL_ERROR, "CPU range returned an invalid phase outcome");
            return false;
        }
        return true;
    }

    bool runWholeGroup(size_t groupLinear) noexcept {
        GroupState group(workgroupVolume_, 1, groupLinear, 0);
        for (uint64_t phase = 0;; ++phase) {
            VernonCpuRangeV1 range{};
            if (!invokeRange(group, 0, workgroupVolume_, phase, range))
                return false;
            if (range.outcome == VERNON_CPU_RANGE_COMPLETE_V1)
                return true;
            group.context.seal(range.yielded_site);
        }
    }

    void runGroupRanges(size_t runnerOrdinal) noexcept {
        constexpr size_t groupsPerClaim = 8;
        if (!runWholeGroup(groupBegin_ + runnerOrdinal))
            return;
        while (status() == VERNON_STATUS_OK) {
            const size_t begin = nextGroupClaim_.fetch_add(groupsPerClaim, std::memory_order_relaxed);
            if (begin >= groupCount_)
                return;
            const size_t end = std::min(begin + groupsPerClaim, groupCount_);
            for (size_t groupLinear = begin; groupLinear < end && status() == VERNON_STATUS_OK; ++groupLinear)
                if (!runWholeGroup(groupBegin_ + groupLinear))
                    return;
        }
    }

    void runRange(const Task &task) noexcept {
        GroupState &groupState = *groups_[task.group];
        VernonCpuRangeV1 range{};
        if (!invokeRange(groupState, task.laneBegin, task.laneEnd, task.phase, range))
            return;

        {
            std::lock_guard lock(taskMutex_);
            if (task.phase != groupState.phase)
                return;
            if (!groupState.hasOutcome) {
                groupState.hasOutcome = true;
                groupState.outcome = range.outcome;
                groupState.yieldedSite = range.yielded_site;
            } else if (groupState.outcome != range.outcome ||
                       (range.outcome == VERNON_CPU_RANGE_YIELDED_V1 && groupState.yieldedSite != range.yielded_site)) {
                failRange(groupState, range, VERNON_STATUS_INTERNAL_ERROR,
                          "CPU lane ranges disagreed on phase completion or barrier site");
                taskAvailable_.notify_all();
                return;
            }
            if (--groupState.outstanding)
                return;
            if (groupState.outcome == VERNON_CPU_RANGE_COMPLETE_V1) {
                --groupsRemaining_;
            } else {
                groupState.context.seal(groupState.yieldedSite);
                enqueuePhase(task.group, task.phase + 1);
            }
        }
        taskAvailable_.notify_all();
    }

    size_t groupCount_{};
    size_t workgroupVolume_{};
    uint32_t grid_[3]{};
    uint32_t workgroup_[3]{};
    CpuRangeCallback callback_;
    std::vector<std::unique_ptr<GroupState>> groups_;
    std::mutex taskMutex_;
    std::condition_variable taskAvailable_;
    std::vector<std::deque<Task>> tasks_;
    size_t groupsRemaining_{};
    size_t groupBegin_{};
    bool groupRangeMode_{};
    std::atomic<size_t> nextGroupClaim_{};
    std::atomic<size_t> nextRunnerOrdinal_{};
};

} // namespace

class CpuWorkgroupScheduler::Impl {
public:
    explicit Impl(CpuWorkgroupSchedulerConfig config) : config_(config) {}

    bool start(std::string &error) noexcept {
#if defined(VERNON_RUNTIME_PROFILE_WEB)
        if (config_.executionPolicy != CpuSchedulerExecutionPolicy::CallingThread) {
            error = "the web Runtime requires the calling-thread CPU scheduler policy";
            return false;
        }
        return true;
#else
        if (config_.executionPolicy == CpuSchedulerExecutionPolicy::CallingThread)
            return true;
        if (ensureWorkers(config_.threadBudget))
            return true;
        try {
            error = schedulerDiagnostic.empty() ? "cannot start CPU workgroup scheduler workers" : schedulerDiagnostic;
        } catch (...) {
        }
        return false;
#endif
    }

    ~Impl() {
#if !defined(VERNON_RUNTIME_PROFILE_WEB)
        {
            std::lock_guard lock(queueMutex_);
            stopping_ = true;
        }
        workAvailable_.notify_all();
        for (std::thread &worker : workers_)
            if (worker.joinable())
                worker.join();
#endif
    }

    VernonStatus dispatch(const uint32_t grid[3], const uint32_t workgroup[3], const CpuRangeCallback &callback,
                          bool forceCallingThread, size_t groupBegin = 0,
                          size_t requestedGroupCount = std::numeric_limits<size_t>::max()) noexcept {
        try {
            schedulerDiagnostic.clear();
            if (activeWorkgroup)
                return fail("nested CPU workgroup dispatch is unsupported", VERNON_STATUS_INVALID_ARGUMENT);
            if (!grid || !workgroup || !callback)
                return fail("CPU workgroup dispatch arguments are incomplete", VERNON_STATUS_INVALID_ARGUMENT);
            for (size_t dimension = 0; dimension < 3; ++dimension) {
                if (!grid[dimension] || !workgroup[dimension] ||
                    grid[dimension] > std::numeric_limits<uint32_t>::max() / workgroup[dimension])
                    return fail("CPU workgroup dispatch dimensions are invalid", VERNON_STATUS_INVALID_ARGUMENT);
            }
            size_t workgroupVolume = 0;
            size_t groupCount = 0;
            if (!checkedVolume(workgroup, workgroupVolume) || !checkedVolume(grid, groupCount))
                return fail("CPU workgroup dispatch volume overflows", VERNON_STATUS_INVALID_ARGUMENT);
            if (groupBegin > groupCount || (requestedGroupCount != std::numeric_limits<size_t>::max() &&
                                            requestedGroupCount > groupCount - groupBegin))
                return fail("CPU workgroup dispatch group range is invalid", VERNON_STATUS_INVALID_ARGUMENT);
            groupCount = requestedGroupCount == std::numeric_limits<size_t>::max() ? groupCount - groupBegin
                                                                                   : requestedGroupCount;
            if (!groupCount)
                return fail("CPU workgroup dispatch group range is empty", VERNON_STATUS_INVALID_ARGUMENT);
            if (workgroupVolume > config_.maxWorkgroupVolume)
                return fail("CPU workgroup volume " + std::to_string(workgroupVolume) + " exceeds configured maximum " +
                                std::to_string(config_.maxWorkgroupVolume),
                            VERNON_STATUS_INVALID_ARGUMENT);
            constexpr size_t inlineInvocationLimit = 64;
            if (groupCount == 1 && workgroupVolume <= inlineInvocationLimit) {
                auto job =
                    std::make_shared<RangeJob>(1, groupBegin, groupCount, workgroupVolume, grid, workgroup, callback);
                job->run();
                if (job->status() != VERNON_STATUS_OK)
                    schedulerDiagnostic = job->diagnostic();
                return job->status();
            }
            const size_t runners =
                forceCallingThread || config_.executionPolicy == CpuSchedulerExecutionPolicy::CallingThread
                    ? 1
                    : std::min(config_.threadBudget, groupCount > std::numeric_limits<size_t>::max() / workgroupVolume
                                                         ? config_.threadBudget
                                                         : groupCount * workgroupVolume);
            auto job =
                std::make_shared<RangeJob>(runners, groupBegin, groupCount, workgroupVolume, grid, workgroup, callback);
            if (forceCallingThread || config_.executionPolicy == CpuSchedulerExecutionPolicy::CallingThread) {
                job->run();
            } else {
#if defined(VERNON_RUNTIME_PROFILE_WEB)
                return fail("worker-pool scheduling is unavailable in the web Runtime",
                            VERNON_STATUS_UNSUPPORTED_TARGET);
#else
                if (!enqueue(job, runners))
                    return VERNON_STATUS_INTERNAL_ERROR;
                job->wait();
#endif
            }
            if (job->status() != VERNON_STATUS_OK)
                schedulerDiagnostic = job->diagnostic();
            return job->status();
        } catch (const std::bad_alloc &) {
            return fail("CPU workgroup scheduler ran out of memory", VERNON_STATUS_INTERNAL_ERROR);
        } catch (const std::system_error &exception) {
            return fail(std::string("CPU workgroup scheduler system error: ") + exception.what(),
                        VERNON_STATUS_INTERNAL_ERROR);
        } catch (const std::exception &exception) {
            return fail(std::string("CPU workgroup scheduler failed: ") + exception.what(),
                        VERNON_STATUS_INTERNAL_ERROR);
        } catch (...) {
            return fail("CPU workgroup scheduler failed with an unknown exception", VERNON_STATUS_INTERNAL_ERROR);
        }
    }

    const std::string &lastDiagnostic() const noexcept { return schedulerDiagnostic; }

private:
#if !defined(VERNON_RUNTIME_PROFILE_WEB)
    bool ensureWorkers(size_t required) {
        std::lock_guard workersLock(workersMutex_);
        if (workers_.size() >= required)
            return true;
        if (required > config_.threadBudget) {
            fail("CPU workgroup dispatch exceeds scheduler thread capacity", VERNON_STATUS_INVALID_ARGUMENT);
            return false;
        }
        try {
            workers_.reserve(required);
            while (workers_.size() < required)
                workers_.emplace_back([this] { workerLoop(); });
            return true;
        } catch (const std::system_error &exception) {
            fail(std::string("cannot create CPU scheduler worker: ") + exception.what(), VERNON_STATUS_INTERNAL_ERROR);
        } catch (const std::bad_alloc &) {
            fail("cannot allocate CPU scheduler workers", VERNON_STATUS_INTERNAL_ERROR);
        }
        return false;
    }

    bool enqueue(const std::shared_ptr<SchedulerJob> &job, size_t taskCount) {
        std::lock_guard lock(queueMutex_);
        const size_t originalSize = queue_.size();
        try {
            for (size_t task = 0; task < taskCount; ++task)
                queue_.push_back(job);
        } catch (...) {
            while (queue_.size() > originalSize)
                queue_.pop_back();
            fail("cannot allocate CPU scheduler work queue", VERNON_STATUS_INTERNAL_ERROR);
            return false;
        }
        workAvailable_.notify_all();
        return true;
    }

    void workerLoop() noexcept {
        while (true) {
            std::shared_ptr<SchedulerJob> job;
            try {
                std::unique_lock lock(queueMutex_);
                workAvailable_.wait(lock, [&] { return stopping_ || !queue_.empty(); });
                if (stopping_ && queue_.empty())
                    return;
                job = std::move(queue_.front());
                queue_.pop_front();
            } catch (...) {
                return;
            }
            job->run();
        }
    }
#endif

    VernonStatus fail(std::string message, VernonStatus status) noexcept {
        try {
            schedulerDiagnostic = std::move(message);
        } catch (...) {
        }
        return status;
    }

    CpuWorkgroupSchedulerConfig config_;
#if !defined(VERNON_RUNTIME_PROFILE_WEB)
    std::mutex workersMutex_;
    std::mutex queueMutex_;
    std::condition_variable workAvailable_;
    std::deque<std::shared_ptr<SchedulerJob>> queue_;
    bool stopping_{};
    std::vector<std::thread> workers_;
#endif
};

CpuLaneCoordinates cpuRangeCoordinates(const VernonCpuRangeV1 &range, size_t localLinear) noexcept {
    CpuLaneCoordinates coordinates{};
    std::copy_n(range.group, 3, coordinates.group);
    coordinates.local[0] = static_cast<uint32_t>(localLinear % range.workgroup[0]);
    coordinates.local[1] = static_cast<uint32_t>((localLinear / range.workgroup[0]) % range.workgroup[1]);
    coordinates.local[2] =
        static_cast<uint32_t>(localLinear / (static_cast<size_t>(range.workgroup[0]) * range.workgroup[1]));
    for (size_t dimension = 0; dimension < 3; ++dimension)
        coordinates.global[dimension] =
            coordinates.group[dimension] * range.workgroup[dimension] + coordinates.local[dimension];
    const size_t globalX = static_cast<size_t>(range.grid[0]) * range.workgroup[0];
    const size_t globalY = static_cast<size_t>(range.grid[1]) * range.workgroup[1];
    coordinates.linearIndex =
        coordinates.global[0] + globalX * (coordinates.global[1] + globalY * coordinates.global[2]);
    return coordinates;
}

CpuWorkgroupSchedulerConfig CpuWorkgroupScheduler::defaultConfig() noexcept {
#if defined(VERNON_RUNTIME_PROFILE_WEB)
    return {1, environmentSize("VERNON_CPU_MAX_WORKGROUP_VOLUME", kDefaultMaxWorkgroupVolume),
            CpuSchedulerExecutionPolicy::CallingThread};
#else
    const size_t detected = std::max(1u, std::thread::hardware_concurrency());
    return {environmentSize("VERNON_CPU_THREAD_BUDGET", detected),
            environmentSize("VERNON_CPU_MAX_WORKGROUP_VOLUME", kDefaultMaxWorkgroupVolume)};
#endif
}

std::unique_ptr<CpuWorkgroupScheduler> CpuWorkgroupScheduler::create(CpuWorkgroupSchedulerConfig config,
                                                                     std::string &error) noexcept {
    try {
        if (!config.threadBudget || config.threadBudget > kImplementationMaxThreadBudget) {
            error =
                "CPU scheduler thread budget must be between 1 and " + std::to_string(kImplementationMaxThreadBudget);
            return nullptr;
        }
        if (!config.maxWorkgroupVolume || config.maxWorkgroupVolume > kImplementationMaxWorkgroupVolume) {
            error = "CPU scheduler maximum workgroup volume must be between 1 and " +
                    std::to_string(kImplementationMaxWorkgroupVolume);
            return nullptr;
        }
        auto impl = std::unique_ptr<Impl>(new (std::nothrow) Impl(config));
        if (!impl) {
            error = "cannot allocate CPU workgroup scheduler";
            return nullptr;
        }
        if (!impl->start(error))
            return nullptr;
        auto scheduler =
            std::unique_ptr<CpuWorkgroupScheduler>(new (std::nothrow) CpuWorkgroupScheduler(std::move(impl)));
        if (!scheduler)
            error = "cannot allocate CPU workgroup scheduler";
        return scheduler;
    } catch (...) {
        try {
            error = "cannot initialize CPU workgroup scheduler";
        } catch (...) {
        }
        return nullptr;
    }
}

CpuWorkgroupScheduler::CpuWorkgroupScheduler(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}
CpuWorkgroupScheduler::~CpuWorkgroupScheduler() = default;

VernonStatus CpuWorkgroupScheduler::dispatch(const uint32_t grid[3], const uint32_t workgroup[3],
                                             const CpuRangeCallback &callback) noexcept {
    return impl_->dispatch(grid, workgroup, callback, false);
}

VernonStatus CpuWorkgroupScheduler::dispatchInline(const uint32_t grid[3], const uint32_t workgroup[3],
                                                   const CpuRangeCallback &callback) noexcept {
    return impl_->dispatch(grid, workgroup, callback, true);
}

VernonStatus CpuWorkgroupScheduler::dispatchGroupInline(const uint32_t grid[3], const uint32_t workgroup[3],
                                                        size_t groupLinear, const CpuRangeCallback &callback) noexcept {
    return impl_->dispatch(grid, workgroup, callback, true, groupLinear, 1);
}

const std::string &CpuWorkgroupScheduler::lastDiagnostic() const noexcept { return impl_->lastDiagnostic(); }

} // namespace vernon::runtime

extern "C" VERNON_RUNTIME_CAPI uint64_t vernonCpuWorkgroupAddressV1(uint64_t site, uint64_t size, uint64_t alignment,
                                                                    uint64_t offset) {
    using namespace vernon::runtime;
    if (!activeWorkgroup)
        failInactiveWorkgroupHelper("vernonCpuWorkgroupAddressV1");
    if (size > std::numeric_limits<size_t>::max() || alignment > std::numeric_limits<size_t>::max() ||
        offset > std::numeric_limits<size_t>::max()) {
        activeWorkgroup->fail(VERNON_STATUS_INTERNAL_ERROR, "workgroup allocation request exceeds host size limits");
        return 0;
    }
    return activeWorkgroup->address(site, static_cast<size_t>(size), static_cast<size_t>(alignment),
                                    static_cast<size_t>(offset));
}

extern "C" VERNON_RUNTIME_CAPI uint64_t vernonCpuLaneAddressV1(uint64_t site, uint64_t size, uint64_t alignment,
                                                               uint64_t offset) {
    using namespace vernon::runtime;
    if (!activeWorkgroup || !activeRange || !activeLaneArena || activeRange->active_lane < activeRange->lane_begin ||
        activeRange->active_lane >= activeRange->lane_end)
        failInactiveWorkgroupHelper("vernonCpuLaneAddressV1");
    if (size > std::numeric_limits<size_t>::max() || alignment > std::numeric_limits<size_t>::max() ||
        offset > std::numeric_limits<size_t>::max()) {
        activeWorkgroup->fail(VERNON_STATUS_INTERNAL_ERROR, "lane allocation request exceeds host size limits");
        return 0;
    }
    return activeLaneArena->address(*activeWorkgroup, site, static_cast<size_t>(size), static_cast<size_t>(alignment),
                                    static_cast<size_t>(offset), activeRange->active_lane);
}

extern "C" VERNON_RUNTIME_CAPI void vernonCpuWorkgroupBarrierV1(uint64_t site) {
    using namespace vernon::runtime;
    if (!activeWorkgroup || !activeRange)
        failInactiveWorkgroupHelper("vernonCpuWorkgroupBarrierV1");
    if (activeRange->completed_lanes) {
        activeWorkgroup->fail(VERNON_STATUS_INTERNAL_ERROR, "lane completed while peer lanes yielded at a barrier");
        activeRange->outcome = UINT32_MAX;
        return;
    }
    if (activeRange->outcome == VERNON_CPU_RANGE_COMPLETE_V1) {
        activeRange->yielded_site = site;
        activeRange->outcome = VERNON_CPU_RANGE_YIELDED_V1;
        return;
    }
    if (activeRange->outcome != VERNON_CPU_RANGE_YIELDED_V1 || activeRange->yielded_site != site) {
        activeWorkgroup->fail(VERNON_STATUS_INTERNAL_ERROR, "barrier site mismatch within one workgroup phase");
        activeRange->outcome = UINT32_MAX;
    }
}

extern "C" VERNON_RUNTIME_CAPI bool vernonCpuWorkgroupIsLeaderV1() {
    if (!vernon::runtime::activeWorkgroup || !vernon::runtime::activeRange)
        vernon::runtime::failInactiveWorkgroupHelper("vernonCpuWorkgroupIsLeaderV1");
    return vernon::runtime::activeRange->active_lane == 0;
}
