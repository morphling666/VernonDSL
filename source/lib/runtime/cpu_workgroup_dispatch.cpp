#include "cpu_workgroup_dispatch.h"

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

class CpuBarrier {
public:
    explicit CpuBarrier(size_t participants) : participants_(participants) {}

    bool arriveAndWait(uint64_t site) noexcept {
        if (participants_ <= 1)
            return true;
        try {
            std::unique_lock lock(mutex_);
            if (cancelled_ || completed_) {
                if (completed_) {
                    try {
                        diagnostic_ = "barrier site " + std::to_string(site) + " reached after another lane completed";
                    } catch (...) {
                    }
                }
                cancelLocked("barrier reached after workgroup cancellation or lane completion");
                return false;
            }
            const size_t generation = generation_;
            if (!arrived_)
                site_ = site;
            else if (site_ != site) {
                try {
                    diagnostic_ = "barrier site mismatch: expected " + std::to_string(site_) + ", received " +
                                  std::to_string(site);
                } catch (...) {
                }
                cancelLocked("lanes reached different barrier sites in one generation");
                return false;
            }
            if (++arrived_ == participants_) {
                arrived_ = 0;
                ++generation_;
                condition_.notify_all();
                return true;
            }
            condition_.wait(lock, [&] { return generation_ != generation || cancelled_; });
            return !cancelled_;
        } catch (...) {
            cancel("barrier synchronization failed");
            return false;
        }
    }

    bool completeLane() noexcept {
        try {
            std::lock_guard lock(mutex_);
            ++completed_;
            if (arrived_) {
                try {
                    diagnostic_ =
                        "lane completed while peer lanes were waiting at barrier site " + std::to_string(site_);
                } catch (...) {
                }
                cancelLocked("lane completed while peer lanes were waiting at a barrier");
                return false;
            }
            return !cancelled_;
        } catch (...) {
            cancel("lane completion synchronization failed");
            return false;
        }
    }

    void cancel(const char *reason) noexcept {
        try {
            std::lock_guard lock(mutex_);
            cancelLocked(reason);
        } catch (...) {
            cancelled_ = true;
            condition_.notify_all();
        }
    }

    std::string diagnostic() const noexcept {
        try {
            std::lock_guard lock(mutex_);
            return diagnostic_;
        } catch (...) {
            return {};
        }
    }

private:
    void cancelLocked(const char *reason) noexcept {
        cancelled_ = true;
        if (diagnostic_.empty()) {
            try {
                diagnostic_ = reason;
            } catch (...) {
            }
        }
        condition_.notify_all();
    }

    mutable std::mutex mutex_;
    std::condition_variable condition_;
    size_t participants_{};
    size_t arrived_{};
    size_t completed_{};
    size_t generation_{};
    uint64_t site_{};
    bool cancelled_{};
    std::string diagnostic_;
};

struct SharedAllocation {
    uint64_t site{};
    size_t offset{};
    size_t size{};
    size_t alignment{};
};

struct CpuWorkgroupAbort {};

class CpuWorkgroupContext {
public:
    explicit CpuWorkgroupContext(size_t participants) noexcept : barrier_(participants) {
        if (!primal_.initialize() || !pullback_.initialize())
            fail(VERNON_STATUS_INTERNAL_ERROR, "cannot allocate workgroup storage plan");
    }

    uint64_t address(uint64_t site, size_t size, size_t alignment, size_t offset) noexcept {
        if (site & kPullbackAllocationSiteBit)
            return pullback_.address(*this, site, size, alignment, offset);
        return primal_.address(*this, site, size, alignment, offset, "primal workgroup");
    }

    void barrier(uint64_t site) {
        primal_.seal();
        if (site != std::numeric_limits<uint64_t>::max())
            pullback_.seal();
        if (barrier_.arriveAndWait(site))
            return;
        const std::string detail = barrier_.diagnostic();
        fail(VERNON_STATUS_INTERNAL_ERROR, detail.empty() ? "workgroup barrier failed" : detail.c_str());
        throw CpuWorkgroupAbort{};
    }

    void completeLane(VernonStatus laneStatus) noexcept {
        if (laneStatus != VERNON_STATUS_OK)
            fail(laneStatus, "CPU lane returned an error");
        if (!barrier_.completeLane()) {
            const std::string detail = barrier_.diagnostic();
            fail(VERNON_STATUS_INTERNAL_ERROR, detail.empty() ? "CPU lane completed before its peers" : detail.c_str());
        }
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
        barrier_.cancel(message);
    }

    VernonStatus status() const noexcept { return static_cast<VernonStatus>(status_.load(std::memory_order_acquire)); }

    std::string diagnostic() const noexcept {
        try {
            std::lock_guard lock(diagnosticMutex_);
            if (!diagnostic_.empty())
                return diagnostic_;
        } catch (...) {
        }
        return barrier_.diagnostic();
    }

private:
    static constexpr size_t kArenaSize = 16 * 1024;

    class Arena {
    public:
        bool initialize() noexcept {
            storage.reset(new (std::nothrow) Storage);
            if (!storage)
                return false;
            addressBase = storage->bytes;
            std::memset(addressBase, 0, kArenaSize);
            try {
                allocations.reserve(16);
            } catch (...) {
                return false;
            }
            return true;
        }

        uint64_t address(CpuWorkgroupContext &workgroup, uint64_t site, size_t size, size_t alignment, size_t offset,
                         const char *kind) noexcept {
            try {
                std::lock_guard lock(mutex);
                if (workgroup.status() != VERNON_STATUS_OK || !addressBase)
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
        bool initialize() noexcept {
            try {
                allocations.reserve(16);
                return true;
            } catch (...) {
                return false;
            }
        }

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

    CpuBarrier barrier_;
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
    std::unique_ptr<unsigned char[]> storage;
    unsigned char *address{};
    bool initialized{};
};

class CpuLaneScratch {
public:
    void begin() noexcept {
        for (LaneAllocation &allocation : allocations_)
            allocation.initialized = false;
    }

    uint64_t address(CpuWorkgroupContext &workgroup, uint64_t site, size_t size, size_t alignment,
                     size_t offset) noexcept {
        try {
            if (!alignment || (alignment & (alignment - 1)) || (size ? offset >= size : offset != 0) ||
                size > std::numeric_limits<size_t>::max() - (alignment - 1)) {
                workgroup.fail(VERNON_STATUS_INTERNAL_ERROR, "invalid lane-private allocation request");
                return 0;
            }
            if (!size)
                return reinterpret_cast<uint64_t>(&empty_);
            auto found = std::find_if(allocations_.begin(), allocations_.end(),
                                      [&](const LaneAllocation &allocation) { return allocation.site == site; });
            if (found == allocations_.end()) {
                allocations_.push_back({site, size, alignment});
                found = std::prev(allocations_.end());
            } else if (found->size != size || found->alignment != alignment) {
                if (found->initialized) {
                    workgroup.fail(VERNON_STATUS_INTERNAL_ERROR, "inconsistent lane-private allocation site");
                    return 0;
                }
                found->size = size;
                found->alignment = alignment;
                found->storage.reset();
                found->address = nullptr;
            }
            if (!found->storage) {
                found->storage.reset(new (std::nothrow) unsigned char[size + alignment - 1]);
                if (!found->storage) {
                    workgroup.fail(VERNON_STATUS_INTERNAL_ERROR, "cannot allocate lane-private adjoint storage");
                    return 0;
                }
                const uintptr_t raw = reinterpret_cast<uintptr_t>(found->storage.get());
                found->address = reinterpret_cast<unsigned char *>((raw + alignment - 1) & ~(alignment - 1));
            }
            if (!found->initialized) {
                std::memset(found->address, 0, size);
                found->initialized = true;
            }
            return reinterpret_cast<uint64_t>(found->address + offset);
        } catch (...) {
            workgroup.fail(VERNON_STATUS_INTERNAL_ERROR, "lane-private allocation bookkeeping failed");
            return 0;
        }
    }

private:
    std::vector<LaneAllocation> allocations_;
    unsigned char empty_{};
};

thread_local CpuWorkgroupContext *activeWorkgroup;
thread_local bool activeWorkgroupLeader;
thread_local CpuLaneScratch activeLaneScratch;

[[noreturn]] void failInactiveWorkgroupHelper(const char *helper) noexcept {
    std::fputs("Vernon CPU runtime fatal error: ", stderr);
    std::fputs(helper, stderr);
    std::fputs(" called outside an active CPU workgroup\n", stderr);
    std::fflush(stderr);
    std::abort();
}

class ActiveWorkgroupScope {
public:
    ActiveWorkgroupScope(CpuWorkgroupContext &context, bool leader) noexcept
        : previous_(std::exchange(activeWorkgroup, &context)),
          previousLeader_(std::exchange(activeWorkgroupLeader, leader)) {
        activeLaneScratch.begin();
    }
    ~ActiveWorkgroupScope() {
        activeWorkgroup = previous_;
        activeWorkgroupLeader = previousLeader_;
    }

private:
    CpuWorkgroupContext *previous_{};
    bool previousLeader_{};
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
        try {
            std::unique_lock lock(completionMutex_);
            completion_.wait(lock, [&] { return remaining_.load(std::memory_order_acquire) == 0; });
        } catch (...) {
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

    bool recordFailure(VernonStatus failureStatus, const CpuLaneCoordinates &coordinates,
                       const std::string &detail) noexcept {
        int expected = VERNON_STATUS_OK;
        if (!status_.compare_exchange_strong(expected, failureStatus, std::memory_order_acq_rel,
                                             std::memory_order_relaxed))
            return false;
        try {
            std::lock_guard lock(diagnosticMutex_);
            diagnostic_ =
                "CPU workgroup (" + std::to_string(coordinates.group[0]) + "," + std::to_string(coordinates.group[1]) +
                "," + std::to_string(coordinates.group[2]) + "), local lane (" + std::to_string(coordinates.local[0]) +
                "," + std::to_string(coordinates.local[1]) + "," + std::to_string(coordinates.local[2]) + ") failed";
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

CpuLaneCoordinates independentCoordinates(size_t index, const uint32_t grid[3]) noexcept {
    CpuLaneCoordinates coordinates{};
    coordinates.linearIndex = index;
    coordinates.global[0] = static_cast<uint32_t>(index % grid[0]);
    coordinates.global[1] = static_cast<uint32_t>((index / grid[0]) % grid[1]);
    coordinates.global[2] = static_cast<uint32_t>(index / (static_cast<size_t>(grid[0]) * grid[1]));
    std::copy_n(coordinates.global, 3, coordinates.group);
    return coordinates;
}

class IndependentJob final : public CompletedJob {
public:
    IndependentJob(size_t runners, size_t count, const uint32_t grid[3], const CpuLaneCallback &callback)
        : CompletedJob(runners), count_(count), callback_(callback) {
        std::copy_n(grid, 3, grid_);
    }

    void run() noexcept override {
        CompletionScope completion(*this);
        while (status() == VERNON_STATUS_OK) {
            const size_t index = next_.fetch_add(1, std::memory_order_relaxed);
            if (index >= count_)
                return;
            const CpuLaneCoordinates coordinates = independentCoordinates(index, grid_);
            CpuWorkgroupContext context(1);
            if (context.status() != VERNON_STATUS_OK) {
                recordFailure(context.status(), coordinates, context.diagnostic());
                return;
            }
            ActiveWorkgroupScope scope(context, true);
            VernonStatus laneStatus = VERNON_STATUS_INTERNAL_ERROR;
            std::string exceptionDiagnostic;
            try {
                laneStatus = callback_(coordinates);
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
            if (!exceptionDiagnostic.empty())
                context.fail(VERNON_STATUS_INTERNAL_ERROR, exceptionDiagnostic.c_str());
            context.completeLane(laneStatus);
            if (context.status() != VERNON_STATUS_OK) {
                recordFailure(context.status(), coordinates, context.diagnostic());
                return;
            }
        }
    }

private:
    size_t count_{};
    uint32_t grid_[3]{};
    const CpuLaneCallback &callback_;
    std::atomic<size_t> next_{};
};

class CooperativeJob final : public CompletedJob {
public:
    CooperativeJob(size_t taskCount, size_t firstGroup, size_t groupCount, size_t workgroupVolume,
                   const uint32_t grid[3], const uint32_t workgroup[3], const uint32_t global[3],
                   const CpuLaneCallback &callback)
        : CompletedJob(taskCount), firstGroup_(firstGroup), groupCount_(groupCount), workgroupVolume_(workgroupVolume),
          taskCount_(taskCount), callback_(callback) {
        std::copy_n(grid, 3, grid_);
        std::copy_n(workgroup, 3, workgroup_);
        std::copy_n(global, 3, global_);
        contexts_.reserve(groupCount_);
        for (size_t group = 0; group < groupCount_; ++group)
            contexts_.push_back(std::make_unique<CpuWorkgroupContext>(workgroupVolume_));
    }

    bool ready(std::string &diagnostic) const noexcept {
        for (const auto &context : contexts_) {
            if (context->status() == VERNON_STATUS_OK)
                continue;
            diagnostic = context->diagnostic();
            return false;
        }
        return true;
    }

    void run() noexcept override {
        CompletionScope completion(*this);
        const size_t task = next_.fetch_add(1, std::memory_order_relaxed);
        if (task >= groupCount_ * workgroupVolume_)
            return;
        const size_t localLinear = task % workgroupVolume_;
        const size_t batchGroup = task / workgroupVolume_;
        CpuWorkgroupContext &context = *contexts_[batchGroup];
        const CpuLaneCoordinates coordinates = coordinatesFor(firstGroup_ + batchGroup, localLinear);
        ActiveWorkgroupScope scope(context,
                                   coordinates.local[0] == 0 && coordinates.local[1] == 0 && coordinates.local[2] == 0);
        if (!startWave()) {
            context.fail(VERNON_STATUS_INTERNAL_ERROR, "CPU cooperative team launch synchronization failed");
            recordFailure(context.status(), coordinates, context.diagnostic());
            return;
        }
        VernonStatus laneStatus = VERNON_STATUS_INTERNAL_ERROR;
        std::string exceptionDiagnostic;
        try {
            laneStatus = callback_(coordinates);
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
        if (!exceptionDiagnostic.empty())
            context.fail(VERNON_STATUS_INTERNAL_ERROR, exceptionDiagnostic.c_str());
        context.completeLane(laneStatus);
        if (context.status() != VERNON_STATUS_OK)
            recordFailure(context.status(), coordinates, context.diagnostic());
    }

private:
    bool startWave() noexcept {
        try {
            std::unique_lock lock(startMutex_);
            if (++started_ == taskCount_) {
                startCondition_.notify_all();
                return true;
            }
            startCondition_.wait(lock, [&] { return started_ == taskCount_; });
            return true;
        } catch (...) {
            return false;
        }
    }

    CpuLaneCoordinates coordinatesFor(size_t groupLinear, size_t localLinear) const noexcept {
        CpuLaneCoordinates coordinates{};
        coordinates.group[0] = static_cast<uint32_t>(groupLinear % grid_[0]);
        coordinates.group[1] = static_cast<uint32_t>((groupLinear / grid_[0]) % grid_[1]);
        coordinates.group[2] = static_cast<uint32_t>(groupLinear / (static_cast<size_t>(grid_[0]) * grid_[1]));
        coordinates.local[0] = static_cast<uint32_t>(localLinear % workgroup_[0]);
        coordinates.local[1] = static_cast<uint32_t>((localLinear / workgroup_[0]) % workgroup_[1]);
        coordinates.local[2] =
            static_cast<uint32_t>(localLinear / (static_cast<size_t>(workgroup_[0]) * workgroup_[1]));
        for (size_t dimension = 0; dimension < 3; ++dimension)
            coordinates.global[dimension] =
                coordinates.group[dimension] * workgroup_[dimension] + coordinates.local[dimension];
        coordinates.linearIndex = coordinates.global[0] +
                                  static_cast<size_t>(global_[0]) *
                                      (coordinates.global[1] + static_cast<size_t>(global_[1]) * coordinates.global[2]);
        return coordinates;
    }

    size_t firstGroup_{};
    size_t groupCount_{};
    size_t workgroupVolume_{};
    size_t taskCount_{};
    uint32_t grid_[3]{};
    uint32_t workgroup_[3]{};
    uint32_t global_[3]{};
    const CpuLaneCallback &callback_;
    std::vector<std::unique_ptr<CpuWorkgroupContext>> contexts_;
    std::atomic<size_t> next_{};
    std::mutex startMutex_;
    std::condition_variable startCondition_;
    size_t started_{};
};

} // namespace

class CpuWorkgroupScheduler::Impl {
public:
    explicit Impl(CpuWorkgroupSchedulerConfig config) : config_(config) {}

    bool start(std::string &error) noexcept {
        if (ensureWorkers(config_.threadBudget))
            return true;
        try {
            error = schedulerDiagnostic.empty() ? "cannot start CPU workgroup scheduler workers" : schedulerDiagnostic;
        } catch (...) {
        }
        return false;
    }

    ~Impl() {
        {
            std::lock_guard lock(queueMutex_);
            stopping_ = true;
        }
        workAvailable_.notify_all();
        for (std::thread &worker : workers_)
            if (worker.joinable())
                worker.join();
    }

    VernonStatus dispatch(const uint32_t grid[3], const uint32_t workgroup[3],
                          const CpuLaneCallback &callback) noexcept {
        try {
            schedulerDiagnostic.clear();
            if (activeWorkgroup)
                return fail("nested CPU workgroup dispatch is unsupported", VERNON_STATUS_INVALID_ARGUMENT);
            if (!grid || !workgroup || !callback)
                return fail("CPU workgroup dispatch arguments are incomplete", VERNON_STATUS_INVALID_ARGUMENT);
            uint32_t global[3]{};
            for (size_t dimension = 0; dimension < 3; ++dimension) {
                if (!grid[dimension] || !workgroup[dimension] ||
                    grid[dimension] > std::numeric_limits<uint32_t>::max() / workgroup[dimension])
                    return fail("CPU workgroup dispatch dimensions are invalid", VERNON_STATUS_INVALID_ARGUMENT);
                global[dimension] = grid[dimension] * workgroup[dimension];
            }
            size_t workgroupVolume = 0;
            size_t groupCount = 0;
            if (!checkedVolume(workgroup, workgroupVolume) || !checkedVolume(grid, groupCount))
                return fail("CPU workgroup dispatch volume overflows", VERNON_STATUS_INVALID_ARGUMENT);
            if (workgroupVolume > config_.maxWorkgroupVolume)
                return fail("CPU workgroup volume " + std::to_string(workgroupVolume) + " exceeds configured maximum " +
                                std::to_string(config_.maxWorkgroupVolume),
                            VERNON_STATUS_INVALID_ARGUMENT);
            return workgroupVolume == 1
                       ? dispatchIndependent(grid, groupCount, callback)
                       : dispatchCooperative(grid, workgroup, global, groupCount, workgroupVolume, callback);
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
    size_t workerCount() const noexcept { return workers_.size(); }
    size_t maxWorkgroupVolume() const noexcept { return config_.maxWorkgroupVolume; }

private:
    VernonStatus dispatchIndependent(const uint32_t grid[3], size_t count, const CpuLaneCallback &callback) {
        const size_t runners = std::min(count, config_.threadBudget);
        if (!ensureWorkers(runners))
            return VERNON_STATUS_INTERNAL_ERROR;
        auto job = std::make_shared<IndependentJob>(runners, count, grid, callback);
        if (!enqueue(job, runners))
            return VERNON_STATUS_INTERNAL_ERROR;
        job->wait();
        if (job->status() != VERNON_STATUS_OK)
            schedulerDiagnostic = job->diagnostic();
        return job->status();
    }

    VernonStatus dispatchCooperative(const uint32_t grid[3], const uint32_t workgroup[3], const uint32_t global[3],
                                     size_t groupCount, size_t workgroupVolume, const CpuLaneCallback &callback) {
        const size_t teamCapacity = std::max<size_t>(1, config_.threadBudget / workgroupVolume);
        const size_t groupsPerWave = std::min(groupCount, teamCapacity);
        const size_t requiredWorkers = groupsPerWave * workgroupVolume;
        if (!ensureWorkers(requiredWorkers))
            return VERNON_STATUS_INTERNAL_ERROR;
        for (size_t firstGroup = 0; firstGroup < groupCount; firstGroup += groupsPerWave) {
            const size_t waveGroups = std::min(groupsPerWave, groupCount - firstGroup);
            const size_t taskCount = waveGroups * workgroupVolume;
            auto job = std::make_shared<CooperativeJob>(taskCount, firstGroup, waveGroups, workgroupVolume, grid,
                                                        workgroup, global, callback);
            if (!job->ready(schedulerDiagnostic))
                return VERNON_STATUS_INTERNAL_ERROR;
            if (!enqueue(job, taskCount))
                return VERNON_STATUS_INTERNAL_ERROR;
            job->wait();
            if (job->status() != VERNON_STATUS_OK) {
                schedulerDiagnostic = job->diagnostic();
                return job->status();
            }
        }
        return VERNON_STATUS_OK;
    }

    bool ensureWorkers(size_t required) {
        std::lock_guard workersLock(workersMutex_);
        if (workers_.size() >= required)
            return true;
        const size_t hardLimit = std::max(config_.threadBudget, config_.maxWorkgroupVolume);
        if (required > hardLimit) {
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

    VernonStatus fail(std::string message, VernonStatus status) noexcept {
        try {
            schedulerDiagnostic = std::move(message);
        } catch (...) {
        }
        return status;
    }

    CpuWorkgroupSchedulerConfig config_;
    std::mutex workersMutex_;
    std::mutex queueMutex_;
    std::condition_variable workAvailable_;
    std::deque<std::shared_ptr<SchedulerJob>> queue_;
    bool stopping_{};
    std::vector<std::thread> workers_;
};

CpuWorkgroupSchedulerConfig CpuWorkgroupScheduler::defaultConfig() noexcept {
    const size_t detected = std::max(1u, std::thread::hardware_concurrency());
    return {environmentSize("VERNON_CPU_THREAD_BUDGET", detected),
            environmentSize("VERNON_CPU_MAX_WORKGROUP_VOLUME", kDefaultMaxWorkgroupVolume)};
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
                                             const CpuLaneCallback &callback) noexcept {
    return impl_->dispatch(grid, workgroup, callback);
}

const std::string &CpuWorkgroupScheduler::lastDiagnostic() const noexcept { return impl_->lastDiagnostic(); }
size_t CpuWorkgroupScheduler::workerCount() const noexcept { return impl_->workerCount(); }
size_t CpuWorkgroupScheduler::maxWorkgroupVolume() const noexcept { return impl_->maxWorkgroupVolume(); }

} // namespace vernon::runtime

extern "C" VERNON_RUNTIME_CAPI uint64_t vernonCpuWorkgroupAddressV1(uint64_t site, uint64_t size, uint64_t alignment,
                                                                    uint64_t offset) {
    using namespace vernon::runtime;
    if (!activeWorkgroup)
        failInactiveWorkgroupHelper("vernonCpuWorkgroupAddressV1");
    return activeWorkgroup->address(site, static_cast<size_t>(size), static_cast<size_t>(alignment),
                                    static_cast<size_t>(offset));
}

extern "C" VERNON_RUNTIME_CAPI uint64_t vernonCpuLaneAddressV1(uint64_t site, uint64_t size, uint64_t alignment,
                                                               uint64_t offset) {
    using namespace vernon::runtime;
    if (!activeWorkgroup)
        failInactiveWorkgroupHelper("vernonCpuLaneAddressV1");
    return activeLaneScratch.address(*activeWorkgroup, site, static_cast<size_t>(size), static_cast<size_t>(alignment),
                                     static_cast<size_t>(offset));
}

extern "C" VERNON_RUNTIME_CAPI void vernonCpuWorkgroupBarrierV1(uint64_t site) {
    using namespace vernon::runtime;
    if (!activeWorkgroup)
        failInactiveWorkgroupHelper("vernonCpuWorkgroupBarrierV1");
    activeWorkgroup->barrier(site);
}

extern "C" VERNON_RUNTIME_CAPI bool vernonCpuWorkgroupIsLeaderV1() {
    if (!vernon::runtime::activeWorkgroup)
        vernon::runtime::failInactiveWorkgroupHelper("vernonCpuWorkgroupIsLeaderV1");
    return vernon::runtime::activeWorkgroupLeader;
}
