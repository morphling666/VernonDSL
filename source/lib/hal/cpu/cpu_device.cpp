#include "hal/cpu/cpu_device.h"

#include "runtime/autodiff/host_tape_allocator.h"
#include "runtime/cpu_workgroup_dispatch.h"
#include "runtime/schedule/tape.h"

#include "VernonCpuWorkgroupABI.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <limits>
#include <mutex>
#include <new>
#include <utility>
#include <vector>

namespace vernon::hal {
namespace {

struct CpuExecutable final : Executable {
    VernonCpuEntryPoint entry{};
    VernonCpuChunkEntryPoint chunkEntry{};
    uint32_t workgroup[3]{};
};

struct CpuCommandBuffer final : CommandBuffer {
    std::vector<DispatchCommand> commands;
    std::atomic<bool> timed{false};

    void setTimed(bool enabled) override { timed.store(enabled, std::memory_order_relaxed); }
};

class CpuBuffer final : public Buffer {
public:
    explicit CpuBuffer(void *data) : data_(data) {}
    CpuBuffer(const CpuBuffer &) = delete;
    CpuBuffer &operator=(const CpuBuffer &) = delete;
    ~CpuBuffer() override { ::operator delete(data_, std::align_val_t{kAlignment}); }

    void *address() const override { return data_; }

    static constexpr size_t kAlignment = 256;

private:
    void *data_;
};

class CpuDevice final : public Device {
public:
    explicit CpuDevice(runtime::CpuWorkgroupScheduler &scheduler) : scheduler_(scheduler) {}

    Result<std::unique_ptr<Executable>, std::string> createExecutable(const ExecutableSource &source) override {
        if (!source.cpuEntry)
            return Result<std::unique_ptr<Executable>, std::string>{err(std::string("CPU executable has no entry"))};
        auto executable = std::unique_ptr<CpuExecutable>(new (std::nothrow) CpuExecutable());
        if (!executable)
            return Result<std::unique_ptr<Executable>, std::string>{
                err(std::string("cannot allocate a CPU executable"))};
        executable->entry = source.cpuEntry;
        executable->chunkEntry = source.cpuChunkEntry;
        std::memcpy(executable->workgroup, source.workgroup, sizeof(executable->workgroup));
        return Result<std::unique_ptr<Executable>, std::string>{ok(std::unique_ptr<Executable>(std::move(executable)))};
    }

    Result<std::unique_ptr<Buffer>, std::string> allocate(uint64_t size) override {
        using Allocated = Result<std::unique_ptr<Buffer>, std::string>;
        void *data = ::operator new(static_cast<size_t>(std::max<uint64_t>(size, 1)),
                                    std::align_val_t{CpuBuffer::kAlignment}, std::nothrow);
        if (!data)
            return Allocated{err(std::string("cannot allocate a CPU buffer"))};
        auto buffer = std::unique_ptr<CpuBuffer>(new (std::nothrow) CpuBuffer(data));
        if (!buffer) {
            ::operator delete(data, std::align_val_t{CpuBuffer::kAlignment});
            return Allocated{err(std::string("cannot allocate a CPU buffer"))};
        }
        return Allocated{ok(std::unique_ptr<Buffer>(std::move(buffer)))};
    }

    // Submissions complete before they return, so no earlier one is still reading the buffer.
    Result<void, std::string> upload(Buffer &buffer, const void *bytes, size_t size) override {
        if (size)
            std::memcpy(buffer.address(), bytes, size);
        return Result<void, std::string>{ok()};
    }

    Result<void, std::string> enqueueUpload(Buffer &buffer, const void *bytes, size_t size) override {
        return upload(buffer, bytes, size);
    }

    Result<void, std::string> download(const Buffer &buffer, void *bytes, size_t size) override {
        if (size)
            std::memcpy(bytes, buffer.address(), size);
        return Result<void, std::string>{ok()};
    }

    Result<std::unique_ptr<CommandBuffer>, std::string> record(const std::vector<DispatchCommand> &commands) override {
        using Recorded = Result<std::unique_ptr<CommandBuffer>, std::string>;
        auto buffer = std::unique_ptr<CpuCommandBuffer>(new (std::nothrow) CpuCommandBuffer());
        if (!buffer)
            return Recorded{err(std::string("cannot allocate a CPU command buffer"))};
        for (const DispatchCommand &command : commands) {
            if (command.kind == CommandKind::Dispatch && !dynamic_cast<const CpuExecutable *>(command.executable))
                return Recorded{err(std::string("CPU command buffer names an executable of another device"))};
            if (!command.recordAddresses.empty())
                return Recorded{err(std::string("CPU kernels take no record addresses"))};
            const LeafCopy &copy = command.copy;
            if (command.kind == CommandKind::Copy &&
                (command.argumentsSize != copy.argumentsSize() || !copy.bytes ||
                 copy.storageOffset > copy.storageElement || copy.bytes > copy.storageElement - copy.storageOffset ||
                 copy.carrierOffset > copy.carrierElement || copy.bytes > copy.carrierElement - copy.carrierOffset))
                return Recorded{err(std::string("CPU leaf copy is malformed"))};
            for (const ResultCommit &commit : command.results)
                if (commit.frameOffset > command.resultsSize || commit.size > command.resultsSize - commit.frameOffset)
                    return Recorded{err(std::string("CPU result commit reads outside its results frame"))};
        }
        try {
            buffer->commands = commands;
        } catch (const std::bad_alloc &) {
            return Recorded{err(std::string("cannot allocate a CPU command buffer"))};
        }
        return Recorded{ok(std::unique_ptr<CommandBuffer>(std::move(buffer)))};
    }

    Result<TimelinePoint, std::string> submit(const CommandBuffer &commands, const uint8_t *record,
                                              size_t size) override {
        using Submitted = Result<TimelinePoint, std::string>;
        const auto &tape = static_cast<const CpuCommandBuffer &>(commands);
        // The scheduler runs one dispatch at a time, and the queue is in order.
        std::lock_guard<std::mutex> lock(mutex_);
        const bool timed = tape.timed.load(std::memory_order_relaxed);
        std::vector<ScopeTiming> scopes;
        Clock::time_point origin{};
        for (const DispatchCommand &command : tape.commands) {
            if (command.kind == CommandKind::Fill) {
                if (command.argumentsOffset > size || 2 * sizeof(uint64_t) > size - command.argumentsOffset)
                    return Submitted{err(std::string("CPU fill reads outside its record"))};
                void *target = nullptr;
                uint64_t bytes = 0;
                std::memcpy(&target, record + command.argumentsOffset, sizeof(target));
                std::memcpy(&bytes, record + command.argumentsOffset + sizeof(uint64_t), sizeof(bytes));
                if (bytes)
                    std::memset(target, 0, static_cast<size_t>(bytes));
                continue;
            }
            if (command.kind == CommandKind::Copy) {
                if (command.argumentsOffset > size || command.argumentsSize > size - command.argumentsOffset)
                    return Submitted{err(std::string("CPU leaf copy reads outside its record"))};
                copyLeaf(command.copy, record + command.argumentsOffset);
                continue;
            }
            if (command.argumentsOffset > size || command.argumentsSize > size - command.argumentsOffset ||
                command.gridOffset > size || 3 * sizeof(uint32_t) > size - command.gridOffset)
                return Submitted{err(std::string("CPU dispatch reads outside its record"))};
            const auto &executable = static_cast<const CpuExecutable &>(*command.executable);
            uint32_t grid[3];
            std::memcpy(grid, record + command.gridOffset, sizeof(grid));
            const uint8_t *arguments = record + command.argumentsOffset;
            const size_t argumentsSize = static_cast<size_t>(command.argumentsSize);
            const size_t runners = runnerLimit(grid, executable.workgroup);
            for (const ResultCommit &commit : command.results)
                if (commit.addressOffset > size || sizeof(void *) > size - commit.addressOffset)
                    return Submitted{err(std::string("CPU result commit reads outside its record"))};
            const uint64_t volume = uint64_t{grid[0]} * grid[1] * grid[2] * executable.workgroup[0] *
                                    executable.workgroup[1] * executable.workgroup[2];
            if (auto prepared = prepareResults(command, volume); prepared.isErr())
                return Submitted{err(std::move(prepared).error())};
            const Clock::time_point begin = timed ? Clock::now() : Clock::time_point{};
            if (timed && scopes.empty())
                origin = begin;
            const auto timeDispatch = [&]() {
                if (!timed)
                    return;
                const auto since = [&](Clock::time_point at) {
                    return static_cast<uint64_t>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(at - origin).count());
                };
                scopes.push_back({command.node, 1, since(begin), since(Clock::now())});
            };
            if (command.tape.present()) {
                if (auto taped = dispatchTaped(command, executable, grid, record, size, runners); taped.isErr())
                    return Submitted{err(std::move(taped).error())};
                timeDispatch();
                commitResults(command, record);
                continue;
            }
            const uint64_t groups = uint64_t{grid[0]} * grid[1] * grid[2];
            const VernonStatus status =
                executable.chunkEntry && groups && results_.empty()
                    ? scheduler_.dispatchGroupSpans(
                          static_cast<size_t>(groups), runners,
                          [&](size_t begin, size_t end) {
                              VernonCpuChunk chunk{
                                  sizeof(VernonCpuChunk), arguments, argumentsSize, {}, {}, begin, end};
                              std::memcpy(chunk.grid, grid, sizeof(grid));
                              std::memcpy(chunk.workgroup, executable.workgroup, sizeof(chunk.workgroup));
                              return executable.chunkEntry(&chunk);
                          })
                    : scheduler_.dispatch(grid, executable.workgroup, runners, [&](VernonCpuRange &range) {
                          range.arguments = arguments;
                          range.arguments_size = argumentsSize;
                          range.results = results_.empty() ? nullptr : results_.data();
                          range.results_size = results_.size();
                          range.lane_arguments = nullptr;
                          range.lane_results = laneResults_.empty() ? nullptr : laneResults_.data();
                          range.lane_table_count = laneResults_.size();
                          range.textures = nullptr;
                          const VernonCpuInvocation invocation{&range, VERNON_CPU_RANGE_ARGUMENTS_SIZE, nullptr, 0,
                                                               nullptr};
                          return executable.entry(&invocation);
                      });
            if (status != VERNON_STATUS_OK)
                return Submitted{err(scheduler_.lastDiagnostic().empty() ? std::string("CPU dispatch failed")
                                                                         : scheduler_.lastDiagnostic())};
            timeDispatch();
            commitResults(command, record);
        }
        ++completed_;
        if (timed) {
            try {
                if (timings_.size() == kKeptTimings)
                    timings_.pop_front();
                timings_.push_back({completed_, std::move(scopes)});
            } catch (const std::bad_alloc &) {
            }
        }
        return Submitted{ok(completed_)};
    }

    Result<void, std::string> wait(TimelinePoint point) override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (point > completed_)
            return Result<void, std::string>{err(std::string("CPU timeline point was never submitted"))};
        return Result<void, std::string>{ok()};
    }

    SubmissionTimings timings(TimelinePoint point) override {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto &[submitted, scopes] : timings_)
            if (submitted == point)
                return {true, scopes};
        return {true, std::nullopt};
    }

private:
    using Clock = std::chrono::steady_clock;
    static constexpr size_t kKeptTimings = 8;

    static void copyLeaf(const LeafCopy &copy, const uint8_t *arguments) {
        uint8_t *storage = nullptr;
        uint8_t *carrier = nullptr;
        int64_t offset = 0;
        std::memcpy(&storage, arguments, sizeof(storage));
        std::memcpy(&carrier, arguments + sizeof(uint64_t), sizeof(carrier));
        std::memcpy(&offset, arguments + 2 * sizeof(uint64_t), sizeof(offset));
        std::vector<int64_t> extents(copy.rank);
        std::vector<int64_t> strides(copy.rank);
        for (uint32_t axis = 0; axis < copy.rank; ++axis) {
            std::memcpy(&extents[axis], arguments + (3 + axis) * sizeof(uint64_t), sizeof(int64_t));
            std::memcpy(&strides[axis], arguments + (3 + copy.rank + axis) * sizeof(uint64_t), sizeof(int64_t));
        }
        if (std::any_of(extents.begin(), extents.end(), [](int64_t extent) { return extent <= 0; }))
            return;
        std::vector<int64_t> index(copy.rank, 0);
        int64_t element = offset;
        for (uint8_t *packed = carrier + copy.carrierOffset;; packed += copy.carrierElement) {
            uint8_t *leaf = storage + element * static_cast<int64_t>(copy.storageElement) +
                            static_cast<int64_t>(copy.storageOffset);
            std::memcpy(copy.scatter ? leaf : packed, copy.scatter ? packed : leaf, static_cast<size_t>(copy.bytes));
            uint32_t axis = copy.rank;
            while (axis && ++index[axis - 1] == extents[axis - 1]) {
                element -= (extents[axis - 1] - 1) * strides[axis - 1];
                index[--axis] = 0;
            }
            if (!axis)
                return;
            element += strides[axis - 1];
        }
    }

    // A dispatch summing gradient results gives each of its `volume` lanes a zeroed frame of its own.
    Result<void, std::string> prepareResults(const DispatchCommand &command, uint64_t volume) {
        using Done = Result<void, std::string>;
        const auto frame = static_cast<size_t>(command.resultsSize);
        try {
            results_.assign(frame, 0);
            laneResults_.clear();
            if (command.resultSums.empty() || !frame)
                return Done{ok()};
            if (volume > std::numeric_limits<size_t>::max() / frame)
                return Done{err(std::string("CPU result lanes overflow"))};
            laneResultFrames_.assign(static_cast<size_t>(volume) * frame, 0);
            laneResults_.resize(static_cast<size_t>(volume));
        } catch (const std::bad_alloc &) {
            return Done{err(std::string("cannot allocate CPU results frames"))};
        }
        for (size_t lane = 0; lane < laneResults_.size(); ++lane)
            laneResults_[lane] = laneResultFrames_.data() + lane * frame;
        return Done{ok()};
    }

    // The last lane's frame, with each gradient range the sum of the lanes', in lane order.
    void reduceResults(const DispatchCommand &command) {
        if (laneResults_.empty())
            return;
        std::memcpy(results_.data(), laneResults_.back(), results_.size());
        for (const ResultSum &sum : command.resultSums) {
            const auto accumulate = [&](auto scalar) {
                using Scalar = decltype(scalar);
                for (uint64_t at = sum.frameOffset; at + sizeof(Scalar) <= sum.frameOffset + sum.size;
                     at += sizeof(Scalar)) {
                    Scalar total{};
                    for (const void *lane : laneResults_) {
                        Scalar value;
                        std::memcpy(&value, static_cast<const uint8_t *>(lane) + at, sizeof(Scalar));
                        total += value;
                    }
                    std::memcpy(results_.data() + at, &total, sizeof(Scalar));
                }
            };
            if (sum.dtype == VERNON_DATA_F64)
                accumulate(double{});
            else
                accumulate(float{});
        }
    }

    void commitResults(const DispatchCommand &command, const uint8_t *record) {
        reduceResults(command);
        for (const ResultCommit &commit : command.results) {
            uint8_t *buffer = nullptr;
            std::memcpy(&buffer, record + commit.addressOffset, sizeof(buffer));
            std::memcpy(buffer + commit.bufferOffset, results_.data() + commit.frameOffset,
                        static_cast<size_t>(commit.size));
        }
    }

    // A kernel keeping a tape runs one invocation per lane, each with its own frame naming its lane's allocator (a
    // writer in the forward schedule, a reader of the sealed batch in the backward); the batch is sealed only once
    // every lane recorded without overflow.
    Result<void, std::string> dispatchTaped(const DispatchCommand &command, const CpuExecutable &executable,
                                            const uint32_t grid[3], const uint8_t *record, size_t size,
                                            size_t runners) {
        using Done = Result<void, std::string>;
        const DispatchTape &use = command.tape;
        const uint64_t argumentsEnd = command.argumentsOffset + command.argumentsSize;
        const auto inArguments = [&](uint64_t offset, uint64_t bytes) {
            return offset >= command.argumentsOffset && offset <= argumentsEnd && bytes <= argumentsEnd - offset;
        };
        if (use.tapeOffset > size || sizeof(void *) > size - use.tapeOffset ||
            !inArguments(use.allocatorOffset, sizeof(void *)) ||
            (use.rootOffset != UINT64_MAX && !inArguments(use.rootOffset, sizeof(VernonAdRegionHandle))))
            return Done{err(std::string("CPU tape dispatch reads outside its record"))};
        VernonScheduleTape *tape = nullptr;
        std::memcpy(&tape, record + use.tapeOffset, sizeof(tape));
        if (!tape)
            return Done{err(std::string("CPU tape dispatch has no tape"))};
        const uint64_t volume = uint64_t{grid[0]} * grid[1] * grid[2] * executable.workgroup[0] *
                                executable.workgroup[1] * executable.workgroup[2];
        if (!volume)
            return Done{ok()};
        if (use.writes) {
            if (auto prepared = prepareBatch(*tape, static_cast<size_t>(volume), use.stride); prepared.isErr())
                return prepared;
        } else if (!tape->batch || !tape->batch->isCompacted() || tape->batch->size() != volume) {
            return Done{err(std::string("CPU tape dispatch reads a tape its forward never sealed"))};
        }
        runtime::ad::HostStaticTapeBatch &batch = *tape->batch;
        const size_t argumentsSize = static_cast<size_t>(command.argumentsSize);
        const size_t lanes = static_cast<size_t>(volume);
        try {
            laneFrames_.resize(lanes * argumentsSize);
            laneArguments_.resize(lanes);
            if (!use.writes)
                readers_.resize(lanes);
        } catch (const std::bad_alloc &) {
            return Done{err(std::string("cannot allocate CPU tape lane frames"))};
        }
        const uint8_t *arguments = record + command.argumentsOffset;
        const size_t allocatorAt = static_cast<size_t>(use.allocatorOffset - command.argumentsOffset);
        const size_t rootAt =
            use.rootOffset == UINT64_MAX ? SIZE_MAX : static_cast<size_t>(use.rootOffset - command.argumentsOffset);
        for (size_t lane = 0; lane < lanes; ++lane) {
            uint8_t *frame = laneFrames_.data() + lane * argumentsSize;
            std::memcpy(frame, arguments, argumentsSize);
            VernonAdTapeAllocator *allocator = nullptr;
            VernonAdRegionHandle root = VERNON_AD_INVALID_REGION_HANDLE;
            if (use.writes) {
                allocator = batch.descriptor(lane);
                root = batch.rootRegion(lane);
            } else {
                if (!batch.initializeReader(lane, readers_[lane]))
                    return Done{err(std::string("CPU tape dispatch cannot read its tape"))};
                allocator = readers_[lane].descriptor();
                root = readers_[lane].rootRegion();
                if (rootAt != SIZE_MAX && !root)
                    return Done{err(std::string("CPU tape dispatch's tape has no sealed root region"))};
            }
            if (!allocator)
                return Done{err(std::string("CPU tape dispatch is missing a lane allocator"))};
            std::memcpy(frame + allocatorAt, &allocator, sizeof(allocator));
            if (rootAt != SIZE_MAX)
                std::memcpy(frame + rootAt, &root, sizeof(root));
            laneArguments_[lane] = frame;
        }
        const VernonStatus status =
            scheduler_.dispatch(grid, executable.workgroup, runners, [&](VernonCpuRange &range) {
                range.arguments = laneArguments_.front();
                range.arguments_size = argumentsSize;
                range.results = results_.empty() ? nullptr : results_.data();
                range.results_size = results_.size();
                range.lane_results = laneResults_.empty() ? nullptr : laneResults_.data();
                range.textures = nullptr;
                range.lane_arguments = laneArguments_.data();
                range.lane_table_count = lanes;
                const VernonCpuInvocation invocation{&range, VERNON_CPU_RANGE_ARGUMENTS_SIZE, nullptr, 0, nullptr};
                return executable.entry(&invocation);
            });
        if (status != VERNON_STATUS_OK)
            return Done{err(scheduler_.lastDiagnostic().empty() ? std::string("CPU tape dispatch failed")
                                                                : scheduler_.lastDiagnostic())};
        if (!use.writes)
            return Done{ok()};
        for (size_t lane = 0; lane < lanes; ++lane) {
            const VernonAdTapeAllocatorStatus laneStatus = batch.descriptor(lane)->status;
            if (laneStatus == VERNON_AD_TAPE_ALLOCATOR_OK)
                continue;
            return Done{err(std::string("autodiff tape allocator ") +
                            (laneStatus == VERNON_AD_TAPE_ALLOCATOR_CAPACITY_EXHAUSTED        ? "capacity exhausted"
                             : laneStatus == VERNON_AD_TAPE_ALLOCATOR_ARITHMETIC_OVERFLOW     ? "arithmetic overflow"
                             : laneStatus == VERNON_AD_TAPE_ALLOCATOR_HOST_ALLOCATION_FAILURE ? "host allocation failed"
                             : laneStatus == VERNON_AD_TAPE_ALLOCATOR_INVALID_ABI             ? "invalid ABI"
                                                                                              : "invalid state"))};
        }
        if (!batch.compact(true))
            return Done{err(std::string("autodiff tape could not be sealed"))};
        return Done{ok()};
    }

    // A tape's earlier batch of the same shape is recycled; nothing reads it any more once its forward reruns.
    static Result<void, std::string> prepareBatch(VernonScheduleTape &tape, size_t lanes, uint64_t stride) {
        using Done = Result<void, std::string>;
        runtime::ad::HostStaticTapeBatch *batch = tape.batch.get();
        if (batch && tape.batch.use_count() == 1 && batch->size() == lanes && batch->payloadStride() == stride &&
            batch->markConstructionRecyclable() && batch->resetRecyclableConstruction())
            return Done{ok()};
        tape.batch.reset();
        static const auto policy = std::make_shared<runtime::ad::AutodiffMemoryPolicy>();
        auto created = runtime::ad::HostStaticTapeBatch::create(lanes, static_cast<size_t>(stride),
                                                                policy->invocationLimit(), policy, nullptr);
        if (created.isErr())
            return Done{err(std::string(created.error() == runtime::ad::HostTapeError::BudgetExceeded
                                            ? "autodiff tape exceeds its memory budget"
                                            : "autodiff tape cannot be allocated"))};
        tape.batch = std::move(created).value();
        return Done{ok()};
    }

    // Whole workgroups per runner, since splitting one across threads runs the barrier-phase protocol, and enough
    // invocations per runner to pay for waking it (specs/performance.md section 3.2).
    size_t runnerLimit(const uint32_t grid[3], const uint32_t workgroup[3]) const {
        const uint64_t groups = uint64_t{grid[0]} * grid[1] * grid[2];
        const uint64_t invocations = groups * workgroup[0] * workgroup[1] * workgroup[2];
        return static_cast<size_t>(std::min(groups, (invocations + invocationsPerRunner_ - 1) / invocationsPerRunner_));
    }

    static uint64_t configuredInvocationsPerRunner() {
        constexpr uint64_t fallback = 4096;
        const char *configured = std::getenv("VERNON_CPU_INVOCATIONS_PER_RUNNER");
        if (!configured || !*configured)
            return fallback;
        char *end = nullptr;
        const unsigned long long value = std::strtoull(configured, &end, 10);
        return end && !*end && value ? value : fallback;
    }

    runtime::CpuWorkgroupScheduler &scheduler_;
    uint64_t invocationsPerRunner_{configuredInvocationsPerRunner()};
    std::mutex mutex_;
    TimelinePoint completed_{};
    std::deque<std::pair<TimelinePoint, std::vector<ScopeTiming>>> timings_;
    std::vector<uint8_t> laneFrames_;
    // The results frame of the dispatch running, when its kernel has results, and its lanes' when it sums gradients.
    std::vector<uint8_t> results_;
    std::vector<uint8_t> laneResultFrames_;
    std::vector<void *> laneResults_;
    std::vector<const void *> laneArguments_;
    std::vector<runtime::ad::HostStaticTapeBatch::Reader> readers_;
};

} // namespace

std::unique_ptr<Device> createCpuDevice(runtime::CpuWorkgroupScheduler &scheduler) {
    return std::unique_ptr<Device>(new (std::nothrow) CpuDevice(scheduler));
}

} // namespace vernon::hal
