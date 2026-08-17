#include "host_tape_allocator.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <functional>
#include <new>
#include <stdexcept>
#include <utility>

namespace vernon::runtime::ad {
namespace {

#ifdef VERNON_HOST_TAPE_INSTRUMENTATION
thread_local HostTapeTraversalMetrics *activeTraversalMetrics;
std::mutex traversalMetricsMutex;
#endif

bool checkedAdd(size_t left, size_t right, size_t &result) {
    if (right > std::numeric_limits<size_t>::max() - left)
        return false;
    result = left + right;
    return true;
}

bool checkedMultiply(size_t left, size_t right, size_t &result) {
    if (left && right > std::numeric_limits<size_t>::max() / left)
        return false;
    result = left * right;
    return true;
}

template <typename T> class ChunkedArena {
public:
    using ReserveMemory = bool (*)(void *, size_t);
    using ReleaseMemory = void (*)(void *, size_t);

    explicit ChunkedArena(size_t maximumBytes) : maximumElements_(std::max<size_t>(maximumBytes / sizeof(T), 1)) {
        const size_t requestedChunks =
            maximumElements_ / kElementsPerChunk + (maximumElements_ % kElementsPerChunk != 0);
        maximumChunks_ = std::min(requestedChunks, kMaximumChunks);
        maximumElements_ = std::min(maximumElements_, maximumChunks_ * kElementsPerChunk);
        directoryCount_ = maximumChunks_ / kChunksPerDirectory + (maximumChunks_ % kChunksPerDirectory != 0);
        directories_ = std::make_unique<std::atomic<Directory *>[]>(directoryCount_);
        for (size_t index = 0; index < directoryCount_; ++index)
            directories_[index].store(nullptr, std::memory_order_relaxed);
    }

    ~ChunkedArena() { clear(); }

    void clear() {
        if (!directories_)
            return;
        for (size_t directoryIndex = 0; directoryIndex < directoryCount_; ++directoryIndex) {
            Directory *directory = directories_[directoryIndex].exchange(nullptr, std::memory_order_acq_rel);
            if (!directory)
                continue;
            for (auto &chunk : directory->chunks)
                delete[] chunk.load(std::memory_order_relaxed);
            delete directory;
        }
        size_.store(0, std::memory_order_release);
        dynamicBytes_.store(0, std::memory_order_release);
    }

    void release() {
        clear();
        directories_.reset();
        directoryCount_ = 0;
        maximumChunks_ = 0;
        maximumElements_ = 0;
    }

    void rewind() { size_.store(0, std::memory_order_release); }

    ChunkedArena(const ChunkedArena &) = delete;
    ChunkedArena &operator=(const ChunkedArena &) = delete;

    void setMemoryCallbacks(void *context, ReserveMemory reserveMemory, ReleaseMemory releaseMemory) {
        memoryContext_ = context;
        reserveMemory_ = reserveMemory;
        releaseMemory_ = releaseMemory;
    }

    bool allocate(size_t count, size_t &offset) {
        size_t current = size_.load(std::memory_order_relaxed);
        for (;;) {
            size_t end = 0;
            if (!checkedAdd(current, count, end) || end > maximumElements_)
                return false;
            if (count) {
                const size_t firstChunk = current / kElementsPerChunk;
                const size_t lastChunk = (end - 1) / kElementsPerChunk;
                for (size_t chunk = firstChunk; chunk <= lastChunk; ++chunk)
                    if (!ensureChunk(chunk))
                        return false;
            }
            const size_t begin = current;
            if (size_.compare_exchange_weak(current, end, std::memory_order_release, std::memory_order_relaxed)) {
                offset = begin;
                return true;
            }
        }
    }

    T *at(size_t index) {
        if (index >= size())
            return nullptr;
        T *chunk = chunkAt(index / kElementsPerChunk);
        return chunk ? chunk + index % kElementsPerChunk : nullptr;
    }

    const T *at(size_t index) const {
        if (index >= size())
            return nullptr;
        const T *chunk = chunkAt(index / kElementsPerChunk);
        return chunk ? chunk + index % kElementsPerChunk : nullptr;
    }

    bool write(size_t offset, const T *source, size_t count) {
        size_t copied = 0;
        while (copied < count) {
            T *destination = at(offset + copied);
            if (!destination)
                return false;
            const size_t inChunk = (offset + copied) % kElementsPerChunk;
            const size_t span = std::min(count - copied, kElementsPerChunk - inChunk);
            std::copy_n(source + copied, span, destination);
            copied += span;
        }
        return true;
    }

    bool read(size_t offset, T *destination, size_t count) const {
        size_t copied = 0;
        while (copied < count) {
            const T *source = at(offset + copied);
            if (!source)
                return false;
            const size_t inChunk = (offset + copied) % kElementsPerChunk;
            const size_t span = std::min(count - copied, kElementsPerChunk - inChunk);
            std::copy_n(source, span, destination + copied);
            copied += span;
        }
        return true;
    }

    size_t size() const { return size_.load(std::memory_order_acquire); }
    size_t allocatedBytes() const {
        return (directories_ ? directoryCount_ * sizeof(std::atomic<Directory *>) : 0) +
               dynamicBytes_.load(std::memory_order_relaxed);
    }

private:
    static constexpr size_t kChunkBytes = 64u * 1024u;
    static constexpr size_t kElementsPerChunk = std::max<size_t>(kChunkBytes / sizeof(T), 1);
    static constexpr size_t kChunksPerDirectory = 1024;
    static constexpr size_t kMaximumDirectories = 4096;
    static constexpr size_t kMaximumChunks = kChunksPerDirectory * kMaximumDirectories;

    struct Directory {
        Directory() {
            for (auto &chunk : chunks)
                chunk.store(nullptr, std::memory_order_relaxed);
        }
        std::array<std::atomic<T *>, kChunksPerDirectory> chunks;
    };

    T *chunkAt(size_t index) const {
        Directory *directory = directories_[index / kChunksPerDirectory].load(std::memory_order_acquire);
        return directory ? directory->chunks[index % kChunksPerDirectory].load(std::memory_order_acquire) : nullptr;
    }

    bool ensureChunk(size_t index) {
        const size_t directoryIndex = index / kChunksPerDirectory;
        Directory *directory = directories_[directoryIndex].load(std::memory_order_acquire);
        if (!directory) {
            if (reserveMemory_ && !reserveMemory_(memoryContext_, sizeof(Directory)))
                return false;
            Directory *candidate = new (std::nothrow) Directory;
            if (!candidate) {
                if (releaseMemory_)
                    releaseMemory_(memoryContext_, sizeof(Directory));
                return false;
            }
            Directory *expected = nullptr;
            if (!directories_[directoryIndex].compare_exchange_strong(expected, candidate, std::memory_order_release,
                                                                      std::memory_order_acquire)) {
                delete candidate;
                if (releaseMemory_)
                    releaseMemory_(memoryContext_, sizeof(Directory));
            } else {
                dynamicBytes_.fetch_add(sizeof(Directory), std::memory_order_relaxed);
            }
            directory = expected ? expected : candidate;
        }
        std::atomic<T *> &slot = directory->chunks[index % kChunksPerDirectory];
        if (slot.load(std::memory_order_acquire))
            return true;
        constexpr size_t chunkBytes = kElementsPerChunk * sizeof(T);
        if (reserveMemory_ && !reserveMemory_(memoryContext_, chunkBytes))
            return false;
        T *candidate = new (std::nothrow) T[kElementsPerChunk]{};
        if (!candidate) {
            if (releaseMemory_)
                releaseMemory_(memoryContext_, chunkBytes);
            return false;
        }
        T *expected = nullptr;
        if (!slot.compare_exchange_strong(expected, candidate, std::memory_order_release, std::memory_order_acquire)) {
            delete[] candidate;
            if (releaseMemory_)
                releaseMemory_(memoryContext_, chunkBytes);
        } else {
            dynamicBytes_.fetch_add(chunkBytes, std::memory_order_relaxed);
        }
        return true;
    }

    size_t maximumElements_{};
    size_t maximumChunks_{};
    size_t directoryCount_{};
    std::unique_ptr<std::atomic<Directory *>[]> directories_;
    std::atomic<size_t> size_{};
    std::atomic<size_t> dynamicBytes_{};
    void *memoryContext_{};
    ReserveMemory reserveMemory_{};
    ReleaseMemory releaseMemory_{};
};

template <typename T> void releaseVectorStorage(std::vector<T> &values) { std::vector<T>().swap(values); }

} // namespace

#ifdef VERNON_HOST_TAPE_INSTRUMENTATION
HostTapeTraversalScope::HostTapeTraversalScope(HostTapeTraversalMetrics &metrics)
    : previous_(std::exchange(activeTraversalMetrics, &metrics)) {}

HostTapeTraversalScope::~HostTapeTraversalScope() { activeTraversalMetrics = previous_; }

HostTapeTraversalMetrics *currentHostTapeTraversalMetrics() { return activeTraversalMetrics; }

VernonStatus withHostTapeTraversalMetrics(HostTapeTraversalMetrics *destination,
                                          const std::function<VernonStatus()> &callback) {
    if (!destination)
        return callback();
    HostTapeTraversalMetrics local;
    VernonStatus status;
    {
        HostTapeTraversalScope scope(local);
        status = callback();
    }
    std::lock_guard lock(traversalMetricsMutex);
    destination->regionLookups += local.regionLookups;
    destination->recordResolutions += local.recordResolutions;
    destination->leafReads += local.leafReads;
    destination->childReads += local.childReads;
    destination->executedCountReads += local.executedCountReads;
    destination->exitKindReads += local.exitKindReads;
    return status;
}
#endif

bool AutodiffMemoryPolicy::reserveContext(size_t additionalBytes) {
    std::lock_guard lock(mutex_);
    size_t contextBytes = 0;
    if (!checkedAdd(contextBytes_, additionalBytes, contextBytes) || contextBytes > contextLimit_)
        return false;
    contextBytes_ = contextBytes;
    peakContextBytes_ = std::max(peakContextBytes_, contextBytes_);
    return true;
}

void AutodiffMemoryPolicy::release(size_t bytes) {
    std::lock_guard lock(mutex_);
    contextBytes_ = bytes > contextBytes_ ? 0 : contextBytes_ - bytes;
}

HostTapeMemoryUsage AutodiffMemoryPolicy::usage() const {
    std::lock_guard lock(mutex_);
    return {contextBytes_, peakContextBytes_};
}

std::shared_ptr<AutodiffMemoryReservation>
AutodiffMemoryReservation::reserve(std::shared_ptr<AutodiffMemoryPolicy> policy, size_t bytes) {
    if (!policy)
        return {};
    if (!bytes)
        return std::shared_ptr<AutodiffMemoryReservation>(new AutodiffMemoryReservation(std::move(policy), 0));
    if (!policy->reserveContext(bytes))
        return {};
    try {
        return std::shared_ptr<AutodiffMemoryReservation>(new AutodiffMemoryReservation(std::move(policy), bytes));
    } catch (...) {
        policy->release(bytes);
        return {};
    }
}

AutodiffMemoryReservation::~AutodiffMemoryReservation() {
    if (policy_ && bytes_)
        policy_->release(bytes_);
}

std::shared_ptr<HostTapeDispatchBudget> HostTapeDispatchBudget::reserve(std::shared_ptr<HostTapeMemoryPolicy> policy,
                                                                        size_t capacity) {
    if (!policy || !capacity)
        return {};
    capacity = std::min(capacity, policy->contextLimit());
    if (!capacity)
        return {};
    std::shared_ptr<HostTapeDispatchBudget> budget;
    try {
        budget = std::shared_ptr<HostTapeDispatchBudget>(new HostTapeDispatchBudget(policy, capacity));
    } catch (const std::bad_alloc &) {
        return {};
    }
    return budget;
}

HostTapeDispatchBudget::~HostTapeDispatchBudget() {
    size_t usedBytes = 0;
    {
        std::lock_guard lock(mutex_);
        usedBytes = std::exchange(usedBytes_, 0);
    }
    if (policy_ && usedBytes)
        policy_->release(usedBytes);
}

size_t HostTapeDispatchBudget::usedBytes() const {
    std::lock_guard lock(mutex_);
    return usedBytes_;
}

HostTapeMemoryUsage HostTapeDispatchBudget::usage() const {
    std::lock_guard lock(mutex_);
    return {usedBytes_, peakUsedBytes_};
}

bool HostTapeDispatchBudget::reserveBytes(size_t additionalBytes) {
    std::lock_guard lock(mutex_);
    size_t required = 0;
    if (committed_ || !checkedAdd(usedBytes_, additionalBytes, required) || required > capacity_ || !policy_ ||
        !policy_->reserveContext(additionalBytes))
        return false;
    usedBytes_ = required;
    peakUsedBytes_ = std::max(peakUsedBytes_, usedBytes_);
    return true;
}

void HostTapeDispatchBudget::releaseBytes(size_t bytes) {
    size_t released = 0;
    {
        std::lock_guard lock(mutex_);
        released = std::min(bytes, usedBytes_);
        usedBytes_ -= released;
    }
    if (policy_ && released)
        policy_->release(released);
}

void HostTapeDispatchBudget::commit() {
    std::lock_guard lock(mutex_);
    committed_ = true;
}

bool HostTapeDispatchBudget::beginRecycledConstruction() {
    std::lock_guard lock(mutex_);
    if (!committed_)
        return false;
    committed_ = false;
    return true;
}

#ifdef VERNON_HOST_TAPE_INSTRUMENTATION
size_t hostTapeMemoryPolicyChargedBytesForTesting(HostTapeMemoryPolicy &policy) {
    std::lock_guard lock(policy.mutex_);
    return policy.contextBytes_;
}
#endif

class HostDynamicTapeBatch::Impl {
public:
    using SnapshotRegion = detail::HostTapeSnapshotRegion;
    using SnapshotRecord = detail::HostTapeSnapshotRecord;

    struct Lane {
        VernonAdTapeAllocatorStatus status{VERNON_AD_TAPE_ALLOCATOR_OK};
        size_t requiredBytes{};
        size_t payloadBytes{};
        VernonAdRegionHandle root{VERNON_AD_INVALID_REGION_HANDLE};
        VernonAdRegionHandle current{VERNON_AD_INVALID_REGION_HANDLE};
        uint32_t generation{1};
        bool sealed{};
    };
    struct Region {
        VernonAdRegionHandle handle{};
        VernonAdRegionHandle parent{};
        size_t lane{};
        uint32_t generation{};
        size_t firstRecord{std::numeric_limits<size_t>::max()};
        size_t lastRecord{std::numeric_limits<size_t>::max()};
        size_t recordCount{};
        size_t executedCount{};
        uint32_t exitKind{};
        bool ended{};
    };
    struct Record {
        VernonAdRecordHandle handle{};
        size_t regionIndex{};
        size_t lane{};
        uint32_t generation{};
        size_t payloadOffset{};
        size_t payloadSize{};
        size_t childOffset{};
        size_t childCount{};
        size_t next{std::numeric_limits<size_t>::max()};
    };
    Impl(size_t laneCount, size_t invocationCapacity, std::shared_ptr<HostTapeMemoryPolicy> memoryPolicy,
         std::shared_ptr<HostTapeDispatchBudget> budget)
        : lanes(laneCount), payload(budget         ? budget->capacity()
                                    : memoryPolicy ? memoryPolicy->contextLimit()
                                                   : 1),
          children(budget         ? budget->capacity()
                   : memoryPolicy ? memoryPolicy->contextLimit()
                                  : 1),
          regions(budget         ? budget->capacity()
                  : memoryPolicy ? memoryPolicy->contextLimit()
                                 : 1),
          records(budget         ? budget->capacity()
                  : memoryPolicy ? memoryPolicy->contextLimit()
                                 : 1),
          invocationCapacity(invocationCapacity), policy(std::move(memoryPolicy)), dispatchBudget(std::move(budget)) {
        payload.setMemoryCallbacks(this, reserveArenaMemory, releaseArenaMemory);
        children.setMemoryCallbacks(this, reserveArenaMemory, releaseArenaMemory);
        regions.setMemoryCallbacks(this, reserveArenaMemory, releaseArenaMemory);
        records.setMemoryCallbacks(this, reserveArenaMemory, releaseArenaMemory);
        size_t baseBytes = lanes.capacity() * sizeof(Lane);
        if (!checkedAdd(baseBytes, payload.allocatedBytes(), baseBytes) ||
            !checkedAdd(baseBytes, children.allocatedBytes(), baseBytes) ||
            !checkedAdd(baseBytes, regions.allocatedBytes(), baseBytes) ||
            !checkedAdd(baseBytes, records.allocatedBytes(), baseBytes) || !reservePhysical(baseBytes))
            throw std::runtime_error("dynamic tape arena metadata exceeds context budget");
    }

    ~Impl() { releasePhysical(totalCharge.load(std::memory_order_relaxed)); }

    VernonAdTapeAllocatorStatus fail(size_t lane, VernonAdTapeAllocatorStatus status) {
        Lane &state = lanes[lane];
        if (state.status == VERNON_AD_TAPE_ALLOCATOR_OK ||
            (state.status == VERNON_AD_TAPE_ALLOCATOR_CAPACITY_EXHAUSTED &&
             status == VERNON_AD_TAPE_ALLOCATOR_ARITHMETIC_OVERFLOW))
            state.status = status;
        return state.status;
    }

    bool reservePhysical(size_t additional) {
        if (dispatchBudget ? !dispatchBudget->reserveTransientBytes(additional)
                           : !policy || !policy->reserveContext(additional))
            return false;
        totalCharge.fetch_add(additional, std::memory_order_relaxed);
        return true;
    }

    void releasePhysical(size_t bytes) {
        if (!bytes)
            return;
        if (dispatchBudget)
            dispatchBudget->releaseTransientBytes(bytes);
        else if (policy)
            policy->release(bytes);
        totalCharge.fetch_sub(bytes, std::memory_order_relaxed);
    }

    static bool reserveArenaMemory(void *context, size_t bytes) {
        return static_cast<Impl *>(context)->reservePhysical(bytes);
    }

    static void releaseArenaMemory(void *context, size_t bytes) {
        static_cast<Impl *>(context)->releasePhysical(bytes);
    }

    template <typename T> void releaseArena(ChunkedArena<T> &arena) {
        releasePhysical(arena.allocatedBytes());
        arena.release();
    }

    Region *findRegion(size_t lane, VernonAdRegionHandle handle) {
        if (!handle || handle > regions.size())
            return nullptr;
        Region *region = regions.at(static_cast<size_t>(handle - 1));
        if (!region)
            return nullptr;
        const Lane &state = lanes[lane];
        return region->handle == handle && region->lane == lane && region->generation == state.generation ? region
                                                                                                          : nullptr;
    }

    const SnapshotRegion *findSnapshotRegion(size_t lane, VernonAdRegionHandle handle) const {
        constexpr uint64_t indexMask = std::numeric_limits<uint32_t>::max();
        const uint64_t encodedLane = handle >> 32;
        const uint64_t encodedIndex = handle & indexMask;
        if (!compacted || encodedLane != lane + 1 || !encodedIndex || encodedIndex > snapshotRegions.size())
            return nullptr;
        return &snapshotRegions[static_cast<size_t>(encodedIndex - 1)];
    }

    Record *findRecord(size_t lane, VernonAdRecordHandle handle) {
        if (!handle || handle > records.size())
            return nullptr;
        Record *record = records.at(static_cast<size_t>(handle - 1));
        if (!record)
            return nullptr;
        const Lane &state = lanes[lane];
        return record->handle == handle && record->lane == lane && record->generation == state.generation ? record
                                                                                                          : nullptr;
    }

    const SnapshotRecord *resolveSnapshotRecord(size_t lane, VernonAdRegionHandle handle, size_t ordinal) const {
        const SnapshotRegion *region = findSnapshotRegion(lane, handle);
        if (!region || ordinal >= region->recordCount || region->recordOffset > snapshotRecords.size() ||
            region->recordCount > snapshotRecords.size() - region->recordOffset)
            return nullptr;
        return &snapshotRecords[region->recordOffset + ordinal];
    }

    mutable std::mutex mutex;
    std::vector<Lane> lanes;
    ChunkedArena<std::byte> payload;
    ChunkedArena<VernonAdRegionHandle> children;
    ChunkedArena<Region> regions;
    ChunkedArena<Record> records;

    std::vector<VernonAdRegionHandle> snapshotChildren;
    std::vector<SnapshotRegion> snapshotRegions;
    std::vector<SnapshotRecord> snapshotRecords;
    std::vector<VernonAdRegionHandle> laneRoots;

    size_t invocationCapacity{};
    std::shared_ptr<HostTapeMemoryPolicy> policy;
    std::shared_ptr<HostTapeDispatchBudget> dispatchBudget;
    std::atomic<size_t> totalCharge{};
    size_t compactedLogicalBytes{};
    bool compacted{};
};

HostDynamicTapeBatch::HostDynamicTapeBatch(size_t laneCount, size_t invocationCapacity,
                                           std::shared_ptr<HostTapeMemoryPolicy> policy,
                                           std::shared_ptr<HostTapeDispatchBudget> dispatchBudget)
    : impl_(std::make_unique<Impl>(laneCount, invocationCapacity, std::move(policy), std::move(dispatchBudget))) {}

HostDynamicTapeBatch::~HostDynamicTapeBatch() = default;

VernonAdTapeAllocatorStatus HostDynamicTapeBatch::reset(size_t lane) {
    if (lane >= impl_->lanes.size() || impl_->compacted)
        return VERNON_AD_TAPE_ALLOCATOR_INVALID_STATE;
    auto &state = impl_->lanes[lane];
    const uint32_t generation = state.generation == std::numeric_limits<uint32_t>::max() ? 1 : state.generation + 1;
    state = {};
    state.generation = generation;
    return VERNON_AD_TAPE_ALLOCATOR_OK;
}

VernonAdTapeAllocatorStatus HostDynamicTapeBatch::beginRegion(size_t lane, VernonAdRegionHandle parent,
                                                              VernonAdRegionHandle *result) {
    if (result)
        *result = VERNON_AD_INVALID_REGION_HANDLE;
    if (lane >= impl_->lanes.size() || !result || impl_->compacted)
        return VERNON_AD_TAPE_ALLOCATOR_INVALID_STATE;
    auto &state = impl_->lanes[lane];
    if (state.status != VERNON_AD_TAPE_ALLOCATOR_OK)
        return state.status;
    const VernonAdRegionHandle resolvedParent = parent == VernonAdRegionHandle{1} && state.root ? state.root : parent;
    if (state.sealed || resolvedParent != state.current)
        return impl_->fail(lane, VERNON_AD_TAPE_ALLOCATOR_INVALID_STATE);
    size_t required = 0;
    if (!checkedAdd(state.requiredBytes, kHostTapeRegionResidentBytes, required))
        return impl_->fail(lane, VERNON_AD_TAPE_ALLOCATOR_ARITHMETIC_OVERFLOW);
    state.requiredBytes = required;
    if (required > impl_->invocationCapacity)
        return impl_->fail(lane, VERNON_AD_TAPE_ALLOCATOR_CAPACITY_EXHAUSTED);
    size_t index = 0;
    if (!impl_->regions.allocate(1, index) || index == std::numeric_limits<VernonAdRegionHandle>::max()) {
        return impl_->fail(lane, VERNON_AD_TAPE_ALLOCATOR_HOST_ALLOCATION_FAILURE);
    }
    const auto handle = static_cast<VernonAdRegionHandle>(index + 1);
    Impl::Region *region = impl_->regions.at(index);
    if (!region)
        return impl_->fail(lane, VERNON_AD_TAPE_ALLOCATOR_HOST_ALLOCATION_FAILURE);
    *region = {handle, resolvedParent, lane, state.generation};
    state.current = handle;
    if (!state.root)
        state.root = handle;
    *result = handle;
    return VERNON_AD_TAPE_ALLOCATOR_OK;
}

VernonAdTapeAllocatorStatus HostDynamicTapeBatch::reserveRecord(size_t lane, VernonAdRegionHandle regionHandle,
                                                                size_t payloadSize, size_t payloadAlignment,
                                                                size_t childCount, VernonAdRecordHandle *result) {
    if (result)
        *result = VERNON_AD_INVALID_RECORD_HANDLE;
    if (lane >= impl_->lanes.size() || !result || impl_->compacted)
        return VERNON_AD_TAPE_ALLOCATOR_INVALID_STATE;
    auto &state = impl_->lanes[lane];
    const VernonAdRegionHandle resolvedRegion =
        regionHandle == VernonAdRegionHandle{1} && state.root ? state.root : regionHandle;
    if (state.status != VERNON_AD_TAPE_ALLOCATOR_OK || state.sealed || state.current != resolvedRegion ||
        !payloadAlignment || (payloadAlignment & (payloadAlignment - 1)))
        return impl_->fail(lane, VERNON_AD_TAPE_ALLOCATOR_INVALID_STATE);
    auto *region = impl_->findRegion(lane, resolvedRegion);
    if (!region)
        return impl_->fail(lane, VERNON_AD_TAPE_ALLOCATOR_INVALID_STATE);
    size_t childBytes = 0;
    size_t required = 0;
    if (!checkedMultiply(childCount, sizeof(VernonAdRegionHandle), childBytes) ||
        !checkedAdd(state.requiredBytes, kHostTapeRecordResidentBytes, required) ||
        !checkedAdd(required, childBytes, required) || !checkedAdd(required, payloadAlignment - 1, required))
        return impl_->fail(lane, VERNON_AD_TAPE_ALLOCATOR_ARITHMETIC_OVERFLOW);
    required &= ~(payloadAlignment - 1);
    if (!checkedAdd(required, payloadSize, required))
        return impl_->fail(lane, VERNON_AD_TAPE_ALLOCATOR_ARITHMETIC_OVERFLOW);
    state.requiredBytes = required;
    if (required > impl_->invocationCapacity)
        return impl_->fail(lane, VERNON_AD_TAPE_ALLOCATOR_CAPACITY_EXHAUSTED);
    size_t payloadOffset = 0;
    size_t childOffset = 0;
    size_t index = 0;
    if (!impl_->payload.allocate(payloadSize, payloadOffset) || !impl_->children.allocate(childCount, childOffset) ||
        !impl_->records.allocate(1, index) || index == std::numeric_limits<VernonAdRecordHandle>::max()) {
        return impl_->fail(lane, VERNON_AD_TAPE_ALLOCATOR_HOST_ALLOCATION_FAILURE);
    }
    const auto handle = static_cast<VernonAdRecordHandle>(index + 1);
    Impl::Record *newRecord = impl_->records.at(index);
    if (!newRecord)
        return impl_->fail(lane, VERNON_AD_TAPE_ALLOCATOR_HOST_ALLOCATION_FAILURE);
    for (size_t childIndex = 0; childIndex < childCount; ++childIndex) {
        VernonAdRegionHandle *child = impl_->children.at(childOffset + childIndex);
        if (!child)
            return impl_->fail(lane, VERNON_AD_TAPE_ALLOCATOR_HOST_ALLOCATION_FAILURE);
        *child = VERNON_AD_INVALID_REGION_HANDLE;
    }
    *newRecord = {handle,        static_cast<size_t>(resolvedRegion - 1),
                  lane,          state.generation,
                  payloadOffset, payloadSize,
                  childOffset,   childCount};
    region = impl_->findRegion(lane, resolvedRegion);
    if (region->lastRecord != std::numeric_limits<size_t>::max()) {
        Impl::Record *previous = impl_->records.at(region->lastRecord);
        if (!previous)
            return impl_->fail(lane, VERNON_AD_TAPE_ALLOCATOR_HOST_ALLOCATION_FAILURE);
        previous->next = index;
    } else {
        region->firstRecord = index;
    }
    region->lastRecord = index;
    ++region->recordCount;
    state.payloadBytes += payloadSize;
    *result = handle;
    return VERNON_AD_TAPE_ALLOCATOR_OK;
}

VernonAdTapeAllocatorStatus HostDynamicTapeBatch::writeLeaf(size_t lane, VernonAdRecordHandle recordHandle,
                                                            size_t leafOffset, const void *data, size_t byteSize) {
    if (lane >= impl_->lanes.size() || impl_->compacted || (byteSize && !data))
        return VERNON_AD_TAPE_ALLOCATOR_INVALID_STATE;
    auto &state = impl_->lanes[lane];
    auto *record = impl_->findRecord(lane, recordHandle);
    size_t end = 0;
    if (state.status != VERNON_AD_TAPE_ALLOCATOR_OK || state.sealed || !record ||
        state.current != record->regionIndex + 1 || !checkedAdd(leafOffset, byteSize, end) || end > record->payloadSize)
        return impl_->fail(lane, VERNON_AD_TAPE_ALLOCATOR_INVALID_STATE);
    if (byteSize &&
        !impl_->payload.write(record->payloadOffset + leafOffset, static_cast<const std::byte *>(data), byteSize))
        return impl_->fail(lane, VERNON_AD_TAPE_ALLOCATOR_HOST_ALLOCATION_FAILURE);
    return VERNON_AD_TAPE_ALLOCATOR_OK;
}

VernonAdTapeAllocatorStatus HostDynamicTapeBatch::setChild(size_t lane, VernonAdRecordHandle recordHandle,
                                                           size_t ordinal, VernonAdRegionHandle childHandle) {
    if (lane >= impl_->lanes.size() || impl_->compacted)
        return VERNON_AD_TAPE_ALLOCATOR_INVALID_STATE;
    auto &state = impl_->lanes[lane];
    auto *record = impl_->findRecord(lane, recordHandle);
    auto *child = impl_->findRegion(lane, childHandle);
    VernonAdRegionHandle *destination =
        record && ordinal < record->childCount ? impl_->children.at(record->childOffset + ordinal) : nullptr;
    if (state.status != VERNON_AD_TAPE_ALLOCATOR_OK || state.sealed || !record || !child || !destination ||
        *destination || child->parent != record->regionIndex + 1)
        return impl_->fail(lane, VERNON_AD_TAPE_ALLOCATOR_INVALID_STATE);
    *destination = childHandle;
    return VERNON_AD_TAPE_ALLOCATOR_OK;
}

VernonAdTapeAllocatorStatus HostDynamicTapeBatch::endRegion(size_t lane, VernonAdRegionHandle regionHandle,
                                                            size_t executedCount, uint32_t exitKind) {
    if (lane >= impl_->lanes.size() || impl_->compacted)
        return VERNON_AD_TAPE_ALLOCATOR_INVALID_STATE;
    auto &state = impl_->lanes[lane];
    const VernonAdRegionHandle resolvedRegion =
        regionHandle == VernonAdRegionHandle{1} && state.root ? state.root : regionHandle;
    auto *region = impl_->findRegion(lane, resolvedRegion);
    if (state.status != VERNON_AD_TAPE_ALLOCATOR_OK || state.sealed || !region || region->ended ||
        state.current != resolvedRegion)
        return impl_->fail(lane, VERNON_AD_TAPE_ALLOCATOR_INVALID_STATE);
    region->executedCount = executedCount;
    region->exitKind = exitKind;
    region->ended = true;
    state.current = region->parent;
    return VERNON_AD_TAPE_ALLOCATOR_OK;
}

VernonAdTapeAllocatorStatus HostDynamicTapeBatch::seal(size_t lane) {
    if (lane >= impl_->lanes.size() || impl_->compacted)
        return VERNON_AD_TAPE_ALLOCATOR_INVALID_STATE;
    auto &state = impl_->lanes[lane];
    if (state.status != VERNON_AD_TAPE_ALLOCATOR_OK || state.sealed || state.current || !state.root)
        return impl_->fail(lane, VERNON_AD_TAPE_ALLOCATOR_INVALID_STATE);
    state.sealed = true;
    return VERNON_AD_TAPE_ALLOCATOR_OK;
}

VernonAdTapeAllocatorStatus HostDynamicTapeBatch::status(size_t lane) const {
    return lane < impl_->lanes.size() ? impl_->lanes[lane].status : VERNON_AD_TAPE_ALLOCATOR_INVALID_STATE;
}

size_t HostDynamicTapeBatch::requiredBytes(size_t lane) const {
    return lane < impl_->lanes.size() ? impl_->lanes[lane].requiredBytes : 0;
}

bool HostDynamicTapeBatch::compact(bool retainConstructionStorage) {
    std::lock_guard lock(impl_->mutex);
    if (impl_->compacted)
        return true;
    size_t pendingCompactCharge = 0;
    struct CompactRollback {
        Impl *impl;
        size_t &pendingCharge;
        bool committed{};
        ~CompactRollback() {
            if (committed)
                return;
            releaseVectorStorage(impl->snapshotChildren);
            releaseVectorStorage(impl->snapshotRegions);
            releaseVectorStorage(impl->snapshotRecords);
            releaseVectorStorage(impl->laneRoots);
            impl->releasePhysical(pendingCharge);
        }
    } rollback{impl_.get(), pendingCompactCharge};
    try {
        bool hasDynamicLane = false;
        for (size_t lane = 0; lane < impl_->lanes.size(); ++lane) {
            const auto &state = impl_->lanes[lane];
            if (!state.sealed || state.status != VERNON_AD_TAPE_ALLOCATOR_OK)
                continue;
            hasDynamicLane = true;
            impl_->compactedLogicalBytes += state.payloadBytes;
        }
        if (!hasDynamicLane) {
            if (!retainConstructionStorage) {
                impl_->releasePhysical(impl_->lanes.capacity() * sizeof(Impl::Lane));
                releaseVectorStorage(impl_->lanes);
                impl_->releaseArena(impl_->payload);
                impl_->releaseArena(impl_->children);
                impl_->releaseArena(impl_->regions);
                impl_->releaseArena(impl_->records);
            }
            releaseVectorStorage(impl_->snapshotRegions);
            impl_->compacted = true;
            return true;
        }
        size_t regionCount = 0;
        size_t recordCount = 0;
        size_t childCount = 0;
        for (size_t sourceRegionIndex = 0; sourceRegionIndex < impl_->regions.size(); ++sourceRegionIndex) {
            const Impl::Region *sourceRegion = impl_->regions.at(sourceRegionIndex);
            if (!sourceRegion || sourceRegion->lane >= impl_->lanes.size() ||
                impl_->lanes[sourceRegion->lane].generation != sourceRegion->generation ||
                !impl_->lanes[sourceRegion->lane].sealed)
                continue;
            if (!checkedAdd(regionCount, 1, regionCount) ||
                !checkedAdd(recordCount, sourceRegion->recordCount, recordCount))
                return false;
            size_t recordIndex = sourceRegion->firstRecord;
            for (size_t ordinal = 0; ordinal < sourceRegion->recordCount; ++ordinal) {
                if (recordIndex >= impl_->records.size())
                    return false;
                const Impl::Record *sourceRecord = impl_->records.at(recordIndex);
                if (!sourceRecord || !checkedAdd(childCount, sourceRecord->childCount, childCount))
                    return false;
                recordIndex = sourceRecord->next;
            }
            if (recordIndex != std::numeric_limits<size_t>::max())
                return false;
        }
        size_t compactReservation = 0;
        size_t bytes = 0;
        if (!checkedMultiply(regionCount, sizeof(Impl::SnapshotRegion), compactReservation) ||
            !checkedMultiply(recordCount, sizeof(Impl::SnapshotRecord), bytes) ||
            !checkedAdd(compactReservation, bytes, compactReservation) ||
            !checkedMultiply(childCount, sizeof(VernonAdRegionHandle), bytes) ||
            !checkedAdd(compactReservation, bytes, compactReservation) ||
            !checkedMultiply(impl_->lanes.size(), sizeof(VernonAdRegionHandle), bytes) ||
            !checkedAdd(compactReservation, bytes, compactReservation) ||
            !checkedMultiply(impl_->regions.size(), sizeof(VernonAdRegionHandle), bytes) ||
            !checkedAdd(compactReservation, bytes, compactReservation) || !impl_->reservePhysical(compactReservation))
            return false;
        pendingCompactCharge = compactReservation;
        impl_->laneRoots.resize(impl_->lanes.size(), VERNON_AD_INVALID_REGION_HANDLE);
        impl_->snapshotRegions.reserve(regionCount);
        impl_->snapshotRecords.reserve(recordCount);
        impl_->snapshotChildren.reserve(childCount);
        std::vector<VernonAdRegionHandle> compactHandles(impl_->regions.size(), VERNON_AD_INVALID_REGION_HANDLE);
        for (size_t sourceRegionIndex = 0; sourceRegionIndex < impl_->regions.size(); ++sourceRegionIndex) {
            const Impl::Region *sourceRegion = impl_->regions.at(sourceRegionIndex);
            if (!sourceRegion || sourceRegion->lane >= impl_->lanes.size() ||
                impl_->lanes[sourceRegion->lane].generation != sourceRegion->generation ||
                !impl_->lanes[sourceRegion->lane].sealed)
                continue;
            if (sourceRegion->lane >= std::numeric_limits<uint32_t>::max() ||
                impl_->snapshotRegions.size() >= std::numeric_limits<uint32_t>::max())
                return false;
            const uint64_t encodedLane = static_cast<uint64_t>(sourceRegion->lane + 1) << 32;
            const uint64_t encodedIndex = static_cast<uint64_t>(impl_->snapshotRegions.size() + 1);
            compactHandles[sourceRegionIndex] = encodedLane | encodedIndex;
            impl_->snapshotRegions.emplace_back();
        }
        for (size_t sourceRegionIndex = 0; sourceRegionIndex < impl_->regions.size(); ++sourceRegionIndex) {
            const VernonAdRegionHandle compactHandle = compactHandles[sourceRegionIndex];
            if (!compactHandle)
                continue;
            const Impl::Region *sourceRegion = impl_->regions.at(sourceRegionIndex);
            if (!sourceRegion)
                return false;
            const size_t destinationRegionIndex =
                static_cast<size_t>((compactHandle & std::numeric_limits<uint32_t>::max()) - 1);
            auto &destinationRegion = impl_->snapshotRegions[destinationRegionIndex];
            destinationRegion.recordOffset = impl_->snapshotRecords.size();
            destinationRegion.recordCount = sourceRegion->recordCount;
            destinationRegion.executedCount = sourceRegion->executedCount;
            destinationRegion.exitKind = sourceRegion->exitKind;
            size_t recordIndex = sourceRegion->firstRecord;
            for (size_t ordinal = 0; ordinal < sourceRegion->recordCount; ++ordinal) {
                if (recordIndex >= impl_->records.size())
                    return false;
                const Impl::Record *sourceRecord = impl_->records.at(recordIndex);
                if (!sourceRecord)
                    return false;
                const size_t childOffset = impl_->snapshotChildren.size();
                impl_->snapshotChildren.resize(childOffset + sourceRecord->childCount);
                if (!impl_->children.read(sourceRecord->childOffset, impl_->snapshotChildren.data() + childOffset,
                                          sourceRecord->childCount))
                    return false;
                impl_->snapshotRecords.push_back(
                    {sourceRecord->payloadOffset, sourceRecord->payloadSize, childOffset, sourceRecord->childCount});
                recordIndex = sourceRecord->next;
            }
            if (recordIndex != std::numeric_limits<size_t>::max())
                return false;
        }
        for (VernonAdRegionHandle &child : impl_->snapshotChildren) {
            if (!child || child > compactHandles.size() || !compactHandles[static_cast<size_t>(child - 1)])
                return false;
            child = compactHandles[static_cast<size_t>(child - 1)];
        }
        for (size_t lane = 0; lane < impl_->lanes.size(); ++lane) {
            const auto &state = impl_->lanes[lane];
            if (!state.sealed || state.status != VERNON_AD_TAPE_ALLOCATOR_OK)
                continue;
            if (!state.root || state.root > compactHandles.size() ||
                !compactHandles[static_cast<size_t>(state.root - 1)])
                return false;
            impl_->laneRoots[lane] = compactHandles[static_cast<size_t>(state.root - 1)];
        }
        size_t compactCharge = 0;
        bytes = 0;
        if (!checkedMultiply(impl_->snapshotChildren.capacity(), sizeof(VernonAdRegionHandle), bytes) ||
            !checkedAdd(compactCharge, bytes, compactCharge) ||
            !checkedMultiply(impl_->snapshotRegions.capacity(), sizeof(Impl::SnapshotRegion), bytes) ||
            !checkedAdd(compactCharge, bytes, compactCharge) ||
            !checkedMultiply(impl_->snapshotRecords.capacity(), sizeof(Impl::SnapshotRecord), bytes) ||
            !checkedAdd(compactCharge, bytes, compactCharge) ||
            !checkedMultiply(impl_->laneRoots.capacity(), sizeof(VernonAdRegionHandle), bytes) ||
            !checkedAdd(compactCharge, bytes, compactCharge) ||
            !checkedMultiply(compactHandles.capacity(), sizeof(VernonAdRegionHandle), bytes) ||
            !checkedAdd(compactCharge, bytes, compactCharge))
            return false;
        if (compactCharge > compactReservation) {
            if (!impl_->reservePhysical(compactCharge - compactReservation))
                return false;
        } else {
            impl_->releasePhysical(compactReservation - compactCharge);
        }
        pendingCompactCharge = compactCharge;
        const size_t handleBytes = compactHandles.capacity() * sizeof(VernonAdRegionHandle);
        releaseVectorStorage(compactHandles);
        impl_->releasePhysical(handleBytes);
        pendingCompactCharge -= handleBytes;
        if (!retainConstructionStorage) {
            impl_->releasePhysical(impl_->lanes.capacity() * sizeof(Impl::Lane));
            releaseVectorStorage(impl_->lanes);
            impl_->releaseArena(impl_->children);
            impl_->releaseArena(impl_->regions);
            impl_->releaseArena(impl_->records);
        }
        impl_->compacted = true;
        rollback.committed = true;
        return true;
    } catch (const std::exception &) {
        return false;
    }
}

bool HostDynamicTapeBatch::resetCompactedReplay() {
    std::lock_guard lock(impl_->mutex);
    if (!impl_->compacted || impl_->lanes.empty())
        return false;
    size_t snapshotBytes = impl_->snapshotChildren.capacity() * sizeof(VernonAdRegionHandle) +
                           impl_->snapshotRegions.capacity() * sizeof(Impl::SnapshotRegion) +
                           impl_->snapshotRecords.capacity() * sizeof(Impl::SnapshotRecord) +
                           impl_->laneRoots.capacity() * sizeof(VernonAdRegionHandle);
    releaseVectorStorage(impl_->snapshotChildren);
    releaseVectorStorage(impl_->snapshotRegions);
    releaseVectorStorage(impl_->snapshotRecords);
    releaseVectorStorage(impl_->laneRoots);
    impl_->releasePhysical(snapshotBytes);
    std::fill(impl_->lanes.begin(), impl_->lanes.end(), Impl::Lane{});
    impl_->payload.rewind();
    impl_->children.rewind();
    impl_->regions.rewind();
    impl_->records.rewind();
    impl_->compactedLogicalBytes = 0;
    impl_->compacted = false;
    return true;
}

size_t HostDynamicTapeBatch::constructionBytes() const {
    return impl_->lanes.capacity() * sizeof(Impl::Lane) + impl_->payload.allocatedBytes() +
           impl_->children.allocatedBytes() + impl_->regions.allocatedBytes() + impl_->records.allocatedBytes();
}

bool HostDynamicTapeBatch::isCompacted() const { return impl_->compacted; }

VernonAdRegionHandle HostDynamicTapeBatch::rootRegion(size_t lane) const {
    if (!impl_->compacted) {
        if (lane >= impl_->lanes.size() || !impl_->lanes[lane].sealed ||
            impl_->lanes[lane].status != VERNON_AD_TAPE_ALLOCATOR_OK)
            return VERNON_AD_INVALID_REGION_HANDLE;
        return impl_->lanes[lane].root;
    }
    return lane < impl_->laneRoots.size() ? impl_->laneRoots[lane] : VERNON_AD_INVALID_REGION_HANDLE;
}

VernonAdTapeAllocatorStatus HostDynamicTapeBatch::readLeaf(size_t lane, VernonAdRegionHandle region, size_t recordIndex,
                                                           size_t leafOffset, void *data, size_t byteSize) const {
    const auto *record = impl_->resolveSnapshotRecord(lane, region, recordIndex);
    size_t end = 0;
    if (!record || (byteSize && !data) || !checkedAdd(leafOffset, byteSize, end) || end > record->payloadSize ||
        record->payloadOffset > impl_->payload.size() ||
        record->payloadSize > impl_->payload.size() - record->payloadOffset)
        return VERNON_AD_TAPE_ALLOCATOR_INVALID_STATE;
#ifdef VERNON_HOST_TAPE_INSTRUMENTATION
    if (activeTraversalMetrics) {
        ++activeTraversalMetrics->regionLookups;
        ++activeTraversalMetrics->recordResolutions;
        ++activeTraversalMetrics->leafReads;
    }
#endif
    if (byteSize && !impl_->payload.read(record->payloadOffset + leafOffset, static_cast<std::byte *>(data), byteSize))
        return VERNON_AD_TAPE_ALLOCATOR_INVALID_STATE;
    return VERNON_AD_TAPE_ALLOCATOR_OK;
}

VernonAdTapeAllocatorStatus HostDynamicTapeBatch::readChild(size_t lane, VernonAdRegionHandle region,
                                                            size_t recordIndex, size_t childOrdinal,
                                                            VernonAdRegionHandle *child) const {
    if (child)
        *child = VERNON_AD_INVALID_REGION_HANDLE;
    const auto *record = impl_->resolveSnapshotRecord(lane, region, recordIndex);
    if (!child || !record || childOrdinal >= record->childCount ||
        record->childOffset > impl_->snapshotChildren.size() ||
        record->childCount > impl_->snapshotChildren.size() - record->childOffset)
        return VERNON_AD_TAPE_ALLOCATOR_INVALID_STATE;
#ifdef VERNON_HOST_TAPE_INSTRUMENTATION
    if (activeTraversalMetrics) {
        ++activeTraversalMetrics->regionLookups;
        ++activeTraversalMetrics->recordResolutions;
        ++activeTraversalMetrics->childReads;
    }
#endif
    const auto value = impl_->snapshotChildren[record->childOffset + childOrdinal];
    if (!impl_->findSnapshotRegion(lane, value))
        return VERNON_AD_TAPE_ALLOCATOR_INVALID_STATE;
    *child = value;
    return VERNON_AD_TAPE_ALLOCATOR_OK;
}

VernonAdTapeAllocatorStatus HostDynamicTapeBatch::readCount(size_t lane, VernonAdRegionHandle region,
                                                            size_t *count) const {
    if (count)
        *count = 0;
    const auto *value = impl_->findSnapshotRegion(lane, region);
    if (!count || !value)
        return VERNON_AD_TAPE_ALLOCATOR_INVALID_STATE;
#ifdef VERNON_HOST_TAPE_INSTRUMENTATION
    if (activeTraversalMetrics) {
        ++activeTraversalMetrics->regionLookups;
        ++activeTraversalMetrics->executedCountReads;
    }
#endif
    *count = value->executedCount;
    return VERNON_AD_TAPE_ALLOCATOR_OK;
}

VernonAdTapeAllocatorStatus HostDynamicTapeBatch::readExit(size_t lane, VernonAdRegionHandle region,
                                                           uint32_t *exitKind) const {
    if (exitKind)
        *exitKind = 0;
    const auto *value = impl_->findSnapshotRegion(lane, region);
    if (!exitKind || !value)
        return VERNON_AD_TAPE_ALLOCATOR_INVALID_STATE;
#ifdef VERNON_HOST_TAPE_INSTRUMENTATION
    if (activeTraversalMetrics) {
        ++activeTraversalMetrics->regionLookups;
        ++activeTraversalMetrics->exitKindReads;
    }
#endif
    *exitKind = value->exitKind;
    return VERNON_AD_TAPE_ALLOCATOR_OK;
}

size_t HostDynamicTapeBatch::logicalBytes() const {
    if (impl_->compacted)
        return impl_->compactedLogicalBytes;
    size_t total = 0;
    for (const auto &lane : impl_->lanes)
        total += lane.payloadBytes;
    return total;
}

size_t HostDynamicTapeBatch::residentBytes() const { return impl_->totalCharge.load(std::memory_order_relaxed); }

size_t HostDynamicTapeBatch::allocatedBytes() const { return residentBytes(); }

HostStaticTapeBatch::Reader *HostStaticTapeBatch::Reader::owner(VernonAdTapeAllocator *allocator) {
    if (!allocator || allocator->struct_size != sizeof(VernonAdTapeAllocator) ||
        allocator->abi_version != VERNON_AD_TAPE_ALLOCATOR_ABI_VERSION || !allocator->user_data) {
        if (allocator)
            allocator->status = VERNON_AD_TAPE_ALLOCATOR_INVALID_ABI;
        return nullptr;
    }
    auto *reader = static_cast<Reader *>(allocator->user_data);
    if (&reader->descriptor_ != allocator || !reader->batch_ || !reader->batch_->compacted_ ||
        reader->lane_ >= reader->batch_->laneCount_) {
        allocator->status = VERNON_AD_TAPE_ALLOCATOR_INVALID_STATE;
        return nullptr;
    }
    return reader;
}

VernonAdTapeAllocatorStatus HostStaticTapeBatch::Reader::reject(VernonAdTapeAllocator *allocator) {
    if (owner(allocator))
        allocator->status = VERNON_AD_TAPE_ALLOCATOR_INVALID_STATE;
    return allocator ? allocator->status : VERNON_AD_TAPE_ALLOCATOR_INVALID_ABI;
}

VernonAdTapeAllocatorStatus HostStaticTapeBatch::Reader::rejectBegin(VernonAdTapeAllocator *allocator,
                                                                     VernonAdRegionHandle,
                                                                     VernonAdRegionHandle *region) {
    if (region)
        *region = VERNON_AD_INVALID_REGION_HANDLE;
    return reject(allocator);
}

VernonAdTapeAllocatorStatus HostStaticTapeBatch::Reader::rejectReserve(VernonAdTapeAllocator *allocator,
                                                                       VernonAdRegionHandle, size_t, size_t, size_t,
                                                                       VernonAdRecordHandle *record) {
    if (record)
        *record = VERNON_AD_INVALID_RECORD_HANDLE;
    return reject(allocator);
}

VernonAdTapeAllocatorStatus HostStaticTapeBatch::Reader::rejectWrite(VernonAdTapeAllocator *allocator,
                                                                     VernonAdRecordHandle, size_t, const void *,
                                                                     size_t) {
    return reject(allocator);
}

VernonAdTapeAllocatorStatus HostStaticTapeBatch::Reader::rejectChild(VernonAdTapeAllocator *allocator,
                                                                     VernonAdRecordHandle, size_t,
                                                                     VernonAdRegionHandle) {
    return reject(allocator);
}

VernonAdTapeAllocatorStatus HostStaticTapeBatch::Reader::rejectEnd(VernonAdTapeAllocator *allocator,
                                                                   VernonAdRegionHandle, size_t, uint32_t) {
    return reject(allocator);
}

VernonAdTapeAllocatorStatus HostStaticTapeBatch::Reader::readLeaf(VernonAdTapeAllocator *allocator,
                                                                  VernonAdRegionHandle region, size_t recordIndex,
                                                                  size_t leafOffset, void *data, size_t byteSize) {
    Reader *reader = owner(allocator);
    size_t end = 0;
    if (!reader)
        return allocator ? allocator->status : VERNON_AD_TAPE_ALLOCATOR_INVALID_ABI;
    if (reader->lane_ < reader->batch_->compactedLaneKinds_.size() &&
        reader->batch_->compactedLaneKinds_[reader->lane_]) {
        const auto status =
            reader->batch_->dynamicBatch()->readLeaf(reader->lane_, region, recordIndex, leafOffset, data, byteSize);
        allocator->status = status;
        return status;
    }
    if (region != 1 || recordIndex != 0 || (byteSize && !data) || !checkedAdd(leafOffset, byteSize, end) ||
        end > reader->batch_->payloadStride_) {
        allocator->status = VERNON_AD_TAPE_ALLOCATOR_INVALID_STATE;
        return allocator->status;
    }
    if (byteSize)
        std::memcpy(data, reader->batch_->payload(reader->lane_) + leafOffset, byteSize);
    return VERNON_AD_TAPE_ALLOCATOR_OK;
}

VernonAdTapeAllocatorStatus HostStaticTapeBatch::Reader::readChild(VernonAdTapeAllocator *allocator,
                                                                   VernonAdRegionHandle region, size_t recordIndex,
                                                                   size_t childOrdinal, VernonAdRegionHandle *child) {
    if (child)
        *child = VERNON_AD_INVALID_REGION_HANDLE;
    Reader *reader = owner(allocator);
    if (reader && reader->lane_ < reader->batch_->compactedLaneKinds_.size() &&
        reader->batch_->compactedLaneKinds_[reader->lane_]) {
        const auto status =
            reader->batch_->dynamicBatch()->readChild(reader->lane_, region, recordIndex, childOrdinal, child);
        allocator->status = status;
        return status;
    }
    return reject(allocator);
}

VernonAdTapeAllocatorStatus HostStaticTapeBatch::Reader::readCount(VernonAdTapeAllocator *allocator,
                                                                   VernonAdRegionHandle region, size_t *count) {
    Reader *reader = owner(allocator);
    if (!reader)
        return allocator ? allocator->status : VERNON_AD_TAPE_ALLOCATOR_INVALID_ABI;
    if (reader->lane_ < reader->batch_->compactedLaneKinds_.size() &&
        reader->batch_->compactedLaneKinds_[reader->lane_]) {
        const auto status = reader->batch_->dynamicBatch()->readCount(reader->lane_, region, count);
        allocator->status = status;
        return status;
    }
    if (region != 1 || !count) {
        allocator->status = VERNON_AD_TAPE_ALLOCATOR_INVALID_STATE;
        return allocator->status;
    }
    *count = 1;
    return VERNON_AD_TAPE_ALLOCATOR_OK;
}

VernonAdTapeAllocatorStatus HostStaticTapeBatch::Reader::readExit(VernonAdTapeAllocator *allocator,
                                                                  VernonAdRegionHandle region, uint32_t *exitKind) {
    Reader *reader = owner(allocator);
    if (!reader)
        return allocator ? allocator->status : VERNON_AD_TAPE_ALLOCATOR_INVALID_ABI;
    if (reader->lane_ < reader->batch_->compactedLaneKinds_.size() &&
        reader->batch_->compactedLaneKinds_[reader->lane_]) {
        const auto status = reader->batch_->dynamicBatch()->readExit(reader->lane_, region, exitKind);
        allocator->status = status;
        return status;
    }
    if (region != 1 || !exitKind) {
        allocator->status = VERNON_AD_TAPE_ALLOCATOR_INVALID_STATE;
        return allocator->status;
    }
    *exitKind = 0;
    return VERNON_AD_TAPE_ALLOCATOR_OK;
}

std::shared_ptr<HostStaticTapeBatch>
HostStaticTapeBatch::create(size_t laneCount, size_t payloadStride, size_t invocationCapacity,
                            std::shared_ptr<HostTapeMemoryPolicy> policy,
                            std::shared_ptr<HostTapeDispatchBudget> dispatchBudget) {
    if (!laneCount || !payloadStride || !policy)
        return {};
    try {
        return std::shared_ptr<HostStaticTapeBatch>(new HostStaticTapeBatch(
            laneCount, payloadStride, invocationCapacity, std::move(policy), std::move(dispatchBudget)));
    } catch (const std::exception &) {
        return {};
    }
}

HostStaticTapeBatch::HostStaticTapeBatch(size_t laneCount, size_t payloadStride, size_t invocationCapacity,
                                         std::shared_ptr<HostTapeMemoryPolicy> policy,
                                         std::shared_ptr<HostTapeDispatchBudget> dispatchBudget)
    : descriptors_(laneCount), lanes_(laneCount), laneCount_(laneCount), payloadStride_(payloadStride),
      invocationCapacity_(invocationCapacity), policy_(std::move(policy)), dispatchBudget_(std::move(dispatchBudget)) {
    size_t payloadBytes = 0;
    if (!checkedMultiply(laneCount, payloadStride, payloadBytes))
        throw std::length_error("static tape batch payload size overflows");
    payload_.resize(payloadBytes);
    policyCharge_ = descriptors_.capacity() * sizeof(VernonAdTapeAllocator) + lanes_.capacity() * sizeof(LaneState) +
                    payload_.capacity() * sizeof(std::byte);
    if (!(dispatchBudget_ ? dispatchBudget_->reserveTransientBytes(policyCharge_)
                          : policy_->reserveContext(policyCharge_)))
        throw std::runtime_error("static tape batch exceeds context budget");

    initializeDescriptors();
}

void HostStaticTapeBatch::initializeDescriptors() {
    for (VernonAdTapeAllocator &descriptor : descriptors_) {
        descriptor = {};
        descriptor.struct_size = sizeof(descriptor);
        descriptor.abi_version = VERNON_AD_TAPE_ALLOCATOR_ABI_VERSION;
        descriptor.status = VERNON_AD_TAPE_ALLOCATOR_OK;
        descriptor.user_data = this;
        descriptor.capacity_bytes = invocationCapacity_;
        descriptor.reset = reset;
        descriptor.begin_region = beginRegion;
        descriptor.reserve_record = reserveRecord;
        descriptor.write_leaf = writeLeaf;
        descriptor.set_child = setChild;
        descriptor.end_region = endRegion;
        descriptor.seal = seal;
        descriptor.read_leaf = readLeaf;
        descriptor.read_child = readChild;
        descriptor.read_executed_count = readCount;
        descriptor.read_exit_kind = readExit;
    }
}

HostStaticTapeBatch::~HostStaticTapeBatch() {
    constructionState_ = ConstructionState::Released;
    if (dispatchBudget_ && policyCharge_)
        dispatchBudget_->releaseTransientBytes(policyCharge_);
    else if (policy_ && policyCharge_)
        policy_->release(policyCharge_);
}

bool hostStaticTapeBatchPureStaticBytes(size_t laneCount, size_t payloadStride, size_t &result) {
    const size_t fixedBytes = sizeof(VernonAdTapeAllocator) + sizeof(HostStaticTapeBatch::LaneState);
    if (payloadStride > SIZE_MAX - fixedBytes || laneCount > SIZE_MAX / (fixedBytes + payloadStride))
        return false;
    result = laneCount * (fixedBytes + payloadStride);
    return true;
}

size_t HostStaticTapeBatch::allocatedBytes() const {
    const HostDynamicTapeBatch *dynamic = dynamicBatch();
    return descriptors_.capacity() * sizeof(VernonAdTapeAllocator) + lanes_.capacity() * sizeof(LaneState) +
           payload_.capacity() * sizeof(std::byte) + compactedLaneKinds_.capacity() * sizeof(uint8_t) +
           (dynamic ? dynamic->allocatedBytes() : 0);
}

bool HostStaticTapeBatch::compact(bool retainConstructionStorage) {
    if (compacted_)
        return true;
    if (constructionState_ != ConstructionState::Constructing)
        return false;
    if (lanes_.size() != laneCount_ || std::any_of(lanes_.begin(), lanes_.end(), [](const LaneState &lane) {
            return lane.phase != LanePhase::Sealed && lane.phase != LanePhase::Promoted;
        }))
        return false;
    compactedLogicalBytes_ = logicalBytes();
    bool hasStaticLane = false;
    const bool hasDynamicLane = std::any_of(lanes_.begin(), lanes_.end(),
                                            [](const LaneState &lane) { return lane.phase == LanePhase::Promoted; });
    if (hasDynamicLane && (!dynamicBatch() || !dynamicBatch()->compact(retainConstructionStorage)))
        return false;
    if (hasDynamicLane) {
        const size_t reservation = laneCount_ * sizeof(uint8_t);
        if (!(dispatchBudget_ ? dispatchBudget_->reserveTransientBytes(reservation)
                              : policy_->reserveContext(reservation)))
            return false;
        try {
            compactedLaneKinds_.resize(laneCount_);
        } catch (...) {
            if (dispatchBudget_)
                dispatchBudget_->releaseTransientBytes(reservation);
            else
                policy_->release(reservation);
            throw;
        }
        const size_t actual = compactedLaneKinds_.capacity() * sizeof(uint8_t);
        if (actual > reservation && !(dispatchBudget_ ? dispatchBudget_->reserveTransientBytes(actual - reservation)
                                                      : policy_->reserveContext(actual - reservation))) {
            releaseVectorStorage(compactedLaneKinds_);
            if (dispatchBudget_)
                dispatchBudget_->releaseTransientBytes(reservation);
            else
                policy_->release(reservation);
            return false;
        }
        if (actual < reservation) {
            if (dispatchBudget_)
                dispatchBudget_->releaseTransientBytes(reservation - actual);
            else
                policy_->release(reservation - actual);
        }
        policyCharge_ += actual;
    }
    for (size_t lane = 0; lane < laneCount_; ++lane) {
        if (hasDynamicLane)
            compactedLaneKinds_[lane] = lanes_[lane].phase == LanePhase::Promoted;
        hasStaticLane |= lanes_[lane].phase == LanePhase::Sealed;
    }
    const size_t previousCharge = policyCharge_;
    if (!retainConstructionStorage) {
        std::vector<VernonAdTapeAllocator>().swap(descriptors_);
        std::vector<LaneState>().swap(lanes_);
    }
    if (!hasStaticLane && !retainConstructionStorage)
        releaseVectorStorage(payload_);
    compacted_ = true;
    retainsConstructionStorage_ = retainConstructionStorage;
    constructionState_ = ConstructionState::FrozenReader;
    if (!retainConstructionStorage)
        policyCharge_ = payload_.capacity() * sizeof(std::byte) + compactedLaneKinds_.capacity() * sizeof(uint8_t);
    const size_t released = previousCharge - policyCharge_;
    if (dispatchBudget_)
        dispatchBudget_->releaseTransientBytes(released);
    else
        policy_->release(released);
    return true;
}

bool HostStaticTapeBatch::markConstructionRecyclable() {
    if (!compacted_ || !retainsConstructionStorage_ || constructionState_ != ConstructionState::FrozenReader)
        return false;
    constructionState_ = ConstructionState::Recyclable;
    return true;
}

bool HostStaticTapeBatch::resetRecyclableConstruction() {
    if (!compacted_ || !retainsConstructionStorage_ || constructionState_ != ConstructionState::Recyclable ||
        descriptors_.size() != laneCount_ || lanes_.size() != laneCount_ ||
        (dynamicBatch() && !dynamicBatch()->resetCompactedReplay()))
        return false;
    std::fill(lanes_.begin(), lanes_.end(), LaneState{});
    const size_t laneKindBytes = compactedLaneKinds_.capacity() * sizeof(uint8_t);
    releaseVectorStorage(compactedLaneKinds_);
    if (laneKindBytes) {
        policyCharge_ -= std::min(policyCharge_, laneKindBytes);
        if (dispatchBudget_)
            dispatchBudget_->releaseTransientBytes(laneKindBytes);
        else
            policy_->release(laneKindBytes);
    }
    if (dispatchBudget_ && !dispatchBudget_->beginRecycledConstruction())
        return false;
    initializeDescriptors();
    compactedLogicalBytes_ = 0;
    compacted_ = false;
    retainsConstructionStorage_ = false;
    constructionState_ = ConstructionState::Constructing;
    return true;
}

size_t HostStaticTapeBatch::constructionBytes() const {
    size_t bytes = descriptors_.capacity() * sizeof(VernonAdTapeAllocator) + lanes_.capacity() * sizeof(LaneState) +
                   payload_.capacity() * sizeof(std::byte);
    const HostDynamicTapeBatch *dynamic = dynamicBatch();
    if (dynamic && dynamic->constructionBytes() <= std::numeric_limits<size_t>::max() - bytes)
        bytes += dynamic->constructionBytes();
    else if (dynamic)
        return std::numeric_limits<size_t>::max();
    return bytes;
}

VernonAdRegionHandle HostStaticTapeBatch::Reader::rootRegion() const {
    if (!batch_ || lane_ >= laneCount_)
        return VERNON_AD_INVALID_REGION_HANDLE;
    if (lane_ < batch_->compactedLaneKinds_.size() && batch_->compactedLaneKinds_[lane_])
        return batch_->dynamicBatch()->rootRegion(lane_);
    return VernonAdRegionHandle{1};
}

bool HostStaticTapeBatch::initializeReader(size_t lane, Reader &reader) const {
    if (!compacted_ || lane >= laneCount_)
        return false;
    reader = {};
    reader.batch_ = this;
    reader.lane_ = lane;
    reader.laneCount_ = laneCount_;
    VernonAdTapeAllocator &descriptor = reader.descriptor_;
    descriptor.struct_size = sizeof(descriptor);
    descriptor.abi_version = VERNON_AD_TAPE_ALLOCATOR_ABI_VERSION;
    descriptor.status = VERNON_AD_TAPE_ALLOCATOR_OK;
    descriptor.user_data = &reader;
    descriptor.capacity_bytes = payloadStride_;
    descriptor.required_bytes = payloadStride_;
    descriptor.reset = Reader::reject;
    descriptor.begin_region = Reader::rejectBegin;
    descriptor.reserve_record = Reader::rejectReserve;
    descriptor.write_leaf = Reader::rejectWrite;
    descriptor.set_child = Reader::rejectChild;
    descriptor.end_region = Reader::rejectEnd;
    descriptor.seal = Reader::reject;
    descriptor.read_leaf = Reader::readLeaf;
    descriptor.read_child = Reader::readChild;
    descriptor.read_executed_count = Reader::readCount;
    descriptor.read_exit_kind = Reader::readExit;
    return true;
}

VernonAdTapeAllocator *HostStaticTapeBatch::descriptor(size_t lane) {
    return lane < descriptors_.size() ? &descriptors_[lane] : nullptr;
}

VernonAdRegionHandle HostStaticTapeBatch::rootRegion(size_t lane) const {
    if (compacted_) {
        if (lane >= compactedLaneKinds_.size())
            return VERNON_AD_INVALID_REGION_HANDLE;
        return compactedLaneKinds_[lane] ? dynamicBatch()->rootRegion(lane) : VernonAdRegionHandle{1};
    }
    if (lane >= lanes_.size())
        return VERNON_AD_INVALID_REGION_HANDLE;
    if (lanes_[lane].phase == LanePhase::Promoted)
        return dynamicBatch()->rootRegion(lane);
    return lanes_[lane].phase == LanePhase::Sealed ? VernonAdRegionHandle{1} : VERNON_AD_INVALID_REGION_HANDLE;
}

size_t HostStaticTapeBatch::logicalBytes() const {
    if (compacted_)
        return compactedLogicalBytes_;
    size_t bytes = 0;
    for (const LaneState &lane : lanes_)
        if (lane.phase != LanePhase::Promoted)
            bytes += lane.payloadSize;
    const HostDynamicTapeBatch *dynamic = dynamicBatch();
    return bytes + (dynamic ? dynamic->logicalBytes() : 0);
}

size_t HostStaticTapeBatch::residentBytes() const {
    const HostDynamicTapeBatch *dynamic = dynamicBatch();
    const size_t dynamicBytes = dynamic ? dynamic->residentBytes() : 0;
    return policyCharge_ + dynamicBytes;
}

std::pair<HostStaticTapeBatch *, size_t> HostStaticTapeBatch::owner(VernonAdTapeAllocator *allocator) {
    if (!allocator)
        return {};
    if (allocator->struct_size != sizeof(VernonAdTapeAllocator) ||
        allocator->abi_version != VERNON_AD_TAPE_ALLOCATOR_ABI_VERSION) {
        allocator->status = VERNON_AD_TAPE_ALLOCATOR_INVALID_ABI;
        return {};
    }
    auto *self = static_cast<HostStaticTapeBatch *>(allocator->user_data);
    if (!self || self->compacted_ || self->descriptors_.empty()) {
        allocator->status = VERNON_AD_TAPE_ALLOCATOR_INVALID_STATE;
        return {};
    }
    const auto address = reinterpret_cast<uintptr_t>(allocator);
    const auto begin = reinterpret_cast<uintptr_t>(self->descriptors_.data());
    size_t descriptorBytes = 0;
    if (!checkedMultiply(self->descriptors_.size(), sizeof(VernonAdTapeAllocator), descriptorBytes) ||
        address < begin || address - begin >= descriptorBytes ||
        (address - begin) % sizeof(VernonAdTapeAllocator) != 0) {
        allocator->status = VERNON_AD_TAPE_ALLOCATOR_INVALID_STATE;
        return {};
    }
    return {self, (address - begin) / sizeof(VernonAdTapeAllocator)};
}

VernonAdTapeAllocatorStatus HostStaticTapeBatch::fail(size_t lane, VernonAdTapeAllocatorStatus status) {
    VernonAdTapeAllocator &descriptor = descriptors_[lane];
    if (descriptor.status == VERNON_AD_TAPE_ALLOCATOR_OK ||
        (descriptor.status == VERNON_AD_TAPE_ALLOCATOR_CAPACITY_EXHAUSTED &&
         status == VERNON_AD_TAPE_ALLOCATOR_ARITHMETIC_OVERFLOW))
        descriptor.status = status;
    return descriptor.status;
}

HostDynamicTapeBatch *HostStaticTapeBatch::ensureDynamicBatch() {
    std::call_once(dynamicBatchOnce_, [this] {
        dynamicBatchOwner_ =
            std::make_unique<HostDynamicTapeBatch>(laneCount_, invocationCapacity_, policy_, dispatchBudget_);
        dynamicBatchAddress_.store(dynamicBatchOwner_.get(), std::memory_order_release);
    });
    return dynamicBatch();
}

VernonAdTapeAllocatorStatus HostStaticTapeBatch::syncDynamic(size_t lane, VernonAdTapeAllocatorStatus status) {
    HostDynamicTapeBatch *dynamic = dynamicBatch();
    descriptors_[lane].status = dynamic->status(lane);
    descriptors_[lane].required_bytes = dynamic->requiredBytes(lane);
    return status;
}

VernonAdTapeAllocatorStatus HostStaticTapeBatch::promote(size_t lane, size_t payloadSize, size_t payloadAlignment,
                                                         size_t childCount, VernonAdRecordHandle *record) {
    VernonAdRegionHandle root = VERNON_AD_INVALID_REGION_HANDLE;
    HostDynamicTapeBatch *dynamic = nullptr;
    try {
        dynamic = ensureDynamicBatch();
    } catch (const std::exception &) {
        return fail(lane, VERNON_AD_TAPE_ALLOCATOR_HOST_ALLOCATION_FAILURE);
    }
    VernonAdTapeAllocatorStatus status = dynamic->beginRegion(lane, VERNON_AD_INVALID_REGION_HANDLE, &root);
    if (status == VERNON_AD_TAPE_ALLOCATOR_OK)
        status = dynamic->reserveRecord(lane, root, payloadSize, payloadAlignment, childCount, record);
    lanes_[lane].phase = LanePhase::Promoted;
    return syncDynamic(lane, status);
}

VernonAdTapeAllocatorStatus HostStaticTapeBatch::reset(VernonAdTapeAllocator *allocator) {
    auto [self, lane] = owner(allocator);
    if (!self)
        return allocator ? allocator->status : VERNON_AD_TAPE_ALLOCATOR_INVALID_ABI;
    if (HostDynamicTapeBatch *dynamic = self->dynamicBatch())
        dynamic->reset(lane);
    self->lanes_[lane] = {};
    allocator->required_bytes = 0;
    allocator->status = VERNON_AD_TAPE_ALLOCATOR_OK;
    return VERNON_AD_TAPE_ALLOCATOR_OK;
}

VernonAdTapeAllocatorStatus HostStaticTapeBatch::beginRegion(VernonAdTapeAllocator *allocator,
                                                             VernonAdRegionHandle parent,
                                                             VernonAdRegionHandle *region) {
    if (region)
        *region = VERNON_AD_INVALID_REGION_HANDLE;
    auto [self, lane] = owner(allocator);
    if (!self)
        return allocator ? allocator->status : VERNON_AD_TAPE_ALLOCATOR_INVALID_ABI;
    if (self->lanes_[lane].phase == LanePhase::Promoted) {
        return self->syncDynamic(lane, self->dynamicBatch()->beginRegion(lane, parent, region));
    }
    if (!region || allocator->status != VERNON_AD_TAPE_ALLOCATOR_OK || self->lanes_[lane].phase != LanePhase::Empty ||
        parent != VERNON_AD_INVALID_REGION_HANDLE)
        return self->fail(lane, VERNON_AD_TAPE_ALLOCATOR_INVALID_STATE);
    allocator->required_bytes = kHostTapeRegionResidentBytes;
    if (allocator->required_bytes > allocator->capacity_bytes)
        return self->fail(lane, VERNON_AD_TAPE_ALLOCATOR_CAPACITY_EXHAUSTED);
    self->lanes_[lane].phase = LanePhase::RegionOpen;
    *region = 1;
    return VERNON_AD_TAPE_ALLOCATOR_OK;
}

VernonAdTapeAllocatorStatus HostStaticTapeBatch::reserveRecord(VernonAdTapeAllocator *allocator,
                                                               VernonAdRegionHandle region, size_t payloadSize,
                                                               size_t payloadAlignment, size_t childCount,
                                                               VernonAdRecordHandle *record) {
    if (record)
        *record = VERNON_AD_INVALID_RECORD_HANDLE;
    auto [self, lane] = owner(allocator);
    if (!self)
        return allocator ? allocator->status : VERNON_AD_TAPE_ALLOCATOR_INVALID_ABI;
    if (self->lanes_[lane].phase == LanePhase::Promoted) {
        return self->syncDynamic(
            lane, self->dynamicBatch()->reserveRecord(lane, region, payloadSize, payloadAlignment, childCount, record));
    }
    if (!record || allocator->status != VERNON_AD_TAPE_ALLOCATOR_OK || region != 1 ||
        self->lanes_[lane].phase != LanePhase::RegionOpen || !payloadAlignment ||
        (payloadAlignment & (payloadAlignment - 1)))
        return self->fail(lane, VERNON_AD_TAPE_ALLOCATOR_INVALID_STATE);
    if (childCount)
        return self->promote(lane, payloadSize, payloadAlignment, childCount, record);
    size_t required = 0;
    if (!checkedAdd(kHostTapeRegionResidentBytes, kHostTapeRecordResidentBytes, required) ||
        !checkedAdd(required, payloadAlignment - 1, required))
        return self->fail(lane, VERNON_AD_TAPE_ALLOCATOR_ARITHMETIC_OVERFLOW);
    required &= ~(payloadAlignment - 1);
    if (!checkedAdd(required, payloadSize, required))
        return self->fail(lane, VERNON_AD_TAPE_ALLOCATOR_ARITHMETIC_OVERFLOW);
    allocator->required_bytes = required;
    if (required > allocator->capacity_bytes || payloadSize > self->payloadStride_)
        return self->fail(lane, VERNON_AD_TAPE_ALLOCATOR_CAPACITY_EXHAUSTED);
    self->lanes_[lane].payloadSize = payloadSize;
    self->lanes_[lane].phase = LanePhase::RecordOpen;
    *record = 1;
    return VERNON_AD_TAPE_ALLOCATOR_OK;
}

VernonAdTapeAllocatorStatus HostStaticTapeBatch::writeLeaf(VernonAdTapeAllocator *allocator,
                                                           VernonAdRecordHandle record, size_t leafOffset,
                                                           const void *data, size_t byteSize) {
    auto [self, lane] = owner(allocator);
    if (!self)
        return allocator ? allocator->status : VERNON_AD_TAPE_ALLOCATOR_INVALID_ABI;
    if (self->lanes_[lane].phase == LanePhase::Promoted) {
        return self->syncDynamic(lane, self->dynamicBatch()->writeLeaf(lane, record, leafOffset, data, byteSize));
    }
    size_t end = 0;
    if (allocator->status != VERNON_AD_TAPE_ALLOCATOR_OK || record != 1 ||
        self->lanes_[lane].phase != LanePhase::RecordOpen || (byteSize && !data) ||
        !checkedAdd(leafOffset, byteSize, end) || end > self->lanes_[lane].payloadSize)
        return self->fail(lane, VERNON_AD_TAPE_ALLOCATOR_INVALID_STATE);
    if (byteSize)
        std::memcpy(self->payload(lane) + leafOffset, data, byteSize);
    return VERNON_AD_TAPE_ALLOCATOR_OK;
}

VernonAdTapeAllocatorStatus HostStaticTapeBatch::setChild(VernonAdTapeAllocator *allocator, VernonAdRecordHandle record,
                                                          size_t childOrdinal, VernonAdRegionHandle child) {
    auto [self, lane] = owner(allocator);
    if (!self)
        return allocator ? allocator->status : VERNON_AD_TAPE_ALLOCATOR_INVALID_ABI;
    if (self->lanes_[lane].phase != LanePhase::Promoted)
        return self->fail(lane, VERNON_AD_TAPE_ALLOCATOR_INVALID_STATE);
    return self->syncDynamic(lane, self->dynamicBatch()->setChild(lane, record, childOrdinal, child));
}

VernonAdTapeAllocatorStatus HostStaticTapeBatch::endRegion(VernonAdTapeAllocator *allocator,
                                                           VernonAdRegionHandle region, size_t executedCount,
                                                           uint32_t exitKind) {
    auto [self, lane] = owner(allocator);
    if (!self)
        return allocator ? allocator->status : VERNON_AD_TAPE_ALLOCATOR_INVALID_ABI;
    if (self->lanes_[lane].phase == LanePhase::Promoted) {
        return self->syncDynamic(lane, self->dynamicBatch()->endRegion(lane, region, executedCount, exitKind));
    }
    if (allocator->status != VERNON_AD_TAPE_ALLOCATOR_OK || region != 1 ||
        self->lanes_[lane].phase != LanePhase::RecordOpen)
        return self->fail(lane, VERNON_AD_TAPE_ALLOCATOR_INVALID_STATE);
    self->lanes_[lane].executedCount = executedCount;
    self->lanes_[lane].exitKind = exitKind;
    self->lanes_[lane].phase = LanePhase::RegionEnded;
    return VERNON_AD_TAPE_ALLOCATOR_OK;
}

VernonAdTapeAllocatorStatus HostStaticTapeBatch::seal(VernonAdTapeAllocator *allocator) {
    auto [self, lane] = owner(allocator);
    if (!self)
        return allocator ? allocator->status : VERNON_AD_TAPE_ALLOCATOR_INVALID_ABI;
    if (self->lanes_[lane].phase == LanePhase::Promoted) {
        return self->syncDynamic(lane, self->dynamicBatch()->seal(lane));
    }
    if (allocator->status != VERNON_AD_TAPE_ALLOCATOR_OK || self->lanes_[lane].phase != LanePhase::RegionEnded)
        return self->fail(lane, VERNON_AD_TAPE_ALLOCATOR_INVALID_STATE);
    self->lanes_[lane].phase = LanePhase::Sealed;
    return VERNON_AD_TAPE_ALLOCATOR_OK;
}

VernonAdTapeAllocatorStatus HostStaticTapeBatch::readLeaf(VernonAdTapeAllocator *allocator, VernonAdRegionHandle region,
                                                          size_t recordIndex, size_t leafOffset, void *data,
                                                          size_t byteSize) {
    auto [self, lane] = owner(allocator);
    if (!self)
        return allocator ? allocator->status : VERNON_AD_TAPE_ALLOCATOR_INVALID_ABI;
    if (self->lanes_[lane].phase == LanePhase::Promoted) {
        return self->syncDynamic(lane,
                                 self->dynamicBatch()->readLeaf(lane, region, recordIndex, leafOffset, data, byteSize));
    }
    size_t end = 0;
    if (allocator->status != VERNON_AD_TAPE_ALLOCATOR_OK || self->lanes_[lane].phase != LanePhase::Sealed ||
        region != 1 || recordIndex != 0 || (byteSize && !data) || !checkedAdd(leafOffset, byteSize, end) ||
        end > self->lanes_[lane].payloadSize)
        return self->fail(lane, VERNON_AD_TAPE_ALLOCATOR_INVALID_STATE);
#ifdef VERNON_HOST_TAPE_INSTRUMENTATION
    if (activeTraversalMetrics) {
        ++activeTraversalMetrics->regionLookups;
        ++activeTraversalMetrics->recordResolutions;
        ++activeTraversalMetrics->leafReads;
    }
#endif
    if (byteSize)
        std::memcpy(data, self->payload(lane) + leafOffset, byteSize);
    return VERNON_AD_TAPE_ALLOCATOR_OK;
}

VernonAdTapeAllocatorStatus HostStaticTapeBatch::readChild(VernonAdTapeAllocator *allocator,
                                                           VernonAdRegionHandle region, size_t recordIndex,
                                                           size_t childOrdinal, VernonAdRegionHandle *child) {
    if (child)
        *child = VERNON_AD_INVALID_REGION_HANDLE;
    auto [self, lane] = owner(allocator);
    if (!self)
        return allocator ? allocator->status : VERNON_AD_TAPE_ALLOCATOR_INVALID_ABI;
    if (self->lanes_[lane].phase != LanePhase::Promoted)
        return self->fail(lane, VERNON_AD_TAPE_ALLOCATOR_INVALID_STATE);
    return self->syncDynamic(lane, self->dynamicBatch()->readChild(lane, region, recordIndex, childOrdinal, child));
}

VernonAdTapeAllocatorStatus HostStaticTapeBatch::readCount(VernonAdTapeAllocator *allocator,
                                                           VernonAdRegionHandle region, size_t *count) {
    if (count)
        *count = 0;
    auto [self, lane] = owner(allocator);
    if (!self)
        return allocator ? allocator->status : VERNON_AD_TAPE_ALLOCATOR_INVALID_ABI;
    if (self->lanes_[lane].phase == LanePhase::Promoted) {
        return self->syncDynamic(lane, self->dynamicBatch()->readCount(lane, region, count));
    }
    if (!count || allocator->status != VERNON_AD_TAPE_ALLOCATOR_OK || self->lanes_[lane].phase != LanePhase::Sealed ||
        region != 1)
        return self->fail(lane, VERNON_AD_TAPE_ALLOCATOR_INVALID_STATE);
    *count = self->lanes_[lane].executedCount;
    return VERNON_AD_TAPE_ALLOCATOR_OK;
}

VernonAdTapeAllocatorStatus HostStaticTapeBatch::readExit(VernonAdTapeAllocator *allocator, VernonAdRegionHandle region,
                                                          uint32_t *exitKind) {
    if (exitKind)
        *exitKind = 0;
    auto [self, lane] = owner(allocator);
    if (!self)
        return allocator ? allocator->status : VERNON_AD_TAPE_ALLOCATOR_INVALID_ABI;
    if (self->lanes_[lane].phase == LanePhase::Promoted) {
        return self->syncDynamic(lane, self->dynamicBatch()->readExit(lane, region, exitKind));
    }
    if (!exitKind || allocator->status != VERNON_AD_TAPE_ALLOCATOR_OK ||
        self->lanes_[lane].phase != LanePhase::Sealed || region != 1)
        return self->fail(lane, VERNON_AD_TAPE_ALLOCATOR_INVALID_STATE);
    *exitKind = self->lanes_[lane].exitKind;
    return VERNON_AD_TAPE_ALLOCATOR_OK;
}

} // namespace vernon::runtime::ad
