#ifndef VERNON_RUNTIME_AUTODIFF_HOST_TAPE_ALLOCATOR_H
#define VERNON_RUNTIME_AUTODIFF_HOST_TAPE_ALLOCATOR_H

#include "runtime/autodiff/tape_allocator_abi.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <type_traits>
#include <vector>

#ifdef VERNON_HOST_TAPE_INSTRUMENTATION
#include "VernonCommon.h"
#include <functional>
#endif

namespace vernon::runtime::ad {

inline constexpr size_t kDefaultHostTapeInvocationLimit = 64u * 1024u * 1024u;
inline constexpr size_t kDefaultHostTapeContextLimit = 256u * 1024u * 1024u;

namespace detail {
struct HostTapeRegion {
    VernonAdRegionHandle handle{};
    VernonAdRegionHandle parent{};
    size_t firstRecordIndex{std::numeric_limits<size_t>::max()};
    size_t lastRecordIndex{std::numeric_limits<size_t>::max()};
    size_t recordOffset{};
    size_t recordCount{};
    size_t executedCount{};
    uint32_t exitKind{};
    bool open{};
    bool ended{};
};

struct HostTapeRecord {
    VernonAdRecordHandle handle{};
    size_t regionIndex{};
    size_t payloadOffset{};
    size_t payloadSize{};
    size_t childOffset{};
    size_t childCount{};
    size_t nextRegionRecordIndex{std::numeric_limits<size_t>::max()};
};

struct HostTapeSnapshotRegion {
    size_t recordOffset{};
    size_t recordCount{};
    size_t executedCount{};
    uint32_t exitKind{};
};

struct HostTapeSnapshotRecord {
    size_t payloadOffset{};
    size_t payloadSize{};
    size_t childOffset{};
    size_t childCount{};
};
} // namespace detail

inline constexpr uint32_t kHostTapePageLayoutVersion = 1;
static_assert(std::is_standard_layout_v<detail::HostTapeSnapshotRegion>);
static_assert(std::is_trivially_copyable_v<detail::HostTapeSnapshotRegion>);
static_assert(offsetof(detail::HostTapeSnapshotRegion, recordOffset) == 0);
static_assert(offsetof(detail::HostTapeSnapshotRegion, recordCount) == sizeof(size_t));
static_assert(offsetof(detail::HostTapeSnapshotRegion, executedCount) == 2 * sizeof(size_t));
static_assert(offsetof(detail::HostTapeSnapshotRegion, exitKind) == 3 * sizeof(size_t));
static_assert(sizeof(detail::HostTapeSnapshotRegion) == 4 * sizeof(size_t));
static_assert(std::is_standard_layout_v<detail::HostTapeSnapshotRecord>);
static_assert(std::is_trivially_copyable_v<detail::HostTapeSnapshotRecord>);
static_assert(offsetof(detail::HostTapeSnapshotRecord, payloadOffset) == 0);
static_assert(offsetof(detail::HostTapeSnapshotRecord, payloadSize) == sizeof(size_t));
static_assert(offsetof(detail::HostTapeSnapshotRecord, childOffset) == 2 * sizeof(size_t));
static_assert(offsetof(detail::HostTapeSnapshotRecord, childCount) == 3 * sizeof(size_t));
static_assert(sizeof(detail::HostTapeSnapshotRecord) == 4 * sizeof(size_t));

inline constexpr size_t kHostTapeRegionResidentBytes = sizeof(detail::HostTapeRegion) + sizeof(size_t);
inline constexpr size_t kHostTapeRecordResidentBytes = sizeof(detail::HostTapeRecord) + sizeof(size_t);
inline constexpr size_t kHostTapeSnapshotRegionResidentBytes = sizeof(detail::HostTapeSnapshotRegion);
inline constexpr size_t kHostTapeSnapshotRecordResidentBytes = sizeof(detail::HostTapeSnapshotRecord);

struct HostTapeMemoryUsage {
    size_t currentBytes{};
    size_t peakBytes{};
};

#ifdef VERNON_HOST_TAPE_INSTRUMENTATION
struct HostTapeTraversalMetrics {
    size_t regionLookups{};
    size_t recordResolutions{};
    size_t leafReads{};
    size_t childReads{};
    size_t executedCountReads{};
    size_t exitKindReads{};

    void reset() { *this = {}; }
};

class HostTapeTraversalScope {
public:
    explicit HostTapeTraversalScope(HostTapeTraversalMetrics &metrics);
    ~HostTapeTraversalScope();

    HostTapeTraversalScope(const HostTapeTraversalScope &) = delete;
    HostTapeTraversalScope &operator=(const HostTapeTraversalScope &) = delete;

private:
    HostTapeTraversalMetrics *previous_{};
};

HostTapeTraversalMetrics *currentHostTapeTraversalMetrics();
VernonStatus withHostTapeTraversalMetrics(HostTapeTraversalMetrics *destination,
                                          const std::function<VernonStatus()> &callback);
#endif

class HostTapeDispatchBudget;
class HostDynamicTapeBatch;
class HostStaticTapeBatch;
bool hostStaticTapeBatchPureStaticBytes(size_t laneCount, size_t payloadStride, size_t &result);

class AutodiffMemoryPolicy {
public:
    AutodiffMemoryPolicy(size_t invocationLimit = kDefaultHostTapeInvocationLimit,
                         size_t contextLimit = kDefaultHostTapeContextLimit)
        : invocationLimit_(invocationLimit), contextLimit_(contextLimit) {}

    size_t invocationLimit() const { return invocationLimit_; }
    size_t contextLimit() const { return contextLimit_; }
    HostTapeMemoryUsage usage() const;

private:
    friend class AutodiffMemoryReservation;
    friend class HostDynamicTapeBatch;
    friend class HostStaticTapeBatch;
    friend class HostTapeDispatchBudget;
#ifdef VERNON_HOST_TAPE_INSTRUMENTATION
    friend size_t hostTapeMemoryPolicyChargedBytesForTesting(AutodiffMemoryPolicy &policy);
#endif

    bool reserveContext(size_t additionalBytes);
    void release(size_t bytes);

    mutable std::mutex mutex_;
    size_t invocationLimit_;
    size_t contextLimit_;
    size_t contextBytes_{};
    size_t peakContextBytes_{};
};

using HostTapeMemoryPolicy = AutodiffMemoryPolicy;

class AutodiffMemoryReservation {
public:
    static std::shared_ptr<AutodiffMemoryReservation> reserve(std::shared_ptr<AutodiffMemoryPolicy> policy,
                                                              size_t bytes);
    ~AutodiffMemoryReservation();

    size_t bytes() const { return bytes_; }
    bool shrink(size_t bytes);

private:
    AutodiffMemoryReservation(std::shared_ptr<AutodiffMemoryPolicy> policy, size_t bytes)
        : policy_(std::move(policy)), bytes_(bytes) {}

    std::shared_ptr<AutodiffMemoryPolicy> policy_;
    size_t bytes_{};
};

class HostTapeDispatchBudget {
public:
    static std::shared_ptr<HostTapeDispatchBudget> reserve(std::shared_ptr<HostTapeMemoryPolicy> policy,
                                                           size_t capacity);
    ~HostTapeDispatchBudget();

    size_t capacity() const { return capacity_; }
    size_t usedBytes() const;
    HostTapeMemoryUsage usage() const;
    bool reserveTransientBytes(size_t bytes) { return reserveBytes(bytes); }
    void releaseTransientBytes(size_t bytes) { releaseBytes(bytes); }
    void commit();
    bool beginRecycledConstruction();

private:
    HostTapeDispatchBudget(std::shared_ptr<HostTapeMemoryPolicy> policy, size_t capacity)
        : policy_(std::move(policy)), capacity_(capacity) {}

    friend class HostDynamicTapeBatch;

    bool reserveBytes(size_t additionalBytes);
    void releaseBytes(size_t bytes);

    std::shared_ptr<HostTapeMemoryPolicy> policy_;
    mutable std::mutex mutex_;
    size_t capacity_{};
    size_t usedBytes_{};
    size_t peakUsedBytes_{};
    bool committed_{};
};

class HostDynamicTapeBatch {
public:
    HostDynamicTapeBatch(size_t laneCount, size_t invocationCapacity, std::shared_ptr<HostTapeMemoryPolicy> policy,
                         std::shared_ptr<HostTapeDispatchBudget> dispatchBudget);
    ~HostDynamicTapeBatch();

    HostDynamicTapeBatch(const HostDynamicTapeBatch &) = delete;
    HostDynamicTapeBatch &operator=(const HostDynamicTapeBatch &) = delete;

    VernonAdTapeAllocatorStatus reset(size_t lane);
    VernonAdTapeAllocatorStatus beginRegion(size_t lane, VernonAdRegionHandle parent, VernonAdRegionHandle *region);
    VernonAdTapeAllocatorStatus reserveRecord(size_t lane, VernonAdRegionHandle region, size_t payloadSize,
                                              size_t payloadAlignment, size_t childCount, VernonAdRecordHandle *record);
    VernonAdTapeAllocatorStatus writeLeaf(size_t lane, VernonAdRecordHandle record, size_t leafOffset, const void *data,
                                          size_t byteSize);
    VernonAdTapeAllocatorStatus setChild(size_t lane, VernonAdRecordHandle record, size_t childOrdinal,
                                         VernonAdRegionHandle child);
    VernonAdTapeAllocatorStatus endRegion(size_t lane, VernonAdRegionHandle region, size_t executedCount,
                                          uint32_t exitKind);
    VernonAdTapeAllocatorStatus seal(size_t lane);
    VernonAdTapeAllocatorStatus status(size_t lane) const;
    size_t requiredBytes(size_t lane) const;

    bool compact(bool retainConstructionStorage = false);
    bool resetCompactedReplay();
    size_t constructionBytes() const;
    bool isCompacted() const;
    VernonAdRegionHandle rootRegion(size_t lane) const;
    VernonAdTapeAllocatorStatus readLeaf(size_t lane, VernonAdRegionHandle region, size_t recordIndex,
                                         size_t leafOffset, void *data, size_t byteSize) const;
    VernonAdTapeAllocatorStatus readChild(size_t lane, VernonAdRegionHandle region, size_t recordIndex,
                                          size_t childOrdinal, VernonAdRegionHandle *child) const;
    VernonAdTapeAllocatorStatus readCount(size_t lane, VernonAdRegionHandle region, size_t *count) const;
    VernonAdTapeAllocatorStatus readExit(size_t lane, VernonAdRegionHandle region, uint32_t *exitKind) const;

    size_t logicalBytes() const;
    size_t residentBytes() const;
    size_t allocatedBytes() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

/// Dispatch-owned allocator for both straight-line and dynamic VJP profiles.
/// Descriptors, lane state, and static payloads are contiguous. Dynamic lanes
/// append to one shared arena and seal into one immutable batch.
class HostStaticTapeBatch : public std::enable_shared_from_this<HostStaticTapeBatch> {
public:
    enum class ConstructionState : uint8_t { Constructing, FrozenReader, Recyclable, Released };

    class Reader {
    public:
        Reader() = default;
        VernonAdTapeAllocator *descriptor() { return &descriptor_; }
        VernonAdRegionHandle rootRegion() const;

    private:
        friend class HostStaticTapeBatch;

        static VernonAdTapeAllocatorStatus reject(VernonAdTapeAllocator *allocator);
        static VernonAdTapeAllocatorStatus rejectBegin(VernonAdTapeAllocator *allocator, VernonAdRegionHandle,
                                                       VernonAdRegionHandle *);
        static VernonAdTapeAllocatorStatus rejectReserve(VernonAdTapeAllocator *allocator, VernonAdRegionHandle, size_t,
                                                         size_t, size_t, VernonAdRecordHandle *);
        static VernonAdTapeAllocatorStatus rejectWrite(VernonAdTapeAllocator *allocator, VernonAdRecordHandle, size_t,
                                                       const void *, size_t);
        static VernonAdTapeAllocatorStatus rejectChild(VernonAdTapeAllocator *allocator, VernonAdRecordHandle, size_t,
                                                       VernonAdRegionHandle);
        static VernonAdTapeAllocatorStatus rejectEnd(VernonAdTapeAllocator *allocator, VernonAdRegionHandle, size_t,
                                                     uint32_t);
        static VernonAdTapeAllocatorStatus readLeaf(VernonAdTapeAllocator *allocator, VernonAdRegionHandle region,
                                                    size_t recordIndex, size_t leafOffset, void *data, size_t byteSize);
        static VernonAdTapeAllocatorStatus readChild(VernonAdTapeAllocator *allocator, VernonAdRegionHandle, size_t,
                                                     size_t, VernonAdRegionHandle *);
        static VernonAdTapeAllocatorStatus readCount(VernonAdTapeAllocator *allocator, VernonAdRegionHandle region,
                                                     size_t *count);
        static VernonAdTapeAllocatorStatus readExit(VernonAdTapeAllocator *allocator, VernonAdRegionHandle region,
                                                    uint32_t *exitKind);
        static Reader *owner(VernonAdTapeAllocator *allocator);

        VernonAdTapeAllocator descriptor_{};
        const HostStaticTapeBatch *batch_{};
        size_t lane_{std::numeric_limits<size_t>::max()};
        size_t laneCount_{};
    };

    static std::shared_ptr<HostStaticTapeBatch> create(size_t laneCount, size_t payloadStride,
                                                       size_t invocationCapacity,
                                                       std::shared_ptr<HostTapeMemoryPolicy> policy,
                                                       std::shared_ptr<HostTapeDispatchBudget> dispatchBudget);
    static HostStaticTapeBatch *fromWriteDescriptor(VernonAdTapeAllocator *allocator);
    ~HostStaticTapeBatch();

    HostStaticTapeBatch(const HostStaticTapeBatch &) = delete;
    HostStaticTapeBatch &operator=(const HostStaticTapeBatch &) = delete;

    size_t size() const { return laneCount_; }
    VernonAdTapeAllocator *descriptor(size_t lane);
    VernonAdRegionHandle rootRegion(size_t lane) const;
    bool compact(bool retainConstructionStorage = false);
    bool markConstructionRecyclable();
    bool resetRecyclableConstruction();
    bool initializeReader(size_t lane, Reader &reader) const;
    bool isCompacted() const { return compacted_; }
    bool hasDynamicLanes() const;
    ConstructionState constructionState() const { return constructionState_; }
    size_t constructionBytes() const;
    size_t logicalBytes() const;
    size_t residentBytes() const;
    size_t allocatedBytes() const;

private:
    friend bool hostStaticTapeBatchPureStaticBytes(size_t laneCount, size_t payloadStride, size_t &result);
    enum class LanePhase : uint8_t {
        Empty,
        RegionOpen,
        RecordOpen,
        RegionEnded,
        Sealed,
        Promoted,
    };

    struct LaneState {
        LanePhase phase{LanePhase::Empty};
        size_t payloadSize{};
        size_t executedCount{};
        uint32_t exitKind{};
    };

    HostStaticTapeBatch(size_t laneCount, size_t payloadStride, size_t invocationCapacity,
                        std::shared_ptr<HostTapeMemoryPolicy> policy,
                        std::shared_ptr<HostTapeDispatchBudget> dispatchBudget);
    void initializeDescriptors();

    static std::pair<HostStaticTapeBatch *, size_t> owner(VernonAdTapeAllocator *allocator);
    static VernonAdTapeAllocatorStatus reset(VernonAdTapeAllocator *allocator);
    static VernonAdTapeAllocatorStatus beginRegion(VernonAdTapeAllocator *allocator, VernonAdRegionHandle parent,
                                                   VernonAdRegionHandle *region);
    static VernonAdTapeAllocatorStatus reserveRecord(VernonAdTapeAllocator *allocator, VernonAdRegionHandle region,
                                                     size_t payloadSize, size_t payloadAlignment, size_t childCount,
                                                     VernonAdRecordHandle *record);
    static VernonAdTapeAllocatorStatus writeLeaf(VernonAdTapeAllocator *allocator, VernonAdRecordHandle record,
                                                 size_t leafOffset, const void *data, size_t byteSize);
    static VernonAdTapeAllocatorStatus setChild(VernonAdTapeAllocator *allocator, VernonAdRecordHandle record,
                                                size_t childOrdinal, VernonAdRegionHandle child);
    static VernonAdTapeAllocatorStatus endRegion(VernonAdTapeAllocator *allocator, VernonAdRegionHandle region,
                                                 size_t executedCount, uint32_t exitKind);
    static VernonAdTapeAllocatorStatus seal(VernonAdTapeAllocator *allocator);
    static VernonAdTapeAllocatorStatus readLeaf(VernonAdTapeAllocator *allocator, VernonAdRegionHandle region,
                                                size_t recordIndex, size_t leafOffset, void *data, size_t byteSize);
    static VernonAdTapeAllocatorStatus readChild(VernonAdTapeAllocator *allocator, VernonAdRegionHandle region,
                                                 size_t recordIndex, size_t childOrdinal, VernonAdRegionHandle *child);
    static VernonAdTapeAllocatorStatus readCount(VernonAdTapeAllocator *allocator, VernonAdRegionHandle region,
                                                 size_t *count);
    static VernonAdTapeAllocatorStatus readExit(VernonAdTapeAllocator *allocator, VernonAdRegionHandle region,
                                                uint32_t *exitKind);

    VernonAdTapeAllocatorStatus fail(size_t lane, VernonAdTapeAllocatorStatus status);
    HostDynamicTapeBatch *dynamicBatch() const { return dynamicBatchAddress_.load(std::memory_order_acquire); }
    HostDynamicTapeBatch *ensureDynamicBatch();
    VernonAdTapeAllocatorStatus promote(size_t lane, size_t payloadSize, size_t payloadAlignment, size_t childCount,
                                        VernonAdRecordHandle *record);
    VernonAdTapeAllocatorStatus syncDynamic(size_t lane, VernonAdTapeAllocatorStatus status);
    std::byte *payload(size_t lane) { return payload_.data() + lane * payloadStride_; }
    const std::byte *payload(size_t lane) const { return payload_.data() + lane * payloadStride_; }

    std::vector<VernonAdTapeAllocator> descriptors_;
    std::vector<LaneState> lanes_;
    std::unique_ptr<HostDynamicTapeBatch> dynamicBatchOwner_;
    std::atomic<HostDynamicTapeBatch *> dynamicBatchAddress_{};
    std::once_flag dynamicBatchOnce_;
    std::vector<std::byte> payload_;
    std::vector<uint8_t> compactedLaneKinds_;
    size_t laneCount_{};
    size_t payloadStride_{};
    size_t invocationCapacity_{};
    std::shared_ptr<HostTapeMemoryPolicy> policy_;
    std::shared_ptr<HostTapeDispatchBudget> dispatchBudget_;
    size_t policyCharge_{};
    size_t compactedLogicalBytes_{};
    bool compacted_{};
    bool retainsConstructionStorage_{};
    ConstructionState constructionState_{ConstructionState::Constructing};
};

} // namespace vernon::runtime::ad

#endif
