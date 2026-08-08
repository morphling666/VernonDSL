#ifndef VERNON_RUNTIME_AUTODIFF_HOST_TAPE_ALLOCATOR_H
#define VERNON_RUNTIME_AUTODIFF_HOST_TAPE_ALLOCATOR_H

#include "runtime/autodiff/tape_allocator_abi.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

namespace vernon::runtime::ad {

inline constexpr size_t kDefaultHostTapeInvocationLimit = 64u * 1024u * 1024u;
inline constexpr size_t kDefaultHostTapeContextLimit = 256u * 1024u * 1024u;

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
#endif

class HostTapeMemoryPolicy {
public:
    HostTapeMemoryPolicy(size_t invocationLimit = kDefaultHostTapeInvocationLimit,
                         size_t contextLimit = kDefaultHostTapeContextLimit)
        : invocationLimit_(invocationLimit), contextLimit_(contextLimit) {}

    size_t invocationLimit() const { return invocationLimit_; }

private:
    friend class HostDynamicTape;
    friend class HostTapeSnapshot;
#ifdef VERNON_HOST_TAPE_INSTRUMENTATION
    friend size_t hostTapeMemoryPolicyChargedBytesForTesting(HostTapeMemoryPolicy &policy);
#endif

    bool reserve(size_t currentInvocationBytes, size_t additionalBytes);
    void release(size_t bytes);

    std::mutex mutex_;
    size_t invocationLimit_;
    size_t contextLimit_;
    size_t contextBytes_{};
};

class HostTapeSnapshot {
public:
    ~HostTapeSnapshot();

    VernonAdTapeAllocator *descriptor() const { return &descriptor_; }
    VernonAdRegionHandle rootRegion() const;

private:
    friend class HostDynamicTape;

    struct Region {
        VernonAdRegionHandle handle{};
        std::vector<size_t> records;
        size_t executedCount{};
        uint32_t exitKind{};
    };

    struct Record {
        VernonAdRecordHandle handle{};
        size_t payloadOffset{};
        size_t payloadSize{};
        std::vector<VernonAdRegionHandle> children;
    };

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
    static VernonAdTapeAllocatorStatus readChild(VernonAdTapeAllocator *allocator, VernonAdRegionHandle region,
                                                 size_t recordIndex, size_t childOrdinal, VernonAdRegionHandle *child);
    static VernonAdTapeAllocatorStatus readExecutedCount(VernonAdTapeAllocator *allocator, VernonAdRegionHandle region,
                                                         size_t *executedCount);
    static VernonAdTapeAllocatorStatus readExitKind(VernonAdTapeAllocator *allocator, VernonAdRegionHandle region,
                                                    uint32_t *exitKind);
    void initializeDescriptor();
    const Region *findRegion(VernonAdRegionHandle handle) const;

    std::vector<std::byte> payload_;
    std::vector<Region> regions_;
    std::vector<Record> records_;
    std::unordered_map<VernonAdRegionHandle, size_t> regionIndex_;
    std::shared_ptr<HostTapeMemoryPolicy> policy_;
    size_t policyCharge_{};
    mutable VernonAdTapeAllocator descriptor_{};
};

class HostDynamicTape {
public:
    explicit HostDynamicTape(size_t capacityBytes = std::numeric_limits<size_t>::max(),
                             std::shared_ptr<HostTapeMemoryPolicy> policy = std::make_shared<HostTapeMemoryPolicy>());
    ~HostDynamicTape();

    HostDynamicTape(const HostDynamicTape &) = delete;
    HostDynamicTape &operator=(const HostDynamicTape &) = delete;

    VernonAdTapeAllocator &descriptor() { return descriptor_; }
    std::shared_ptr<const HostTapeSnapshot> takeSnapshot();

private:
    struct Region {
        VernonAdRegionHandle handle{};
        VernonAdRegionHandle parent{};
        std::vector<size_t> records;
        size_t executedCount{};
        uint32_t exitKind{};
        bool open{};
        bool ended{};
    };

    struct Record {
        VernonAdRecordHandle handle{};
        size_t regionIndex{};
        size_t payloadOffset{};
        size_t payloadSize{};
        std::vector<VernonAdRegionHandle> children;
    };

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
    static VernonAdTapeAllocatorStatus readLeaf(VernonAdTapeAllocator *allocator, VernonAdRegionHandle, size_t, size_t,
                                                void *, size_t);
    static VernonAdTapeAllocatorStatus readChild(VernonAdTapeAllocator *allocator, VernonAdRegionHandle, size_t, size_t,
                                                 VernonAdRegionHandle *);
    static VernonAdTapeAllocatorStatus readCount(VernonAdTapeAllocator *allocator, VernonAdRegionHandle, size_t *);
    static VernonAdTapeAllocatorStatus readExit(VernonAdTapeAllocator *allocator, VernonAdRegionHandle, uint32_t *);

    static HostDynamicTape *owner(VernonAdTapeAllocator *allocator);
    VernonAdTapeAllocatorStatus fail(VernonAdTapeAllocatorStatus status);
    Region *findRegion(VernonAdRegionHandle handle);
    Record *findRecord(VernonAdRecordHandle handle);
    bool callable() const;
    void releaseCharge();

    VernonAdTapeAllocator descriptor_{};
    std::vector<std::byte> payload_;
    std::vector<Region> regions_;
    std::vector<Record> records_;
    std::unordered_map<VernonAdRegionHandle, size_t> regionIndex_;
    std::unordered_map<VernonAdRecordHandle, size_t> recordIndex_;
    std::vector<size_t> openRegions_;
    std::shared_ptr<HostTapeMemoryPolicy> policy_;
    std::thread::id ownerThread_;
    VernonAdRegionHandle nextRegionHandle_{1};
    VernonAdRecordHandle nextRecordHandle_{1};
    size_t policyCharge_{};
    bool sealed_{};
};

} // namespace vernon::runtime::ad

#endif
