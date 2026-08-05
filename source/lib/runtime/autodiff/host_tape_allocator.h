#ifndef VERNON_RUNTIME_AUTODIFF_HOST_TAPE_ALLOCATOR_H
#define VERNON_RUNTIME_AUTODIFF_HOST_TAPE_ALLOCATOR_H

#include "runtime/autodiff/tape_allocator_abi.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <thread>
#include <vector>

namespace vernon::runtime::ad {

class HostTapeSnapshot {
public:
    bool readRegion(VernonAdRegionHandle handle, const std::byte *&data, size_t &size) const;

private:
    friend class HostTapeAllocator;

    struct Region {
        VernonAdRegionHandle handle{};
        size_t begin{};
        size_t end{};
    };

    std::vector<std::byte> storage_;
    std::vector<Region> regions_;
};

class HostTapeAllocator {
public:
    explicit HostTapeAllocator(size_t capacityBytes = std::numeric_limits<size_t>::max());
    ~HostTapeAllocator();

    HostTapeAllocator(const HostTapeAllocator &) = delete;
    HostTapeAllocator &operator=(const HostTapeAllocator &) = delete;

    VernonAdTapeAllocator &descriptor() { return descriptor_; }
    std::shared_ptr<const HostTapeSnapshot> takeSnapshot();

private:
    struct Region {
        VernonAdRegionHandle handle{};
        size_t begin{};
        size_t end{};
        bool open{};
    };

    static VernonAdTapeAllocatorStatus reset(VernonAdTapeAllocator *allocator);
    static VernonAdTapeAllocatorStatus beginRegion(VernonAdTapeAllocator *allocator, VernonAdRegionHandle parent,
                                                   VernonAdRegionHandle *region);
    static VernonAdTapeAllocatorStatus reserve(VernonAdTapeAllocator *allocator, VernonAdRegionHandle region,
                                               size_t byteSize, size_t alignment, size_t *byteOffset,
                                               void **writeAddress);
    static VernonAdTapeAllocatorStatus endRegion(VernonAdTapeAllocator *allocator, VernonAdRegionHandle region);
    static VernonAdTapeAllocatorStatus seal(VernonAdTapeAllocator *allocator);
    static VernonAdTapeAllocatorStatus readRegion(VernonAdTapeAllocator *allocator, VernonAdRegionHandle region,
                                                  const void **data, size_t *byteSize);

    static HostTapeAllocator *owner(VernonAdTapeAllocator *allocator);
    VernonAdTapeAllocatorStatus fail(VernonAdTapeAllocatorStatus status);
    Region *findRegion(VernonAdRegionHandle handle);
    bool callable() const;

    VernonAdTapeAllocator descriptor_{};
    std::vector<std::byte> storage_;
    std::vector<Region> regions_;
    std::vector<size_t> openRegions_;
    std::thread::id ownerThread_;
    VernonAdRegionHandle nextRegionHandle_{1};
    bool sealed_{};
};

} // namespace vernon::runtime::ad

#endif
