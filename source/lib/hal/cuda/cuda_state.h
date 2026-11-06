#ifndef VERNON_HAL_CUDA_STATE_H
#define VERNON_HAL_CUDA_STATE_H

#include "VernonHal.h"
#include "hal/cuda/cuda_driver.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace vernon::hal::cuda {

// One CUDA device's primary context and the stream that all of the device's work runs on, in order.
struct DeviceState {
    struct PinnedBlock {
        void *data{};
        size_t size{};
        bool inUse{};
        // Recorded after the stream work that reads the block; the block is reusable once it completes.
        Event retirement{};
    };

    DeviceState() = default;
    DeviceState(const DeviceState &) = delete;
    DeviceState &operator=(const DeviceState &) = delete;
    ~DeviceState();

    Result initialize(uint32_t deviceIndex);
    void shutdown();
    Result makeCurrent() const;
    Result synchronize() const;
    Result allocate(DevicePointer &pointer, size_t size) const;
    Result free(DevicePointer pointer) const;
    // Uploads enqueue on the stream from pinned staging and return; downloads wait for the stream.
    Result upload(DevicePointer destination, const void *source, size_t size);
    Result uploadRanges(DevicePointer destination, const VernonHalBufferWrite *writes, size_t count);
    Result download(void *destination, DevicePointer source, size_t size);
    Result downloadRanges(DevicePointer source, const VernonHalBufferRead *reads, size_t count);

    Device device{};
    Context context{};
    Stream stream{};
    uint32_t computeCapabilityMajor{};
    uint32_t computeCapabilityMinor{};
    uint32_t driverVersion{};

private:
    // `cuMemFree` waits for the whole device, so freed allocations are kept by rounded size and handed to the next
    // allocation of that size. Reuse is safe without an event: all of this device's work runs on `stream` in order,
    // and synchronous copies on the legacy stream wait for it, so work issued after the reuse follows the work that
    // read the memory before it was freed.
    void releaseCachedAllocations() const;

    mutable std::mutex cacheMutex;
    // The rounded size of each live allocation `allocate` made.
    mutable std::unordered_map<DevicePointer, size_t> allocationSizes;
    mutable std::unordered_map<size_t, std::vector<DevicePointer>> cachedAllocations;
    mutable size_t cachedBytes{};

    PinnedBlock *acquirePinned(size_t size, Result &status);
    void releasePinned(PinnedBlock &block);
    // Keeps `block` in use until the stream work enqueued so far completes.
    Result retirePinned(PinnedBlock &block);
    Result acquireEvent(Event &event);
    void recycleEvent(Event event, bool reusable);

    std::mutex poolMutex;
    std::deque<PinnedBlock> stagingPool;
    std::vector<Event> eventPool;
};

struct PreparedFunction {
    Module module{};
    Function function{};

    Result create(DeviceState &device, const void *artifact, size_t artifactSize, const char *entry);
    Result destroy(DeviceState &device);
    Result launch(DeviceState &device, const uint32_t groupCount[3], const uint32_t blockSize[3],
                  void **parameters) const;
};

std::string describeResult(Result result, const char *operation);

} // namespace vernon::hal::cuda

#endif
