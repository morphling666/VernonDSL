#ifndef VERNON_RHI_CUDA_BACKEND_H
#define VERNON_RHI_CUDA_BACKEND_H

#include "VernonRHI.h"
#include "cuda_driver.h"
#include "staging_pool.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <vector>

namespace vernon::rhi::cuda {

struct VERNON_RHI_CAPI DeviceState {
    struct PinnedBlock {
        void *data{};
        size_t size{};
        bool inUse{};
        // Recorded after the stream work that reads the block; the block is reusable once it completes.
        Event retirement{};
    };
    struct TransientChunk {
        DevicePointer pointer{};
        size_t capacity{};
    };
    // The transient chunks of a submitted or abandoned recording, reusable once `retirement` completes.
    struct RetiringTransients {
        StagingArena<TransientChunk> arena;
        Event retirement{};
    };

    ~DeviceState();

    Result initialize(uint32_t deviceIndex);
    void shutdown();
    Result makeCurrent() const;
    Result synchronize() const;
    // Completes after all work enqueued on the stream before it. Every recorded completion must be passed to
    // waitCompletion exactly once.
    Result recordCompletion(Event &completion);
    // kSuccess, kNotReady, or the error of the work it covers.
    Result queryCompletion(Event completion) const;
    Result waitCompletion(Event completion);
    Result allocate(DevicePointer &pointer, size_t size) const;
    Result free(DevicePointer pointer) const;
    Result upload(DevicePointer destination, const void *source, size_t size);
    Result uploadRanges(DevicePointer destination, const VernonRhiBufferUploadRange *ranges, size_t rangeCount);
    Result download(void *destination, DevicePointer source, size_t size);
    Result downloadRanges(DevicePointer source, const VernonRhiBufferDownloadRange *ranges, size_t rangeCount);
    Result copy(DevicePointer destination, DevicePointer source, size_t size);
    // Uploads `size` bytes into device memory that the open recording holds until `retireTransients`, at `offset` in
    // `chunk`, a multiple of `alignment`.
    Result writeTransient(const void *data, size_t size, size_t alignment, DevicePointer &chunk, size_t &offset);
    // Ends the open recording's hold on its transient memory, which is reused once the stream work enqueued so far
    // completes. A device records one encoder at a time, so this runs when it is submitted or abandoned.
    Result retireTransients();

    Device device{};
    Context context{};
    Stream stream{};
    uint32_t computeCapabilityMajor{};
    uint32_t computeCapabilityMinor{};
    uint32_t driverVersion{};
    // Device allocations made through `allocate`, transient chunks included.
    mutable std::atomic<size_t> allocations{};

private:
    PinnedBlock *acquirePinned(size_t size, Result &status);
    void releasePinned(PinnedBlock &block);
    // Keeps `block` in use until the stream work enqueued so far completes.
    Result retirePinned(PinnedBlock &block);
    Result acquireEvent(Event &event);
    void recycleEvent(Event event, bool reusable);
    // Returns the chunks of the retiring recordings whose work has completed to `transientPool`.
    Result reclaimTransients();

    std::mutex poolMutex;
    std::deque<PinnedBlock> stagingPool;
    std::vector<Event> eventPool;
    StagingPool<TransientChunk> transientPool;
    StagingArena<TransientChunk> recordingTransients;
    std::deque<RetiringTransients> retiringTransients;
};

struct VERNON_RHI_CAPI PreparedFunction {
    Module module{};
    Function function{};

    Result create(DeviceState &device, const void *artifact, size_t artifactSize, const char *entry);
    Result destroy(DeviceState &device);
    Result launch(DeviceState &device, const uint32_t groupCount[3], const uint32_t blockSize[3],
                  void **parameters) const;
};

VERNON_RHI_CAPI std::string describeResult(Result result, const char *operation);

} // namespace vernon::rhi::cuda

#endif
