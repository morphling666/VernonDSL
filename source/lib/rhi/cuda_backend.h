#ifndef VERNON_RHI_CUDA_BACKEND_H
#define VERNON_RHI_CUDA_BACKEND_H

#include "VernonRHI.h"
#include "cuda_driver.h"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>

namespace vernon::rhi::cuda {

struct VERNON_RHI_CAPI DeviceState {
    struct PinnedBlock {
        void *data{};
        size_t size{};
        bool inUse{};
    };

    ~DeviceState();

    Result initialize(uint32_t deviceIndex);
    void shutdown();
    Result makeCurrent() const;
    Result synchronize() const;
    Result allocate(DevicePointer &pointer, size_t size) const;
    Result free(DevicePointer pointer) const;
    Result upload(DevicePointer destination, const void *source, size_t size);
    Result uploadRanges(DevicePointer destination, const VernonRhiBufferUploadRange *ranges, size_t rangeCount);
    Result download(void *destination, DevicePointer source, size_t size);
    Result copy(DevicePointer destination, DevicePointer source, size_t size);

    Device device{};
    Context context{};
    Stream stream{};
    uint32_t computeCapabilityMajor{};
    uint32_t computeCapabilityMinor{};
    uint32_t driverVersion{};

private:
    PinnedBlock *acquirePinned(size_t size, Result &status);
    void releasePinned(PinnedBlock &block);

    std::mutex stagingMutex;
    std::deque<PinnedBlock> stagingPool;
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
