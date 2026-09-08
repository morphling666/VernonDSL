#ifndef VERNON_RUNTIME_PROGRAM_EXECUTION_DEVICE_BUFFER_H
#define VERNON_RUNTIME_PROGRAM_EXECUTION_DEVICE_BUFFER_H

#include "VernonRuntime.h"

#include <memory>
#include <vector>

namespace vernon::runtime::program_execution {

class DeviceBuffer {
public:
    DeviceBuffer() = default;
    DeviceBuffer(VernonRuntimeContext &context, size_t size);
    DeviceBuffer(VernonRuntimeContext &context, VernonRhiBuffer handle, size_t size);
    DeviceBuffer(DeviceBuffer &&other) noexcept;
    DeviceBuffer &operator=(DeviceBuffer &&other) noexcept;
    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;
    ~DeviceBuffer();

    bool valid() const;
    bool upload(const void *source, size_t size) const;
    bool upload(VernonRhiCommandEncoder encoder, const void *source, size_t size) const;
    bool upload(VernonRhiCommandEncoder encoder, size_t offset, const void *source, size_t size) const;
    bool upload(size_t offset, const void *source, size_t size) const;
    bool uploadRanges(const std::vector<VernonRhiBufferUploadRange> &ranges) const;
    bool download(void *destination, size_t size) const;
    bool download(size_t offset, void *destination, size_t size) const;
    bool reference(VernonRuntimeProviderResourceReference &output) const;
    bool reference(size_t offset, size_t size, VernonRuntimeProviderResourceReference &output) const;
    size_t size() const { return size_; }
    VernonRhiBuffer handle() const { return handle_; }

private:
    void reset();

    VernonRuntimeContext *context_{};
    VernonRhiBuffer handle_{static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
    size_t size_{};
    bool owned_{true};
};

} // namespace vernon::runtime::program_execution

#endif
