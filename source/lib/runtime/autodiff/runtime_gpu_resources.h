#ifndef VERNON_RUNTIME_AUTODIFF_RUNTIME_GPU_RESOURCES_H
#define VERNON_RUNTIME_AUTODIFF_RUNTIME_GPU_RESOURCES_H

#include "runtime/pipeline_metadata.h"
#include "runtime/runtime_dispatch.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

struct VernonRuntimeContext;

namespace vernon::runtime::ad::gpu {

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

struct DeviceValue {
    std::shared_ptr<DeviceBuffer> retainedBuffer;
    DeviceBuffer buffer;
    VernonDataType dtype{};
    std::vector<uint64_t> shape;
    std::vector<int64_t> strides;
    size_t byteOffset{};
    bool physicalLayout{};

    DeviceValue(VernonRuntimeContext &context, const VernonAdValue &value);
    DeviceValue(VernonRuntimeContext &context, const VernonAdDeviceValue &value);
    DeviceValue(VernonRuntimeContext &context, std::shared_ptr<DeviceBuffer> owner, const VernonAdDeviceValue &value);
    DeviceValue(VernonRuntimeContext &context, size_t allocationSize, VernonDataType dtype, std::vector<uint64_t> shape,
                std::vector<int64_t> strides, size_t byteOffset);
};

struct HostValue {
    VernonDataType dtype{};
    std::vector<uint8_t> bytes;
    std::vector<uint64_t> shape;
    std::vector<int64_t> strides;

    explicit HostValue(const VernonAdValue &value);
};

class OwnedPipeline {
public:
    OwnedPipeline() = default;
    OwnedPipeline(OwnedPipeline &&other) noexcept;
    OwnedPipeline &operator=(OwnedPipeline &&other) noexcept;
    OwnedPipeline(const OwnedPipeline &) = delete;
    OwnedPipeline &operator=(const OwnedPipeline &) = delete;
    ~OwnedPipeline();

    VernonLoadedPipeline *get() const { return value_; }
    VernonLoadedPipeline &operator*() const { return *value_; }
    VernonLoadedPipeline *operator->() const { return value_; }
    bool create(VernonRuntimeContext &context, const Stage &stage);

private:
    void reset();

    VernonLoadedPipeline *value_{};
};

using DeviceValues = std::unordered_map<std::string, DeviceValue>;
using HostValues = std::unordered_map<std::string, HostValue>;

} // namespace vernon::runtime::ad::gpu

#endif
