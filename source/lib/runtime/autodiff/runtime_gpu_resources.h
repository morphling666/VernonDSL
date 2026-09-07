#ifndef VERNON_RUNTIME_AUTODIFF_RUNTIME_GPU_RESOURCES_H
#define VERNON_RUNTIME_AUTODIFF_RUNTIME_GPU_RESOURCES_H

#include "runtime/pipeline_metadata.h"
#include "runtime/program_execution/device_buffer.h"
#include "runtime/runtime_dispatch.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

struct VernonRuntimeContext;

namespace vernon::runtime::ad::gpu {

struct DeviceValue {
    std::shared_ptr<program_execution::DeviceBuffer> retainedBuffer;
    program_execution::DeviceBuffer buffer;
    VernonDataType dtype{};
    std::vector<uint64_t> shape;
    std::vector<int64_t> strides;
    size_t byteOffset{};
    bool physicalLayout{};

    DeviceValue(VernonRuntimeContext &context, const VernonAdValue &value);
    DeviceValue(VernonRuntimeContext &context, const VernonAdDeviceValue &value);
    DeviceValue(VernonRuntimeContext &context, std::shared_ptr<program_execution::DeviceBuffer> owner,
                const VernonAdDeviceValue &value);
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

    VernonStageExecutable *get() const { return value_; }
    VernonStageExecutable &operator*() const { return *value_; }
    VernonStageExecutable *operator->() const { return value_; }
    bool create(VernonRuntimeContext &context, const Stage &stage);

private:
    void reset();

    VernonStageExecutable *value_{};
};

using DeviceValues = std::unordered_map<std::string, DeviceValue>;
using HostValues = std::unordered_map<std::string, HostValue>;

} // namespace vernon::runtime::ad::gpu

#endif
