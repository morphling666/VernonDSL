#include "runtime/autodiff/runtime_gpu_resources.h"

#include "runtime_autodiff_internal.h"

#include "runtime/runtime_state.h"

#include <cstring>
#include <limits>
#include <memory>
#include <utility>

namespace vernon::runtime::ad::gpu {
namespace {

bool multiply(size_t left, size_t right, size_t &result) {
    if (left && right > std::numeric_limits<size_t>::max() / left)
        return false;
    result = left * right;
    return true;
}

} // namespace

DeviceValue::DeviceValue(VernonRuntimeContext &context, const VernonAdValue &value)
    : buffer(context, value.size), dtype(value.dtype) {
    if (value.rank)
        shape.assign(value.shape, value.shape + value.rank);
    strides.resize(shape.size());
    size_t stride = dtypeSize(dtype);
    for (size_t dimension = shape.size(); dimension-- > 0;) {
        if (stride > static_cast<size_t>(std::numeric_limits<int64_t>::max()))
            return;
        strides[dimension] = static_cast<int64_t>(stride);
        if (!multiply(stride, static_cast<size_t>(shape[dimension]), stride)) {
            strides.clear();
            return;
        }
    }
}

DeviceValue::DeviceValue(VernonRuntimeContext &context, const VernonAdDeviceValue &value)
    : buffer(context, value.buffer, static_cast<size_t>(value.buffer_size)), dtype(value.dtype),
      byteOffset(static_cast<size_t>(value.offset)), physicalLayout(true) {
    if (value.rank)
        shape.assign(value.shape, value.shape + value.rank);
    if (value.rank)
        strides.assign(value.byte_strides, value.byte_strides + value.rank);
}

DeviceValue::DeviceValue(VernonRuntimeContext &context, std::shared_ptr<program_execution::DeviceBuffer> owner,
                         const VernonAdDeviceValue &value)
    : retainedBuffer(std::move(owner)),
      buffer(context,
             retainedBuffer ? retainedBuffer->handle()
                            : VernonRhiBuffer{static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0},
             retainedBuffer ? retainedBuffer->size() : 0),
      dtype(value.dtype), byteOffset(static_cast<size_t>(value.offset)), physicalLayout(true) {
    if (value.rank) {
        shape.assign(value.shape, value.shape + value.rank);
        strides.assign(value.byte_strides, value.byte_strides + value.rank);
    }
}

DeviceValue::DeviceValue(VernonRuntimeContext &context, size_t allocationSize, VernonDataType dtype,
                         std::vector<uint64_t> shape, std::vector<int64_t> strides, size_t byteOffset)
    : buffer(context, allocationSize), dtype(dtype), shape(std::move(shape)), strides(std::move(strides)),
      byteOffset(byteOffset), physicalLayout(true) {}

HostValue::HostValue(const VernonAdValue &value) : dtype(value.dtype), bytes(value.size) {
    if (value.size)
        std::memcpy(bytes.data(), value.data, value.size);
    if (value.rank)
        shape.assign(value.shape, value.shape + value.rank);
    strides.resize(shape.size());
    size_t stride = dtypeSize(dtype);
    for (size_t dimension = shape.size(); dimension-- > 0;) {
        if (stride > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
            strides.clear();
            return;
        }
        strides[dimension] = static_cast<int64_t>(stride);
        if (!multiply(stride, static_cast<size_t>(shape[dimension]), stride)) {
            strides.clear();
            return;
        }
    }
}

OwnedPipeline::OwnedPipeline(OwnedPipeline &&other) noexcept : value_(std::exchange(other.value_, nullptr)) {}

OwnedPipeline &OwnedPipeline::operator=(OwnedPipeline &&other) noexcept {
    if (this != &other) {
        reset();
        value_ = std::exchange(other.value_, nullptr);
    }
    return *this;
}

OwnedPipeline::~OwnedPipeline() { reset(); }

bool OwnedPipeline::create(VernonRuntimeContext &context, const Stage &stage) {
    reset();
    auto pipeline = std::make_unique<VernonStageExecutable>();
    pipeline->context = &context;
    if (!buildReflectedComputeVariant(stage, context.backend, pipeline->bindingProjection,
                                      invocationDiagnostic(context)))
        return false;
    pipeline->workgroupSize = {stage.workgroup[0], stage.workgroup[1], stage.workgroup[2]};
    pipeline->dispatchContract = stage.dispatchContract;
    pipeline->readFootprints = stage.readFootprints;
    pipeline->writeFootprints = stage.writeFootprints;
    BackendPipelineBundle bundle;
    bundle.context = &context;
    bundle.stages.emplace(stage.entry, stage);
    if (!resolveBackendPipeline(bundle, pipeline->bindingProjection, *pipeline))
        return false;
    value_ = pipeline.release();
    return true;
}

void OwnedPipeline::reset() {
    if (!value_)
        return;
    delete value_;
    value_ = nullptr;
}

} // namespace vernon::runtime::ad::gpu
