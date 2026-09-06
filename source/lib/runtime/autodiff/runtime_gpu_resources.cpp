#include "runtime/autodiff/runtime_gpu_resources.h"

#include "runtime_autodiff_internal.h"
#include "runtime_gpu_failure_injection.h"

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

DeviceBuffer::DeviceBuffer(VernonRuntimeContext &context, size_t size) : context_(&context), size_(size) {
    VernonRhiBufferDescriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.size = size;
    descriptor.alignment = 4;
    descriptor.usage =
        VERNON_RHI_BUFFER_TRANSFER_SOURCE | VERNON_RHI_BUFFER_TRANSFER_DESTINATION | VERNON_RHI_BUFFER_STORAGE;
    descriptor.memory_class = VERNON_RHI_MEMORY_DEVICE;
    if (!size || injectFailure(FailureBoundary::Allocation) ||
        vernonRhiDeviceCreateBuffer(context.rhiDevice, &descriptor, &handle_) != VERNON_RHI_STATUS_OK)
        handle_ = {static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
}

DeviceBuffer::DeviceBuffer(VernonRuntimeContext &context, VernonRhiBuffer handle, size_t size)
    : context_(&context), handle_(handle), size_(size), owned_(false) {}

DeviceBuffer::DeviceBuffer(DeviceBuffer &&other) noexcept
    : context_(std::exchange(other.context_, nullptr)),
      handle_(std::exchange(other.handle_, VernonRhiBuffer{static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0})),
      size_(std::exchange(other.size_, 0)), owned_(std::exchange(other.owned_, true)) {}

DeviceBuffer &DeviceBuffer::operator=(DeviceBuffer &&other) noexcept {
    if (this == &other)
        return *this;
    reset();
    context_ = std::exchange(other.context_, nullptr);
    handle_ = std::exchange(other.handle_, VernonRhiBuffer{static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0});
    size_ = std::exchange(other.size_, 0);
    owned_ = std::exchange(other.owned_, true);
    return *this;
}

DeviceBuffer::~DeviceBuffer() { reset(); }

bool DeviceBuffer::valid() const {
    return context_ && handle_.index != VERNON_RHI_INVALID_HANDLE_INDEX &&
           vernonRhiDeviceIsBufferValid(context_->rhiDevice, handle_);
}

bool DeviceBuffer::upload(const void *source, size_t size) const {
    return valid() && size == size_ && !injectFailure(FailureBoundary::Upload) &&
           vernonRhiDeviceUploadBuffer(context_->rhiDevice, handle_, 0, source, size) == VERNON_RHI_STATUS_OK;
}

bool DeviceBuffer::upload(VernonRhiCommandEncoder encoder, const void *source, size_t size) const {
    return valid() && size == size_ && !injectFailure(FailureBoundary::Upload) &&
           vernonRhiCommandEncoderUploadBuffer(context_->rhiDevice, encoder, handle_, 0, source, size) ==
               VERNON_RHI_STATUS_OK;
}

bool DeviceBuffer::upload(VernonRhiCommandEncoder encoder, size_t offset, const void *source, size_t size) const {
    return valid() && offset <= size_ && size <= size_ - offset && !injectFailure(FailureBoundary::Upload) &&
           vernonRhiCommandEncoderUploadBuffer(context_->rhiDevice, encoder, handle_, offset, source, size) ==
               VERNON_RHI_STATUS_OK;
}

bool DeviceBuffer::upload(size_t offset, const void *source, size_t size) const {
    return valid() && offset <= size_ && size <= size_ - offset && !injectFailure(FailureBoundary::Upload) &&
           vernonRhiDeviceUploadBuffer(context_->rhiDevice, handle_, offset, source, size) == VERNON_RHI_STATUS_OK;
}

bool DeviceBuffer::uploadRanges(const std::vector<VernonRhiBufferUploadRange> &ranges) const {
    return valid() && !injectFailure(FailureBoundary::Upload) &&
           vernonRhiDeviceUploadBufferRanges(context_->rhiDevice, handle_, ranges.data(), ranges.size()) ==
               VERNON_RHI_STATUS_OK;
}

bool DeviceBuffer::download(void *destination, size_t size) const {
    return valid() && size == size_ && !injectFailure(FailureBoundary::Download) &&
           vernonRhiDeviceDownloadBuffer(context_->rhiDevice, handle_, 0, destination, size) == VERNON_RHI_STATUS_OK;
}

bool DeviceBuffer::download(size_t offset, void *destination, size_t size) const {
    return valid() && offset <= size_ && size <= size_ - offset && !injectFailure(FailureBoundary::Download) &&
           vernonRhiDeviceDownloadBuffer(context_->rhiDevice, handle_, offset, destination, size) ==
               VERNON_RHI_STATUS_OK;
}

bool DeviceBuffer::reference(VernonRuntimeProviderResourceReference &output) const {
    return valid() && referenceBackendRhiBuffer(*context_, handle_, 0, size_, output) == VERNON_STATUS_OK;
}

bool DeviceBuffer::reference(size_t offset, size_t size, VernonRuntimeProviderResourceReference &output) const {
    return valid() && offset <= size_ && size <= size_ - offset &&
           referenceBackendRhiBuffer(*context_, handle_, offset, size, output) == VERNON_STATUS_OK;
}

void DeviceBuffer::reset() {
    if (owned_ && context_ && handle_.index != VERNON_RHI_INVALID_HANDLE_INDEX)
        (void)vernonRhiDeviceDestroyBuffer(context_->rhiDevice, handle_);
    context_ = nullptr;
    handle_ = {static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
    size_ = 0;
    owned_ = true;
}

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

DeviceValue::DeviceValue(VernonRuntimeContext &context, std::shared_ptr<DeviceBuffer> owner,
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
