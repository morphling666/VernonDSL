#include "device_buffer.h"

#include "failure_injection.h"
#include "runtime/runtime_dispatch.h"
#include "runtime/runtime_state.h"

#include <utility>

namespace vernon::runtime::program_execution {

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

} // namespace vernon::runtime::program_execution
