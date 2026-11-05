#include "device_buffer.h"

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
    if (!size || vernonRhiDeviceCreateBuffer(context.rhiDevice, &descriptor, &handle_) != VERNON_RHI_STATUS_OK)
        handle_ = {static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
}

DeviceBuffer::DeviceBuffer(DeviceBuffer &&other) noexcept
    : context_(std::exchange(other.context_, nullptr)),
      handle_(std::exchange(other.handle_, VernonRhiBuffer{static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0})),
      size_(std::exchange(other.size_, 0)) {}

DeviceBuffer &DeviceBuffer::operator=(DeviceBuffer &&other) noexcept {
    if (this == &other)
        return *this;
    reset();
    context_ = std::exchange(other.context_, nullptr);
    handle_ = std::exchange(other.handle_, VernonRhiBuffer{static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0});
    size_ = std::exchange(other.size_, 0);
    return *this;
}

DeviceBuffer::~DeviceBuffer() { reset(); }

bool DeviceBuffer::valid() const {
    return context_ && handle_.index != VERNON_RHI_INVALID_HANDLE_INDEX &&
           vernonRhiDeviceIsBufferValid(context_->rhiDevice, handle_);
}

bool DeviceBuffer::reference(VernonRuntimeProviderResourceReference &output) const {
    if (!valid())
        return false;
    auto referenced = referenceBackendRhiBuffer(*context_, handle_, 0, size_);
    if (referenced.isErr())
        return false;
    output = referenced.value();
    return true;
}

void DeviceBuffer::reset() {
    if (context_ && handle_.index != VERNON_RHI_INVALID_HANDLE_INDEX)
        (void)vernonRhiDeviceDestroyBuffer(context_->rhiDevice, handle_);
    context_ = nullptr;
    handle_ = {static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
    size_ = 0;
}

} // namespace vernon::runtime::program_execution
