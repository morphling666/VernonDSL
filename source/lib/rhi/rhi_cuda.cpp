#include "backend_dispatch.h"
#include "cuda_backend.h"

#include <cstdint>
#include <limits>
#include <mutex>
#include <string>
#include <utility>

namespace vernon::rhi {
const BackendDispatch &cudaBackendDispatch();
}

namespace {

using vernon::rhi::invalidArgument;
using vernon::rhi::unsupported;
using BufferHandle = vernon::rhi::ResourceHandle<vernon::rhi::BufferResourceTag>;
using BufferLifecycle = vernon::rhi::ResourceLifecycleSlot<vernon::rhi::BufferResourceTag>;

struct CudaBufferSlot {
    explicit CudaBufferSlot(BufferLifecycle lifecycle) noexcept : lifecycle(std::move(lifecycle)) {}

    BufferLifecycle lifecycle;
    vernon::rhi::cuda::DevicePointer pointer{};
    VernonRhiBufferDescriptor descriptor{};
};

struct CudaDevice {
    CudaDevice() noexcept = default;
    CudaDevice(const CudaDevice &) = delete;
    CudaDevice &operator=(const CudaDevice &) = delete;

    vernon::rhi::cuda::DeviceState state;
    vernon::rhi::StableResourceSlotContainer<CudaBufferSlot> buffers;
    vernon::rhi::CommandDeviceStateRef commandState;
    std::string error;
    std::mutex mutex;
    std::mutex slotMutex;
    std::mutex creationMutex;
};

constexpr uint32_t cudaDeviceBit = uint32_t{1} << 29;
constexpr size_t cudaDeviceCapacity = 256;
using CudaDeviceRegistry = vernon::rhi::DeviceRegistry<CudaDevice, cudaDeviceCapacity>;
using CudaDeviceAnchor = vernon::rhi::DeviceRegistryAnchor<CudaDevice>;

struct CudaRegistryStorage {
    CudaRegistryStorage() noexcept {
        // Initialize the loader after the registry member and before this
        // object's destructor is registered. Devices therefore die first.
        (void)vernon::rhi::cuda::driver();
    }

    CudaDeviceRegistry registry;
};

CudaDeviceRegistry &cudaDevices() {
    static CudaRegistryStorage storage;
    return storage.registry;
}

VernonRhiDevice invalidDevice() noexcept { return {static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0}; }

template <typename Handle> uint64_t resourceKey(Handle handle) noexcept {
    return (static_cast<uint64_t>(handle.generation) << 32) | (static_cast<uint64_t>(handle.index) + 1);
}

vernon::RhiError backendFailure(CudaDevice &device, vernon::rhi::cuda::Result status, const char *operation) noexcept {
    device.error = vernon::rhi::cuda::describeResult(status, operation);
    return {vernon::RhiErrorCode::BackendFailure, {operation, static_cast<uint64_t>(status), 0}};
}

vernon::Result<BufferHandle, vernon::RhiError> decodeBufferKey(uint64_t key, const char *operation) noexcept {
    const uint64_t encodedIndex = key & UINT32_MAX;
    const uint32_t generation = static_cast<uint32_t>(key >> 32);
    if (!encodedIndex || !generation)
        return vernon::Result<BufferHandle, vernon::RhiError>{vernon::err(invalidArgument(operation, key))};
    return vernon::Result<BufferHandle, vernon::RhiError>{
        vernon::ok(BufferHandle{static_cast<uint32_t>(encodedIndex - 1), generation})};
}

vernon::Result<CudaDeviceAnchor, vernon::RhiError> lookupCudaDevice(VernonRhiDevice handle) noexcept {
    if ((handle.index & cudaDeviceBit) == 0)
        return vernon::Result<CudaDeviceAnchor, vernon::RhiError>{
            vernon::err(invalidArgument("lookup_cuda_device", handle.index, handle.generation))};
    return cudaDevices().lookup({handle.index & ~cudaDeviceBit, handle.generation});
}

CudaBufferSlot *findBufferSlot(CudaDevice &device, uint32_t index) noexcept {
    std::lock_guard<std::mutex> guard(device.slotMutex);
    return device.buffers.get(index);
}

struct PinnedCudaBuffer {
    CudaBufferSlot *slot;
    vernon::OperationPin pin;
};

vernon::Result<PinnedCudaBuffer, vernon::RhiError> pinBuffer(CudaDevice &device, BufferHandle handle) noexcept {
    CudaBufferSlot *slot = findBufferSlot(device, handle.index);
    if (!slot)
        return vernon::Result<PinnedCudaBuffer, vernon::RhiError>{
            vernon::err(invalidArgument("pin_cuda_buffer", handle.generation, handle.index))};
    auto pin = slot->lifecycle.pin(handle);
    if (pin.isErr())
        return vernon::Result<PinnedCudaBuffer, vernon::RhiError>{vernon::err(std::move(pin).error())};
    return vernon::Result<PinnedCudaBuffer, vernon::RhiError>{
        vernon::ok(PinnedCudaBuffer{slot, std::move(pin).value()})};
}

vernon::Result<void, vernon::RhiError> teardownCudaBuffer(void *context, BufferHandle handle) noexcept {
    auto &device = *static_cast<CudaDevice *>(context);
    CudaBufferSlot *slot = findBufferSlot(device, handle.index);
    if (!slot)
        return vernon::Result<void, vernon::RhiError>{
            vernon::err(invalidArgument("teardown_cuda_buffer", handle.generation, handle.index))};
    std::lock_guard<std::mutex> guard(device.mutex);
    if (!slot->pointer)
        return vernon::Result<void, vernon::RhiError>{
            vernon::err(invalidArgument("teardown_cuda_buffer", handle.generation, handle.index))};
    const auto status = device.state.free(slot->pointer);
    if (status != vernon::rhi::cuda::kSuccess)
        return vernon::Result<void, vernon::RhiError>{vernon::err(backendFailure(device, status, "cuMemFree"))};
    slot->pointer = 0;
    slot->descriptor = {};
    return vernon::Result<void, vernon::RhiError>{vernon::ok()};
}

vernon::Result<void, vernon::RhiError> teardownCudaDevice(CudaDevice &device) noexcept {
    std::lock_guard<std::mutex> slotGuard(device.slotMutex);
    std::lock_guard<std::mutex> nativeGuard(device.mutex);
    for (std::size_t index = 0; index < device.buffers.size(); ++index) {
        CudaBufferSlot &slot = device.buffers[index];
        if (!slot.pointer)
            continue;
        const auto status = device.state.free(slot.pointer);
        if (status != vernon::rhi::cuda::kSuccess)
            return vernon::Result<void, vernon::RhiError>{vernon::err(backendFailure(device, status, "cuMemFree"))};
        slot.pointer = 0;
        slot.descriptor = {};
    }
    device.state.shutdown();
    return vernon::Result<void, vernon::RhiError>{vernon::ok()};
}

vernon::Result<VernonRhiDevice, vernon::RhiError>
createOwnedDeviceResult(const VernonRhiOwnedDeviceDescriptor *descriptor) noexcept {
    if (!descriptor || descriptor->struct_size < sizeof(*descriptor))
        return vernon::Result<VernonRhiDevice, vernon::RhiError>{vernon::err(invalidArgument("create_cuda_device"))};
    auto reserved = cudaDevices().reserve();
    if (reserved.isErr())
        return vernon::Result<VernonRhiDevice, vernon::RhiError>{vernon::err(std::move(reserved).error())};
    CudaDevice &device = reserved.value().device();
    auto owner = reserved.value().retainOwner();
    if (owner.isErr())
        return vernon::Result<VernonRhiDevice, vernon::RhiError>{vernon::err(std::move(owner).error())};
    auto commandState = vernon::rhi::createCommandDeviceState(std::move(owner).value());
    if (commandState.isErr())
        return vernon::Result<VernonRhiDevice, vernon::RhiError>{vernon::err(std::move(commandState).error())};
    device.commandState = std::move(commandState).value();
    const auto status = device.state.initialize(descriptor->device_index);
    if (status != vernon::rhi::cuda::kSuccess)
        return vernon::Result<VernonRhiDevice, vernon::RhiError>{
            vernon::err(backendFailure(device, status, "cuDevicePrimaryCtxRetain"))};
    auto published = cudaDevices().publish(std::move(reserved).value());
    if (published.isErr())
        return vernon::Result<VernonRhiDevice, vernon::RhiError>{vernon::err(std::move(published).error())};
    const auto registryHandle = published.value();
    return vernon::Result<VernonRhiDevice, vernon::RhiError>{
        vernon::ok(VernonRhiDevice{registryHandle.index | cudaDeviceBit, registryHandle.generation})};
}

vernon::Result<void, vernon::RhiError> destroyDeviceResult(VernonRhiDevice handle) noexcept {
    if ((handle.index & cudaDeviceBit) == 0)
        return vernon::Result<void, vernon::RhiError>{
            vernon::err(invalidArgument("destroy_cuda_device", handle.index, handle.generation))};
    return cudaDevices().remove({handle.index & ~cudaDeviceBit, handle.generation}, teardownCudaDevice);
}

} // namespace

namespace vernon::rhi::cuda_api {

Result<VernonRhiDevice, RhiError> createOwnedDevice(const VernonRhiOwnedDeviceDescriptor *descriptor) noexcept {
    return createOwnedDeviceResult(descriptor);
}

Result<void, RhiError> destroyDevice(VernonRhiDevice handle) noexcept { return destroyDeviceResult(handle); }

Result<vernon::rhi::CommandDeviceStateRef, RhiError> commandState(VernonRhiDevice handle) noexcept {
    auto anchored = lookupCudaDevice(handle);
    if (anchored.isErr())
        return Result<vernon::rhi::CommandDeviceStateRef, RhiError>{err(std::move(anchored).error())};
    return anchored.value().device().commandState.retain();
}

Option<VernonStringView> lastError(VernonRhiDevice handle) {
    auto anchored = lookupCudaDevice(handle);
    if (anchored.isErr())
        return Option<VernonStringView>{};
    CudaDevice &device = anchored.value().device();
    std::lock_guard<std::mutex> guard(device.mutex);
    return Option<VernonStringView>{some(VernonStringView{device.error.data(), device.error.size()})};
}

Result<void, RhiError> synchronize(VernonRhiDevice handle) noexcept {
    auto anchored = lookupCudaDevice(handle);
    if (anchored.isErr())
        return Result<void, RhiError>{err(std::move(anchored).error())};
    CudaDevice &device = anchored.value().device();
    std::lock_guard<std::mutex> guard(device.mutex);
    const auto status = device.state.synchronize();
    if (status == vernon::rhi::cuda::kSuccess)
        return Result<void, RhiError>{ok()};
    return Result<void, RhiError>{err(backendFailure(device, status, "cuStreamSynchronize"))};
}

Result<VernonRhiBuffer, RhiError> createBuffer(VernonRhiDevice handle, const VernonRhiBufferDescriptor *descriptor) {
    if (!descriptor || descriptor->struct_size < sizeof(*descriptor) || descriptor->size == 0 ||
        descriptor->size > (std::numeric_limits<size_t>::max)() ||
        descriptor->memory_class > VERNON_RHI_MEMORY_READBACK)
        return Result<VernonRhiBuffer, RhiError>{
            err(invalidArgument("create_cuda_buffer", descriptor ? descriptor->size : 0))};

    auto anchored = lookupCudaDevice(handle);
    if (anchored.isErr())
        return Result<VernonRhiBuffer, RhiError>{err(std::move(anchored).error())};
    CudaDevice &device = anchored.value().device();
    auto owner = anchored.value().retainOwner();
    if (owner.isErr())
        return Result<VernonRhiBuffer, RhiError>{err(std::move(owner).error())};
    auto creation = ResourceCreationReservation::create(owner.value());
    if (creation.isErr())
        return Result<VernonRhiBuffer, RhiError>{err(std::move(creation).error())};

    std::lock_guard<std::mutex> creationGuard(device.creationMutex);
    CudaBufferSlot *slot = nullptr;
    uint32_t index = 0;
    for (;;) {
        slot = findBufferSlot(device, index);
        if (!slot)
            break;
        if (!slot->lifecycle.snapshot().occupied) {
            std::lock_guard<std::mutex> nativeGuard(device.mutex);
            if (!slot->pointer)
                break;
        }
        ++index;
    }
    if (!slot) {
        auto lifecycle = BufferLifecycle::create(index);
        if (lifecycle.isErr())
            return Result<VernonRhiBuffer, RhiError>{err(std::move(lifecycle).error())};
        std::lock_guard<std::mutex> slotGuard(device.slotMutex);
        auto appended = device.buffers.emplace("allocate_cuda_resource_slot", std::move(lifecycle).value());
        if (appended.isErr())
            return Result<VernonRhiBuffer, RhiError>{err(std::move(appended).error())};
        slot = appended.value();
    }

    {
        std::lock_guard<std::mutex> nativeGuard(device.mutex);
        const auto status = device.state.allocate(slot->pointer, static_cast<size_t>(descriptor->size));
        if (status != vernon::rhi::cuda::kSuccess)
            return Result<VernonRhiBuffer, RhiError>{err(backendFailure(device, status, "cuMemAlloc"))};
        slot->descriptor = *descriptor;
    }

    auto published = slot->lifecycle.publish(std::move(creation).value(), &device, teardownCudaBuffer);
    if (published.isErr()) {
        std::lock_guard<std::mutex> nativeGuard(device.mutex);
        const auto status = device.state.free(slot->pointer);
        if (status != vernon::rhi::cuda::kSuccess)
            return Result<VernonRhiBuffer, RhiError>{err(backendFailure(device, status, "cuMemFree"))};
        slot->pointer = 0;
        slot->descriptor = {};
        return Result<VernonRhiBuffer, RhiError>{err(std::move(published).error())};
    }
    const BufferHandle buffer = published.value();
    return Result<VernonRhiBuffer, RhiError>{ok(VernonRhiBuffer{buffer.index, buffer.generation})};
}

Result<void, RhiError> uploadBuffer(VernonRhiDevice handle, VernonRhiBuffer buffer, uint64_t offset, const void *source,
                                    uint64_t size) noexcept {
    if (!source || size > (std::numeric_limits<size_t>::max)())
        return Result<void, RhiError>{err(invalidArgument("upload_cuda_buffer", size))};
    auto anchored = lookupCudaDevice(handle);
    if (anchored.isErr())
        return Result<void, RhiError>{err(std::move(anchored).error())};
    CudaDevice &device = anchored.value().device();
    auto pinned = pinBuffer(device, {buffer.index, buffer.generation});
    if (pinned.isErr())
        return Result<void, RhiError>{err(std::move(pinned).error())};
    std::lock_guard<std::mutex> guard(device.mutex);
    CudaBufferSlot &slot = *pinned.value().slot;
    if (offset > slot.descriptor.size || size > slot.descriptor.size - offset)
        return Result<void, RhiError>{err(invalidArgument("upload_cuda_buffer", offset))};
    const auto status = device.state.upload(slot.pointer + offset, source, static_cast<size_t>(size));
    return status == vernon::rhi::cuda::kSuccess
               ? Result<void, RhiError>{ok()}
               : Result<void, RhiError>{err(backendFailure(device, status, "cuMemcpyHtoDAsync"))};
}

Result<void, RhiError> uploadBufferRanges(VernonRhiDevice handle, VernonRhiBuffer buffer,
                                          const VernonRhiBufferUploadRange *ranges, size_t rangeCount) noexcept {
    if (!ranges || rangeCount == 0)
        return Result<void, RhiError>{err(invalidArgument("upload_cuda_buffer_ranges", rangeCount))};
    auto anchored = lookupCudaDevice(handle);
    if (anchored.isErr())
        return Result<void, RhiError>{err(std::move(anchored).error())};
    CudaDevice &device = anchored.value().device();
    auto pinned = pinBuffer(device, {buffer.index, buffer.generation});
    if (pinned.isErr())
        return Result<void, RhiError>{err(std::move(pinned).error())};
    std::lock_guard<std::mutex> guard(device.mutex);
    CudaBufferSlot &slot = *pinned.value().slot;
    size_t stagingSize = 0;
    for (size_t index = 0; index < rangeCount; ++index) {
        const VernonRhiBufferUploadRange &range = ranges[index];
        if (!range.source || range.size == 0 || range.size > (std::numeric_limits<size_t>::max)() ||
            range.offset > slot.descriptor.size || range.size > slot.descriptor.size - range.offset ||
            static_cast<size_t>(range.size) > (std::numeric_limits<size_t>::max)() - stagingSize)
            return Result<void, RhiError>{err(invalidArgument("upload_cuda_buffer_ranges", index))};
        stagingSize += static_cast<size_t>(range.size);
    }
    const auto status = device.state.uploadRanges(slot.pointer, ranges, rangeCount);
    return status == vernon::rhi::cuda::kSuccess
               ? Result<void, RhiError>{ok()}
               : Result<void, RhiError>{err(backendFailure(device, status, "batched cuMemcpyHtoDAsync"))};
}

Result<void, RhiError> downloadBuffer(VernonRhiDevice handle, VernonRhiBuffer buffer, uint64_t offset,
                                      void *destination, uint64_t size) noexcept {
    if (!destination || size > (std::numeric_limits<size_t>::max)())
        return Result<void, RhiError>{err(invalidArgument("download_cuda_buffer", size))};
    auto anchored = lookupCudaDevice(handle);
    if (anchored.isErr())
        return Result<void, RhiError>{err(std::move(anchored).error())};
    CudaDevice &device = anchored.value().device();
    auto pinned = pinBuffer(device, {buffer.index, buffer.generation});
    if (pinned.isErr())
        return Result<void, RhiError>{err(std::move(pinned).error())};
    std::lock_guard<std::mutex> guard(device.mutex);
    CudaBufferSlot &slot = *pinned.value().slot;
    if (offset > slot.descriptor.size || size > slot.descriptor.size - offset)
        return Result<void, RhiError>{err(invalidArgument("download_cuda_buffer", offset))};
    const auto status = device.state.download(destination, slot.pointer + offset, static_cast<size_t>(size));
    return status == vernon::rhi::cuda::kSuccess
               ? Result<void, RhiError>{ok()}
               : Result<void, RhiError>{err(backendFailure(device, status, "cuMemcpyDtoHAsync"))};
}

Result<void, RhiError> downloadBufferRanges(VernonRhiDevice handle, VernonRhiBuffer buffer,
                                            const VernonRhiBufferDownloadRange *ranges, size_t rangeCount) {
    if (!ranges || rangeCount == 0)
        return Result<void, RhiError>{err(invalidArgument("download_cuda_buffer_ranges", rangeCount))};
    auto anchored = lookupCudaDevice(handle);
    if (anchored.isErr())
        return Result<void, RhiError>{err(std::move(anchored).error())};
    CudaDevice &device = anchored.value().device();
    auto pinned = pinBuffer(device, {buffer.index, buffer.generation});
    if (pinned.isErr())
        return Result<void, RhiError>{err(std::move(pinned).error())};
    std::lock_guard<std::mutex> guard(device.mutex);
    CudaBufferSlot &slot = *pinned.value().slot;
    for (size_t index = 0; index < rangeCount; ++index) {
        const VernonRhiBufferDownloadRange &range = ranges[index];
        if (!range.destination || range.size == 0 || range.size > (std::numeric_limits<size_t>::max)() ||
            range.offset > slot.descriptor.size || range.size > slot.descriptor.size - range.offset)
            return Result<void, RhiError>{err(invalidArgument("download_cuda_buffer_ranges", index))};
    }
    const auto status = device.state.downloadRanges(slot.pointer, ranges, rangeCount);
    return status == vernon::rhi::cuda::kSuccess
               ? Result<void, RhiError>{ok()}
               : Result<void, RhiError>{err(backendFailure(device, status, "batched cuMemcpyDtoHAsync"))};
}

Result<bool, RhiError> isBufferValid(VernonRhiDevice handle, VernonRhiBuffer buffer) {
    auto anchored = lookupCudaDevice(handle);
    if (anchored.isErr())
        return Result<bool, RhiError>{err(std::move(anchored).error())};
    auto pinned = pinBuffer(anchored.value().device(), {buffer.index, buffer.generation});
    if (pinned.isErr())
        return Result<bool, RhiError>{err(std::move(pinned).error())};
    return Result<bool, RhiError>{ok(true)};
}

Result<void, RhiError> destroyBuffer(VernonRhiDevice handle, VernonRhiBuffer buffer) noexcept {
    auto anchored = lookupCudaDevice(handle);
    if (anchored.isErr())
        return Result<void, RhiError>{err(std::move(anchored).error())};
    CudaBufferSlot *slot = findBufferSlot(anchored.value().device(), buffer.index);
    if (!slot)
        return Result<void, RhiError>{err(invalidArgument("destroy_cuda_buffer", buffer.generation, buffer.index))};
    return slot->lifecycle.destroyPublic({buffer.index, buffer.generation});
}

Result<void *, RhiError> getBufferNativeHandle(VernonRhiDevice handle, VernonRhiBuffer buffer) {
    auto anchored = lookupCudaDevice(handle);
    if (anchored.isErr())
        return Result<void *, RhiError>{err(std::move(anchored).error())};
    CudaDevice &device = anchored.value().device();
    auto pinned = pinBuffer(device, {buffer.index, buffer.generation});
    if (pinned.isErr())
        return Result<void *, RhiError>{err(std::move(pinned).error())};
    std::lock_guard<std::mutex> guard(device.mutex);
    return Result<void *, RhiError>{ok(reinterpret_cast<void *>(static_cast<uintptr_t>(pinned.value().slot->pointer)))};
}

bool ownsDevice(VernonRhiDevice handle) { return lookupCudaDevice(handle).isOk(); }

Result<void *, RhiError> deviceStateForBackend(VernonRhiDevice handle) {
    auto anchored = lookupCudaDevice(handle);
    if (anchored.isErr())
        return Result<void *, RhiError>{err(std::move(anchored).error())};
    return Result<void *, RhiError>{ok(static_cast<void *>(&anchored.value().device().state))};
}

Result<CommandRecording, RhiError> beginCommands(VernonRhiDevice handle) noexcept {
    auto anchored = lookupCudaDevice(handle);
    if (anchored.isErr())
        return Result<CommandRecording, RhiError>{err(std::move(anchored).error())};
    return Result<CommandRecording, RhiError>{
        ok(CommandRecording{reinterpret_cast<uintptr_t>(&anchored.value().device().state), VERNON_RHI_BACKEND_CUDA})};
}

Result<CommandSubmission, RhiError> submitCommands(VernonRhiDevice handle, uint64_t native, bool) noexcept {
    auto anchored = lookupCudaDevice(handle);
    if (anchored.isErr() || !native)
        return Result<CommandSubmission, RhiError>{err(invalidArgument("submit_cuda_commands", native, handle.index))};
    CudaDevice &device = anchored.value().device();
    std::lock_guard<std::mutex> guard(device.mutex);
    const auto status = device.state.synchronize();
    if (status != vernon::rhi::cuda::kSuccess)
        return Result<CommandSubmission, RhiError>{err(backendFailure(device, status, "cuStreamSynchronize"))};
    return Result<CommandSubmission, RhiError>{ok(CommandSubmission{true, false})};
}

Result<void, RhiError> recordBarriers(VernonRhiDevice handle, uint64_t, uint64_t native, const VernonRhiBarrier *,
                                      size_t) noexcept {
    if (lookupCudaDevice(handle).isErr() || !native)
        return Result<void, RhiError>{err(invalidArgument("record_cuda_barriers", native, handle.index))};
    return Result<void, RhiError>{ok()};
}

Result<void, RhiError> recordBufferCopy(VernonRhiDevice handle, uint64_t native, VernonRhiBuffer source,
                                        uint64_t sourceOffset, VernonRhiBuffer destination, uint64_t destinationOffset,
                                        uint64_t size) noexcept {
    auto anchored = lookupCudaDevice(handle);
    if (anchored.isErr() || !size)
        return Result<void, RhiError>{err(invalidArgument("record_cuda_buffer_copy", native, handle.index))};
    CudaDevice &device = anchored.value().device();
    if (native != reinterpret_cast<uintptr_t>(&device.state))
        return Result<void, RhiError>{err(invalidArgument("record_cuda_buffer_copy", native, handle.index))};
    auto sourcePinned = pinBuffer(device, {source.index, source.generation});
    if (sourcePinned.isErr())
        return Result<void, RhiError>{err(std::move(sourcePinned).error())};
    auto destinationPinned = pinBuffer(device, {destination.index, destination.generation});
    if (destinationPinned.isErr())
        return Result<void, RhiError>{err(std::move(destinationPinned).error())};
    std::lock_guard<std::mutex> guard(device.mutex);
    CudaBufferSlot &sourceSlot = *sourcePinned.value().slot;
    CudaBufferSlot &destinationSlot = *destinationPinned.value().slot;
    if (sourceOffset > sourceSlot.descriptor.size || size > sourceSlot.descriptor.size - sourceOffset ||
        destinationOffset > destinationSlot.descriptor.size ||
        size > destinationSlot.descriptor.size - destinationOffset)
        return Result<void, RhiError>{err(invalidArgument("record_cuda_buffer_copy", size, 0))};
    const auto status =
        device.state.copy(destinationSlot.pointer + destinationOffset, sourceSlot.pointer + sourceOffset, size);
    if (status == vernon::rhi::cuda::kSuccess)
        return Result<void, RhiError>{ok()};
    return Result<void, RhiError>{err(backendFailure(device, status, "cuMemcpyDtoDAsync"))};
}

Result<uint64_t, RhiError> bufferResource(VernonRhiDevice handle, VernonRhiBuffer buffer) noexcept {
    auto anchored = lookupCudaDevice(handle);
    if (anchored.isErr())
        return Result<uint64_t, RhiError>{err(std::move(anchored).error())};
    auto pinned = pinBuffer(anchored.value().device(), {buffer.index, buffer.generation});
    if (pinned.isErr())
        return Result<uint64_t, RhiError>{err(std::move(pinned).error())};
    return Result<uint64_t, RhiError>{ok(resourceKey(buffer))};
}

Result<RetainedRhiResourceLease, RhiError> retainResource(VernonRhiDevice handle, ResourceKind kind,
                                                          uint64_t key) noexcept {
    if (kind != ResourceKind::Buffer)
        return Result<RetainedRhiResourceLease, RhiError>{
            err(unsupported("retain_cuda_resource", key, static_cast<uint32_t>(kind)))};
    auto anchored = lookupCudaDevice(handle);
    if (anchored.isErr())
        return Result<RetainedRhiResourceLease, RhiError>{err(std::move(anchored).error())};
    auto decoded = decodeBufferKey(key, "retain_cuda_resource");
    if (decoded.isErr())
        return Result<RetainedRhiResourceLease, RhiError>{err(std::move(decoded).error())};
    CudaBufferSlot *slot = findBufferSlot(anchored.value().device(), decoded.value().index);
    if (!slot)
        return Result<RetainedRhiResourceLease, RhiError>{err(invalidArgument("retain_cuda_resource", key))};
    auto retained = slot->lifecycle.retain(decoded.value());
    if (retained.isErr())
        return Result<RetainedRhiResourceLease, RhiError>{err(std::move(retained).error())};
    return Result<RetainedRhiResourceLease, RhiError>{ok(RetainedRhiResourceLease{std::move(retained).value()})};
}

Result<uint64_t, RhiError> resolveResource(VernonRhiDevice handle, ResourceKind kind, uint64_t key) noexcept {
    if (kind != ResourceKind::Buffer)
        return Result<uint64_t, RhiError>{err(unsupported("resolve_cuda_resource", key, static_cast<uint32_t>(kind)))};
    auto anchored = lookupCudaDevice(handle);
    if (anchored.isErr())
        return Result<uint64_t, RhiError>{err(std::move(anchored).error())};
    auto decoded = decodeBufferKey(key, "resolve_cuda_resource");
    if (decoded.isErr())
        return Result<uint64_t, RhiError>{err(std::move(decoded).error())};
    CudaDevice &device = anchored.value().device();
    CudaBufferSlot *slot = findBufferSlot(device, decoded.value().index);
    if (!slot)
        return Result<uint64_t, RhiError>{err(invalidArgument("resolve_cuda_resource", key))};
    std::lock_guard<std::mutex> guard(device.mutex);
    if (!slot->pointer)
        return Result<uint64_t, RhiError>{
            err(RhiError{RhiErrorCode::LifecycleFailure, {"resolve_cuda_resource", key, decoded.value().index}})};
    return Result<uint64_t, RhiError>{ok(static_cast<uint64_t>(slot->pointer))};
}

} // namespace vernon::rhi::cuda_api

const vernon::rhi::BackendDispatch &vernon::rhi::cudaBackendDispatch() {
    using namespace cuda_api;
    static const BackendDispatch dispatch = [] {
        BackendDispatch result{};
        result.backend = VERNON_RHI_BACKEND_CUDA;
        result.ownsDevice = ownsDevice;
        result.createOwnedDevice = createOwnedDevice;
        result.destroyDevice = destroyDevice;
        result.lastError = lastError;
        result.synchronize = synchronize;
        result.deviceState = deviceStateForBackend;
        result.commandState = commandState;
        result.createBuffer = createBuffer;
        result.uploadBuffer = uploadBuffer;
        result.uploadBufferRanges = uploadBufferRanges;
        result.downloadBufferRanges = downloadBufferRanges;
        result.downloadBuffer = downloadBuffer;
        result.destroyBuffer = destroyBuffer;
        result.isBufferValid = isBufferValid;
        result.getBufferNativeHandle = getBufferNativeHandle;
        result.bufferResource = bufferResource;
        result.retainResource = retainResource;
        result.resolveRetainedResource = resolveResource;
        result.beginCommands = beginCommands;
        result.submitCommands = submitCommands;
        result.recordBarriers.emplace(recordBarriers);
        result.recordBufferCopy.emplace(recordBufferCopy);
        return result;
    }();
    return dispatch;
}
