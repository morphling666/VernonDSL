#include "VernonTextureTypes.h"
#include "backend_dispatch.h"
#include "image_data_layout.h"
#include "image_descriptor_validation.h"
#include "metal_backend.h"
#include "sampler_filter.h"

#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <unordered_map>
#include <utility>

namespace vernon::rhi {
uint32_t metalTexturePixelFormat(VernonTextureFormat format);
bool deviceHasActiveCommandEncoder(VernonRhiDevice device);
} // namespace vernon::rhi

namespace {

using BufferHandle = vernon::rhi::ResourceHandle<vernon::rhi::BufferResourceTag>;
using ImageHandle = vernon::rhi::ResourceHandle<vernon::rhi::ImageResourceTag>;
using ImageViewHandle = vernon::rhi::ResourceHandle<vernon::rhi::ImageViewResourceTag>;
using SamplerHandle = vernon::rhi::ResourceHandle<vernon::rhi::SamplerResourceTag>;

template <typename Tag, typename Native> struct MetalResourcePayload {
    using Lifecycle = vernon::rhi::ResourceLifecycleSlot<Tag>;

    explicit MetalResourcePayload(Lifecycle slot) noexcept : lifecycle(std::move(slot)) {}
    MetalResourcePayload(const MetalResourcePayload &) = delete;
    MetalResourcePayload &operator=(const MetalResourcePayload &) = delete;

    Lifecycle lifecycle;
    Native native{};
};

struct MetalBufferPayload : MetalResourcePayload<vernon::rhi::BufferResourceTag, vernon::rhi::metal::Buffer> {
    using MetalResourcePayload::MetalResourcePayload;
    VernonRhiBufferDescriptor descriptor{};
};

struct MetalImagePayload : MetalResourcePayload<vernon::rhi::ImageResourceTag, vernon::rhi::metal::Image> {
    using MetalResourcePayload::MetalResourcePayload;
    VernonRhiImageDescriptor descriptor{};
};

struct MetalImageViewPayload : MetalResourcePayload<vernon::rhi::ImageViewResourceTag, vernon::rhi::metal::ImageView> {
    using MetalResourcePayload::MetalResourcePayload;
    VernonRhiImageViewDescriptor descriptor{};
    VernonRhiImageDescriptor imageDescriptor{};
    uint64_t parentKey{};
    vernon::Option<vernon::rhi::RetainedResourceLease<vernon::rhi::ImageResourceTag>> imageLease;
};

using MetalSamplerPayload = MetalResourcePayload<vernon::rhi::SamplerResourceTag, vernon::rhi::metal::Sampler>;

template <typename Payload> using MetalPayloadSlots = vernon::rhi::StableResourceSlotContainer<Payload>;

struct MetalDevice {
    MetalDevice() noexcept = default;
    MetalDevice(const MetalDevice &) = delete;
    MetalDevice &operator=(const MetalDevice &) = delete;
    MetalDevice(MetalDevice &&) = delete;
    MetalDevice &operator=(MetalDevice &&) = delete;

    vernon::rhi::metal::DeviceState state;
    MetalPayloadSlots<MetalBufferPayload> buffers;
    MetalPayloadSlots<MetalImagePayload> images;
    MetalPayloadSlots<MetalImageViewPayload> imageViews;
    MetalPayloadSlots<MetalSamplerPayload> samplers;
    vernon::rhi::CommandDeviceStateRef commandState;
    std::string error;
    std::mutex mutex;
    std::mutex slotMutex;
    std::mutex creationMutex;
};

constexpr uint32_t metalDeviceBit = uint32_t{1} << 28;
constexpr size_t maximumMetalDevices = 256;
vernon::rhi::DeviceRegistry<MetalDevice, maximumMetalDevices> metalDevices;

template <typename Handle> uint64_t resourceKey(Handle handle) {
    return (static_cast<uint64_t>(handle.generation) << 32) | (static_cast<uint64_t>(handle.index) + 1);
}

template <typename Handle>
vernon::Result<Handle, vernon::RhiError> decodeResourceKey(uint64_t key, const char *operation) noexcept {
    const uint64_t encodedIndex = key & UINT32_MAX;
    const uint32_t generation = static_cast<uint32_t>(key >> 32);
    if (!encodedIndex || !generation)
        return vernon::Result<Handle, vernon::RhiError>{
            vernon::err(vernon::RhiError{vernon::RhiErrorCode::InvalidArgument, {operation, key, 0}})};
    return vernon::Result<Handle, vernon::RhiError>{
        vernon::ok(Handle{static_cast<uint32_t>(encodedIndex - 1), generation})};
}

bool supportsBufferState(const VernonRhiBufferDescriptor &descriptor, VernonRhiResourceState state) {
    switch (state) {
    case VERNON_RHI_STATE_UNDEFINED:
    case VERNON_RHI_STATE_COMMON:
        return true;
    case VERNON_RHI_STATE_TRANSFER_SOURCE:
        return (descriptor.usage & VERNON_RHI_BUFFER_TRANSFER_SOURCE) != 0;
    case VERNON_RHI_STATE_TRANSFER_DESTINATION:
        return (descriptor.usage & VERNON_RHI_BUFFER_TRANSFER_DESTINATION) != 0;
    case VERNON_RHI_STATE_SHADER_READ:
        return (descriptor.usage & (VERNON_RHI_BUFFER_UNIFORM | VERNON_RHI_BUFFER_STORAGE)) != 0;
    case VERNON_RHI_STATE_SHADER_WRITE:
        return (descriptor.usage & VERNON_RHI_BUFFER_STORAGE) != 0;
    default:
        return false;
    }
}

bool supportsImageState(const VernonRhiImageDescriptor &descriptor, VernonRhiResourceState state) {
    switch (state) {
    case VERNON_RHI_STATE_UNDEFINED:
    case VERNON_RHI_STATE_COMMON:
        return true;
    case VERNON_RHI_STATE_TRANSFER_SOURCE:
        return (descriptor.usage & VERNON_RHI_IMAGE_TRANSFER_SOURCE) != 0;
    case VERNON_RHI_STATE_TRANSFER_DESTINATION:
        return (descriptor.usage & VERNON_RHI_IMAGE_TRANSFER_DESTINATION) != 0;
    case VERNON_RHI_STATE_SHADER_READ:
        return (descriptor.usage & (VERNON_RHI_IMAGE_SAMPLED | VERNON_RHI_IMAGE_STORAGE)) != 0;
    case VERNON_RHI_STATE_SHADER_WRITE:
        return (descriptor.usage & VERNON_RHI_IMAGE_STORAGE) != 0;
    case VERNON_RHI_STATE_COLOR_ATTACHMENT:
        return (descriptor.usage & VERNON_RHI_IMAGE_COLOR_ATTACHMENT) != 0;
    case VERNON_RHI_STATE_DEPTH_STENCIL_ATTACHMENT:
        return (descriptor.usage & VERNON_RHI_IMAGE_DEPTH_STENCIL_ATTACHMENT) != 0;
    default:
        return false;
    }
}

vernon::rhi::DeviceRegistryHandle decodeDeviceHandle(VernonRhiDevice handle) noexcept {
    return {handle.index & ~metalDeviceBit, handle.generation};
}

VernonRhiDevice encodeDeviceHandle(vernon::rhi::DeviceRegistryHandle handle) noexcept {
    return {handle.index | metalDeviceBit, handle.generation};
}

using MetalDeviceAnchor = vernon::rhi::DeviceRegistryAnchor<MetalDevice>;

vernon::Result<MetalDeviceAnchor, vernon::RhiError> lookupMetalDevice(VernonRhiDevice handle) noexcept {
    if ((handle.index & metalDeviceBit) == 0)
        return vernon::Result<MetalDeviceAnchor, vernon::RhiError>{vernon::err(
            vernon::RhiError{vernon::RhiErrorCode::InvalidArgument, {"lookup_metal_device", handle.index, 0}})};
    return metalDevices.lookup(decodeDeviceHandle(handle));
}

template <typename Payload>
vernon::Result<Payload *, vernon::RhiError> payloadAt(MetalPayloadSlots<Payload> &payloads, uint32_t index,
                                                      const char *operation) noexcept {
    Payload *payload = payloads.get(index);
    if (!payload)
        return vernon::Result<Payload *, vernon::RhiError>{
            vernon::err(vernon::RhiError{vernon::RhiErrorCode::InvalidArgument, {operation, index, 0}})};
    return vernon::Result<Payload *, vernon::RhiError>{vernon::ok(payload)};
}

template <typename Payload>
vernon::Result<Payload *, vernon::RhiError> reservePayload(MetalDevice &device, MetalPayloadSlots<Payload> &payloads,
                                                           const char *operation) noexcept {
    for (std::size_t index = 0; index < payloads.size(); ++index)
        if (!payloads[index].lifecycle.snapshot().occupied)
            return vernon::Result<Payload *, vernon::RhiError>{vernon::ok(&payloads[index])};
    if (payloads.size() >= UINT32_MAX)
        return vernon::Result<Payload *, vernon::RhiError>{
            vernon::err(vernon::RhiError{vernon::RhiErrorCode::ResourceExhausted, {operation, payloads.size(), 0}})};
    using Lifecycle = typename Payload::Lifecycle;
    auto lifecycle = Lifecycle::create(static_cast<uint32_t>(payloads.size()));
    if (lifecycle.isErr())
        return vernon::Result<Payload *, vernon::RhiError>{vernon::err(std::move(lifecycle).error())};
    std::lock_guard<std::mutex> guard(device.slotMutex);
    return payloads.emplace(operation, std::move(lifecycle).value());
}

template <typename Payload, typename Handle>
vernon::Result<std::pair<Payload *, vernon::OperationPin>, vernon::RhiError>
pinPayload(MetalDevice &device, MetalPayloadSlots<Payload> &payloads, Handle handle, const char *operation) noexcept {
    Payload *payload = nullptr;
    {
        std::lock_guard<std::mutex> guard(device.slotMutex);
        auto found = payloadAt(payloads, handle.index, operation);
        if (found.isErr())
            return vernon::Result<std::pair<Payload *, vernon::OperationPin>, vernon::RhiError>{
                vernon::err(std::move(found).error())};
        payload = found.value();
    }
    auto pin = payload->lifecycle.pin(handle);
    if (pin.isErr())
        return vernon::Result<std::pair<Payload *, vernon::OperationPin>, vernon::RhiError>{
            vernon::err(std::move(pin).error())};
    return vernon::Result<std::pair<Payload *, vernon::OperationPin>, vernon::RhiError>{
        vernon::ok(std::make_pair(payload, std::move(pin).value()))};
}

vernon::Result<void, vernon::RhiError> teardownBuffer(void *context, BufferHandle handle) noexcept {
    auto &device = *static_cast<MetalDevice *>(context);
    MetalBufferPayload *payload = nullptr;
    {
        std::lock_guard<std::mutex> guard(device.slotMutex);
        if (handle.index >= device.buffers.size())
            return vernon::Result<void, vernon::RhiError>{vernon::err(
                vernon::RhiError{vernon::RhiErrorCode::InvalidArgument, {"teardown_metal_buffer", handle.index, 0}})};
        payload = &device.buffers[handle.index];
    }
    std::lock_guard<std::mutex> guard(device.mutex);
    device.state.destroyBuffer(payload->native);
    payload->descriptor = {};
    return vernon::Result<void, vernon::RhiError>{vernon::ok()};
}

vernon::Result<void, vernon::RhiError> teardownImage(void *context, ImageHandle handle) noexcept {
    auto &device = *static_cast<MetalDevice *>(context);
    MetalImagePayload *payload = nullptr;
    {
        std::lock_guard<std::mutex> guard(device.slotMutex);
        if (handle.index >= device.images.size())
            return vernon::Result<void, vernon::RhiError>{vernon::err(
                vernon::RhiError{vernon::RhiErrorCode::InvalidArgument, {"teardown_metal_image", handle.index, 0}})};
        payload = &device.images[handle.index];
    }
    std::lock_guard<std::mutex> guard(device.mutex);
    device.state.destroyImage(payload->native);
    payload->descriptor = {};
    return vernon::Result<void, vernon::RhiError>{vernon::ok()};
}

vernon::Result<void, vernon::RhiError> teardownImageView(void *context, ImageViewHandle handle) noexcept {
    auto &device = *static_cast<MetalDevice *>(context);
    MetalImageViewPayload *payload = nullptr;
    {
        std::lock_guard<std::mutex> guard(device.slotMutex);
        if (handle.index >= device.imageViews.size())
            return vernon::Result<void, vernon::RhiError>{vernon::err(vernon::RhiError{
                vernon::RhiErrorCode::InvalidArgument, {"teardown_metal_image_view", handle.index, 0}})};
        payload = &device.imageViews[handle.index];
    }
    if (!payload->imageLease)
        return vernon::Result<void, vernon::RhiError>{
            vernon::err(vernon::RhiError{vernon::RhiErrorCode::LifecycleFailure,
                                         {"release_metal_image_view_parent", handle.generation, handle.index}})};
    auto prepared = payload->imageLease.value().prepareRelease();
    if (prepared.isErr())
        return vernon::Result<void, vernon::RhiError>{vernon::err(std::move(prepared).error())};
    {
        std::lock_guard<std::mutex> guard(device.mutex);
        device.state.destroyImageView(payload->native);
        payload->descriptor = {};
        payload->imageDescriptor = {};
        payload->parentKey = 0;
    }
    if (prepared.value().commit().isErr())
        vernon::resultContractViolation();
    payload->imageLease.reset();
    return vernon::Result<void, vernon::RhiError>{vernon::ok()};
}

vernon::Result<void, vernon::RhiError> teardownSampler(void *context, SamplerHandle handle) noexcept {
    auto &device = *static_cast<MetalDevice *>(context);
    MetalSamplerPayload *payload = nullptr;
    {
        std::lock_guard<std::mutex> guard(device.slotMutex);
        if (handle.index >= device.samplers.size())
            return vernon::Result<void, vernon::RhiError>{vernon::err(
                vernon::RhiError{vernon::RhiErrorCode::InvalidArgument, {"teardown_metal_sampler", handle.index, 0}})};
        payload = &device.samplers[handle.index];
    }
    std::lock_guard<std::mutex> guard(device.mutex);
    device.state.destroySampler(payload->native);
    return vernon::Result<void, vernon::RhiError>{vernon::ok()};
}

vernon::Result<void, vernon::RhiError> teardownDevice(MetalDevice &device) noexcept {
    std::lock_guard<std::mutex> guard(device.mutex);
    device.state.shutdown();
    return vernon::Result<void, vernon::RhiError>{vernon::ok()};
}

} // namespace

namespace vernon::rhi::metal_api {

Result<VernonRhiDevice, RhiError> createOwnedDevice(const VernonRhiOwnedDeviceDescriptor *descriptor) {
    if (!descriptor || descriptor->struct_size < sizeof(*descriptor))
        return Result<VernonRhiDevice, RhiError>{
            err(RhiError{RhiErrorCode::InvalidArgument, {"create_metal_device", 0, 0}})};
    auto reservation = metalDevices.reserve();
    if (reservation.isErr())
        return Result<VernonRhiDevice, RhiError>{err(std::move(reservation).error())};
    MetalDevice &device = reservation.value().device();
    auto owner = reservation.value().retainOwner();
    if (owner.isErr())
        return Result<VernonRhiDevice, RhiError>{err(std::move(owner).error())};
    auto commandState = vernon::rhi::createCommandDeviceState(std::move(owner).value());
    if (commandState.isErr())
        return Result<VernonRhiDevice, RhiError>{err(std::move(commandState).error())};
    device.commandState = std::move(commandState).value();
    if (!device.state.initialize(descriptor->device_index, device.error))
        return Result<VernonRhiDevice, RhiError>{
            err(RhiError{RhiErrorCode::BackendFailure, {"initialize_metal_device", descriptor->device_index, 0}})};
    auto published = metalDevices.publish(std::move(reservation).value());
    if (published.isErr()) {
        device.state.shutdown();
        return Result<VernonRhiDevice, RhiError>{err(std::move(published).error())};
    }
    return Result<VernonRhiDevice, RhiError>{ok(encodeDeviceHandle(published.value()))};
}

vernon::Result<void, vernon::RhiError> destroyDevice(VernonRhiDevice handle) noexcept {
    if ((handle.index & metalDeviceBit) == 0)
        return vernon::Result<void, vernon::RhiError>{vernon::err(
            vernon::RhiError{vernon::RhiErrorCode::InvalidArgument, {"destroy_metal_device", handle.index, 0}})};
    return metalDevices.remove(decodeDeviceHandle(handle), teardownDevice);
}

vernon::Result<vernon::rhi::CommandDeviceStateRef, vernon::RhiError> commandState(VernonRhiDevice handle) noexcept {
    auto anchor = lookupMetalDevice(handle);
    if (anchor.isErr())
        return vernon::Result<vernon::rhi::CommandDeviceStateRef, vernon::RhiError>{
            vernon::err(std::move(anchor).error())};
    return anchor.value().device().commandState.retain();
}

bool ownsDevice(VernonRhiDevice handle) { return lookupMetalDevice(handle).isOk(); }

Option<VernonStringView> lastError(VernonRhiDevice handle) {
    auto anchor = lookupMetalDevice(handle);
    if (anchor.isErr())
        return Option<VernonStringView>{};
    MetalDevice &device = anchor.value().device();
    std::lock_guard<std::mutex> guard(device.mutex);
    return Option<VernonStringView>{some(VernonStringView{device.error.data(), device.error.size()})};
}

Result<void *, RhiError> deviceStateForBackend(VernonRhiDevice handle) {
    auto anchor = lookupMetalDevice(handle);
    if (anchor.isErr())
        return Result<void *, RhiError>{err(std::move(anchor).error())};
    return Result<void *, RhiError>{ok(static_cast<void *>(&anchor.value().device().state))};
}

Result<VernonRhiBuffer, RhiError> createBuffer(VernonRhiDevice handle, const VernonRhiBufferDescriptor *descriptor) {
    auto anchor = lookupMetalDevice(handle);
    if (anchor.isErr())
        return Result<VernonRhiBuffer, RhiError>{err(std::move(anchor).error())};
    constexpr uint32_t allUsages = VERNON_RHI_BUFFER_TRANSFER_SOURCE | VERNON_RHI_BUFFER_TRANSFER_DESTINATION |
                                   VERNON_RHI_BUFFER_UNIFORM | VERNON_RHI_BUFFER_STORAGE | VERNON_RHI_BUFFER_VERTEX |
                                   VERNON_RHI_BUFFER_INDEX | VERNON_RHI_BUFFER_INDIRECT;
    if (!descriptor || descriptor->struct_size < sizeof(*descriptor) || descriptor->size == 0 ||
        descriptor->size > (std::numeric_limits<size_t>::max)() ||
        descriptor->memory_class > VERNON_RHI_MEMORY_READBACK || (descriptor->usage & ~allUsages) != 0)
        return Result<VernonRhiBuffer, RhiError>{err(
            RhiError{RhiErrorCode::InvalidArgument, {"create_metal_buffer", descriptor ? descriptor->size : 0, 0}})};
    auto owner = anchor.value().retainOwner();
    if (owner.isErr())
        return Result<VernonRhiBuffer, RhiError>{err(std::move(owner).error())};
    auto reservation = ResourceCreationReservation::create(owner.value());
    if (reservation.isErr())
        return Result<VernonRhiBuffer, RhiError>{err(std::move(reservation).error())};
    MetalDevice &device = anchor.value().device();
    std::lock_guard<std::mutex> creationGuard(device.creationMutex);
    auto reserved = reservePayload(device, device.buffers, "reserve_metal_buffer");
    if (reserved.isErr())
        return Result<VernonRhiBuffer, RhiError>{err(std::move(reserved).error())};
    MetalBufferPayload &payload = *reserved.value();
    {
        std::lock_guard<std::mutex> nativeGuard(device.mutex);
        if (!device.state.createBuffer(payload.native, *descriptor, device.error))
            return Result<VernonRhiBuffer, RhiError>{
                err(RhiError{RhiErrorCode::BackendFailure, {"create_metal_buffer", descriptor->size, 0}})};
        payload.descriptor = *descriptor;
    }
    auto published = payload.lifecycle.publish(std::move(reservation).value(), &device, teardownBuffer);
    if (published.isErr()) {
        std::lock_guard<std::mutex> nativeGuard(device.mutex);
        device.state.destroyBuffer(payload.native);
        payload.descriptor = {};
        return Result<VernonRhiBuffer, RhiError>{err(std::move(published).error())};
    }
    return Result<VernonRhiBuffer, RhiError>{
        ok(VernonRhiBuffer{published.value().index, published.value().generation})};
}

Result<void, RhiError> uploadBuffer(VernonRhiDevice handle, VernonRhiBuffer buffer, uint64_t offset, const void *source,
                                    uint64_t size) {
    auto anchor = lookupMetalDevice(handle);
    if (anchor.isErr())
        return Result<void, RhiError>{err(std::move(anchor).error())};
    if (!source || size == 0 || size > (std::numeric_limits<size_t>::max)())
        return Result<void, RhiError>{err(RhiError{RhiErrorCode::InvalidArgument, {"upload_metal_buffer", size, 0}})};
    MetalDevice &device = anchor.value().device();
    auto pinned = pinPayload(device, device.buffers, BufferHandle{buffer.index, buffer.generation}, "upload_buffer");
    if (pinned.isErr())
        return Result<void, RhiError>{err(std::move(pinned).error())};
    MetalBufferPayload &payload = *pinned.value().first;
    std::lock_guard<std::mutex> guard(device.mutex);
    if (offset > payload.descriptor.size || size > payload.descriptor.size - offset)
        return Result<void, RhiError>{err(RhiError{RhiErrorCode::InvalidArgument, {"upload_metal_buffer", offset, 0}})};
    if (payload.descriptor.memory_class != VERNON_RHI_MEMORY_UPLOAD &&
        vernon::rhi::deviceHasActiveCommandEncoder(handle)) {
        device.error = "device-level Metal upload cannot execute while a command encoder is recording";
        return Result<void, RhiError>{err(RhiError{RhiErrorCode::InvalidArgument, {"upload_metal_buffer", offset, 0}})};
    }
    return device.state.uploadBuffer(payload.native, offset, source, size, device.error)
               ? Result<void, RhiError>{ok()}
               : Result<void, RhiError>{
                     err(RhiError{RhiErrorCode::BackendFailure, {"upload_metal_buffer", offset, 0}})};
}

Result<void, RhiError> uploadBufferRanges(VernonRhiDevice handle, VernonRhiBuffer buffer,
                                          const VernonRhiBufferUploadRange *ranges, size_t rangeCount) {
    auto anchor = lookupMetalDevice(handle);
    if (anchor.isErr())
        return Result<void, RhiError>{err(std::move(anchor).error())};
    if (!ranges || rangeCount == 0)
        return Result<void, RhiError>{
            err(RhiError{RhiErrorCode::InvalidArgument, {"upload_metal_buffer_ranges", rangeCount, 0}})};
    MetalDevice &device = anchor.value().device();
    auto pinned =
        pinPayload(device, device.buffers, BufferHandle{buffer.index, buffer.generation}, "upload_buffer_ranges");
    if (pinned.isErr())
        return Result<void, RhiError>{err(std::move(pinned).error())};
    MetalBufferPayload &payload = *pinned.value().first;
    std::lock_guard<std::mutex> guard(device.mutex);
    if (payload.descriptor.memory_class != VERNON_RHI_MEMORY_UPLOAD &&
        vernon::rhi::deviceHasActiveCommandEncoder(handle)) {
        device.error = "device-level Metal upload cannot execute while a command encoder is recording";
        return Result<void, RhiError>{
            err(RhiError{RhiErrorCode::InvalidArgument, {"upload_metal_buffer_ranges", rangeCount, 0}})};
    }
    for (size_t index = 0; index < rangeCount; ++index) {
        const VernonRhiBufferUploadRange &range = ranges[index];
        if (!range.source || range.size == 0 || range.size > (std::numeric_limits<size_t>::max)() ||
            range.offset > payload.descriptor.size || range.size > payload.descriptor.size - range.offset)
            return Result<void, RhiError>{
                err(RhiError{RhiErrorCode::InvalidArgument, {"upload_metal_buffer_ranges", index, 0}})};
    }
    return device.state.uploadBufferRanges(payload.native, ranges, rangeCount, device.error)
               ? Result<void, RhiError>{ok()}
               : Result<void, RhiError>{
                     err(RhiError{RhiErrorCode::BackendFailure, {"upload_metal_buffer_ranges", rangeCount, 0}})};
}

Result<void, RhiError> downloadBuffer(VernonRhiDevice handle, VernonRhiBuffer buffer, uint64_t offset,
                                      void *destination, uint64_t size) {
    auto anchor = lookupMetalDevice(handle);
    if (anchor.isErr())
        return Result<void, RhiError>{err(std::move(anchor).error())};
    if (!destination || size == 0 || size > (std::numeric_limits<size_t>::max)())
        return Result<void, RhiError>{err(RhiError{RhiErrorCode::InvalidArgument, {"download_metal_buffer", size, 0}})};
    MetalDevice &device = anchor.value().device();
    auto pinned = pinPayload(device, device.buffers, BufferHandle{buffer.index, buffer.generation}, "download_buffer");
    if (pinned.isErr())
        return Result<void, RhiError>{err(std::move(pinned).error())};
    MetalBufferPayload &payload = *pinned.value().first;
    std::lock_guard<std::mutex> guard(device.mutex);
    if (offset > payload.descriptor.size || size > payload.descriptor.size - offset)
        return Result<void, RhiError>{
            err(RhiError{RhiErrorCode::InvalidArgument, {"download_metal_buffer", offset, 0}})};
    return device.state.downloadBuffer(payload.native, offset, destination, size, device.error)
               ? Result<void, RhiError>{ok()}
               : Result<void, RhiError>{
                     err(RhiError{RhiErrorCode::BackendFailure, {"download_metal_buffer", offset, 0}})};
}

Result<void, RhiError> downloadBufferRanges(VernonRhiDevice handle, VernonRhiBuffer buffer,
                                            const VernonRhiBufferDownloadRange *ranges, size_t rangeCount) {
    auto anchor = lookupMetalDevice(handle);
    if (anchor.isErr() || !ranges || rangeCount == 0)
        return anchor.isErr() ? Result<void, RhiError>{err(std::move(anchor).error())}
                              : Result<void, RhiError>{err(RhiError{RhiErrorCode::InvalidArgument,
                                                                    {"download_metal_buffer_ranges", rangeCount, 0}})};
    MetalDevice &device = anchor.value().device();
    auto pinned =
        pinPayload(device, device.buffers, BufferHandle{buffer.index, buffer.generation}, "download_buffer_ranges");
    if (pinned.isErr())
        return Result<void, RhiError>{err(std::move(pinned).error())};
    MetalBufferPayload &payload = *pinned.value().first;
    std::lock_guard<std::mutex> guard(device.mutex);
    for (size_t index = 0; index < rangeCount; ++index) {
        const VernonRhiBufferDownloadRange &range = ranges[index];
        if (!range.destination || range.size == 0 || range.size > (std::numeric_limits<size_t>::max)() ||
            range.offset > payload.descriptor.size || range.size > payload.descriptor.size - range.offset)
            return Result<void, RhiError>{
                err(RhiError{RhiErrorCode::InvalidArgument, {"download_metal_buffer_ranges", index, 0}})};
    }
    return device.state.downloadBufferRanges(payload.native, ranges, rangeCount, device.error)
               ? Result<void, RhiError>{ok()}
               : Result<void, RhiError>{
                     err(RhiError{RhiErrorCode::BackendFailure, {"download_metal_buffer_ranges", rangeCount, 0}})};
}

Result<void, RhiError> destroyBuffer(VernonRhiDevice handle, VernonRhiBuffer buffer) {
    auto anchor = lookupMetalDevice(handle);
    if (anchor.isErr())
        return Result<void, RhiError>{err(std::move(anchor).error())};
    MetalDevice &device = anchor.value().device();
    MetalBufferPayload *payload = nullptr;
    {
        std::lock_guard<std::mutex> guard(device.slotMutex);
        auto found = payloadAt(device.buffers, buffer.index, "destroy_buffer");
        if (found.isErr())
            return Result<void, RhiError>{err(std::move(found).error())};
        payload = found.value();
    }
    auto destroyed = payload->lifecycle.destroyPublic(BufferHandle{buffer.index, buffer.generation});
    return destroyed;
}

Result<bool, RhiError> isBufferValid(VernonRhiDevice handle, VernonRhiBuffer buffer) {
    auto anchor = lookupMetalDevice(handle);
    if (anchor.isErr())
        return Result<bool, RhiError>{err(std::move(anchor).error())};
    MetalDevice &device = anchor.value().device();
    auto pinned = pinPayload(device, device.buffers, BufferHandle{buffer.index, buffer.generation}, "is_buffer_valid");
    if (pinned.isErr())
        return Result<bool, RhiError>{err(std::move(pinned).error())};
    return Result<bool, RhiError>{ok(true)};
}

Result<void *, RhiError> getBufferNativeHandle(VernonRhiDevice handle, VernonRhiBuffer buffer) {
    auto anchor = lookupMetalDevice(handle);
    if (anchor.isErr())
        return Result<void *, RhiError>{err(std::move(anchor).error())};
    MetalDevice &device = anchor.value().device();
    auto pinned =
        pinPayload(device, device.buffers, BufferHandle{buffer.index, buffer.generation}, "get_buffer_native_handle");
    if (pinned.isErr())
        return Result<void *, RhiError>{err(std::move(pinned).error())};
    std::lock_guard<std::mutex> guard(device.mutex);
    return Result<void *, RhiError>{ok((__bridge void *)pinned.value().first->native.buffer)};
}

Result<VernonRhiImage, RhiError> createImage(VernonRhiDevice handle, const VernonRhiImageDescriptor *descriptor) {
    auto anchor = lookupMetalDevice(handle);
    if (anchor.isErr())
        return Result<VernonRhiImage, RhiError>{err(std::move(anchor).error())};
    if (vernon::formats::hasStencil(descriptor->format) && (descriptor->usage & VERNON_RHI_IMAGE_STORAGE))
        return Result<VernonRhiImage, RhiError>{
            err(RhiError{RhiErrorCode::Unsupported, {"create_metal_image", descriptor->format, descriptor->usage}})};
    if (vernon::rhi::metal::pixelFormat(descriptor->format) == MTLPixelFormatInvalid)
        return Result<VernonRhiImage, RhiError>{
            err(RhiError{RhiErrorCode::Unsupported, {"create_metal_image", descriptor->format, 0}})};
    auto owner = anchor.value().retainOwner();
    if (owner.isErr())
        return Result<VernonRhiImage, RhiError>{err(std::move(owner).error())};
    auto reservation = ResourceCreationReservation::create(owner.value());
    if (reservation.isErr())
        return Result<VernonRhiImage, RhiError>{err(std::move(reservation).error())};
    MetalDevice &device = anchor.value().device();
    std::lock_guard<std::mutex> creationGuard(device.creationMutex);
    auto reserved = reservePayload(device, device.images, "reserve_metal_image");
    if (reserved.isErr())
        return Result<VernonRhiImage, RhiError>{err(std::move(reserved).error())};
    MetalImagePayload &payload = *reserved.value();
    {
        std::lock_guard<std::mutex> nativeGuard(device.mutex);
        if (!device.state.createImage(payload.native, *descriptor, device.error))
            return Result<VernonRhiImage, RhiError>{
                err(RhiError{RhiErrorCode::BackendFailure, {"create_metal_image", descriptor->format, 0}})};
        payload.descriptor = *descriptor;
    }
    auto published = payload.lifecycle.publish(std::move(reservation).value(), &device, teardownImage);
    if (published.isErr()) {
        std::lock_guard<std::mutex> nativeGuard(device.mutex);
        device.state.destroyImage(payload.native);
        payload.descriptor = {};
        return Result<VernonRhiImage, RhiError>{err(std::move(published).error())};
    }
    return Result<VernonRhiImage, RhiError>{ok(VernonRhiImage{published.value().index, published.value().generation})};
}

Result<VernonRhiImageView, RhiError> createImageView(VernonRhiDevice handle,
                                                     const VernonRhiImageViewDescriptor *descriptor) {
    auto anchor = lookupMetalDevice(handle);
    if (anchor.isErr())
        return Result<VernonRhiImageView, RhiError>{err(std::move(anchor).error())};
    if (!descriptor || descriptor->struct_size < sizeof(*descriptor) ||
        vernon::rhi::metal::pixelFormat(descriptor->format) == MTLPixelFormatInvalid ||
        descriptor->mip_level_count == 0 || descriptor->array_layer_count == 0 || descriptor->aspects == 0)
        return Result<VernonRhiImageView, RhiError>{
            err(RhiError{RhiErrorCode::InvalidArgument, {"create_metal_image_view", 0, 0}})};
    auto owner = anchor.value().retainOwner();
    if (owner.isErr())
        return Result<VernonRhiImageView, RhiError>{err(std::move(owner).error())};
    auto reservation = ResourceCreationReservation::create(owner.value());
    if (reservation.isErr())
        return Result<VernonRhiImageView, RhiError>{err(std::move(reservation).error())};
    MetalDevice &device = anchor.value().device();
    const ImageHandle imageHandle{descriptor->image.index, descriptor->image.generation};
    MetalImagePayload *image = nullptr;
    {
        std::lock_guard<std::mutex> guard(device.slotMutex);
        auto found = payloadAt(device.images, imageHandle.index, "create_image_view");
        if (found.isErr())
            return Result<VernonRhiImageView, RhiError>{err(std::move(found).error())};
        image = found.value();
    }
    auto imageLease = image->lifecycle.retain(imageHandle);
    if (imageLease.isErr())
        return Result<VernonRhiImageView, RhiError>{err(std::move(imageLease).error())};
    auto imagePin = image->lifecycle.pin(imageHandle);
    if (imagePin.isErr())
        return Result<VernonRhiImageView, RhiError>{err(std::move(imagePin).error())};
    std::unique_lock<std::mutex> creationGuard(device.creationMutex);
    {
        std::lock_guard<std::mutex> nativeGuard(device.mutex);
        if (!vernon::rhi::validImageViewDescriptor(image->descriptor, *descriptor))
            return Result<VernonRhiImageView, RhiError>{
                err(RhiError{RhiErrorCode::InvalidArgument, {"create_metal_image_view", 0, 0}})};
    }
    auto reserved = reservePayload(device, device.imageViews, "reserve_metal_image_view");
    if (reserved.isErr())
        return Result<VernonRhiImageView, RhiError>{err(std::move(reserved).error())};
    MetalImageViewPayload &payload = *reserved.value();
    {
        std::lock_guard<std::mutex> nativeGuard(device.mutex);
        if (!device.state.createImageView(payload.native, image->native, *descriptor, device.error))
            return Result<VernonRhiImageView, RhiError>{
                err(RhiError{RhiErrorCode::Unsupported, {"create_metal_image_view", descriptor->format, 0}})};
        payload.descriptor = *descriptor;
        payload.imageDescriptor = image->descriptor;
        payload.parentKey = resourceKey(imageHandle);
        payload.imageLease.emplace(std::move(imageLease).value());
    }
    auto published = payload.lifecycle.publish(std::move(reservation).value(), &device, teardownImageView);
    if (published.isErr()) {
        RhiError publicationError = std::move(published).error();
        vernon::Option<vernon::rhi::RetainedResourceLease<vernon::rhi::ImageResourceTag>> rollbackLease;
        {
            std::lock_guard<std::mutex> nativeGuard(device.mutex);
            device.state.destroyImageView(payload.native);
            payload.descriptor = {};
            payload.imageDescriptor = {};
            payload.parentKey = 0;
            auto taken = payload.imageLease.take();
            if (taken)
                rollbackLease.emplace(std::move(taken).value());
        }
        creationGuard.unlock();
        if (rollbackLease) {
            auto released = rollbackLease.value().release();
            if (released.isErr())
                return Result<VernonRhiImageView, RhiError>{err(std::move(released).error())};
        }
        return Result<VernonRhiImageView, RhiError>{err(std::move(publicationError))};
    }
    return Result<VernonRhiImageView, RhiError>{
        ok(VernonRhiImageView{published.value().index, published.value().generation})};
}

Result<void, RhiError> destroyImageView(VernonRhiDevice handle, VernonRhiImageView imageView) {
    auto anchor = lookupMetalDevice(handle);
    if (anchor.isErr())
        return Result<void, RhiError>{err(std::move(anchor).error())};
    MetalDevice &device = anchor.value().device();
    MetalImageViewPayload *payload = nullptr;
    {
        std::lock_guard<std::mutex> guard(device.slotMutex);
        auto found = payloadAt(device.imageViews, imageView.index, "destroy_image_view");
        if (found.isErr())
            return Result<void, RhiError>{err(std::move(found).error())};
        payload = found.value();
    }
    return payload->lifecycle.destroyPublic(ImageViewHandle{imageView.index, imageView.generation});
}

Result<uint64_t, RhiError> getImageViewNativeHandle(VernonRhiDevice handle, VernonRhiImageView imageView) {
    auto anchor = lookupMetalDevice(handle);
    if (anchor.isErr())
        return Result<uint64_t, RhiError>{err(std::move(anchor).error())};
    MetalDevice &device = anchor.value().device();
    auto pinned = pinPayload(device, device.imageViews, ImageViewHandle{imageView.index, imageView.generation},
                             "get_image_view_native_handle");
    if (pinned.isErr())
        return Result<uint64_t, RhiError>{err(std::move(pinned).error())};
    std::lock_guard<std::mutex> guard(device.mutex);
    return Result<uint64_t, RhiError>{
        ok(static_cast<uint64_t>(reinterpret_cast<uintptr_t>((__bridge void *)pinned.value().first->native.texture)))};
}

Result<void, RhiError> uploadImage(VernonRhiDevice handle, VernonRhiImage image,
                                   const VernonRhiImageUploadDescriptor *uploads, size_t uploadCount) {
    auto anchor = lookupMetalDevice(handle);
    if (anchor.isErr())
        return Result<void, RhiError>{err(std::move(anchor).error())};
    if (!uploads || uploadCount == 0)
        return Result<void, RhiError>{
            err(RhiError{RhiErrorCode::InvalidArgument, {"upload_metal_image", uploadCount, 0}})};
    MetalDevice &device = anchor.value().device();
    auto pinned = pinPayload(device, device.images, ImageHandle{image.index, image.generation}, "upload_image");
    if (pinned.isErr())
        return Result<void, RhiError>{err(std::move(pinned).error())};
    MetalImagePayload &payload = *pinned.value().first;
    std::lock_guard<std::mutex> guard(device.mutex);
    for (size_t index = 0; index < uploadCount; ++index) {
        const auto &upload = uploads[index];
        if (upload.struct_size < sizeof(upload) || !upload.data || upload.mip_level >= payload.descriptor.mip_levels)
            return Result<void, RhiError>{
                err(RhiError{RhiErrorCode::InvalidArgument, {"upload_metal_image", index, 0}})};
    }
    return device.state.uploadImage(payload.native, payload.descriptor, uploads, uploadCount, device.error)
               ? Result<void, RhiError>{ok()}
               : Result<void, RhiError>{
                     err(RhiError{RhiErrorCode::Unsupported, {"upload_metal_image", uploadCount, 0}})};
}

Result<void, RhiError> downloadImage(VernonRhiDevice handle, VernonRhiImage image,
                                     const VernonRhiImageDownloadDescriptor *download, void *destination, size_t size) {
    auto anchor = lookupMetalDevice(handle);
    if (anchor.isErr() || !download || download->struct_size < sizeof(*download) || !destination)
        return anchor.isErr() ? Result<void, RhiError>{err(std::move(anchor).error())}
                              : Result<void, RhiError>{
                                    err(RhiError{RhiErrorCode::InvalidArgument, {"download_metal_image", size, 0}})};
    MetalDevice &device = anchor.value().device();
    auto pinned = pinPayload(device, device.images, ImageHandle{image.index, image.generation}, "download_image");
    if (pinned.isErr())
        return Result<void, RhiError>{err(std::move(pinned).error())};
    MetalImagePayload &payload = *pinned.value().first;
    std::lock_guard<std::mutex> guard(device.mutex);
    const auto required = vernon::rhi::imageDownloadByteSize(payload.descriptor, *download);
    if (!required || size != *required)
        return Result<void, RhiError>{err(RhiError{RhiErrorCode::InvalidArgument, {"download_metal_image", size, 0}})};
    return device.state.downloadImage(payload.native, payload.descriptor, *download, destination, size, device.error)
               ? Result<void, RhiError>{ok()}
               : Result<void, RhiError>{err(RhiError{RhiErrorCode::BackendFailure, {"download_metal_image", size, 0}})};
}

Result<void, RhiError> downloadImageBatch(VernonRhiDevice handle, VernonRhiImage image,
                                          const VernonRhiImageDownload *downloads, size_t downloadCount) {
    auto anchor = lookupMetalDevice(handle);
    if (anchor.isErr() || !downloads || downloadCount == 0)
        return anchor.isErr() ? Result<void, RhiError>{err(std::move(anchor).error())}
                              : Result<void, RhiError>{err(RhiError{RhiErrorCode::InvalidArgument,
                                                                    {"download_metal_image_batch", downloadCount, 0}})};
    MetalDevice &device = anchor.value().device();
    auto pinned = pinPayload(device, device.images, ImageHandle{image.index, image.generation}, "download_image_batch");
    if (pinned.isErr())
        return Result<void, RhiError>{err(std::move(pinned).error())};
    MetalImagePayload &payload = *pinned.value().first;
    std::lock_guard<std::mutex> guard(device.mutex);
    for (size_t index = 0; index < downloadCount; ++index) {
        const auto required = vernon::rhi::imageDownloadByteSize(payload.descriptor, downloads[index].descriptor);
        if (!downloads[index].destination || !required || downloads[index].size != *required)
            return Result<void, RhiError>{
                err(RhiError{RhiErrorCode::InvalidArgument, {"download_metal_image_batch", index, 0}})};
    }
    return device.state.downloadImageBatch(payload.native, payload.descriptor, downloads, downloadCount, device.error)
               ? Result<void, RhiError>{ok()}
               : Result<void, RhiError>{
                     err(RhiError{RhiErrorCode::BackendFailure, {"download_metal_image_batch", downloadCount, 0}})};
}

Result<void, RhiError> generateImageMipmaps(VernonRhiDevice handle, VernonRhiImage image) {
    auto anchor = lookupMetalDevice(handle);
    if (anchor.isErr())
        return Result<void, RhiError>{err(std::move(anchor).error())};
    MetalDevice &device = anchor.value().device();
    auto pinned =
        pinPayload(device, device.images, ImageHandle{image.index, image.generation}, "generate_image_mipmaps");
    if (pinned.isErr())
        return Result<void, RhiError>{err(std::move(pinned).error())};
    MetalImagePayload &payload = *pinned.value().first;
    std::lock_guard<std::mutex> guard(device.mutex);
    if (payload.descriptor.mip_levels < 2 || !(payload.descriptor.usage & VERNON_RHI_IMAGE_TRANSFER_SOURCE) ||
        !(payload.descriptor.usage & VERNON_RHI_IMAGE_TRANSFER_DESTINATION) ||
        vernon::formats::hasDepth(payload.descriptor.format))
        return Result<void, RhiError>{err(
            RhiError{RhiErrorCode::InvalidArgument, {"generate_metal_image_mipmaps", image.generation, image.index}})};
    return device.state.generateImageMipmaps(payload.native, payload.descriptor.mip_levels, device.error)
               ? Result<void, RhiError>{ok()}
               : Result<void, RhiError>{err(RhiError{RhiErrorCode::BackendFailure,
                                                     {"generate_metal_image_mipmaps", image.generation, image.index}})};
}

Result<void, RhiError> destroyImage(VernonRhiDevice handle, VernonRhiImage image) {
    auto anchor = lookupMetalDevice(handle);
    if (anchor.isErr())
        return Result<void, RhiError>{err(std::move(anchor).error())};
    MetalDevice &device = anchor.value().device();
    MetalImagePayload *payload = nullptr;
    {
        std::lock_guard<std::mutex> guard(device.slotMutex);
        auto found = payloadAt(device.images, image.index, "destroy_image");
        if (found.isErr())
            return Result<void, RhiError>{err(std::move(found).error())};
        payload = found.value();
    }
    return payload->lifecycle.destroyPublic(ImageHandle{image.index, image.generation});
}

Result<bool, RhiError> isImageValid(VernonRhiDevice handle, VernonRhiImage image) {
    auto anchor = lookupMetalDevice(handle);
    if (anchor.isErr())
        return Result<bool, RhiError>{err(std::move(anchor).error())};
    MetalDevice &device = anchor.value().device();
    auto pinned = pinPayload(device, device.images, ImageHandle{image.index, image.generation}, "is_image_valid");
    if (pinned.isErr())
        return Result<bool, RhiError>{err(std::move(pinned).error())};
    return Result<bool, RhiError>{ok(true)};
}

Result<uint64_t, RhiError> getImageNativeHandle(VernonRhiDevice handle, VernonRhiImage image) {
    auto anchor = lookupMetalDevice(handle);
    if (anchor.isErr())
        return Result<uint64_t, RhiError>{err(std::move(anchor).error())};
    MetalDevice &device = anchor.value().device();
    auto pinned =
        pinPayload(device, device.images, ImageHandle{image.index, image.generation},         "get_image_native_handle");
    if (pinned.isErr())
        return Result<uint64_t, RhiError>{err(std::move(pinned).error())};
    std::lock_guard<std::mutex> guard(device.mutex);
    return Result<uint64_t, RhiError>{
        ok(static_cast<uint64_t>(reinterpret_cast<uintptr_t>((__bridge void *)pinned.value().first->native.texture)))};
}

// Metal reports no per-format features; only 32-bit float color filtering and multisampling vary between devices.
Result<uint32_t, RhiError> formatCapabilities(VernonRhiDevice handle, VernonRhiFormat format) {
    auto anchor = lookupMetalDevice(handle);
    if (anchor.isErr())
        return Result<uint32_t, RhiError>{err(std::move(anchor).error())};
    const vernon::formats::Format *entry = vernon::formats::find(format);
    if (!entry || vernon::rhi::metal::pixelFormat(format) == MTLPixelFormatInvalid)
        return Result<uint32_t, RhiError>{ok(0u)};
    uint32_t result = entry->baseline;
    const bool float32Color = entry->aspects() == VERNON_RHI_IMAGE_ASPECT_COLOR &&
                              entry->componentClass == vernon::formats::ComponentClass::Float &&
                              entry->dataType == VERNON_RHI_IMAGE_DATA_FLOAT32;
    id<MTLDevice> native = anchor.value().device().state.device;
    if (float32Color && !native.supports32BitFloatFiltering)
        result &= ~VERNON_RHI_FORMAT_FILTERABLE;
    if (float32Color && !native.supports32BitMSAA)
        result &= ~(VERNON_RHI_FORMAT_MULTISAMPLE | VERNON_RHI_FORMAT_RESOLVABLE);
    return Result<uint32_t, RhiError>{ok(result)};
}

Result<VernonRhiSampler, RhiError> createSampler(VernonRhiDevice handle, const VernonRhiSamplerDescriptor *descriptor) {
    auto anchor = lookupMetalDevice(handle);
    if (anchor.isErr())
        return Result<VernonRhiSampler, RhiError>{err(std::move(anchor).error())};
    vernon::rhi::SamplerFilter filter;
    if (!descriptor || descriptor->struct_size < sizeof(*descriptor) ||
        !vernon::rhi::decodeSamplerFilter(*descriptor, filter))
        return Result<VernonRhiSampler, RhiError>{
            err(RhiError{RhiErrorCode::InvalidArgument, {"create_metal_sampler", 0, 0}})};
    auto owner = anchor.value().retainOwner();
    if (owner.isErr())
        return Result<VernonRhiSampler, RhiError>{err(std::move(owner).error())};
    auto reservation = ResourceCreationReservation::create(owner.value());
    if (reservation.isErr())
        return Result<VernonRhiSampler, RhiError>{err(std::move(reservation).error())};
    MetalDevice &device = anchor.value().device();
    std::lock_guard<std::mutex> creationGuard(device.creationMutex);
    auto reserved = reservePayload(device, device.samplers, "reserve_metal_sampler");
    if (reserved.isErr())
        return Result<VernonRhiSampler, RhiError>{err(std::move(reserved).error())};
    MetalSamplerPayload &payload = *reserved.value();
    {
        std::lock_guard<std::mutex> nativeGuard(device.mutex);
        if (!device.state.createSampler(payload.native, *descriptor, device.error))
            return Result<VernonRhiSampler, RhiError>{
                err(RhiError{RhiErrorCode::BackendFailure, {"create_metal_sampler", 0, 0}})};
    }
    auto published = payload.lifecycle.publish(std::move(reservation).value(), &device, teardownSampler);
    if (published.isErr()) {
        std::lock_guard<std::mutex> nativeGuard(device.mutex);
        device.state.destroySampler(payload.native);
        return Result<VernonRhiSampler, RhiError>{err(std::move(published).error())};
    }
    return Result<VernonRhiSampler, RhiError>{
        ok(VernonRhiSampler{published.value().index, published.value().generation})};
}

Result<void, RhiError> destroySampler(VernonRhiDevice handle, VernonRhiSampler sampler) {
    auto anchor = lookupMetalDevice(handle);
    if (anchor.isErr())
        return Result<void, RhiError>{err(std::move(anchor).error())};
    MetalDevice &device = anchor.value().device();
    MetalSamplerPayload *payload = nullptr;
    {
        std::lock_guard<std::mutex> guard(device.slotMutex);
        auto found = payloadAt(device.samplers, sampler.index, "destroy_sampler");
        if (found.isErr())
            return Result<void, RhiError>{err(std::move(found).error())};
        payload = found.value();
    }
    return payload->lifecycle.destroyPublic(SamplerHandle{sampler.index, sampler.generation});
}

Result<bool, RhiError> isSamplerValid(VernonRhiDevice handle, VernonRhiSampler sampler) {
    auto anchor = lookupMetalDevice(handle);
    if (anchor.isErr())
        return Result<bool, RhiError>{err(std::move(anchor).error())};
    MetalDevice &device = anchor.value().device();
    auto pinned =
        pinPayload(device, device.samplers, SamplerHandle{sampler.index, sampler.generation}, "is_sampler_valid");
    if (pinned.isErr())
        return Result<bool, RhiError>{err(std::move(pinned).error())};
    return Result<bool, RhiError>{ok(true)};
}

template <typename Payload, typename Handle>
Result<uint64_t, RhiError> publicResourceKey(VernonRhiDevice deviceHandle,
                                             MetalPayloadSlots<Payload> MetalDevice::*member, Handle handle) noexcept {
    auto anchor = lookupMetalDevice(deviceHandle);
    if (anchor.isErr())
        return Result<uint64_t, RhiError>{err(std::move(anchor).error())};
    MetalDevice &device = anchor.value().device();
    auto pinned = pinPayload(device, device.*member, handle, "resource_key");
    if (pinned.isErr())
        return Result<uint64_t, RhiError>{err(std::move(pinned).error())};
    return Result<uint64_t, RhiError>{ok(resourceKey(handle))};
}

Result<uint64_t, RhiError> bufferResource(VernonRhiDevice handle, VernonRhiBuffer buffer) noexcept {
    return publicResourceKey(handle, &MetalDevice::buffers, BufferHandle{buffer.index, buffer.generation});
}

Result<uint64_t, RhiError> imageResource(VernonRhiDevice handle, VernonRhiImage image) noexcept {
    return publicResourceKey(handle, &MetalDevice::images, ImageHandle{image.index, image.generation});
}

Result<uint64_t, RhiError> imageViewResource(VernonRhiDevice handle, VernonRhiImageView view) noexcept {
    return publicResourceKey(handle, &MetalDevice::imageViews, ImageViewHandle{view.index, view.generation});
}

Result<uint64_t, RhiError> samplerResource(VernonRhiDevice handle, VernonRhiSampler sampler) noexcept {
    return publicResourceKey(handle, &MetalDevice::samplers, SamplerHandle{sampler.index, sampler.generation});
}

template <typename Payload, typename Handle>
vernon::Result<typename Payload::Lifecycle::RetainedLease, vernon::RhiError>
retainTypedResource(MetalDevice &device, MetalPayloadSlots<Payload> &payloads, Handle handle,
                    const char *operation) noexcept {
    Payload *payload = nullptr;
    {
        std::lock_guard<std::mutex> guard(device.slotMutex);
        auto found = payloadAt(payloads, handle.index, operation);
        if (found.isErr())
            return vernon::Result<typename Payload::Lifecycle::RetainedLease, vernon::RhiError>{
                vernon::err(std::move(found).error())};
        payload = found.value();
    }
    return payload->lifecycle.retain(handle);
}

Result<RetainedRhiResourceLease, RhiError> retainResource(VernonRhiDevice handle, ResourceKind kind,
                                                          uint64_t key) noexcept {
    auto anchor = lookupMetalDevice(handle);
    if (anchor.isErr())
        return Result<RetainedRhiResourceLease, RhiError>{err(std::move(anchor).error())};
    MetalDevice &device = anchor.value().device();
    switch (kind) {
    case ResourceKind::Buffer: {
        auto decoded = decodeResourceKey<BufferHandle>(key, "retain_metal_buffer");
        if (decoded.isErr())
            return Result<RetainedRhiResourceLease, RhiError>{err(std::move(decoded).error())};
        auto retained = retainTypedResource(device, device.buffers, decoded.value(), "retain_metal_buffer");
        if (retained.isErr())
            return Result<RetainedRhiResourceLease, RhiError>{err(std::move(retained).error())};
        return Result<RetainedRhiResourceLease, RhiError>{ok(RetainedRhiResourceLease{std::move(retained).value()})};
    }
    case ResourceKind::Image: {
        auto decoded = decodeResourceKey<ImageHandle>(key, "retain_metal_image");
        if (decoded.isErr())
            return Result<RetainedRhiResourceLease, RhiError>{err(std::move(decoded).error())};
        auto retained = retainTypedResource(device, device.images, decoded.value(), "retain_metal_image");
        if (retained.isErr())
            return Result<RetainedRhiResourceLease, RhiError>{err(std::move(retained).error())};
        return Result<RetainedRhiResourceLease, RhiError>{ok(RetainedRhiResourceLease{std::move(retained).value()})};
    }
    case ResourceKind::ImageView: {
        auto decoded = decodeResourceKey<ImageViewHandle>(key, "retain_metal_image_view");
        if (decoded.isErr())
            return Result<RetainedRhiResourceLease, RhiError>{err(std::move(decoded).error())};
        auto retained = retainTypedResource(device, device.imageViews, decoded.value(), "retain_metal_image_view");
        if (retained.isErr())
            return Result<RetainedRhiResourceLease, RhiError>{err(std::move(retained).error())};
        return Result<RetainedRhiResourceLease, RhiError>{ok(RetainedRhiResourceLease{std::move(retained).value()})};
    }
    case ResourceKind::Sampler: {
        auto decoded = decodeResourceKey<SamplerHandle>(key, "retain_metal_sampler");
        if (decoded.isErr())
            return Result<RetainedRhiResourceLease, RhiError>{err(std::move(decoded).error())};
        auto retained = retainTypedResource(device, device.samplers, decoded.value(), "retain_metal_sampler");
        if (retained.isErr())
            return Result<RetainedRhiResourceLease, RhiError>{err(std::move(retained).error())};
        return Result<RetainedRhiResourceLease, RhiError>{ok(RetainedRhiResourceLease{std::move(retained).value()})};
    }
    }
    return Result<RetainedRhiResourceLease, RhiError>{
        err(RhiError{RhiErrorCode::InvalidArgument, {"retain_metal_resource", key, static_cast<uint32_t>(kind)}})};
}

Result<uint64_t, RhiError> resolveResource(VernonRhiDevice handle, ResourceKind kind, uint64_t key) noexcept {
    auto anchor = lookupMetalDevice(handle);
    if (anchor.isErr())
        return Result<uint64_t, RhiError>{err(std::move(anchor).error())};
    MetalDevice &device = anchor.value().device();
    switch (kind) {
    case ResourceKind::Buffer: {
        auto decoded = decodeResourceKey<BufferHandle>(key, "resolve_metal_buffer");
        if (decoded.isErr())
            return Result<uint64_t, RhiError>{err(std::move(decoded).error())};
        MetalBufferPayload *payload{};
        {
            std::lock_guard<std::mutex> slots(device.slotMutex);
            if (decoded.value().index >= device.buffers.size())
                return Result<uint64_t, RhiError>{
                    err(RhiError{RhiErrorCode::InvalidArgument, {"resolve_metal_buffer", key, 0}})};
            payload = &device.buffers[decoded.value().index];
        }
        std::lock_guard<std::mutex> guard(device.mutex);
        return Result<uint64_t, RhiError>{
            ok(static_cast<uint64_t>(reinterpret_cast<uintptr_t>((__bridge void *)payload->native.buffer)))};
    }
    case ResourceKind::Image: {
        auto decoded = decodeResourceKey<ImageHandle>(key, "resolve_metal_image");
        if (decoded.isErr())
            return Result<uint64_t, RhiError>{err(std::move(decoded).error())};
        MetalImagePayload *payload{};
        {
            std::lock_guard<std::mutex> slots(device.slotMutex);
            if (decoded.value().index >= device.images.size())
                return Result<uint64_t, RhiError>{
                    err(RhiError{RhiErrorCode::InvalidArgument, {"resolve_metal_image", key, 0}})};
            payload = &device.images[decoded.value().index];
        }
        std::lock_guard<std::mutex> guard(device.mutex);
        return Result<uint64_t, RhiError>{
            ok(static_cast<uint64_t>(reinterpret_cast<uintptr_t>((__bridge void *)payload->native.texture)))};
    }
    case ResourceKind::ImageView: {
        auto decoded = decodeResourceKey<ImageViewHandle>(key, "resolve_metal_image_view");
        if (decoded.isErr())
            return Result<uint64_t, RhiError>{err(std::move(decoded).error())};
        MetalImageViewPayload *payload{};
        {
            std::lock_guard<std::mutex> slots(device.slotMutex);
            if (decoded.value().index >= device.imageViews.size())
                return Result<uint64_t, RhiError>{
                    err(RhiError{RhiErrorCode::InvalidArgument, {"resolve_metal_image_view", key, 0}})};
            payload = &device.imageViews[decoded.value().index];
        }
        std::lock_guard<std::mutex> guard(device.mutex);
        return Result<uint64_t, RhiError>{
            ok(static_cast<uint64_t>(reinterpret_cast<uintptr_t>((__bridge void *)payload->native.texture)))};
    }
    case ResourceKind::Sampler: {
        auto decoded = decodeResourceKey<SamplerHandle>(key, "resolve_metal_sampler");
        if (decoded.isErr())
            return Result<uint64_t, RhiError>{err(std::move(decoded).error())};
        MetalSamplerPayload *payload{};
        {
            std::lock_guard<std::mutex> slots(device.slotMutex);
            if (decoded.value().index >= device.samplers.size())
                return Result<uint64_t, RhiError>{
                    err(RhiError{RhiErrorCode::InvalidArgument, {"resolve_metal_sampler", key, 0}})};
            payload = &device.samplers[decoded.value().index];
        }
        std::lock_guard<std::mutex> guard(device.mutex);
        return Result<uint64_t, RhiError>{
            ok(static_cast<uint64_t>(reinterpret_cast<uintptr_t>((__bridge void *)payload->native.sampler)))};
    }
    }
    return Result<uint64_t, RhiError>{
        err(RhiError{RhiErrorCode::InvalidArgument, {"resolve_metal_resource", key, static_cast<uint32_t>(kind)}})};
}

Result<ImageResourceDescription, RhiError> describeImageResource(VernonRhiDevice handle, uint64_t key) noexcept {
    auto anchor = lookupMetalDevice(handle);
    if (anchor.isErr())
        return Result<ImageResourceDescription, RhiError>{err(std::move(anchor).error())};
    auto decoded = decodeResourceKey<ImageHandle>(key, "describe_metal_image");
    if (decoded.isErr())
        return Result<ImageResourceDescription, RhiError>{err(std::move(decoded).error())};
    MetalDevice &device = anchor.value().device();
    MetalImagePayload *payload{};
    {
        std::lock_guard<std::mutex> slots(device.slotMutex);
        auto found = payloadAt(device.images, decoded.value().index, "describe_metal_image");
        if (found.isErr())
            return Result<ImageResourceDescription, RhiError>{err(std::move(found).error())};
        payload = found.value();
    }
    std::lock_guard<std::mutex> guard(device.mutex);
    return Result<ImageResourceDescription, RhiError>{ok(ImageResourceDescription{payload->descriptor})};
}

Result<ImageViewResourceDescription, RhiError> describeImageViewResource(VernonRhiDevice handle,
                                                                         uint64_t key) noexcept {
    auto anchor = lookupMetalDevice(handle);
    if (anchor.isErr())
        return Result<ImageViewResourceDescription, RhiError>{err(std::move(anchor).error())};
    auto decoded = decodeResourceKey<ImageViewHandle>(key, "describe_metal_image_view");
    if (decoded.isErr())
        return Result<ImageViewResourceDescription, RhiError>{err(std::move(decoded).error())};
    MetalDevice &device = anchor.value().device();
    MetalImageViewPayload *payload{};
    {
        std::lock_guard<std::mutex> slots(device.slotMutex);
        auto found = payloadAt(device.imageViews, decoded.value().index, "describe_metal_image_view");
        if (found.isErr())
            return Result<ImageViewResourceDescription, RhiError>{err(std::move(found).error())};
        payload = found.value();
    }
    std::lock_guard<std::mutex> guard(device.mutex);
    return Result<ImageViewResourceDescription, RhiError>{
        ok(ImageViewResourceDescription{payload->descriptor, payload->imageDescriptor, payload->parentKey})};
}

void abandonCommands(VernonRhiDevice handle, uint64_t native) noexcept {
    @try {
        auto anchor = lookupMetalDevice(handle);
        if (anchor.isErr())
            return;
        MetalDevice &device = anchor.value().device();
        std::lock_guard<std::mutex> guard(device.mutex);
        device.state.abandonCommands(native);
    } @catch (NSException *) {
    }
}

Result<void, RhiError> recordBarriers(VernonRhiDevice handle, uint64_t, uint64_t native,
                                      const VernonRhiBarrier *barriers, size_t barrierCount) noexcept {
    @try {
        try {
            auto anchor = lookupMetalDevice(handle);
            if (anchor.isErr() || !native || (barrierCount && !barriers))
                return anchor.isErr() ? Result<void, RhiError>{err(std::move(anchor).error())}
                                      : Result<void, RhiError>{err(RhiError{RhiErrorCode::InvalidArgument,
                                                                            {"recordBarriers_metal", native, 0}})};
            MetalDevice &device = anchor.value().device();
            for (size_t index = 0; index < barrierCount; ++index) {
                const VernonRhiBarrier &barrier = barriers[index];
                if (barrier.is_image) {
                    auto pinned =
                        pinPayload(device, device.images, ImageHandle{barrier.image.index, barrier.image.generation},
                                   "record_barrier");
                    if (pinned.isErr())
                        return Result<void, RhiError>{err(std::move(pinned).error())};
                    std::lock_guard<std::mutex> guard(device.mutex);
                    if (!supportsImageState(pinned.value().first->descriptor, barrier.old_state) ||
                        !supportsImageState(pinned.value().first->descriptor, barrier.new_state))
                        return Result<void, RhiError>{
                            err(RhiError{RhiErrorCode::Unsupported, {"recordBarriers_metal_image_state", native, 0}})};
                } else {
                    auto pinned =
                        pinPayload(device, device.buffers,
                                   BufferHandle{barrier.buffer.index, barrier.buffer.generation}, "record_barrier");
                    if (pinned.isErr())
                        return Result<void, RhiError>{err(std::move(pinned).error())};
                    std::lock_guard<std::mutex> guard(device.mutex);
                    if (!supportsBufferState(pinned.value().first->descriptor, barrier.old_state) ||
                        !supportsBufferState(pinned.value().first->descriptor, barrier.new_state))
                        return Result<void, RhiError>{
                            err(RhiError{RhiErrorCode::Unsupported, {"recordBarriers_metal_buffer_state", native, 0}})};
                }
            }
            return Result<void, RhiError>{ok()};
        } catch (const std::bad_alloc &) {
            return Result<void, RhiError>{
                err(RhiError{RhiErrorCode::ResourceExhausted, {"recordBarriers_metal", native, 0}})};
        }
    } @catch (NSException *) {
        return Result<void, RhiError>{
            err(RhiError{RhiErrorCode::BackendFailure, {"recordBarriers_metal_exception", native, 0}})};
    }
}

Result<void, RhiError> recordBufferCopy(VernonRhiDevice handle, uint64_t native, VernonRhiBuffer source,
                                        uint64_t sourceOffset, VernonRhiBuffer destination, uint64_t destinationOffset,
                                        uint64_t size) noexcept {
    @try {
        try {
            auto anchor = lookupMetalDevice(handle);
            if (anchor.isErr() || !native || !size)
                return anchor.isErr() ? Result<void, RhiError>{err(std::move(anchor).error())}
                                      : Result<void, RhiError>{err(RhiError{RhiErrorCode::InvalidArgument,
                                                                            {"recordBufferCopy_metal", native, 0}})};
            MetalDevice &device = anchor.value().device();
            auto sourcePin =
                pinPayload(device, device.buffers, BufferHandle{source.index, source.generation}, "record_buffer_copy");
            if (sourcePin.isErr())
                return Result<void, RhiError>{err(std::move(sourcePin).error())};
            auto destinationPin = pinPayload(
                device, device.buffers, BufferHandle{destination.index, destination.generation}, "record_buffer_copy");
            if (destinationPin.isErr())
                return Result<void, RhiError>{err(std::move(destinationPin).error())};
            MetalBufferPayload &sourcePayload = *sourcePin.value().first;
            MetalBufferPayload &destinationPayload = *destinationPin.value().first;
            std::lock_guard<std::mutex> guard(device.mutex);
            if (sourceOffset > sourcePayload.descriptor.size || size > sourcePayload.descriptor.size - sourceOffset ||
                destinationOffset > destinationPayload.descriptor.size ||
                size > destinationPayload.descriptor.size - destinationOffset)
                return Result<void, RhiError>{
                    err(RhiError{RhiErrorCode::InvalidArgument, {"recordBufferCopy_metal", native, 0}})};
            if (!device.state.copyBuffer(native, sourcePayload.native, sourceOffset, destinationPayload.native,
                                         destinationOffset, size, device.error))
                return Result<void, RhiError>{
                    err(RhiError{RhiErrorCode::BackendFailure, {"recordBufferCopy_metal", native, 0}})};
            return Result<void, RhiError>{ok()};
        } catch (const std::bad_alloc &) {
            return Result<void, RhiError>{
                err(RhiError{RhiErrorCode::ResourceExhausted, {"recordBufferCopy_metal", native, 0}})};
        }
    } @catch (NSException *) {
        return Result<void, RhiError>{
            err(RhiError{RhiErrorCode::BackendFailure, {"recordBufferCopy_metal_exception", native, 0}})};
    }
}

Result<void, RhiError> recordImageCopy(VernonRhiDevice handle, uint64_t, uint64_t native, VernonRhiImage source,
                                       VernonRhiImage destination, const VernonRhiImageCopyRegion *regions,
                                       size_t regionCount) noexcept {
    @try {
        try {
            auto anchor = lookupMetalDevice(handle);
            if (anchor.isErr() || !native || !regions || !regionCount)
                return anchor.isErr() ? Result<void, RhiError>{err(std::move(anchor).error())}
                                      : Result<void, RhiError>{err(RhiError{RhiErrorCode::InvalidArgument,
                                                                            {"recordImageCopy_metal", native, 0}})};
            MetalDevice &device = anchor.value().device();
            auto sourcePin =
                pinPayload(device, device.images, ImageHandle{source.index, source.generation}, "record_image_copy");
            if (sourcePin.isErr())
                return Result<void, RhiError>{err(std::move(sourcePin).error())};
            auto destinationPin = pinPayload(
                device, device.images, ImageHandle{destination.index, destination.generation}, "record_image_copy");
            if (destinationPin.isErr())
                return Result<void, RhiError>{err(std::move(destinationPin).error())};
            std::lock_guard<std::mutex> guard(device.mutex);
            if (!device.state.copyImage(native, sourcePin.value().first->native, destinationPin.value().first->native,
                                        regions, regionCount, device.error))
                return Result<void, RhiError>{
                    err(RhiError{RhiErrorCode::BackendFailure, {"recordImageCopy_metal", native, 0}})};
            return Result<void, RhiError>{ok()};
        } catch (const std::bad_alloc &) {
            return Result<void, RhiError>{
                err(RhiError{RhiErrorCode::ResourceExhausted, {"recordImageCopy_metal", native, 0}})};
        }
    } @catch (NSException *) {
        return Result<void, RhiError>{
            err(RhiError{RhiErrorCode::BackendFailure, {"recordImageCopy_metal_exception", native, 0}})};
    }
}

bool renderAreaCoversMetalTexture(id<MTLTexture> texture, const RenderingScope &scope) {
    return scope.x == 0 && scope.y == 0 && scope.width == texture.width && scope.height == texture.height;
}

// Metal has no render area: load and store actions apply to the whole texture. A discard outside the render area
// must not reach its texels, so it becomes a load or store there; a clear has no such equivalent.
bool metalLoadAction(VernonRhiLoadOperation operation, bool covers, MTLLoadAction &action) {
    if (operation == VERNON_RHI_LOAD_CLEAR)
        action = MTLLoadActionClear;
    else
        action = operation == VERNON_RHI_LOAD_DISCARD && covers ? MTLLoadActionDontCare : MTLLoadActionLoad;
    return operation != VERNON_RHI_LOAD_CLEAR || covers;
}

MTLStoreAction metalStoreAction(VernonRhiStoreOperation operation, bool covers) {
    return operation == VERNON_RHI_STORE_DISCARD && covers ? MTLStoreActionDontCare : MTLStoreActionStore;
}

// Metal converts the double components to the attachment's format; a double holds every 32-bit integer exactly.
MTLClearColor metalClearColor(const VernonClearColor &color, VernonRhiFormat format) {
    const vernon::formats::Format *entry = vernon::formats::find(format);
    const auto componentClass = entry ? entry->componentClass : vernon::formats::ComponentClass::Float;
    if (componentClass == vernon::formats::ComponentClass::Uint)
        return MTLClearColorMake(color.u32[0], color.u32[1], color.u32[2], color.u32[3]);
    if (componentClass == vernon::formats::ComponentClass::Sint)
        return MTLClearColorMake(color.i32[0], color.i32[1], color.i32[2], color.i32[3]);
    return MTLClearColorMake(color.f32[0], color.f32[1], color.f32[2], color.f32[3]);
}

// Passes are created with MTLStoreActionUnknown so that ending the scope, or restarting it for a clear, chooses the
// store actions.
id<MTLRenderCommandEncoder> beginMetalPass(uint64_t native, MTLRenderPassDescriptor *pass, const RenderingScope &scope) {
    if (scope.layers > 1)
        pass.renderTargetArrayLength = scope.layers;
    id<MTLCommandBuffer> commandBuffer =
        (__bridge id<MTLCommandBuffer>)(reinterpret_cast<void *>(static_cast<uintptr_t>(native)));
    return [commandBuffer renderCommandEncoderWithDescriptor:pass];
}

void endMetalPass(const vernon::rhi::metal::RenderingState &rendering, const RenderingScope &scope, bool final) {
    for (size_t index = 0; index < scope.colorCount; ++index) {
        const uint32_t location = scope.colors[index].location;
        const bool covers = renderAreaCoversMetalTexture(rendering.colorTextures[location], scope);
        [rendering.encoder setColorStoreAction:final ? metalStoreAction(scope.colors[index].store, covers)
                                                     : MTLStoreActionStore
                                       atIndex:location];
    }
    if (scope.hasDepth) {
        const bool covers = renderAreaCoversMetalTexture(rendering.depthStencilTexture, scope);
        [rendering.encoder
            setDepthStoreAction:final ? metalStoreAction(scope.depth.depthStore, covers) : MTLStoreActionStore];
        if (scope.depth.stencil)
            [rendering.encoder
                setStencilStoreAction:final ? metalStoreAction(scope.depth.stencilStore, covers) : MTLStoreActionStore];
    }
    for (size_t index = 0; index < rendering.debugGroups.size(); ++index)
        [rendering.encoder popDebugGroup];
    [rendering.encoder endEncoding];
}

// A clear inside the scope restarts the pass with a clear load action on the cleared attachment, so callers require
// the render area to cover it.
bool restartMetalRendering(uint64_t native, vernon::rhi::metal::RenderingState &rendering, const RenderingScope &scope,
                           int32_t colorLocation, const VernonClearColor *clearColor, float clearDepth,
                           uint32_t clearStencil,
                           uint32_t aspects) {
    id<MTLTexture> target = colorLocation >= 0 ? rendering.colorTextures[colorLocation] : rendering.depthStencilTexture;
    if (!rendering.encoder || !target)
        return false;
    endMetalPass(rendering, scope, false);
    rendering.encoder = nil;
    MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
    for (size_t index = 0; index < scope.colorCount; ++index) {
        const uint32_t location = scope.colors[index].location;
        auto *attachment = pass.colorAttachments[location];
        attachment.texture = rendering.colorTextures[location];
        attachment.loadAction = static_cast<int32_t>(location) == colorLocation ? MTLLoadActionClear : MTLLoadActionLoad;
        attachment.storeAction = MTLStoreActionUnknown;
        if (static_cast<int32_t>(location) == colorLocation)
            attachment.clearColor = metalClearColor(*clearColor, scope.colors[index].format);
    }
    if (scope.hasDepth) {
        pass.depthAttachment.texture = rendering.depthStencilTexture;
        pass.depthAttachment.loadAction = aspects & VERNON_RHI_ATTACHMENT_DEPTH ? MTLLoadActionClear : MTLLoadActionLoad;
        pass.depthAttachment.storeAction = MTLStoreActionUnknown;
        pass.depthAttachment.clearDepth = clearDepth;
        if (scope.depth.stencil) {
            pass.stencilAttachment.texture = rendering.depthStencilTexture;
            pass.stencilAttachment.loadAction =
                aspects & VERNON_RHI_ATTACHMENT_STENCIL ? MTLLoadActionClear : MTLLoadActionLoad;
            pass.stencilAttachment.storeAction = MTLStoreActionUnknown;
            pass.stencilAttachment.clearStencil = clearStencil;
        }
    }
    rendering.encoder = beginMetalPass(native, pass, scope);
    for (NSString *group : rendering.debugGroups)
        [rendering.encoder pushDebugGroup:group];
    return rendering.encoder != nil;
}

Result<OpenedRendering, RhiError> beginRendering(VernonRhiDevice handle, uint64_t encoderKey, uint64_t native,
                                                 const RenderingScope &scope, const RenderingRead *, size_t) noexcept {
    @try {
        try {
            auto anchor = lookupMetalDevice(handle);
            if (anchor.isErr())
                return Result<OpenedRendering, RhiError>{err(std::move(anchor).error())};
            if (!native)
                return Result<OpenedRendering, RhiError>{
                    err(RhiError{RhiErrorCode::InvalidArgument, {"beginRendering_metal", native, 0}})};
            auto rendering = std::make_unique<vernon::rhi::metal::RenderingState>();
            OpenedRendering opened{vernon::rhi::CommandRenderingDynamic};
            MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
            for (size_t index = 0; index < scope.attachmentCount(); ++index) {
                auto resolved = retainResolvedCommandResource(handle, encoderKey, ResourceKind::ImageView,
                                                              scope.attachmentView(index));
                if (resolved.isErr())
                    return Result<OpenedRendering, RhiError>{err(std::move(resolved).error())};
                id<MTLTexture> texture =
                    (__bridge id<MTLTexture>)(reinterpret_cast<void *>(static_cast<uintptr_t>(resolved.value())));
                const bool covers = renderAreaCoversMetalTexture(texture, scope);
                MTLLoadAction load{};
                if (index < scope.colorCount) {
                    const auto &attachment = scope.colors[index];
                    auto *target = pass.colorAttachments[attachment.location];
                    if (!metalLoadAction(attachment.load, covers, load))
                        return Result<OpenedRendering, RhiError>{err(RhiError{
                            RhiErrorCode::Unsupported, {"beginRendering_metal_partial_clear", attachment.location, 0}})};
                    target.texture = texture;
                    target.loadAction = load;
                    target.storeAction = MTLStoreActionUnknown;
                    target.clearColor = metalClearColor(attachment.clear, attachment.format);
                    rendering->colorTextures[attachment.location] = texture;
                    opened.colorTargets[attachment.location] = resolved.value();
                    opened.colorResources[attachment.location] = resolved.value();
                    continue;
                }
                if (!metalLoadAction(scope.depth.depthLoad, covers, load))
                    return Result<OpenedRendering, RhiError>{
                        err(RhiError{RhiErrorCode::Unsupported, {"beginRendering_metal_partial_clear", 0, 1}})};
                pass.depthAttachment.texture = texture;
                pass.depthAttachment.loadAction = load;
                pass.depthAttachment.storeAction = MTLStoreActionUnknown;
                pass.depthAttachment.clearDepth = scope.depth.clearDepth;
                if (scope.depth.stencil) {
                    if (!metalLoadAction(scope.depth.stencilLoad, covers, load))
                        return Result<OpenedRendering, RhiError>{
                            err(RhiError{RhiErrorCode::Unsupported, {"beginRendering_metal_partial_clear", 0, 2}})};
                    pass.stencilAttachment.texture = texture;
                    pass.stencilAttachment.loadAction = load;
                    pass.stencilAttachment.storeAction = MTLStoreActionUnknown;
                    pass.stencilAttachment.clearStencil = scope.depth.clearStencil;
                }
                rendering->depthStencilTexture = texture;
                opened.depthTarget = resolved.value();
                opened.depthResource = resolved.value();
            }
            rendering->encoder = beginMetalPass(native, pass, scope);
            if (!rendering->encoder)
                return Result<OpenedRendering, RhiError>{
                    err(RhiError{RhiErrorCode::BackendFailure, {"beginRendering_metal", native, 0}})};
            opened.object = reinterpret_cast<uintptr_t>(rendering.release());
            return Result<OpenedRendering, RhiError>{ok(opened)};
        } catch (const std::bad_alloc &) {
            return Result<OpenedRendering, RhiError>{
                err(RhiError{RhiErrorCode::ResourceExhausted, {"beginRendering_metal", native, 0}})};
        }
    } @catch (NSException *) {
        return Result<OpenedRendering, RhiError>{
            err(RhiError{RhiErrorCode::BackendFailure, {"beginRendering_metal_exception", native, 0}})};
    }
}

Result<void, RhiError> endRendering(VernonRhiDevice handle, uint64_t native, const RenderingScope &scope,
                                    const OpenedRendering &opened) noexcept {
    @try {
        if (opened.kind != vernon::rhi::CommandRenderingDynamic || !opened.object)
            return Result<void, RhiError>{
                err(RhiError{RhiErrorCode::InvalidArgument, {"endRendering_metal", native, opened.kind}})};
        // The scope ends here even when ending it fails, so the state is released on every path.
        std::unique_ptr<vernon::rhi::metal::RenderingState> rendering(
            reinterpret_cast<vernon::rhi::metal::RenderingState *>(static_cast<uintptr_t>(opened.object)));
        auto anchor = lookupMetalDevice(handle);
        if (anchor.isErr())
            return Result<void, RhiError>{err(std::move(anchor).error())};
        // A failed restart leaves no encoder.
        if (rendering->encoder)
            endMetalPass(*rendering, scope, true);
        return Result<void, RhiError>{ok()};
    } @catch (NSException *) {
        return Result<void, RhiError>{
            err(RhiError{RhiErrorCode::BackendFailure, {"endRendering_metal_exception", native, 0}})};
    }
}

Result<void, RhiError> recordDebugCommand(VernonRhiDevice handle, uint64_t native, const OpenedRendering *opened,
                                          DebugCommand command, std::string_view label) noexcept {
    @try {
        try {
            auto anchor = lookupMetalDevice(handle);
            if (anchor.isErr())
                return Result<void, RhiError>{err(std::move(anchor).error())};
            auto *rendering = opened ? reinterpret_cast<vernon::rhi::metal::RenderingState *>(
                                           static_cast<uintptr_t>(opened->object))
                                     : nullptr;
            if (!native || (opened && (opened->kind != vernon::rhi::CommandRenderingDynamic || !rendering)))
                return Result<void, RhiError>{
                    err(RhiError{RhiErrorCode::InvalidArgument, {"record_debug_command_metal", native, 0}})};
            NSString *text = [[NSString alloc] initWithBytes:label.data()
                                                      length:label.size()
                                                    encoding:NSUTF8StringEncoding];
            if (!text)
                text = @"";
            if (rendering) {
                switch (command) {
                case DebugCommand::PushGroup:
                    rendering->debugGroups.push_back(text);
                    [rendering->encoder pushDebugGroup:text];
                    break;
                case DebugCommand::PopGroup:
                    if (!rendering->debugGroups.empty())
                        rendering->debugGroups.pop_back();
                    [rendering->encoder popDebugGroup];
                    break;
                case DebugCommand::InsertMarker:
                    [rendering->encoder insertDebugSignpost:text];
                    break;
                }
                return Result<void, RhiError>{ok()};
            }
            id<MTLCommandBuffer> commandBuffer =
                (__bridge id<MTLCommandBuffer>)(reinterpret_cast<void *>(static_cast<uintptr_t>(native)));
            // A command buffer has no signpost, so a marker outside a scope is an empty group.
            if (command != DebugCommand::PopGroup)
                [commandBuffer pushDebugGroup:text];
            if (command != DebugCommand::PushGroup)
                [commandBuffer popDebugGroup];
            return Result<void, RhiError>{ok()};
        } catch (const std::bad_alloc &) {
            return Result<void, RhiError>{
                err(RhiError{RhiErrorCode::ResourceExhausted, {"record_debug_command_metal", native, 0}})};
        }
    } @catch (NSException *) {
        return Result<void, RhiError>{
            err(RhiError{RhiErrorCode::BackendFailure, {"record_debug_command_metal_exception", native, 0}})};
    }
}

id<MTLCounterSet> metalTimestampCounterSet(id<MTLDevice> device) {
    if (@available(macOS 11.0, iOS 14.0, *)) {
        if (![device supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary])
            return nil;
        for (id<MTLCounterSet> set in device.counterSets)
            if ([set.name isEqualToString:MTLCommonCounterSetTimestamp])
                return set;
    }
    return nil;
}

uint32_t commandCapabilitiesForDevice(VernonRhiDevice handle) {
    constexpr uint32_t capabilities = BackendCommandIndependentRecording | BackendCommandConcurrentSubmission;
    auto anchor = lookupMetalDevice(handle);
    return anchor.isOk() && metalTimestampCounterSet(anchor.value().device().state.device) ? capabilities |
                                                                                                BackendCommandTimers
                                                                                          : capabilities;
}

// Metal samples counters only where an encoder starts or ends, so each timer edge records an empty blit encoder
// sampled at its start; no encoder is open there, since timers begin and end outside rendering scopes. Sample `s` of
// the recording is sample `s % samplesPerBuffer` of buffer `s / samplesPerBuffer`, and timer `t` begins at sample
// `2 * t` and ends at `2 * t + 1`. GPU ticks are scaled to nanoseconds by the CPU and GPU clocks sampled when the
// storage is created and when it is read.
struct MetalTimers {
    static constexpr uint32_t samplesPerBuffer = 64;
    std::vector<id<MTLCounterSampleBuffer>> buffers;
    MTLTimestamp cpuOrigin{};
    MTLTimestamp gpuOrigin{};
};

Result<void, RhiError> recordTimer(VernonRhiDevice handle, uint64_t native, uint64_t &storage, uint32_t timer,
                                   TimerEdge edge) noexcept {
    @try {
        try {
            auto anchor = lookupMetalDevice(handle);
            if (anchor.isErr())
                return Result<void, RhiError>{err(std::move(anchor).error())};
            id<MTLDevice> device = anchor.value().device().state.device;
            id<MTLCounterSet> counters = metalTimestampCounterSet(device);
            if (!native || !counters || timer > UINT32_MAX / 2 - 1)
                return Result<void, RhiError>{
                    err(RhiError{RhiErrorCode::InvalidArgument, {"record_metal_timer", native, timer}})};
            if (@available(macOS 11.0, iOS 14.0, *)) {
                if (!storage) {
                    auto *created = new MetalTimers{};
                    [device sampleTimestamps:&created->cpuOrigin gpuTimestamp:&created->gpuOrigin];
                    storage = reinterpret_cast<uintptr_t>(created);
                }
                auto &timers = *reinterpret_cast<MetalTimers *>(static_cast<uintptr_t>(storage));
                const uint32_t sample = timer * 2 + (edge == TimerEdge::End ? 1 : 0);
                const uint32_t buffer = sample / MetalTimers::samplesPerBuffer;
                if (buffer == timers.buffers.size()) {
                    timers.buffers.reserve(buffer + 1);
                    MTLCounterSampleBufferDescriptor *descriptor = [MTLCounterSampleBufferDescriptor new];
                    descriptor.counterSet = counters;
                    descriptor.storageMode = MTLStorageModeShared;
                    descriptor.sampleCount = MetalTimers::samplesPerBuffer;
                    NSError *error = nil;
                    id<MTLCounterSampleBuffer> created = [device newCounterSampleBufferWithDescriptor:descriptor
                                                                                                error:&error];
                    if (!created)
                        return Result<void, RhiError>{
                            err(RhiError{RhiErrorCode::BackendFailure, {"record_metal_timer", native, timer}})};
                    timers.buffers.push_back(created);
                }
                MTLBlitPassDescriptor *pass = [MTLBlitPassDescriptor blitPassDescriptor];
                pass.sampleBufferAttachments[0].sampleBuffer = timers.buffers[buffer];
                pass.sampleBufferAttachments[0].startOfEncoderSampleIndex = sample % MetalTimers::samplesPerBuffer;
                pass.sampleBufferAttachments[0].endOfEncoderSampleIndex = MTLCounterDontSample;
                id<MTLCommandBuffer> commandBuffer =
                    (__bridge id<MTLCommandBuffer>)(reinterpret_cast<void *>(static_cast<uintptr_t>(native)));
                id<MTLBlitCommandEncoder> encoder = [commandBuffer blitCommandEncoderWithDescriptor:pass];
                [encoder endEncoding];
            }
            return Result<void, RhiError>{ok()};
        } catch (const std::bad_alloc &) {
            return Result<void, RhiError>{
                err(RhiError{RhiErrorCode::ResourceExhausted, {"record_metal_timer", native, timer}})};
        }
    } @catch (NSException *) {
        return Result<void, RhiError>{
            err(RhiError{RhiErrorCode::BackendFailure, {"record_metal_timer_exception", native, timer}})};
    }
}

Result<void, RhiError> readTimers(VernonRhiDevice handle, uint64_t storage, uint32_t count,
                                  VernonRhiTimerSpan *spans) noexcept {
    @try {
        auto anchor = lookupMetalDevice(handle);
        if (anchor.isErr())
            return Result<void, RhiError>{err(std::move(anchor).error())};
        const auto &timers = *reinterpret_cast<const MetalTimers *>(static_cast<uintptr_t>(storage));
        if (@available(macOS 11.0, iOS 14.0, *)) {
            MTLTimestamp cpu{};
            MTLTimestamp gpu{};
            [anchor.value().device().state.device sampleTimestamps:&cpu gpuTimestamp:&gpu];
            const double nanosecondsPerTick = gpu > timers.gpuOrigin ? static_cast<double>(cpu - timers.cpuOrigin) /
                                                                           static_cast<double>(gpu - timers.gpuOrigin)
                                                                     : 1.0;
            for (uint32_t first = 0; first < count * 2; first += MetalTimers::samplesPerBuffer) {
                const uint32_t samples = std::min(count * 2 - first, MetalTimers::samplesPerBuffer);
                const size_t buffer = first / MetalTimers::samplesPerBuffer;
                NSData *data = buffer < timers.buffers.size()
                                   ? [timers.buffers[buffer] resolveCounterRange:NSMakeRange(0, samples)]
                                   : nil;
                if (!data || data.length < samples * sizeof(MTLCounterResultTimestamp))
                    return Result<void, RhiError>{
                        err(RhiError{RhiErrorCode::BackendFailure, {"read_metal_timers", storage, count}})};
                const auto *results = static_cast<const MTLCounterResultTimestamp *>(data.bytes);
                for (uint32_t sample = 0; sample < samples; ++sample) {
                    // A GPU that dropped an empty encoder reports no sample for it.
                    if (results[sample].timestamp == MTLCounterErrorValue)
                        return Result<void, RhiError>{
                            err(RhiError{RhiErrorCode::Unsupported, {"read_metal_timers_dropped", storage, count}})};
                    const auto nanoseconds =
                        static_cast<uint64_t>(static_cast<double>(results[sample].timestamp) * nanosecondsPerTick);
                    VernonRhiTimerSpan &span = spans[(first + sample) / 2];
                    ((first + sample) % 2 ? span.end_nanoseconds : span.begin_nanoseconds) = nanoseconds;
                }
            }
            return Result<void, RhiError>{ok()};
        }
        return Result<void, RhiError>{err(RhiError{RhiErrorCode::Unsupported, {"read_metal_timers", storage, count}})};
    } @catch (NSException *) {
        return Result<void, RhiError>{
            err(RhiError{RhiErrorCode::BackendFailure, {"read_metal_timers_exception", storage, count}})};
    }
}

void releaseTimers(VernonRhiDevice, uint64_t storage) noexcept {
    delete reinterpret_cast<MetalTimers *>(static_cast<uintptr_t>(storage));
}

Result<void, RhiError> clearColor(VernonRhiDevice handle, uint64_t native, const RenderingScope &scope,
                                  const OpenedRendering &opened, uint32_t location,
                                  const VernonClearColor &color) noexcept {
    @try {
        auto anchor = lookupMetalDevice(handle);
        if (anchor.isErr())
            return Result<void, RhiError>{err(std::move(anchor).error())};
        auto *rendering = reinterpret_cast<vernon::rhi::metal::RenderingState *>(static_cast<uintptr_t>(opened.object));
        if (opened.kind != vernon::rhi::CommandRenderingDynamic || !rendering ||
            location >= VERNON_RHI_MAX_COLOR_ATTACHMENTS || !opened.colorTargets[location])
            return Result<void, RhiError>{
                err(RhiError{RhiErrorCode::InvalidArgument, {"clearColor_metal", native, location}})};
        if (!renderAreaCoversMetalTexture(rendering->colorTextures[location], scope))
            return Result<void, RhiError>{
                err(RhiError{RhiErrorCode::Unsupported, {"clearColor_metal_partial", native, location}})};
        if (!restartMetalRendering(native, *rendering, scope, static_cast<int32_t>(location), &color, 1.0f, 0, 0))
            return Result<void, RhiError>{
                err(RhiError{RhiErrorCode::BackendFailure, {"clearColor_metal", native, location}})};
        return Result<void, RhiError>{ok()};
    } @catch (NSException *) {
        return Result<void, RhiError>{
            err(RhiError{RhiErrorCode::BackendFailure, {"clearColor_metal_exception", native, 0}})};
    }
}

Result<void, RhiError> clearDepthStencil(VernonRhiDevice handle, uint64_t native, const RenderingScope &scope,
                                         const OpenedRendering &opened, float depth, uint32_t stencil,
                                         uint32_t aspects) noexcept {
    @try {
        auto anchor = lookupMetalDevice(handle);
        if (anchor.isErr())
            return Result<void, RhiError>{err(std::move(anchor).error())};
        auto *rendering = reinterpret_cast<vernon::rhi::metal::RenderingState *>(static_cast<uintptr_t>(opened.object));
        if (opened.kind != vernon::rhi::CommandRenderingDynamic || !rendering || !opened.depthTarget)
            return Result<void, RhiError>{
                err(RhiError{RhiErrorCode::InvalidArgument, {"clearDepthStencil_metal", native, aspects}})};
        if (!renderAreaCoversMetalTexture(rendering->depthStencilTexture, scope))
            return Result<void, RhiError>{
                err(RhiError{RhiErrorCode::Unsupported, {"clearDepthStencil_metal_partial", native, aspects}})};
        if (!restartMetalRendering(native, *rendering, scope, -1, nullptr, depth, stencil, aspects))
            return Result<void, RhiError>{
                err(RhiError{RhiErrorCode::BackendFailure, {"clearDepthStencil_metal", native, aspects}})};
        return Result<void, RhiError>{ok()};
    } @catch (NSException *) {
        return Result<void, RhiError>{
            err(RhiError{RhiErrorCode::BackendFailure, {"clearDepthStencil_metal_exception", native, 0}})};
    }
}

Result<CommandRecording, RhiError> beginCommands(VernonRhiDevice handle) noexcept {
    @try {
        auto anchor = lookupMetalDevice(handle);
        if (anchor.isErr())
            return Result<CommandRecording, RhiError>{err(std::move(anchor).error())};
        MetalDevice &device = anchor.value().device();
        std::lock_guard<std::mutex> guard(device.mutex);
        uint64_t native{};
        if (!device.state.beginCommands(native, device.error))
            return Result<CommandRecording, RhiError>{
                err(RhiError{RhiErrorCode::BackendFailure, {"begin_metal_commands", handle.generation, handle.index}})};
        return Result<CommandRecording, RhiError>{ok(CommandRecording{native, VERNON_RHI_BACKEND_METAL})};
    } @catch (NSException *) {
        return Result<CommandRecording, RhiError>{
            err(RhiError{RhiErrorCode::BackendFailure, {"begin_metal_commands_exception", 0, 0}})};
    }
}

Result<CommandSubmission, RhiError> submitCommands(VernonRhiDevice handle, uint64_t native, bool writes) noexcept {
    @try {
        (void)writes;
        auto anchor = lookupMetalDevice(handle);
        if (anchor.isErr())
            return Result<CommandSubmission, RhiError>{err(std::move(anchor).error())};
        MetalDevice &device = anchor.value().device();
        std::lock_guard<std::mutex> guard(device.mutex);
        if (!device.state.submitCommands(native, device.error))
            return Result<CommandSubmission, RhiError>{
                err(RhiError{RhiErrorCode::BackendFailure, {"submit_metal_commands", native, 0}})};
        return Result<CommandSubmission, RhiError>{ok(CommandSubmission{false, false, native})};
    } @catch (NSException *) {
        return Result<CommandSubmission, RhiError>{
            err(RhiError{RhiErrorCode::BackendFailure, {"submit_metal_commands_exception", native, 0}})};
    }
}

Result<CommandPoll, RhiError> pollCommands(VernonRhiDevice handle, uint64_t native) noexcept {
    @try {
        auto anchor = lookupMetalDevice(handle);
        if (anchor.isErr())
            return Result<CommandPoll, RhiError>{err(std::move(anchor).error())};
        MetalDevice &device = anchor.value().device();
        std::lock_guard<std::mutex> guard(device.mutex);
        bool completed{};
        bool succeeded{};
        if (!device.state.pollCommands(native, completed, succeeded, device.error))
            return Result<CommandPoll, RhiError>{
                err(RhiError{RhiErrorCode::BackendFailure, {"poll_metal_commands", native, 0}})};
        return Result<CommandPoll, RhiError>{ok(CommandPoll{completed, succeeded})};
    } @catch (NSException *) {
        return Result<CommandPoll, RhiError>{
            err(RhiError{RhiErrorCode::BackendFailure, {"poll_metal_commands_exception", native, 0}})};
    }
}

Result<void, RhiError> completeCommands(VernonRhiDevice handle, uint64_t native) noexcept {
    @try {
        auto anchor = lookupMetalDevice(handle);
        if (anchor.isErr())
            return Result<void, RhiError>{err(std::move(anchor).error())};
        MetalDevice &device = anchor.value().device();
        std::lock_guard<std::mutex> guard(device.mutex);
        return device.state.completeCommands(native, device.error)
                   ? Result<void, RhiError>{ok()}
                   : Result<void, RhiError>{
                         err(RhiError{RhiErrorCode::BackendFailure, {"complete_metal_commands", native, 0}})};
    } @catch (NSException *) {
        return Result<void, RhiError>{
            err(RhiError{RhiErrorCode::BackendFailure, {"complete_metal_commands_exception", native, 0}})};
    }
}

} // namespace vernon::rhi::metal_api

uint32_t vernon::rhi::metalTexturePixelFormat(VernonTextureFormat format) {
    const vernon::formats::Format *entry = vernon::formats::find(format);
    return static_cast<uint32_t>(entry ? vernon::rhi::metal::pixelFormat(entry->rhi) : MTLPixelFormatInvalid);
}

const vernon::rhi::BackendDispatch &vernon::rhi::metalBackendDispatch() {
    using namespace metal_api;
    static const BackendDispatch dispatch = [] {
        BackendDispatch result{};
        result.backend = VERNON_RHI_BACKEND_METAL;
        result.commandCapabilities = BackendCommandIndependentRecording | BackendCommandConcurrentSubmission;
        result.commandCapabilitiesForDevice.emplace(commandCapabilitiesForDevice);
        result.ownsDevice = ownsDevice;
        result.createOwnedDevice = createOwnedDevice;
        result.destroyDevice = destroyDevice;
        result.lastError = lastError;
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
        result.createImage.emplace(createImage);
        result.uploadImage.emplace(uploadImage);
        result.downloadImage.emplace(downloadImage);
        result.downloadImageBatch.emplace(downloadImageBatch);
        result.generateImageMipmaps.emplace(generateImageMipmaps);
        result.destroyImage.emplace(destroyImage);
        result.isImageValid.emplace(isImageValid);
        result.getImageNativeHandle.emplace(getImageNativeHandle);
        result.formatCapabilities.emplace(formatCapabilities);
        result.createSampler.emplace(createSampler);
        result.destroySampler.emplace(destroySampler);
        result.isSamplerValid.emplace(isSamplerValid);
        result.bufferResource = bufferResource;
        result.imageResource.emplace(imageResource);
        result.samplerResource.emplace(samplerResource);
        result.retainResource = retainResource;
        result.resolveRetainedResource = resolveResource;
        result.describeImageResource.emplace(describeImageResource);
        result.beginCommands = beginCommands;
        result.submitCommands = submitCommands;
        result.pollCommands.emplace(pollCommands);
        result.completeCommands.emplace(completeCommands);
        result.abandonCommands.emplace(abandonCommands);
        result.recordBarriers.emplace(recordBarriers);
        result.recordBufferCopy.emplace(recordBufferCopy);
        result.recordImageCopy.emplace(recordImageCopy);
        result.beginRendering.emplace(beginRendering);
        result.endRendering.emplace(endRendering);
        result.clearColor.emplace(clearColor);
        result.clearDepthStencil.emplace(clearDepthStencil);
        result.recordDebugCommand.emplace(recordDebugCommand);
        result.recordTimer.emplace(recordTimer);
        result.readTimers.emplace(readTimers);
        result.releaseTimers.emplace(releaseTimers);
        result.createImageView.emplace(createImageView);
        result.destroyImageView.emplace(destroyImageView);
        result.getImageViewNativeHandle.emplace(getImageViewNativeHandle);
        result.imageViewResource.emplace(imageViewResource);
        result.describeImageViewResource.emplace(describeImageViewResource);
        return result;
    }();
    return dispatch;
}
