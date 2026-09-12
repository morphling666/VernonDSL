#include "backend_dispatch.h"
#include "image_data_layout.h"
#include "image_descriptor_validation.h"
#include "metal_backend.h"
#include "sampler_filter.h"
#include "VernonTextureTypes.h"

#include <limits>
#include <mutex>
#include <new>
#include <string>
#include <unordered_map>
#include <utility>

namespace vernon::rhi {
uint32_t metalTexturePixelFormat(VernonTextureFormat format);
bool deviceHasActiveCommandEncoder(VernonRhiDevice device);
}

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

struct MetalImageViewPayload
    : MetalResourcePayload<vernon::rhi::ImageViewResourceTag, vernon::rhi::metal::ImageView> {
    using MetalResourcePayload::MetalResourcePayload;
    VernonRhiImageViewDescriptor descriptor{};
    VernonRhiImageDescriptor imageDescriptor{};
    uint64_t parentKey{};
    vernon::Option<vernon::rhi::RetainedResourceLease<vernon::rhi::ImageResourceTag>> imageLease;
};

using MetalSamplerPayload =
    MetalResourcePayload<vernon::rhi::SamplerResourceTag, vernon::rhi::metal::Sampler>;

template <typename Payload>
using MetalPayloadSlots = vernon::rhi::StableResourceSlotContainer<Payload>;

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

VernonRhiDevice invalidDevice() { return {static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0}; }

template <typename Handle> uint64_t resourceKey(Handle handle) {
    return (static_cast<uint64_t>(handle.generation) << 32) | (static_cast<uint64_t>(handle.index) + 1);
}

template <typename Handle> vernon::Result<Handle, vernon::RhiError> decodeResourceKey(uint64_t key,
                                                                                      const char *operation) noexcept {
    const uint64_t encodedIndex = key & UINT32_MAX;
    const uint32_t generation = static_cast<uint32_t>(key >> 32);
    if (!encodedIndex || !generation)
        return vernon::Result<Handle, vernon::RhiError>{
            vernon::err(vernon::RhiError{vernon::RhiErrorCode::InvalidArgument, {operation, key, 0}})};
    return vernon::Result<Handle, vernon::RhiError>{
        vernon::ok(Handle{static_cast<uint32_t>(encodedIndex - 1), generation})};
}

uint32_t maximumMipLevels(const VernonRhiImageDescriptor &descriptor) {
    uint32_t extent = descriptor.width > descriptor.height ? descriptor.width : descriptor.height;
    if (descriptor.dimension == VERNON_RHI_IMAGE_3D && descriptor.depth > extent)
        extent = descriptor.depth;
    uint32_t levels = 1;
    while (extent > 1) {
        extent >>= 1;
        ++levels;
    }
    return levels;
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

VernonRhiStatus statusForError(const vernon::RhiError &error) noexcept {
    switch (error.code) {
    case vernon::RhiErrorCode::InvalidArgument:
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    case vernon::RhiErrorCode::Unsupported:
        return VERNON_RHI_STATUS_UNSUPPORTED;
    case vernon::RhiErrorCode::ResourceExhausted:
        return VERNON_RHI_STATUS_RESOURCE_EXHAUSTED;
    case vernon::RhiErrorCode::BackendFailure:
    case vernon::RhiErrorCode::LifecycleFailure:
    case vernon::RhiErrorCode::SynchronizationFailure:
        return VERNON_RHI_STATUS_INTERNAL_ERROR;
    }
    return VERNON_RHI_STATUS_INTERNAL_ERROR;
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
        return vernon::Result<Payload *, vernon::RhiError>{vernon::err(
            vernon::RhiError{vernon::RhiErrorCode::InvalidArgument, {operation, index, 0}})};
    return vernon::Result<Payload *, vernon::RhiError>{vernon::ok(payload)};
}

template <typename Payload>
vernon::Result<Payload *, vernon::RhiError> reservePayload(MetalDevice &device, MetalPayloadSlots<Payload> &payloads,
                                                           const char *operation) noexcept {
    for (std::size_t index = 0; index < payloads.size(); ++index)
        if (!payloads[index].lifecycle.snapshot().occupied)
            return vernon::Result<Payload *, vernon::RhiError>{vernon::ok(&payloads[index])};
    if (payloads.size() >= UINT32_MAX)
        return vernon::Result<Payload *, vernon::RhiError>{vernon::err(
            vernon::RhiError{vernon::RhiErrorCode::ResourceExhausted, {operation, payloads.size(), 0}})};
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
        return vernon::Result<void, vernon::RhiError>{vernon::err(vernon::RhiError{
            vernon::RhiErrorCode::LifecycleFailure,
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

VernonRhiDevice createOwnedDevice(const VernonRhiOwnedDeviceDescriptor *descriptor) {
    if (!descriptor)
        return invalidDevice();
    auto reservation = metalDevices.reserve();
    if (reservation.isErr()) {
        setDeviceCreationError("Metal device registry reservation failed");
        return invalidDevice();
    }
    MetalDevice &device = reservation.value().device();
    auto owner = reservation.value().retainOwner();
    if (owner.isErr()) {
        setDeviceCreationError("Metal device owner retention failed");
        return invalidDevice();
    }
    auto commandState = vernon::rhi::createCommandDeviceState(std::move(owner).value());
    if (commandState.isErr()) {
        setDeviceCreationError("Metal command state allocation failed");
        return invalidDevice();
    }
    device.commandState = std::move(commandState).value();
    if (!device.state.initialize(descriptor->device_index, device.error)) {
        setDeviceCreationError(device.error);
        return invalidDevice();
    }
    auto published = metalDevices.publish(std::move(reservation).value());
    if (published.isErr()) {
        device.state.shutdown();
        setDeviceCreationError("Metal device registry publication failed");
        return invalidDevice();
    }
    return encodeDeviceHandle(published.value());
}

vernon::Result<void, vernon::RhiError> destroyDevice(VernonRhiDevice handle) noexcept {
    if ((handle.index & metalDeviceBit) == 0)
        return vernon::Result<void, vernon::RhiError>{vernon::err(
            vernon::RhiError{vernon::RhiErrorCode::InvalidArgument, {"destroy_metal_device", handle.index, 0}})};
    return metalDevices.remove(decodeDeviceHandle(handle), teardownDevice);
}

vernon::Result<vernon::rhi::CommandDeviceStateRef, vernon::RhiError>
commandState(VernonRhiDevice handle) noexcept {
    auto anchor = lookupMetalDevice(handle);
    if (anchor.isErr())
        return vernon::Result<vernon::rhi::CommandDeviceStateRef, vernon::RhiError>{
            vernon::err(std::move(anchor).error())};
    return anchor.value().device().commandState.retain();
}

bool ownsDevice(VernonRhiDevice handle) { return lookupMetalDevice(handle).isOk(); }

VernonStringView lastError(VernonRhiDevice handle) {
    auto anchor = lookupMetalDevice(handle);
    if (anchor.isErr())
        return {};
    MetalDevice &device = anchor.value().device();
    std::lock_guard<std::mutex> guard(device.mutex);
    return {device.error.data(), device.error.size()};
}

VernonRhiStatus synchronize(VernonRhiDevice handle) {
    auto anchor = lookupMetalDevice(handle);
    if (anchor.isErr())
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    MetalDevice &device = anchor.value().device();
    std::lock_guard<std::mutex> guard(device.mutex);
    return device.state.synchronize(device.error) ? VERNON_RHI_STATUS_OK : VERNON_RHI_STATUS_INTERNAL_ERROR;
}

void *deviceStateForBackend(VernonRhiDevice handle) {
    auto anchor = lookupMetalDevice(handle);
    return anchor.isOk() ? &anchor.value().device().state : nullptr;
}

VernonRhiStatus createBuffer(VernonRhiDevice handle, const VernonRhiBufferDescriptor *descriptor,
                             VernonRhiBuffer *output) {
    auto anchor = lookupMetalDevice(handle);
    if (anchor.isErr())
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    constexpr uint32_t allUsages =
        VERNON_RHI_BUFFER_TRANSFER_SOURCE | VERNON_RHI_BUFFER_TRANSFER_DESTINATION | VERNON_RHI_BUFFER_UNIFORM |
        VERNON_RHI_BUFFER_STORAGE | VERNON_RHI_BUFFER_VERTEX | VERNON_RHI_BUFFER_INDEX | VERNON_RHI_BUFFER_INDIRECT;
    if (!descriptor || !output || descriptor->struct_size < sizeof(*descriptor) || descriptor->size == 0 ||
        descriptor->size > (std::numeric_limits<size_t>::max)() ||
        descriptor->memory_class > VERNON_RHI_MEMORY_READBACK || (descriptor->usage & ~allUsages) != 0)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    auto owner = anchor.value().retainOwner();
    if (owner.isErr())
        return statusForError(owner.error());
    auto reservation = ResourceCreationReservation::create(owner.value());
    if (reservation.isErr())
        return statusForError(reservation.error());
    MetalDevice &device = anchor.value().device();
    std::lock_guard<std::mutex> creationGuard(device.creationMutex);
    auto reserved = reservePayload(device, device.buffers, "reserve_metal_buffer");
    if (reserved.isErr())
        return statusForError(reserved.error());
    MetalBufferPayload &payload = *reserved.value();
    {
        std::lock_guard<std::mutex> nativeGuard(device.mutex);
        if (!device.state.createBuffer(payload.native, *descriptor, device.error))
            return VERNON_RHI_STATUS_INTERNAL_ERROR;
        payload.descriptor = *descriptor;
    }
    auto published = payload.lifecycle.publish(std::move(reservation).value(), &device, teardownBuffer);
    if (published.isErr()) {
        std::lock_guard<std::mutex> nativeGuard(device.mutex);
        device.state.destroyBuffer(payload.native);
        payload.descriptor = {};
        return statusForError(published.error());
    }
    *output = {published.value().index, published.value().generation};
    return VERNON_RHI_STATUS_OK;
}

VernonRhiStatus uploadBuffer(VernonRhiDevice handle, VernonRhiBuffer buffer, uint64_t offset, const void *source,
                             uint64_t size) {
    auto anchor = lookupMetalDevice(handle);
    if (anchor.isErr())
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    if (!source || size == 0 || size > (std::numeric_limits<size_t>::max)())
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    MetalDevice &device = anchor.value().device();
    auto pinned = pinPayload(device, device.buffers, BufferHandle{buffer.index, buffer.generation}, "upload_buffer");
    if (pinned.isErr())
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    MetalBufferPayload &payload = *pinned.value().first;
    std::lock_guard<std::mutex> guard(device.mutex);
    if (offset > payload.descriptor.size || size > payload.descriptor.size - offset)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    if (payload.descriptor.memory_class != VERNON_RHI_MEMORY_UPLOAD &&
        vernon::rhi::deviceHasActiveCommandEncoder(handle)) {
        device.error = "device-level Metal upload cannot execute while a command encoder is recording";
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    }
    return device.state.uploadBuffer(payload.native, offset, source, size, device.error)
               ? VERNON_RHI_STATUS_OK
               : VERNON_RHI_STATUS_INTERNAL_ERROR;
}

VernonRhiStatus uploadBufferRanges(VernonRhiDevice handle, VernonRhiBuffer buffer,
                                   const VernonRhiBufferUploadRange *ranges, size_t rangeCount) {
    auto anchor = lookupMetalDevice(handle);
    if (anchor.isErr())
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    if (!ranges || rangeCount == 0)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    MetalDevice &device = anchor.value().device();
    auto pinned =
        pinPayload(device, device.buffers, BufferHandle{buffer.index, buffer.generation}, "upload_buffer_ranges");
    if (pinned.isErr())
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    MetalBufferPayload &payload = *pinned.value().first;
    std::lock_guard<std::mutex> guard(device.mutex);
    if (payload.descriptor.memory_class != VERNON_RHI_MEMORY_UPLOAD &&
        vernon::rhi::deviceHasActiveCommandEncoder(handle)) {
        device.error = "device-level Metal upload cannot execute while a command encoder is recording";
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    }
    for (size_t index = 0; index < rangeCount; ++index) {
        const VernonRhiBufferUploadRange &range = ranges[index];
        if (!range.source || range.size == 0 || range.size > (std::numeric_limits<size_t>::max)() ||
            range.offset > payload.descriptor.size || range.size > payload.descriptor.size - range.offset)
            return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    }
    return device.state.uploadBufferRanges(payload.native, ranges, rangeCount, device.error)
               ? VERNON_RHI_STATUS_OK
               : VERNON_RHI_STATUS_INTERNAL_ERROR;
}

VernonRhiStatus downloadBuffer(VernonRhiDevice handle, VernonRhiBuffer buffer, uint64_t offset, void *destination,
                               uint64_t size) {
    auto anchor = lookupMetalDevice(handle);
    if (anchor.isErr())
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    if (!destination || size == 0 || size > (std::numeric_limits<size_t>::max)())
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    MetalDevice &device = anchor.value().device();
    auto pinned = pinPayload(device, device.buffers, BufferHandle{buffer.index, buffer.generation}, "download_buffer");
    if (pinned.isErr())
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    MetalBufferPayload &payload = *pinned.value().first;
    std::lock_guard<std::mutex> guard(device.mutex);
    if (offset > payload.descriptor.size || size > payload.descriptor.size - offset)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    return device.state.downloadBuffer(payload.native, offset, destination, size, device.error)
               ? VERNON_RHI_STATUS_OK
               : VERNON_RHI_STATUS_INTERNAL_ERROR;
}

VernonRhiStatus downloadBufferRanges(VernonRhiDevice handle, VernonRhiBuffer buffer,
                                     const VernonRhiBufferDownloadRange *ranges, size_t rangeCount) {
    auto anchor = lookupMetalDevice(handle);
    if (anchor.isErr() || !ranges || rangeCount == 0)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    MetalDevice &device = anchor.value().device();
    auto pinned =
        pinPayload(device, device.buffers, BufferHandle{buffer.index, buffer.generation}, "download_buffer_ranges");
    if (pinned.isErr())
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    MetalBufferPayload &payload = *pinned.value().first;
    std::lock_guard<std::mutex> guard(device.mutex);
    for (size_t index = 0; index < rangeCount; ++index) {
        const VernonRhiBufferDownloadRange &range = ranges[index];
        if (!range.destination || range.size == 0 || range.size > (std::numeric_limits<size_t>::max)() ||
            range.offset > payload.descriptor.size || range.size > payload.descriptor.size - range.offset)
            return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    }
    return device.state.downloadBufferRanges(payload.native, ranges, rangeCount, device.error)
               ? VERNON_RHI_STATUS_OK
               : VERNON_RHI_STATUS_INTERNAL_ERROR;
}

VernonRhiStatus destroyBuffer(VernonRhiDevice handle, VernonRhiBuffer buffer) {
    auto anchor = lookupMetalDevice(handle);
    if (anchor.isErr())
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    MetalDevice &device = anchor.value().device();
    MetalBufferPayload *payload = nullptr;
    {
        std::lock_guard<std::mutex> guard(device.slotMutex);
        auto found = payloadAt(device.buffers, buffer.index, "destroy_buffer");
        if (found.isErr())
            return VERNON_RHI_STATUS_INVALID_ARGUMENT;
        payload = found.value();
    }
    auto destroyed = payload->lifecycle.destroyPublic(BufferHandle{buffer.index, buffer.generation});
    if (destroyed.isErr())
        return statusForError(destroyed.error());
    return VERNON_RHI_STATUS_OK;
}

uint32_t isBufferValid(VernonRhiDevice handle, VernonRhiBuffer buffer) {
    auto anchor = lookupMetalDevice(handle);
    if (anchor.isErr())
        return 0;
    MetalDevice &device = anchor.value().device();
    auto pinned = pinPayload(device, device.buffers, BufferHandle{buffer.index, buffer.generation}, "is_buffer_valid");
    return pinned.isOk();
}

VernonRhiStatus getBufferNativeHandle(VernonRhiDevice handle, VernonRhiBuffer buffer, void **output) {
    auto anchor = lookupMetalDevice(handle);
    if (anchor.isErr() || !output)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    MetalDevice &device = anchor.value().device();
    auto pinned =
        pinPayload(device, device.buffers, BufferHandle{buffer.index, buffer.generation}, "get_buffer_native_handle");
    if (pinned.isErr())
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(device.mutex);
    *output = (__bridge void *)pinned.value().first->native.buffer;
    return VERNON_RHI_STATUS_OK;
}

VernonRhiStatus createImage(VernonRhiDevice handle, const VernonRhiImageDescriptor *descriptor,
                            VernonRhiImage *output) {
    auto anchor = lookupMetalDevice(handle);
    if (anchor.isErr())
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    const bool validCube = descriptor && descriptor->dimension == VERNON_RHI_IMAGE_CUBE &&
                           descriptor->width == descriptor->height && descriptor->depth == 1 &&
                           descriptor->array_layers == 6;
    constexpr uint32_t allUsages =
        VERNON_RHI_IMAGE_TRANSFER_SOURCE | VERNON_RHI_IMAGE_TRANSFER_DESTINATION | VERNON_RHI_IMAGE_SAMPLED |
        VERNON_RHI_IMAGE_STORAGE | VERNON_RHI_IMAGE_COLOR_ATTACHMENT | VERNON_RHI_IMAGE_DEPTH_STENCIL_ATTACHMENT;
    if (!descriptor || !output || descriptor->struct_size < sizeof(*descriptor) || descriptor->width == 0 ||
        descriptor->height == 0 || descriptor->depth == 0 || descriptor->mip_levels == 0 ||
        descriptor->array_layers == 0 || descriptor->sample_count != 1 ||
        descriptor->dimension > VERNON_RHI_IMAGE_CUBE ||
        (descriptor->usage & ~allUsages) != 0 ||
        descriptor->mip_levels > maximumMipLevels(*descriptor) ||
        (descriptor->dimension == VERNON_RHI_IMAGE_2D && descriptor->depth != 1) ||
        (descriptor->dimension == VERNON_RHI_IMAGE_3D && descriptor->array_layers != 1) ||
        (descriptor->dimension == VERNON_RHI_IMAGE_CUBE && !validCube))
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    const bool depthFormat = descriptor->format == VERNON_RHI_FORMAT_D32_FLOAT ||
                             descriptor->format == VERNON_RHI_FORMAT_D32_FLOAT_S8_UINT;
    if ((depthFormat && (descriptor->usage & VERNON_RHI_IMAGE_COLOR_ATTACHMENT)) ||
        (!depthFormat && (descriptor->usage & VERNON_RHI_IMAGE_DEPTH_STENCIL_ATTACHMENT)) ||
        (descriptor->format == VERNON_RHI_FORMAT_D32_FLOAT_S8_UINT &&
         (descriptor->usage & VERNON_RHI_IMAGE_STORAGE)))
        return VERNON_RHI_STATUS_UNSUPPORTED;
    if (vernon::rhi::metal::pixelFormat(descriptor->format) == MTLPixelFormatInvalid)
        return VERNON_RHI_STATUS_UNSUPPORTED;
    auto owner = anchor.value().retainOwner();
    if (owner.isErr())
        return statusForError(owner.error());
    auto reservation = ResourceCreationReservation::create(owner.value());
    if (reservation.isErr())
        return statusForError(reservation.error());
    MetalDevice &device = anchor.value().device();
    std::lock_guard<std::mutex> creationGuard(device.creationMutex);
    auto reserved = reservePayload(device, device.images, "reserve_metal_image");
    if (reserved.isErr())
        return statusForError(reserved.error());
    MetalImagePayload &payload = *reserved.value();
    {
        std::lock_guard<std::mutex> nativeGuard(device.mutex);
        if (!device.state.createImage(payload.native, *descriptor, device.error))
            return VERNON_RHI_STATUS_INTERNAL_ERROR;
        payload.descriptor = *descriptor;
    }
    auto published = payload.lifecycle.publish(std::move(reservation).value(), &device, teardownImage);
    if (published.isErr()) {
        std::lock_guard<std::mutex> nativeGuard(device.mutex);
        device.state.destroyImage(payload.native);
        payload.descriptor = {};
        return statusForError(published.error());
    }
    *output = {published.value().index, published.value().generation};
    return VERNON_RHI_STATUS_OK;
}

VernonRhiStatus createImageView(VernonRhiDevice handle, const VernonRhiImageViewDescriptor *descriptor,
                                VernonRhiImageView *output) {
    auto anchor = lookupMetalDevice(handle);
    if (anchor.isErr())
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    if (!descriptor || !output || descriptor->struct_size < sizeof(*descriptor) ||
        vernon::rhi::metal::pixelFormat(descriptor->format) == MTLPixelFormatInvalid ||
        descriptor->mip_level_count == 0 || descriptor->array_layer_count == 0 || descriptor->aspects == 0)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    auto owner = anchor.value().retainOwner();
    if (owner.isErr())
        return statusForError(owner.error());
    auto reservation = ResourceCreationReservation::create(owner.value());
    if (reservation.isErr())
        return statusForError(reservation.error());
    MetalDevice &device = anchor.value().device();
    const ImageHandle imageHandle{descriptor->image.index, descriptor->image.generation};
    MetalImagePayload *image = nullptr;
    {
        std::lock_guard<std::mutex> guard(device.slotMutex);
        auto found = payloadAt(device.images, imageHandle.index, "create_image_view");
        if (found.isErr())
            return VERNON_RHI_STATUS_INVALID_ARGUMENT;
        image = found.value();
    }
    auto imageLease = image->lifecycle.retain(imageHandle);
    if (imageLease.isErr())
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    auto imagePin = image->lifecycle.pin(imageHandle);
    if (imagePin.isErr())
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    VernonRhiStatus result = VERNON_RHI_STATUS_OK;
    vernon::Option<vernon::rhi::RetainedResourceLease<vernon::rhi::ImageResourceTag>> rollbackLease;
    MetalImageViewPayload *payload = nullptr;
    {
        std::lock_guard<std::mutex> creationGuard(device.creationMutex);
        {
            std::lock_guard<std::mutex> nativeGuard(device.mutex);
            if (!vernon::rhi::validImageViewDescriptor(image->descriptor, *descriptor))
                result = VERNON_RHI_STATUS_INVALID_ARGUMENT;
        }
        if (result == VERNON_RHI_STATUS_OK) {
            auto reserved = reservePayload(device, device.imageViews, "reserve_metal_image_view");
            if (reserved.isErr()) {
                result = statusForError(reserved.error());
            } else {
                payload = reserved.value();
                {
                    std::lock_guard<std::mutex> nativeGuard(device.mutex);
                    if (!device.state.createImageView(payload->native, image->native, *descriptor, device.error)) {
                        result = VERNON_RHI_STATUS_UNSUPPORTED;
                    } else {
                        payload->descriptor = *descriptor;
                        payload->imageDescriptor = image->descriptor;
                        payload->parentKey = resourceKey(imageHandle);
                        payload->imageLease.emplace(std::move(imageLease).value());
                    }
                }
                if (result == VERNON_RHI_STATUS_OK) {
                    auto published = payload->lifecycle.publish(std::move(reservation).value(), &device,
                                                                teardownImageView);
                    if (published.isErr()) {
                        std::lock_guard<std::mutex> nativeGuard(device.mutex);
                        device.state.destroyImageView(payload->native);
                        payload->descriptor = {};
                        payload->imageDescriptor = {};
                        payload->parentKey = 0;
                        auto taken = payload->imageLease.take();
                        if (taken)
                            rollbackLease.emplace(std::move(taken).value());
                        result = statusForError(published.error());
                    } else {
                        *output = {published.value().index, published.value().generation};
                    }
                }
            }
        }
    }
    if (result != VERNON_RHI_STATUS_OK && rollbackLease) {
        auto released = rollbackLease.value().release();
        if (released.isErr())
            return VERNON_RHI_STATUS_INTERNAL_ERROR;
    }
    return result;
}

VernonRhiStatus destroyImageView(VernonRhiDevice handle, VernonRhiImageView imageView) {
    auto anchor = lookupMetalDevice(handle);
    if (anchor.isErr())
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    MetalDevice &device = anchor.value().device();
    MetalImageViewPayload *payload = nullptr;
    {
        std::lock_guard<std::mutex> guard(device.slotMutex);
        auto found = payloadAt(device.imageViews, imageView.index, "destroy_image_view");
        if (found.isErr())
            return VERNON_RHI_STATUS_INVALID_ARGUMENT;
        payload = found.value();
    }
    auto destroyed = payload->lifecycle.destroyPublic(ImageViewHandle{imageView.index, imageView.generation});
    return destroyed.isOk() ? VERNON_RHI_STATUS_OK : statusForError(destroyed.error());
}

VernonRhiStatus getImageViewNativeHandle(VernonRhiDevice handle, VernonRhiImageView imageView, uint64_t *output) {
    auto anchor = lookupMetalDevice(handle);
    if (anchor.isErr() || !output)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    MetalDevice &device = anchor.value().device();
    auto pinned = pinPayload(device, device.imageViews, ImageViewHandle{imageView.index, imageView.generation},
                             "get_image_view_native_handle");
    if (pinned.isErr())
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(device.mutex);
    *output = reinterpret_cast<uintptr_t>((__bridge void *)pinned.value().first->native.texture);
    return VERNON_RHI_STATUS_OK;
}

VernonRhiStatus uploadImage(VernonRhiDevice handle, VernonRhiImage image,
                            const VernonRhiImageUploadDescriptor *uploads, size_t uploadCount) {
    auto anchor = lookupMetalDevice(handle);
    if (anchor.isErr())
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    if (!uploads || uploadCount == 0)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    MetalDevice &device = anchor.value().device();
    auto pinned = pinPayload(device, device.images, ImageHandle{image.index, image.generation}, "upload_image");
    if (pinned.isErr())
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    MetalImagePayload &payload = *pinned.value().first;
    std::lock_guard<std::mutex> guard(device.mutex);
    for (size_t index = 0; index < uploadCount; ++index) {
        const auto &upload = uploads[index];
        if (upload.struct_size < sizeof(upload) || !upload.data || upload.mip_level >= payload.descriptor.mip_levels)
            return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    }
    return device.state.uploadImage(payload.native, payload.descriptor, uploads, uploadCount, device.error)
               ? VERNON_RHI_STATUS_OK
               : VERNON_RHI_STATUS_UNSUPPORTED;
}

VernonRhiStatus downloadImage(VernonRhiDevice handle, VernonRhiImage image,
                              const VernonRhiImageDownloadDescriptor *download, void *destination, size_t size) {
    auto anchor = lookupMetalDevice(handle);
    if (anchor.isErr() || !download || download->struct_size < sizeof(*download) || !destination)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    MetalDevice &device = anchor.value().device();
    auto pinned = pinPayload(device, device.images, ImageHandle{image.index, image.generation}, "download_image");
    if (pinned.isErr())
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    MetalImagePayload &payload = *pinned.value().first;
    std::lock_guard<std::mutex> guard(device.mutex);
    const auto required = vernon::rhi::imageDownloadByteSize(payload.descriptor, *download);
    if (!required || size != *required)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    return device.state.downloadImage(payload.native, payload.descriptor, *download, destination, size, device.error)
               ? VERNON_RHI_STATUS_OK
               : VERNON_RHI_STATUS_INTERNAL_ERROR;
}

VernonRhiStatus downloadImageBatch(VernonRhiDevice handle, VernonRhiImage image,
                                   const VernonRhiImageDownload *downloads, size_t downloadCount) {
    auto anchor = lookupMetalDevice(handle);
    if (anchor.isErr() || !downloads || downloadCount == 0)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    MetalDevice &device = anchor.value().device();
    auto pinned = pinPayload(device, device.images, ImageHandle{image.index, image.generation}, "download_image_batch");
    if (pinned.isErr())
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    MetalImagePayload &payload = *pinned.value().first;
    std::lock_guard<std::mutex> guard(device.mutex);
    for (size_t index = 0; index < downloadCount; ++index) {
        const auto required = vernon::rhi::imageDownloadByteSize(payload.descriptor, downloads[index].descriptor);
        if (!downloads[index].destination || !required || downloads[index].size != *required)
            return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    }
    return device.state.downloadImageBatch(payload.native, payload.descriptor, downloads, downloadCount, device.error)
               ? VERNON_RHI_STATUS_OK
               : VERNON_RHI_STATUS_INTERNAL_ERROR;
}

VernonRhiStatus generateImageMipmaps(VernonRhiDevice handle, VernonRhiImage image) {
    auto anchor = lookupMetalDevice(handle);
    if (anchor.isErr())
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    MetalDevice &device = anchor.value().device();
    auto pinned =
        pinPayload(device, device.images, ImageHandle{image.index, image.generation}, "generate_image_mipmaps");
    if (pinned.isErr())
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    MetalImagePayload &payload = *pinned.value().first;
    std::lock_guard<std::mutex> guard(device.mutex);
    if (payload.descriptor.mip_levels < 2 ||
        !(payload.descriptor.usage & VERNON_RHI_IMAGE_TRANSFER_SOURCE) ||
        !(payload.descriptor.usage & VERNON_RHI_IMAGE_TRANSFER_DESTINATION) ||
        payload.descriptor.format == VERNON_RHI_FORMAT_D32_FLOAT ||
        payload.descriptor.format == VERNON_RHI_FORMAT_D32_FLOAT_S8_UINT)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    return device.state.generateImageMipmaps(payload.native, payload.descriptor.mip_levels, device.error)
               ? VERNON_RHI_STATUS_OK
               : VERNON_RHI_STATUS_INTERNAL_ERROR;
}

VernonRhiStatus destroyImage(VernonRhiDevice handle, VernonRhiImage image) {
    auto anchor = lookupMetalDevice(handle);
    if (anchor.isErr())
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    MetalDevice &device = anchor.value().device();
    MetalImagePayload *payload = nullptr;
    {
        std::lock_guard<std::mutex> guard(device.slotMutex);
        auto found = payloadAt(device.images, image.index, "destroy_image");
        if (found.isErr())
            return VERNON_RHI_STATUS_INVALID_ARGUMENT;
        payload = found.value();
    }
    auto destroyed = payload->lifecycle.destroyPublic(ImageHandle{image.index, image.generation});
    return destroyed.isOk() ? VERNON_RHI_STATUS_OK : statusForError(destroyed.error());
}

uint32_t isImageValid(VernonRhiDevice handle, VernonRhiImage image) {
    auto anchor = lookupMetalDevice(handle);
    if (anchor.isErr())
        return 0;
    MetalDevice &device = anchor.value().device();
    auto pinned = pinPayload(device, device.images, ImageHandle{image.index, image.generation}, "is_image_valid");
    return pinned.isOk();
}

VernonRhiStatus getImageNativeHandle(VernonRhiDevice handle, VernonRhiImage image, uint64_t *output) {
    auto anchor = lookupMetalDevice(handle);
    if (anchor.isErr() || !output)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    MetalDevice &device = anchor.value().device();
    auto pinned =
        pinPayload(device, device.images, ImageHandle{image.index, image.generation}, "get_image_native_handle");
    if (pinned.isErr())
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(device.mutex);
    *output = reinterpret_cast<uintptr_t>((__bridge void *)pinned.value().first->native.texture);
    return VERNON_RHI_STATUS_OK;
}

VernonRhiStatus createSampler(VernonRhiDevice handle, const VernonRhiSamplerDescriptor *descriptor,
                              VernonRhiSampler *output) {
    auto anchor = lookupMetalDevice(handle);
    if (anchor.isErr())
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    vernon::rhi::SamplerFilter filter;
    if (!descriptor || !output || descriptor->struct_size < sizeof(*descriptor) ||
        !vernon::rhi::decodeSamplerFilter(*descriptor, filter))
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    auto owner = anchor.value().retainOwner();
    if (owner.isErr())
        return statusForError(owner.error());
    auto reservation = ResourceCreationReservation::create(owner.value());
    if (reservation.isErr())
        return statusForError(reservation.error());
    MetalDevice &device = anchor.value().device();
    std::lock_guard<std::mutex> creationGuard(device.creationMutex);
    auto reserved = reservePayload(device, device.samplers, "reserve_metal_sampler");
    if (reserved.isErr())
        return statusForError(reserved.error());
    MetalSamplerPayload &payload = *reserved.value();
    {
        std::lock_guard<std::mutex> nativeGuard(device.mutex);
        if (!device.state.createSampler(payload.native, *descriptor, device.error))
            return VERNON_RHI_STATUS_INTERNAL_ERROR;
    }
    auto published = payload.lifecycle.publish(std::move(reservation).value(), &device, teardownSampler);
    if (published.isErr()) {
        std::lock_guard<std::mutex> nativeGuard(device.mutex);
        device.state.destroySampler(payload.native);
        return statusForError(published.error());
    }
    *output = {published.value().index, published.value().generation};
    return VERNON_RHI_STATUS_OK;
}

VernonRhiStatus destroySampler(VernonRhiDevice handle, VernonRhiSampler sampler) {
    auto anchor = lookupMetalDevice(handle);
    if (anchor.isErr())
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    MetalDevice &device = anchor.value().device();
    MetalSamplerPayload *payload = nullptr;
    {
        std::lock_guard<std::mutex> guard(device.slotMutex);
        auto found = payloadAt(device.samplers, sampler.index, "destroy_sampler");
        if (found.isErr())
            return VERNON_RHI_STATUS_INVALID_ARGUMENT;
        payload = found.value();
    }
    auto destroyed = payload->lifecycle.destroyPublic(SamplerHandle{sampler.index, sampler.generation});
    return destroyed.isOk() ? VERNON_RHI_STATUS_OK : statusForError(destroyed.error());
}

uint32_t isSamplerValid(VernonRhiDevice handle, VernonRhiSampler sampler) {
    auto anchor = lookupMetalDevice(handle);
    if (anchor.isErr())
        return 0;
    MetalDevice &device = anchor.value().device();
    auto pinned =
        pinPayload(device, device.samplers, SamplerHandle{sampler.index, sampler.generation}, "is_sampler_valid");
    return pinned.isOk();
}

template <typename Payload, typename Handle>
uint64_t publicResourceKey(VernonRhiDevice deviceHandle, MetalPayloadSlots<Payload> MetalDevice::*member,
                           Handle handle) noexcept {
    auto anchor = lookupMetalDevice(deviceHandle);
    if (anchor.isErr())
        return 0;
    MetalDevice &device = anchor.value().device();
    auto pinned = pinPayload(device, device.*member, handle, "resource_key");
    return pinned.isOk() ? resourceKey(handle) : 0;
}

uint64_t bufferResource(VernonRhiDevice handle, VernonRhiBuffer buffer) {
    return publicResourceKey(handle, &MetalDevice::buffers, BufferHandle{buffer.index, buffer.generation});
}

uint64_t imageResource(VernonRhiDevice handle, VernonRhiImage image) {
    return publicResourceKey(handle, &MetalDevice::images, ImageHandle{image.index, image.generation});
}

uint64_t imageViewResource(VernonRhiDevice handle, VernonRhiImageView view) {
    return publicResourceKey(handle, &MetalDevice::imageViews, ImageViewHandle{view.index, view.generation});
}

uint64_t samplerResource(VernonRhiDevice handle, VernonRhiSampler sampler) {
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
        return Result<RetainedRhiResourceLease, RhiError>{
            ok(RetainedRhiResourceLease{std::move(retained).value()})};
    }
    case ResourceKind::Image: {
        auto decoded = decodeResourceKey<ImageHandle>(key, "retain_metal_image");
        if (decoded.isErr())
            return Result<RetainedRhiResourceLease, RhiError>{err(std::move(decoded).error())};
        auto retained = retainTypedResource(device, device.images, decoded.value(), "retain_metal_image");
        if (retained.isErr())
            return Result<RetainedRhiResourceLease, RhiError>{err(std::move(retained).error())};
        return Result<RetainedRhiResourceLease, RhiError>{
            ok(RetainedRhiResourceLease{std::move(retained).value()})};
    }
    case ResourceKind::ImageView: {
        auto decoded = decodeResourceKey<ImageViewHandle>(key, "retain_metal_image_view");
        if (decoded.isErr())
            return Result<RetainedRhiResourceLease, RhiError>{err(std::move(decoded).error())};
        auto retained =
            retainTypedResource(device, device.imageViews, decoded.value(), "retain_metal_image_view");
        if (retained.isErr())
            return Result<RetainedRhiResourceLease, RhiError>{err(std::move(retained).error())};
        return Result<RetainedRhiResourceLease, RhiError>{
            ok(RetainedRhiResourceLease{std::move(retained).value()})};
    }
    case ResourceKind::Sampler: {
        auto decoded = decodeResourceKey<SamplerHandle>(key, "retain_metal_sampler");
        if (decoded.isErr())
            return Result<RetainedRhiResourceLease, RhiError>{err(std::move(decoded).error())};
        auto retained = retainTypedResource(device, device.samplers, decoded.value(), "retain_metal_sampler");
        if (retained.isErr())
            return Result<RetainedRhiResourceLease, RhiError>{err(std::move(retained).error())};
        return Result<RetainedRhiResourceLease, RhiError>{
            ok(RetainedRhiResourceLease{std::move(retained).value()})};
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
        return Result<uint64_t, RhiError>{ok(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(
            (__bridge void *)payload->native.buffer)))};
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
        return Result<uint64_t, RhiError>{ok(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(
            (__bridge void *)payload->native.texture)))};
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
        return Result<uint64_t, RhiError>{ok(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(
            (__bridge void *)payload->native.texture)))};
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
        return Result<uint64_t, RhiError>{ok(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(
            (__bridge void *)payload->native.sampler)))};
    }
    }
    return Result<uint64_t, RhiError>{
        err(RhiError{RhiErrorCode::InvalidArgument, {"resolve_metal_resource", key, static_cast<uint32_t>(kind)}})};
}

Result<void, RhiError> describeImageResource(VernonRhiDevice handle, uint64_t key,
                                              VernonRhiImageDescriptor *descriptor) noexcept {
    if (!descriptor)
        return Result<void, RhiError>{
            err(RhiError{RhiErrorCode::InvalidArgument, {"describe_metal_image", key, 0}})};
    auto anchor = lookupMetalDevice(handle);
    if (anchor.isErr())
        return Result<void, RhiError>{err(std::move(anchor).error())};
    auto decoded = decodeResourceKey<ImageHandle>(key, "describe_metal_image");
    if (decoded.isErr())
        return Result<void, RhiError>{err(std::move(decoded).error())};
    MetalDevice &device = anchor.value().device();
    MetalImagePayload *payload{};
    {
        std::lock_guard<std::mutex> slots(device.slotMutex);
        auto found = payloadAt(device.images, decoded.value().index, "describe_metal_image");
        if (found.isErr())
            return Result<void, RhiError>{err(std::move(found).error())};
        payload = found.value();
    }
    std::lock_guard<std::mutex> guard(device.mutex);
    *descriptor = payload->descriptor;
    return Result<void, RhiError>{ok()};
}

Result<void, RhiError> describeImageViewResource(VernonRhiDevice handle, uint64_t key,
                                                  VernonRhiImageViewDescriptor *view,
                                                  VernonRhiImageDescriptor *image,
                                                  uint64_t *parentKey) noexcept {
    if (!view || !image || !parentKey)
        return Result<void, RhiError>{
            err(RhiError{RhiErrorCode::InvalidArgument, {"describe_metal_image_view", key, 0}})};
    auto anchor = lookupMetalDevice(handle);
    if (anchor.isErr())
        return Result<void, RhiError>{err(std::move(anchor).error())};
    auto decoded = decodeResourceKey<ImageViewHandle>(key, "describe_metal_image_view");
    if (decoded.isErr())
        return Result<void, RhiError>{err(std::move(decoded).error())};
    MetalDevice &device = anchor.value().device();
    MetalImageViewPayload *payload{};
    {
        std::lock_guard<std::mutex> slots(device.slotMutex);
        auto found = payloadAt(device.imageViews, decoded.value().index, "describe_metal_image_view");
        if (found.isErr())
            return Result<void, RhiError>{err(std::move(found).error())};
        payload = found.value();
    }
    std::lock_guard<std::mutex> guard(device.mutex);
    *view = payload->descriptor;
    *image = payload->imageDescriptor;
    *parentKey = payload->parentKey;
    return Result<void, RhiError>{ok()};
}

bool beginCommands(VernonRhiDevice handle, uint64_t &native, VernonRhiBackend &backend) {
    auto anchor = lookupMetalDevice(handle);
    if (anchor.isErr())
        return false;
    MetalDevice &device = anchor.value().device();
    std::lock_guard<std::mutex> guard(device.mutex);
    if (!device.state.beginCommands(native, device.error))
        return false;
    backend = VERNON_RHI_BACKEND_METAL;
    return true;
}

bool submitCommands(VernonRhiDevice handle, uint64_t native, bool, bool &completed, bool &externalCompletion) {
    externalCompletion = false;
    auto anchor = lookupMetalDevice(handle);
    if (anchor.isErr())
        return false;
    MetalDevice &device = anchor.value().device();
    std::lock_guard<std::mutex> guard(device.mutex);
    completed = false;
    return device.state.submitCommands(native, device.error);
}

bool pollCommands(VernonRhiDevice handle, uint64_t native, bool &completed, bool &succeeded) {
    auto anchor = lookupMetalDevice(handle);
    if (anchor.isErr())
        return false;
    MetalDevice &device = anchor.value().device();
    std::lock_guard<std::mutex> guard(device.mutex);
    return device.state.pollCommands(native, completed, succeeded, device.error);
}

bool completeBorrowedCommands(VernonRhiDevice handle, uint64_t native) {
    auto anchor = lookupMetalDevice(handle);
    if (anchor.isErr())
        return false;
    MetalDevice &device = anchor.value().device();
    std::lock_guard<std::mutex> guard(device.mutex);
    return device.state.completeCommands(native, device.error);
}

void abandonCommands(VernonRhiDevice handle, uint64_t native) {
    auto anchor = lookupMetalDevice(handle);
    if (anchor.isErr())
        return;
    MetalDevice &device = anchor.value().device();
    std::lock_guard<std::mutex> guard(device.mutex);
    device.state.abandonCommands(native);
}

bool recordBarriers(VernonRhiDevice handle, uint64_t, uint64_t native, const VernonRhiBarrier *barriers,
                    size_t barrierCount) {
    auto anchor = lookupMetalDevice(handle);
    if (anchor.isErr() || !native || (barrierCount && !barriers))
        return false;
    MetalDevice &device = anchor.value().device();
    for (size_t index = 0; index < barrierCount; ++index) {
        const VernonRhiBarrier &barrier = barriers[index];
        if (barrier.is_image) {
            auto pinned = pinPayload(device, device.images,
                                     ImageHandle{barrier.image.index, barrier.image.generation}, "record_barrier");
            if (pinned.isErr())
                return false;
            std::lock_guard<std::mutex> guard(device.mutex);
            if (!supportsImageState(pinned.value().first->descriptor, barrier.old_state) ||
                !supportsImageState(pinned.value().first->descriptor, barrier.new_state))
                return false;
        } else {
            auto pinned = pinPayload(device, device.buffers,
                                     BufferHandle{barrier.buffer.index, barrier.buffer.generation}, "record_barrier");
            if (pinned.isErr())
                return false;
            std::lock_guard<std::mutex> guard(device.mutex);
            if (!supportsBufferState(pinned.value().first->descriptor, barrier.old_state) ||
                !supportsBufferState(pinned.value().first->descriptor, barrier.new_state))
                return false;
        }
    }
    return true;
}

bool recordBufferCopy(VernonRhiDevice handle, uint64_t native, VernonRhiBuffer source, uint64_t sourceOffset,
                      VernonRhiBuffer destination, uint64_t destinationOffset, uint64_t size) {
    auto anchor = lookupMetalDevice(handle);
    if (anchor.isErr() || !native || !size)
        return false;
    MetalDevice &device = anchor.value().device();
    auto sourcePin =
        pinPayload(device, device.buffers, BufferHandle{source.index, source.generation}, "record_buffer_copy");
    if (sourcePin.isErr())
        return false;
    auto destinationPin = pinPayload(device, device.buffers,
                                     BufferHandle{destination.index, destination.generation}, "record_buffer_copy");
    if (destinationPin.isErr())
        return false;
    MetalBufferPayload &sourcePayload = *sourcePin.value().first;
    MetalBufferPayload &destinationPayload = *destinationPin.value().first;
    std::lock_guard<std::mutex> guard(device.mutex);
    if (sourceOffset > sourcePayload.descriptor.size ||
        size > sourcePayload.descriptor.size - sourceOffset ||
        destinationOffset > destinationPayload.descriptor.size ||
        size > destinationPayload.descriptor.size - destinationOffset)
        return false;
    return device.state.copyBuffer(native, sourcePayload.native, sourceOffset, destinationPayload.native,
                                   destinationOffset, size, device.error);
}

bool supportsImageCopy(VernonRhiDevice handle) { return lookupMetalDevice(handle).isOk(); }

bool recordImageCopy(VernonRhiDevice handle, uint64_t, uint64_t native, VernonRhiImage source,
                     VernonRhiImage destination, const VernonRhiImageCopyRegion *regions, size_t regionCount) {
    auto anchor = lookupMetalDevice(handle);
    if (anchor.isErr() || !native || !regions || !regionCount)
        return false;
    MetalDevice &device = anchor.value().device();
    auto sourcePin =
        pinPayload(device, device.images, ImageHandle{source.index, source.generation}, "record_image_copy");
    if (sourcePin.isErr())
        return false;
    auto destinationPin = pinPayload(device, device.images,
                                     ImageHandle{destination.index, destination.generation}, "record_image_copy");
    if (destinationPin.isErr())
        return false;
    std::lock_guard<std::mutex> guard(device.mutex);
    return device.state.copyImage(native, sourcePin.value().first->native, destinationPin.value().first->native,
                                  regions, regionCount, device.error);
}

bool restartRendering(uint64_t native, vernon::rhi::metal::RenderingState &rendering, int32_t x, int32_t y,
                      uint32_t width, uint32_t height, uint32_t layers, int32_t colorLocation,
                      const float *clearColor, float clearDepth, uint32_t clearStencil, uint32_t aspects) {
    if (!native || !rendering.encoder || x != 0 || y != 0 || layers != 1)
        return false;
    id<MTLTexture> target = colorLocation >= 0 ? rendering.colorTextures[colorLocation]
                                               : rendering.depthStencilTexture;
    if (!target || width != target.width || height != target.height)
        return false;
    if ((aspects & VERNON_RHI_ATTACHMENT_STENCIL) &&
        target.pixelFormat != MTLPixelFormatDepth32Float_Stencil8)
        return false;
    for (size_t index = 0; index < rendering.colorTextures.size(); ++index)
        if (rendering.colorTextures[index])
            [rendering.encoder setColorStoreAction:MTLStoreActionStore atIndex:index];
    if (rendering.depthStencilTexture) {
        [rendering.encoder setDepthStoreAction:MTLStoreActionStore];
        if (rendering.depthStencilTexture.pixelFormat == MTLPixelFormatDepth32Float_Stencil8)
            [rendering.encoder setStencilStoreAction:MTLStoreActionStore];
    }
    [rendering.encoder endEncoding];

    MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
    for (size_t index = 0; index < rendering.colorTextures.size(); ++index) {
        id<MTLTexture> texture = rendering.colorTextures[index];
        if (!texture)
            continue;
        auto *attachment = pass.colorAttachments[index];
        attachment.texture = texture;
        attachment.loadAction = static_cast<int32_t>(index) == colorLocation ? MTLLoadActionClear
                                                                             : MTLLoadActionLoad;
        attachment.storeAction = MTLStoreActionStore;
        if (static_cast<int32_t>(index) == colorLocation)
            attachment.clearColor =
                MTLClearColorMake(clearColor[0], clearColor[1], clearColor[2], clearColor[3]);
    }
    if (rendering.depthStencilTexture) {
        pass.depthAttachment.texture = rendering.depthStencilTexture;
        pass.depthAttachment.loadAction = aspects & VERNON_RHI_ATTACHMENT_DEPTH ? MTLLoadActionClear
                                                                                : MTLLoadActionLoad;
        pass.depthAttachment.storeAction = MTLStoreActionStore;
        pass.depthAttachment.clearDepth = clearDepth;
        if (rendering.depthStencilTexture.pixelFormat == MTLPixelFormatDepth32Float_Stencil8) {
            pass.stencilAttachment.texture = rendering.depthStencilTexture;
            pass.stencilAttachment.loadAction = aspects & VERNON_RHI_ATTACHMENT_STENCIL ? MTLLoadActionClear
                                                                                        : MTLLoadActionLoad;
            pass.stencilAttachment.storeAction = MTLStoreActionStore;
            pass.stencilAttachment.clearStencil = clearStencil;
        }
    }
    id<MTLCommandBuffer> commandBuffer =
        (__bridge id<MTLCommandBuffer>)(reinterpret_cast<void *>(static_cast<uintptr_t>(native)));
    rendering.encoder = [commandBuffer renderCommandEncoderWithDescriptor:pass];
    return rendering.encoder != nil;
}

bool endRendering(VernonRhiDevice handle, uint64_t native, VernonRhiBackend backend, uint32_t backendKind,
                  uint32_t colorDiscardMask, uint32_t depthStencilDiscard, const uint64_t *, size_t, uint64_t,
                  uint64_t renderingObject) {
    auto anchor = lookupMetalDevice(handle);
    if (anchor.isErr() || backend != VERNON_RHI_BACKEND_METAL ||
        backendKind != vernon::rhi::CommandRenderingDynamic || !renderingObject)
        return false;
    auto *rendering = reinterpret_cast<vernon::rhi::metal::RenderingState *>(
        static_cast<uintptr_t>(renderingObject));
    for (size_t index = 0; index < rendering->colorTextures.size(); ++index)
        if (rendering->colorTextures[index])
            [rendering->encoder setColorStoreAction:(colorDiscardMask & (uint32_t{1} << index))
                                                        ? MTLStoreActionDontCare
                                                        : MTLStoreActionStore
                                            atIndex:index];
    if (rendering->depthStencilTexture) {
        [rendering->encoder setDepthStoreAction:(depthStencilDiscard & VERNON_RHI_ATTACHMENT_DEPTH)
                                                    ? MTLStoreActionDontCare
                                                    : MTLStoreActionStore];
        if (rendering->depthStencilTexture.pixelFormat == MTLPixelFormatDepth32Float_Stencil8)
            [rendering->encoder setStencilStoreAction:(depthStencilDiscard & VERNON_RHI_ATTACHMENT_STENCIL)
                                                          ? MTLStoreActionDontCare
                                                          : MTLStoreActionStore];
    }
    [rendering->encoder endEncoding];
    delete rendering;
    return true;
}

bool clearColor(VernonRhiDevice handle, uint64_t native, VernonRhiBackend backend, uint32_t backendKind,
                uint64_t renderingObject, int32_t x, int32_t y, uint32_t width, uint32_t height, uint32_t layers,
                uint64_t target, uint32_t location, const float color[4]) {
    auto anchor = lookupMetalDevice(handle);
    if (anchor.isErr() || backend != VERNON_RHI_BACKEND_METAL ||
        backendKind != vernon::rhi::CommandRenderingDynamic || !target || location >= 8 || !color)
        return false;
    id<MTLTexture> texture = (__bridge id<MTLTexture>)(reinterpret_cast<void *>(static_cast<uintptr_t>(target)));
    auto *rendering =
        reinterpret_cast<vernon::rhi::metal::RenderingState *>(static_cast<uintptr_t>(renderingObject));
    return rendering && rendering->colorTextures[location] == texture &&
           restartRendering(native, *rendering, x, y, width, height, layers, static_cast<int32_t>(location), color,
                            1.0f, 0, 0);
}

bool clearDepthStencil(VernonRhiDevice handle, uint64_t native, VernonRhiBackend backend, uint32_t backendKind,
                       uint64_t renderingObject, int32_t x, int32_t y, uint32_t width, uint32_t height, uint32_t layers,
                       uint64_t target, float depth, uint32_t stencil, uint32_t aspects) {
    auto anchor = lookupMetalDevice(handle);
    if (anchor.isErr() || backend != VERNON_RHI_BACKEND_METAL ||
        backendKind != vernon::rhi::CommandRenderingDynamic || !target ||
        (aspects & ~(VERNON_RHI_ATTACHMENT_DEPTH | VERNON_RHI_ATTACHMENT_STENCIL)) || !aspects)
        return false;
    id<MTLTexture> texture = (__bridge id<MTLTexture>)(reinterpret_cast<void *>(static_cast<uintptr_t>(target)));
    auto *rendering =
        reinterpret_cast<vernon::rhi::metal::RenderingState *>(static_cast<uintptr_t>(renderingObject));
    return rendering && rendering->depthStencilTexture == texture &&
           restartRendering(native, *rendering, x, y, width, height, layers, -1, nullptr, depth, stencil, aspects);
}

} // namespace vernon::rhi::metal_api

uint32_t vernon::rhi::metalTexturePixelFormat(VernonTextureFormat format) {
    VernonRhiFormat rhiFormat{};
    switch (format) {
    case VERNON_TEXTURE_RGBA8_UNORM:
        rhiFormat = VERNON_RHI_FORMAT_RGBA8_UNORM;
        break;
    case VERNON_TEXTURE_RGBA8_SRGB:
        rhiFormat = VERNON_RHI_FORMAT_RGBA8_SRGB;
        break;
    case VERNON_TEXTURE_RGBA16_FLOAT:
        rhiFormat = VERNON_RHI_FORMAT_RGBA16_FLOAT;
        break;
    case VERNON_TEXTURE_RGBA32_FLOAT:
        rhiFormat = VERNON_RHI_FORMAT_RGBA32_FLOAT;
        break;
    case VERNON_TEXTURE_R8_UNORM:
        rhiFormat = VERNON_RHI_FORMAT_R8_UNORM;
        break;
    case VERNON_TEXTURE_R16_FLOAT:
        rhiFormat = VERNON_RHI_FORMAT_R16_FLOAT;
        break;
    case VERNON_TEXTURE_R32_FLOAT:
        rhiFormat = VERNON_RHI_FORMAT_R32_FLOAT;
        break;
    case VERNON_TEXTURE_RG8_UNORM:
        rhiFormat = VERNON_RHI_FORMAT_RG8_UNORM;
        break;
    case VERNON_TEXTURE_RGB8_UNORM:
        rhiFormat = VERNON_RHI_FORMAT_RGB8_UNORM;
        break;
    case VERNON_TEXTURE_R11G11B10_FLOAT:
        rhiFormat = VERNON_RHI_FORMAT_R11G11B10_FLOAT;
        break;
    case VERNON_TEXTURE_D32_FLOAT:
        rhiFormat = VERNON_RHI_FORMAT_D32_FLOAT;
        break;
    case VERNON_TEXTURE_D32_FLOAT_S8_UINT:
        rhiFormat = VERNON_RHI_FORMAT_D32_FLOAT_S8_UINT;
        break;
    }
    return static_cast<uint32_t>(vernon::rhi::metal::pixelFormat(rhiFormat));
}

const vernon::rhi::BackendDispatch &vernon::rhi::metalBackendDispatch() {
    using namespace metal_api;
    static const BackendDispatch dispatch{
        VERNON_RHI_BACKEND_METAL,
        BackendCommandIndependentRecording | BackendCommandConcurrentSubmission,
        nullptr,
        ownsDevice,
        createOwnedDevice,
        destroyDevice,
        lastError,
        synchronize,
        deviceStateForBackend,
        commandState,
        createBuffer,
        uploadBuffer,
        uploadBufferRanges,
        downloadBufferRanges,
        downloadBuffer,
        destroyBuffer,
        isBufferValid,
        getBufferNativeHandle,
        createImage,
        nullptr,
        uploadImage,
        downloadImage,
        downloadImageBatch,
        generateImageMipmaps,
        nullptr,
        destroyImage,
        isImageValid,
        getImageNativeHandle,
        createSampler,
        destroySampler,
        isSamplerValid,
        bufferResource,
        imageResource,
        samplerResource,
        retainResource,
        resolveResource,
        describeImageResource,
        beginCommands,
        submitCommands,
        pollCommands,
        completeBorrowedCommands,
        abandonCommands,
        recordBarriers,
        recordBufferCopy,
        supportsImageCopy,
        recordImageCopy,
        endRendering,
        clearColor,
        clearDepthStencil,
        nullptr,
        createImageView,
        destroyImageView,
        getImageViewNativeHandle,
        imageViewResource,
        describeImageViewResource,
    };
    return dispatch;
}
