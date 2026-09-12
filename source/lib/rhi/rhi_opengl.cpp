#include "backend_dispatch.h"
#include "image_data_layout.h"
#include "image_descriptor_validation.h"
#include "opengl_backend.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <limits>
#include <mutex>
#include <new>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace {

using vernon::rhi::opengl::DeviceState;
using vernon::rhi::opengl::Enum;
using vernon::rhi::opengl::Image;
using vernon::rhi::opengl::Int;
using vernon::rhi::opengl::Size;
using vernon::rhi::opengl::Uint;

struct FormatInfo {
    Int internal{};
    Enum external{};
    Enum allocationType{};
};

using BufferLifecycle = vernon::rhi::ResourceLifecycleSlot<vernon::rhi::BufferResourceTag>;
using ImageLifecycle = vernon::rhi::ResourceLifecycleSlot<vernon::rhi::ImageResourceTag>;
using ImageViewLifecycle = vernon::rhi::ResourceLifecycleSlot<vernon::rhi::ImageViewResourceTag>;
using SamplerLifecycle = vernon::rhi::ResourceLifecycleSlot<vernon::rhi::SamplerResourceTag>;
using ImageLease = vernon::rhi::RetainedResourceLease<vernon::rhi::ImageResourceTag>;

constexpr std::size_t kMaximumOpenGLDevices = 256;

struct ImageSlot {
    explicit ImageSlot(ImageLifecycle lifecycle) noexcept : lifecycle(std::move(lifecycle)) {}
    ImageLifecycle lifecycle;
    Image image;
    VernonRhiImageDescriptor descriptor{};
    Enum target{};
};

struct IdentityImageViewBacking {
    Uint name{};
};
struct OwnedTextureViewBacking {
    Image image;
};
using ImageViewBacking = std::variant<IdentityImageViewBacking, OwnedTextureViewBacking>;

struct ImageViewSlot {
    explicit ImageViewSlot(ImageViewLifecycle lifecycle) noexcept : lifecycle(std::move(lifecycle)) {}
    ImageViewLifecycle lifecycle;
    ImageViewBacking backing;
    VernonRhiImageViewDescriptor descriptor{};
    VernonRhiImageDescriptor imageDescriptor{};
    uint64_t parentKey{};
    Enum target{};
    vernon::Option<ImageLease> parent;
};

struct BufferSlot {
    explicit BufferSlot(BufferLifecycle lifecycle) noexcept : lifecycle(std::move(lifecycle)) {}
    BufferLifecycle lifecycle;
    vernon::rhi::opengl::Buffer buffer;
    VernonRhiBufferDescriptor descriptor{};
};

struct SamplerSlot {
    explicit SamplerSlot(SamplerLifecycle lifecycle) noexcept : lifecycle(std::move(lifecycle)) {}
    SamplerLifecycle lifecycle;
    vernon::rhi::opengl::Sampler sampler;
};

struct OpenGLDevice {
    OpenGLDevice() noexcept = default;
    OpenGLDevice(const OpenGLDevice &) = delete;
    OpenGLDevice &operator=(const OpenGLDevice &) = delete;
    OpenGLDevice(OpenGLDevice &&) = delete;
    OpenGLDevice &operator=(OpenGLDevice &&) = delete;

    DeviceState state;
    vernon::rhi::StableResourceSlotContainer<BufferSlot> buffers;
    vernon::rhi::StableResourceSlotContainer<ImageSlot> images;
    vernon::rhi::StableResourceSlotContainer<ImageViewSlot> imageViews;
    vernon::rhi::StableResourceSlotContainer<SamplerSlot> samplers;
    vernon::rhi::CommandDeviceStateRef commandState;
    std::string error;
    std::mutex resourceAllocationMutex;
    std::mutex mutex;
};

using OpenGLDeviceRegistry = vernon::rhi::DeviceRegistry<OpenGLDevice, kMaximumOpenGLDevices>;
OpenGLDeviceRegistry devices;

VernonRhiDevice invalidDevice() { return {static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0}; }

template <typename Handle> uint64_t resourceKey(Handle handle) {
    return (static_cast<uint64_t>(handle.generation) << 32) | (static_cast<uint64_t>(handle.index) + 1);
}

template <typename Handle> bool decodeResourceKey(uint64_t key, Handle &handle) {
    const uint64_t encodedIndex = key & UINT32_MAX;
    const uint32_t generation = static_cast<uint32_t>(key >> 32);
    if (!encodedIndex || !generation || encodedIndex - 1 > UINT32_MAX)
        return false;
    handle = {static_cast<uint32_t>(encodedIndex - 1), generation};
    return true;
}

template <typename Slot>
Slot *indexedSlot(vernon::rhi::StableResourceSlotContainer<Slot> &slots, uint32_t slotIndex) noexcept {
    return slots.get(slotIndex);
}

template <typename Slot, typename Lifecycle>
vernon::Result<Slot *, vernon::RhiError> availableSlot(vernon::rhi::StableResourceSlotContainer<Slot> &slots) noexcept {
    for (std::size_t index = 0; index < slots.size(); ++index)
        if (!slots[index].lifecycle.snapshot().occupied)
            return vernon::Result<Slot *, vernon::RhiError>{vernon::ok(&slots[index])};
    auto lifecycle = Lifecycle::create(static_cast<uint32_t>(slots.size()));
    if (lifecycle.isErr())
        return vernon::Result<Slot *, vernon::RhiError>{vernon::err(std::move(lifecycle).error())};
    return slots.emplace("allocate_opengl_resource_slot", std::move(lifecycle).value());
}

auto lookupDevice(VernonRhiDevice handle) noexcept { return devices.lookup({handle.index, handle.generation}); }

template <typename Slot, typename Handle> auto pinSlot(Slot *slot, Handle handle) noexcept {
    using PinResult = decltype(slot->lifecycle.pin({handle.index, handle.generation}));
    if (!slot)
        return PinResult{vernon::err(vernon::RhiError{vernon::RhiErrorCode::InvalidArgument,
                                                      {"pin_opengl_resource", handle.generation, handle.index}})};
    return slot->lifecycle.pin({handle.index, handle.generation});
}

VernonRhiStatus statusFromError(const vernon::RhiError &error) noexcept { return vernon::toVernonRhiStatus(error); }

template <typename Slot, typename Handle> bool publicSlot(Slot *slot, Handle handle) noexcept {
    if (!slot)
        return false;
    auto pin = pinSlot(slot, handle);
    return pin.isOk();
}

Uint imageViewName(const ImageViewSlot &view) {
    if (const auto *owned = std::get_if<OwnedTextureViewBacking>(&view.backing))
        return owned->image.name;
    return std::get<IdentityImageViewBacking>(view.backing).name;
}

vernon::Result<void, vernon::RhiError> teardownBuffer(void *context, BufferLifecycle::Handle handle) noexcept {
    auto &device = *static_cast<OpenGLDevice *>(context);
    BufferSlot *slot = indexedSlot(device.buffers, handle.index);
    if (!slot)
        return vernon::Result<void, vernon::RhiError>{vernon::err(vernon::RhiError{
            vernon::RhiErrorCode::LifecycleFailure, {"teardown_opengl_buffer", handle.generation, handle.index}})};
    std::lock_guard<std::mutex> guard(device.mutex);
    device.state.makeCurrent();
    device.state.destroyBuffer(slot->buffer);
    slot->buffer = {};
    slot->descriptor = {};
    return vernon::Result<void, vernon::RhiError>{vernon::ok()};
}

vernon::Result<void, vernon::RhiError> teardownImage(void *context, ImageLifecycle::Handle handle) noexcept {
    auto &device = *static_cast<OpenGLDevice *>(context);
    ImageSlot *slot = indexedSlot(device.images, handle.index);
    if (!slot)
        return vernon::Result<void, vernon::RhiError>{vernon::err(vernon::RhiError{
            vernon::RhiErrorCode::LifecycleFailure, {"teardown_opengl_image", handle.generation, handle.index}})};
    std::lock_guard<std::mutex> guard(device.mutex);
    device.state.makeCurrent();
    device.state.destroyImage(slot->image);
    slot->image = {};
    slot->descriptor = {};
    slot->target = 0;
    return vernon::Result<void, vernon::RhiError>{vernon::ok()};
}

vernon::Result<void, vernon::RhiError> teardownImageView(void *context, ImageViewLifecycle::Handle handle) noexcept {
    auto &device = *static_cast<OpenGLDevice *>(context);
    ImageViewSlot *slot = indexedSlot(device.imageViews, handle.index);
    if (!slot)
        return vernon::Result<void, vernon::RhiError>{vernon::err(vernon::RhiError{
            vernon::RhiErrorCode::LifecycleFailure, {"teardown_opengl_image_view", handle.generation, handle.index}})};
    if (!slot->parent)
        return vernon::Result<void, vernon::RhiError>{
            vernon::err(vernon::RhiError{vernon::RhiErrorCode::LifecycleFailure,
                                         {"release_opengl_image_view_parent", handle.generation, handle.index}})};
    auto prepared = slot->parent.value().prepareRelease();
    if (prepared.isErr())
        return vernon::Result<void, vernon::RhiError>{vernon::err(std::move(prepared).error())};
    {
        std::lock_guard<std::mutex> guard(device.mutex);
        device.state.makeCurrent();
        if (auto *owned = std::get_if<OwnedTextureViewBacking>(&slot->backing))
            device.state.destroyImage(owned->image);
        slot->backing.emplace<IdentityImageViewBacking>();
        slot->descriptor = {};
        slot->imageDescriptor = {};
        slot->parentKey = 0;
        slot->target = 0;
    }
    if (prepared.value().commit().isErr())
        vernon::resultContractViolation();
    slot->parent.reset();
    return vernon::Result<void, vernon::RhiError>{vernon::ok()};
}

vernon::Result<void, vernon::RhiError> teardownSampler(void *context, SamplerLifecycle::Handle handle) noexcept {
    auto &device = *static_cast<OpenGLDevice *>(context);
    SamplerSlot *slot = indexedSlot(device.samplers, handle.index);
    if (!slot)
        return vernon::Result<void, vernon::RhiError>{vernon::err(vernon::RhiError{
            vernon::RhiErrorCode::LifecycleFailure, {"teardown_opengl_sampler", handle.generation, handle.index}})};
    std::lock_guard<std::mutex> guard(device.mutex);
    device.state.makeCurrent();
    device.state.destroySampler(slot->sampler);
    slot->sampler = {};
    return vernon::Result<void, vernon::RhiError>{vernon::ok()};
}

vernon::Result<void, vernon::RhiError> teardownDevice(OpenGLDevice &device) noexcept {
    std::lock_guard<std::mutex> guard(device.mutex);
    device.state.makeCurrent();
    return vernon::Result<void, vernon::RhiError>{vernon::ok()};
}

bool formatInfo(VernonRhiFormat format, FormatInfo &result) {
    constexpr Enum red = 0x1903;
    constexpr Enum rg = 0x8227;
    constexpr Enum rgb = 0x1907;
    constexpr Enum rgba = 0x1908;
    constexpr Enum depth = 0x1902;
    constexpr Enum depthStencil = 0x84F9;
    constexpr Enum unsignedByte = 0x1401;
    constexpr Enum floating = 0x1406;
    switch (format) {
    case VERNON_RHI_FORMAT_R8_UNORM:
        result = {0x8229, red, unsignedByte};
        return true;
    case VERNON_RHI_FORMAT_RG8_UNORM:
        result = {0x822B, rg, unsignedByte};
        return true;
    case VERNON_RHI_FORMAT_RGB8_UNORM:
        result = {0x8051, rgb, unsignedByte};
        return true;
    case VERNON_RHI_FORMAT_RGBA8_UNORM:
        result = {0x8058, rgba, unsignedByte};
        return true;
    case VERNON_RHI_FORMAT_RGBA8_SRGB:
        result = {0x8C43, rgba, unsignedByte};
        return true;
    case VERNON_RHI_FORMAT_R16_FLOAT:
        result = {0x822D, red, floating};
        return true;
    case VERNON_RHI_FORMAT_RGBA16_FLOAT:
        result = {0x881A, rgba, floating};
        return true;
    case VERNON_RHI_FORMAT_R32_FLOAT:
        result = {0x822E, red, floating};
        return true;
    case VERNON_RHI_FORMAT_RG32_FLOAT:
        result = {0x8230, rg, floating};
        return true;
    case VERNON_RHI_FORMAT_RGB32_FLOAT:
        result = {0x8815, rgb, floating};
        return true;
    case VERNON_RHI_FORMAT_RGBA32_FLOAT:
        result = {0x8814, rgba, floating};
        return true;
    case VERNON_RHI_FORMAT_R11G11B10_FLOAT:
        result = {0x8C3A, rgb, floating};
        return true;
    case VERNON_RHI_FORMAT_D32_FLOAT:
        result = {vernon::rhi::opengl::kDepthComponent32f, depth, floating};
        return true;
    case VERNON_RHI_FORMAT_D32_FLOAT_S8_UINT:
        result = {vernon::rhi::opengl::kDepth32fStencil8, depthStencil, 0x8DAD};
        return true;
    case VERNON_RHI_FORMAT_UNDEFINED:
        return false;
    }
    return false;
}

Enum imageTarget(VernonRhiImageDimension dimension) {
    switch (dimension) {
    case VERNON_RHI_IMAGE_2D:
        return vernon::rhi::opengl::kTexture2D;
    case VERNON_RHI_IMAGE_3D:
        return vernon::rhi::opengl::kTexture3D;
    case VERNON_RHI_IMAGE_CUBE:
        return vernon::rhi::opengl::kTextureCubeMap;
    }
    return 0;
}

Int samplerFilter(uint32_t value) {
    constexpr Int values[] = {0x2600, 0x2601, 0x2700, 0x2703, 0x2701, 0x2702};
    return value < sizeof(values) / sizeof(values[0]) ? values[value] : 0;
}

Int samplerAddress(uint32_t value) {
    constexpr Int values[] = {0x2901, 0x812F, 0x8370};
    return value < sizeof(values) / sizeof(values[0]) ? values[value] : 0;
}

Enum imageDataFormat(VernonRhiImageDataFormat format) {
    constexpr Enum values[] = {0x1903, 0x8227, 0x1907, 0x80E0, 0x1908, 0x80E1, 0x1902, 0x84F9};
    return static_cast<uint32_t>(format) < sizeof(values) / sizeof(values[0]) ? values[static_cast<uint32_t>(format)]
                                                                              : 0;
}

VernonRhiStatus fail(OpenGLDevice &device, std::string message,
                     VernonRhiStatus status = VERNON_RHI_STATUS_INVALID_ARGUMENT) {
    device.error = std::move(message);
    return status;
}

vernon::Result<VernonRhiDevice, vernon::RhiError>
createOpenGLDeviceResult(const VernonOpenGLContextCallbacks *callbacks, bool embeddedProfile) {
    if (!callbacks)
        return vernon::Result<VernonRhiDevice, vernon::RhiError>{
            vernon::err(vernon::RhiError{vernon::RhiErrorCode::InvalidArgument, {"create_opengl_device", 0, 0}})};
    auto reservation = devices.reserve();
    if (reservation.isErr())
        return vernon::Result<VernonRhiDevice, vernon::RhiError>{vernon::err(std::move(reservation).error())};
    auto owner = reservation.value().retainOwner();
    if (owner.isErr())
        return vernon::Result<VernonRhiDevice, vernon::RhiError>{vernon::err(std::move(owner).error())};
    auto commandState = vernon::rhi::createCommandDeviceState(std::move(owner).value());
    if (commandState.isErr())
        return vernon::Result<VernonRhiDevice, vernon::RhiError>{vernon::err(std::move(commandState).error())};
    OpenGLDevice &device = reservation.value().device();
    device.commandState = std::move(commandState).value();
    if (!device.state.initialize(*callbacks, embeddedProfile, device.error))
        return vernon::Result<VernonRhiDevice, vernon::RhiError>{
            vernon::err(vernon::RhiError{vernon::RhiErrorCode::BackendFailure, {"initialize_opengl_device", 0, 0}})};
    auto published = devices.publish(std::move(reservation).value());
    if (published.isErr())
        return vernon::Result<VernonRhiDevice, vernon::RhiError>{vernon::err(std::move(published).error())};
    return vernon::Result<VernonRhiDevice, vernon::RhiError>{
        vernon::ok(VernonRhiDevice{published.value().index, published.value().generation})};
}
} // namespace

namespace vernon::rhi::opengl_api {

extern "C" VERNON_RHI_CAPI VernonRhiDevice vernonRhiCreateOpenGLDevice(const VernonOpenGLContextCallbacks *callbacks,
                                                                       uint32_t embeddedProfile) {
    auto created = createOpenGLDeviceResult(callbacks, embeddedProfile != 0);
    if (created.isOk())
        return created.value();
    setDeviceCreationError("OpenGL device creation failed");
    return invalidDevice();
}

VernonRhiDevice createOwnedDevice(const VernonRhiOwnedDeviceDescriptor *descriptor) {

    if (!descriptor || descriptor->struct_size < sizeof(*descriptor))
        return invalidDevice();
    if (descriptor->flags & ~static_cast<uint32_t>(VERNON_RHI_OWNED_DEVICE_FORCE_SOFTWARE))
        return invalidDevice();
    if (descriptor->backend != VERNON_RHI_BACKEND_DIRECTX12 && descriptor->flags)
        return invalidDevice();
    if (descriptor->backend == VERNON_RHI_BACKEND_OPENGL || descriptor->backend == VERNON_RHI_BACKEND_OPENGL_ES)
        return vernonRhiCreateOpenGLDevice(descriptor->opengl_callbacks,
                                           descriptor->backend == VERNON_RHI_BACKEND_OPENGL_ES);
    return invalidDevice();
}

vernon::Result<void, vernon::RhiError> destroyDevice(VernonRhiDevice handle) noexcept {
    return devices.remove({handle.index, handle.generation}, teardownDevice);
}

vernon::Result<vernon::rhi::CommandDeviceStateRef, vernon::RhiError> commandState(VernonRhiDevice handle) noexcept {
    auto device = lookupDevice(handle);
    if (device.isErr())
        return vernon::Result<vernon::rhi::CommandDeviceStateRef, vernon::RhiError>{
            vernon::err(std::move(device).error())};
    return device.value().device().commandState.retain();
}

VernonStringView lastError(VernonRhiDevice handle) {
    auto device = lookupDevice(handle);
    if (device.isErr())
        return {};
    OpenGLDevice &state = device.value().device();
    std::lock_guard<std::mutex> guard(state.mutex);
    return {state.error.data(), state.error.size()};
}

VernonRhiStatus synchronize(VernonRhiDevice handle) {
    auto device = lookupDevice(handle);
    if (device.isErr())
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    OpenGLDevice &state = device.value().device();
    std::lock_guard<std::mutex> guard(state.mutex);
    state.state.makeCurrent();
    state.state.driver.finish();
    return VERNON_RHI_STATUS_OK;
}

VernonRhiStatus createBuffer(VernonRhiDevice handle, const VernonRhiBufferDescriptor *descriptor,
                             VernonRhiBuffer *output) {
    auto device = lookupDevice(handle);
    if (device.isErr() || !descriptor || !output || descriptor->struct_size < sizeof(*descriptor) ||
        descriptor->size == 0 || descriptor->size > (std::numeric_limits<size_t>::max)() ||
        descriptor->memory_class > VERNON_RHI_MEMORY_READBACK)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    *output = {static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
    auto owner = device.value().retainOwner();
    if (owner.isErr())
        return statusFromError(owner.error());
    auto reservation = vernon::rhi::ResourceCreationReservation::create(owner.value());
    if (reservation.isErr())
        return statusFromError(reservation.error());
    OpenGLDevice &state = device.value().device();
    std::lock_guard<std::mutex> allocationGuard(state.resourceAllocationMutex);
    auto available = availableSlot<BufferSlot, BufferLifecycle>(state.buffers);
    if (available.isErr())
        return statusFromError(available.error());
    BufferSlot &slot = *available.value();
    std::lock_guard<std::mutex> guard(state.mutex);
    state.state.makeCurrent();
    if (!state.state.createBuffer(slot.buffer, static_cast<size_t>(descriptor->size), state.error))
        return VERNON_RHI_STATUS_INTERNAL_ERROR;
    slot.descriptor = *descriptor;
    auto published = slot.lifecycle.publish(std::move(reservation).value(), &state, teardownBuffer);
    if (published.isErr()) {
        state.state.destroyBuffer(slot.buffer);
        slot.buffer = {};
        slot.descriptor = {};
        return statusFromError(published.error());
    }
    *output = {published.value().index, published.value().generation};
    return VERNON_RHI_STATUS_OK;
}

VernonRhiStatus uploadBuffer(VernonRhiDevice handle, VernonRhiBuffer buffer, uint64_t offset, const void *source,
                             uint64_t size) {

    auto device = lookupDevice(handle);
    if (device.isErr() || !source)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    OpenGLDevice &state = device.value().device();
    BufferSlot *slot = indexedSlot(state.buffers, buffer.index);
    auto pin = pinSlot(slot, buffer);
    if (pin.isErr())
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(state.mutex);
    state.state.makeCurrent();
    if (!slot || offset > slot->descriptor.size || size > slot->descriptor.size - offset)
        return fail(state, "OpenGL RHI buffer upload range is invalid");
    return state.state.uploadBuffer(slot->buffer, static_cast<size_t>(offset), source, static_cast<size_t>(size),
                                    state.error)
               ? VERNON_RHI_STATUS_OK
               : VERNON_RHI_STATUS_INTERNAL_ERROR;
}

VernonRhiStatus uploadBufferRanges(VernonRhiDevice handle, VernonRhiBuffer buffer,
                                   const VernonRhiBufferUploadRange *ranges, size_t rangeCount) {
    auto device = lookupDevice(handle);
    if (device.isErr() || !ranges || rangeCount == 0)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    OpenGLDevice &state = device.value().device();
    BufferSlot *slot = indexedSlot(state.buffers, buffer.index);
    auto pin = pinSlot(slot, buffer);
    if (pin.isErr())
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(state.mutex);
    state.state.makeCurrent();
    for (size_t index = 0; index < rangeCount; ++index) {
        const VernonRhiBufferUploadRange &range = ranges[index];
        if (!range.source || range.size == 0 || range.size > (std::numeric_limits<size_t>::max)() ||
            range.offset > slot->descriptor.size || range.size > slot->descriptor.size - range.offset)
            return fail(state, "OpenGL RHI buffer upload range is invalid");
    }
    for (size_t index = 0; index < rangeCount; ++index) {
        const VernonRhiBufferUploadRange &range = ranges[index];
        if (!state.state.uploadBuffer(slot->buffer, static_cast<size_t>(range.offset), range.source,
                                      static_cast<size_t>(range.size), state.error))
            return VERNON_RHI_STATUS_INTERNAL_ERROR;
    }
    return VERNON_RHI_STATUS_OK;
}

VernonRhiStatus downloadBuffer(VernonRhiDevice handle, VernonRhiBuffer buffer, uint64_t offset, void *destination,
                               uint64_t size) {

    auto device = lookupDevice(handle);
    if (device.isErr() || !destination)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    OpenGLDevice &state = device.value().device();
    BufferSlot *slot = indexedSlot(state.buffers, buffer.index);
    auto pin = pinSlot(slot, buffer);
    if (pin.isErr())
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(state.mutex);
    state.state.makeCurrent();
    if (!slot || offset > slot->descriptor.size || size > slot->descriptor.size - offset)
        return fail(state, "OpenGL RHI buffer readback range is invalid");
    return state.state.downloadBuffer(slot->buffer, static_cast<size_t>(offset), destination, static_cast<size_t>(size),
                                      state.error)
               ? VERNON_RHI_STATUS_OK
               : VERNON_RHI_STATUS_INTERNAL_ERROR;
}

uint32_t isBufferValid(VernonRhiDevice handle, VernonRhiBuffer buffer) {

    auto device = lookupDevice(handle);
    if (device.isErr())
        return 0;
    OpenGLDevice &state = device.value().device();
    return publicSlot(indexedSlot(state.buffers, buffer.index), buffer);
}

VernonRhiStatus createImage(VernonRhiDevice handle, const VernonRhiImageDescriptor *descriptor,
                            VernonRhiImage *output) {
    auto device = lookupDevice(handle);
    if (device.isErr() || !descriptor || !output || descriptor->struct_size < sizeof(*descriptor))
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    *output = {static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
    FormatInfo format;
    const Enum target = imageTarget(descriptor->dimension);
    const bool validCube =
        descriptor->dimension != VERNON_RHI_IMAGE_CUBE ||
        (descriptor->width == descriptor->height && descriptor->depth == 1 && descriptor->array_layers == 6);
    const bool depthFormat =
        descriptor->format == VERNON_RHI_FORMAT_D32_FLOAT || descriptor->format == VERNON_RHI_FORMAT_D32_FLOAT_S8_UINT;
    const bool validDimension =
        (descriptor->dimension == VERNON_RHI_IMAGE_2D && descriptor->depth == 1 && descriptor->array_layers == 1) ||
        (descriptor->dimension == VERNON_RHI_IMAGE_3D && descriptor->array_layers == 1) ||
        descriptor->dimension == VERNON_RHI_IMAGE_CUBE;
    OpenGLDevice &state = device.value().device();
    if (!target || !formatInfo(descriptor->format, format) || descriptor->width == 0 || descriptor->height == 0 ||
        descriptor->depth == 0 || descriptor->mip_levels == 0 || descriptor->sample_count != 1 || !validCube ||
        !validDimension || (depthFormat && (descriptor->usage & VERNON_RHI_IMAGE_COLOR_ATTACHMENT)) ||
        (!depthFormat && (descriptor->usage & VERNON_RHI_IMAGE_DEPTH_STENCIL_ATTACHMENT)))
        return fail(state, "OpenGL RHI image descriptor is invalid");
    auto owner = device.value().retainOwner();
    if (owner.isErr())
        return statusFromError(owner.error());
    auto reservation = vernon::rhi::ResourceCreationReservation::create(owner.value());
    if (reservation.isErr())
        return statusFromError(reservation.error());
    std::lock_guard<std::mutex> allocationGuard(state.resourceAllocationMutex);
    auto available = availableSlot<ImageSlot, ImageLifecycle>(state.images);
    if (available.isErr())
        return statusFromError(available.error());
    ImageSlot &slot = *available.value();
    std::lock_guard<std::mutex> guard(state.mutex);
    state.state.makeCurrent();
    state.state.driver.genTextures(1, &slot.image.name);
    if (!slot.image.name)
        return fail(state, "OpenGL RHI image allocation failed", VERNON_RHI_STATUS_INTERNAL_ERROR);
    slot.image.imported = false;
    {
        auto &driver = state.state.driver;
        driver.bindTexture(target, slot.image.name);
        for (uint32_t mip = 0; mip < descriptor->mip_levels; ++mip) {
            const Size mipWidth = static_cast<Size>(vernon::rhi::imageMipExtent(descriptor->width, mip));
            const Size mipHeight = static_cast<Size>(vernon::rhi::imageMipExtent(descriptor->height, mip));
            const Size mipDepth = static_cast<Size>(vernon::rhi::imageMipExtent(descriptor->depth, mip));
            if (descriptor->dimension == VERNON_RHI_IMAGE_3D) {
                driver.texImage3D(target, static_cast<Int>(mip), format.internal, mipWidth, mipHeight, mipDepth, 0,
                                  format.external, format.allocationType, nullptr);
            } else if (descriptor->dimension == VERNON_RHI_IMAGE_CUBE) {
                for (uint32_t face = 0; face < 6; ++face)
                    driver.texImage2D(vernon::rhi::opengl::kTextureCubeMapPositiveX + face, static_cast<Int>(mip),
                                      format.internal, mipWidth, mipHeight, 0, format.external, format.allocationType,
                                      nullptr);
            } else {
                driver.texImage2D(target, static_cast<Int>(mip), format.internal, mipWidth, mipHeight, 0,
                                  format.external, format.allocationType, nullptr);
            }
        }
        driver.bindTexture(target, 0);
    }
    slot.descriptor = *descriptor;
    slot.target = target;
    auto published = slot.lifecycle.publish(std::move(reservation).value(), &state, teardownImage);
    if (published.isErr()) {
        state.state.destroyImage(slot.image);
        slot.image = {};
        slot.descriptor = {};
        slot.target = 0;
        return statusFromError(published.error());
    }
    *output = {published.value().index, published.value().generation};
    return VERNON_RHI_STATUS_OK;
}

VernonRhiStatus setImageSampler(VernonRhiDevice handle, VernonRhiImage image,
                                const VernonRhiSamplerDescriptor *descriptor) {

    auto device = lookupDevice(handle);
    if (device.isErr() || !descriptor || descriptor->struct_size < sizeof(*descriptor))
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    OpenGLDevice &state = device.value().device();
    ImageSlot *slot = indexedSlot(state.images, image.index);
    auto pin = pinSlot(slot, image);
    if (pin.isErr())
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    const Int minFilter = samplerFilter(descriptor->min_filter);
    const Int magFilter = samplerFilter(descriptor->mag_filter);
    const Int addressU = samplerAddress(descriptor->address_u);
    const Int addressV = samplerAddress(descriptor->address_v);
    const Int addressW = samplerAddress(descriptor->address_w);
    std::lock_guard<std::mutex> guard(state.mutex);
    state.state.makeCurrent();
    if (!slot || !minFilter || (descriptor->mag_filter > VERNON_RHI_FILTER_LINEAR) || !magFilter || !addressU ||
        !addressV || !addressW)
        return fail(state, "OpenGL RHI image sampler descriptor is invalid");
    auto &driver = state.state.driver;
    driver.bindTexture(slot->target, slot->image.name);
    driver.texParameteri(slot->target, 0x2802, addressU);
    driver.texParameteri(slot->target, 0x2803, addressV);
    if (slot->descriptor.dimension != VERNON_RHI_IMAGE_2D)
        driver.texParameteri(slot->target, 0x8072, addressW);
    driver.texParameteri(slot->target, 0x2801, minFilter);
    driver.texParameteri(slot->target, 0x2800, magFilter);
    driver.bindTexture(slot->target, 0);
    return VERNON_RHI_STATUS_OK;
}

VernonRhiStatus uploadImage(VernonRhiDevice handle, VernonRhiImage image, const VernonRhiImageUploadDescriptor *uploads,
                            size_t uploadCount) {

    auto device = lookupDevice(handle);
    if (device.isErr() || (uploadCount != 0 && !uploads))
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    OpenGLDevice &state = device.value().device();
    ImageSlot *slot = indexedSlot(state.images, image.index);
    auto pin = pinSlot(slot, image);
    if (pin.isErr())
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(state.mutex);
    state.state.makeCurrent();
    auto &driver = state.state.driver;
    driver.bindTexture(slot->target, slot->image.name);
    driver.pixelStorei(0x0CF5, 1);
    FormatInfo storageFormat;
    if (!formatInfo(slot->descriptor.format, storageFormat)) {
        driver.bindTexture(slot->target, 0);
        return fail(state, "OpenGL RHI image storage format is invalid");
    }
    for (size_t index = 0; index < uploadCount; ++index) {
        const VernonRhiImageUploadDescriptor &upload = uploads[index];
        const Enum sourceFormat = imageDataFormat(upload.source_format);
        const bool validLayer =
            slot->descriptor.dimension == VERNON_RHI_IMAGE_CUBE ? upload.array_layer < 6 : upload.array_layer == 0;
        const bool validMip = upload.mip_level < slot->descriptor.mip_levels;
        const uint32_t mipWidth = validMip ? vernon::rhi::imageMipExtent(slot->descriptor.width, upload.mip_level) : 0;
        const uint32_t mipHeight =
            validMip ? vernon::rhi::imageMipExtent(slot->descriptor.height, upload.mip_level) : 0;
        const uint32_t mipDepth = !validMip ? 0
                                  : slot->descriptor.dimension == VERNON_RHI_IMAGE_3D
                                      ? vernon::rhi::imageMipExtent(slot->descriptor.depth, upload.mip_level)
                                      : slot->descriptor.depth;
        if (upload.struct_size < sizeof(upload) || upload.mip_level >= slot->descriptor.mip_levels ||
            upload.width == 0 || upload.height == 0 || upload.depth == 0 || !validLayer || !sourceFormat ||
            upload.offset_x >= mipWidth || upload.width > mipWidth - upload.offset_x || upload.offset_y >= mipHeight ||
            upload.height > mipHeight - upload.offset_y || upload.offset_z >= mipDepth ||
            upload.depth > mipDepth - upload.offset_z ||
            !vernon::rhi::uploadLayoutMatches(slot->descriptor.format, upload.source_format, upload.source_type)) {
            driver.bindTexture(slot->target, 0);
            return fail(state, "OpenGL RHI image upload descriptor is invalid");
        }
        const Enum type = upload.source_type == VERNON_RHI_IMAGE_DATA_UINT8              ? 0x1401
                          : upload.source_type == VERNON_RHI_IMAGE_DATA_FLOAT16          ? 0x140B
                          : upload.source_type == VERNON_RHI_IMAGE_DATA_FLOAT32          ? 0x1406
                          : slot->descriptor.format == VERNON_RHI_FORMAT_R11G11B10_FLOAT ? 0x8C3B
                                                                                         : 0x1405;
        if (slot->descriptor.dimension == VERNON_RHI_IMAGE_3D)
            driver.texSubImage3D(slot->target, static_cast<Int>(upload.mip_level), static_cast<Int>(upload.offset_x),
                                 static_cast<Int>(upload.offset_y), static_cast<Int>(upload.offset_z),
                                 static_cast<Size>(upload.width), static_cast<Size>(upload.height),
                                 static_cast<Size>(upload.depth), sourceFormat, type, upload.data);
        else
            driver.texSubImage2D(slot->descriptor.dimension == VERNON_RHI_IMAGE_CUBE
                                     ? vernon::rhi::opengl::kTextureCubeMapPositiveX + upload.array_layer
                                     : slot->target,
                                 static_cast<Int>(upload.mip_level), static_cast<Int>(upload.offset_x),
                                 static_cast<Int>(upload.offset_y), static_cast<Size>(upload.width),
                                 static_cast<Size>(upload.height), sourceFormat, type, upload.data);
    }
    driver.bindTexture(slot->target, 0);
    return VERNON_RHI_STATUS_OK;
}

VernonRhiStatus downloadImage(VernonRhiDevice handle, VernonRhiImage image,
                              const VernonRhiImageDownloadDescriptor *download, void *destination, size_t size) {

    if (!download || download->struct_size < sizeof(*download) || !destination)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    auto device = lookupDevice(handle);
    if (device.isErr())
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    OpenGLDevice &state = device.value().device();
    ImageSlot *slot = indexedSlot(state.images, image.index);
    auto pin = pinSlot(slot, image);
    if (pin.isErr())
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(state.mutex);
    state.state.makeCurrent();
    const auto expected = slot ? imageDownloadByteSize(slot->descriptor, *download) : std::nullopt;
    FormatInfo format;
    if (!slot || !expected || size != *expected || !formatInfo(slot->descriptor.format, format))
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    if (slot->descriptor.dimension == VERNON_RHI_IMAGE_3D)
        return state.state.downloadImage3D(slot->image, static_cast<Int>(download->mip_level),
                                           static_cast<Int>(download->offset_x), static_cast<Int>(download->offset_y),
                                           static_cast<Int>(download->offset_z), static_cast<Size>(download->width),
                                           static_cast<Size>(download->height), static_cast<Size>(download->depth),
                                           format.external, format.allocationType, *expected / download->depth,
                                           destination, state.error)
                   ? VERNON_RHI_STATUS_OK
                   : VERNON_RHI_STATUS_INTERNAL_ERROR;
    const Enum target = slot->descriptor.dimension == VERNON_RHI_IMAGE_CUBE
                            ? vernon::rhi::opengl::kTextureCubeMapPositiveX + download->array_layer
                            : slot->target;
    if (slot->descriptor.format != VERNON_RHI_FORMAT_D32_FLOAT_S8_UINT)
        return state.state.downloadImage2D(slot->image, target, static_cast<Int>(download->mip_level),
                                           static_cast<Int>(download->offset_x), static_cast<Int>(download->offset_y),
                                           static_cast<Size>(download->width), static_cast<Size>(download->height),
                                           format.external, format.allocationType, destination, state.error)
                   ? VERNON_RHI_STATUS_OK
                   : VERNON_RHI_STATUS_INTERNAL_ERROR;
    std::vector<uint8_t> native(*expected);
    if (!state.state.downloadImage2D(slot->image, target, static_cast<Int>(download->mip_level),
                                     static_cast<Int>(download->offset_x), static_cast<Int>(download->offset_y),
                                     static_cast<Size>(download->width), static_cast<Size>(download->height),
                                     format.external, format.allocationType, native.data(), state.error))
        return VERNON_RHI_STATUS_INTERNAL_ERROR;
    auto *output = static_cast<uint8_t *>(destination);
    const size_t pixels = static_cast<size_t>(download->width) * download->height;
    for (size_t index = 0; index < pixels; ++index) {
        float depth{};
        uint32_t stencilWord{};
        std::memcpy(&depth, native.data() + index * packedDepthStencilPixelSize, sizeof(depth));
        std::memcpy(&stencilWord, native.data() + index * packedDepthStencilPixelSize + sizeof(depth),
                    sizeof(stencilWord));
        storePackedDepthStencil(output + index * packedDepthStencilPixelSize, depth,
                                static_cast<uint8_t>(stencilWord & 0xff));
    }
    return VERNON_RHI_STATUS_OK;
}

VernonRhiStatus generateImageMipmaps(VernonRhiDevice handle, VernonRhiImage image) {

    auto device = lookupDevice(handle);
    if (device.isErr())
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    OpenGLDevice &state = device.value().device();
    ImageSlot *slot = indexedSlot(state.images, image.index);
    auto pin = pinSlot(slot, image);
    if (pin.isErr())
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(state.mutex);
    if (!slot || slot->descriptor.mip_levels < 2 || !(slot->descriptor.usage & VERNON_RHI_IMAGE_TRANSFER_SOURCE) ||
        !(slot->descriptor.usage & VERNON_RHI_IMAGE_TRANSFER_DESTINATION) ||
        slot->descriptor.format == VERNON_RHI_FORMAT_D32_FLOAT ||
        slot->descriptor.format == VERNON_RHI_FORMAT_D32_FLOAT_S8_UINT)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    state.state.makeCurrent();
    state.state.driver.bindTexture(slot->target, slot->image.name);
    state.state.driver.generateMipmap(slot->target);
    state.state.driver.bindTexture(slot->target, 0);
    return VERNON_RHI_STATUS_OK;
}

VernonRhiStatus bindImage(VernonRhiDevice handle, VernonRhiImage image, uint32_t textureUnit) {

    auto device = lookupDevice(handle);
    if (device.isErr())
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    OpenGLDevice &state = device.value().device();
    ImageSlot *slot = indexedSlot(state.images, image.index);
    auto pin = pinSlot(slot, image);
    if (pin.isErr())
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(state.mutex);
    state.state.makeCurrent();
    state.state.driver.activeTexture(vernon::rhi::opengl::kTexture0 + textureUnit);
    state.state.driver.bindTexture(slot->target, slot->image.name);
    return VERNON_RHI_STATUS_OK;
}

VernonRhiStatus destroyImage(VernonRhiDevice handle, VernonRhiImage image) {

    auto device = lookupDevice(handle);
    if (device.isErr())
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    OpenGLDevice &state = device.value().device();
    ImageSlot *slot = indexedSlot(state.images, image.index);
    if (!slot)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    auto destroyed = slot->lifecycle.destroyPublic({image.index, image.generation});
    return destroyed.isOk() ? VERNON_RHI_STATUS_OK : statusFromError(destroyed.error());
}

uint32_t isImageValid(VernonRhiDevice handle, VernonRhiImage image) {

    auto device = lookupDevice(handle);
    if (device.isErr())
        return 0;
    OpenGLDevice &state = device.value().device();
    return publicSlot(indexedSlot(state.images, image.index), image);
}

VernonRhiStatus createSampler(VernonRhiDevice handle, const VernonRhiSamplerDescriptor *descriptor,
                              VernonRhiSampler *output) {

    auto device = lookupDevice(handle);
    if (device.isErr() || !descriptor || !output || descriptor->struct_size < sizeof(*descriptor))
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    *output = {static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
    const Int minFilter = samplerFilter(descriptor->min_filter);
    const Int magFilter = samplerFilter(descriptor->mag_filter);
    const Int addressU = samplerAddress(descriptor->address_u);
    const Int addressV = samplerAddress(descriptor->address_v);
    const Int addressW = samplerAddress(descriptor->address_w);
    if (!minFilter || descriptor->mag_filter > VERNON_RHI_FILTER_LINEAR || !magFilter || !addressU || !addressV ||
        !addressW)
        return fail(device.value().device(), "OpenGL RHI sampler descriptor is invalid");
    auto owner = device.value().retainOwner();
    if (owner.isErr())
        return statusFromError(owner.error());
    auto reservation = vernon::rhi::ResourceCreationReservation::create(owner.value());
    if (reservation.isErr())
        return statusFromError(reservation.error());
    OpenGLDevice &state = device.value().device();
    std::lock_guard<std::mutex> allocationGuard(state.resourceAllocationMutex);
    auto available = availableSlot<SamplerSlot, SamplerLifecycle>(state.samplers);
    if (available.isErr())
        return statusFromError(available.error());
    SamplerSlot &slot = *available.value();
    std::lock_guard<std::mutex> guard(state.mutex);
    state.state.makeCurrent();
    if (!state.state.createSampler(slot.sampler, addressU, addressV, addressW, minFilter, magFilter, state.error))
        return VERNON_RHI_STATUS_INTERNAL_ERROR;
    auto published = slot.lifecycle.publish(std::move(reservation).value(), &state, teardownSampler);
    if (published.isErr()) {
        state.state.destroySampler(slot.sampler);
        slot.sampler = {};
        return statusFromError(published.error());
    }
    *output = {published.value().index, published.value().generation};
    return VERNON_RHI_STATUS_OK;
}

VernonRhiStatus destroySampler(VernonRhiDevice handle, VernonRhiSampler sampler) {

    auto device = lookupDevice(handle);
    if (device.isErr())
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    OpenGLDevice &state = device.value().device();
    SamplerSlot *slot = indexedSlot(state.samplers, sampler.index);
    if (!slot)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    auto destroyed = slot->lifecycle.destroyPublic({sampler.index, sampler.generation});
    return destroyed.isOk() ? VERNON_RHI_STATUS_OK : statusFromError(destroyed.error());
}

uint32_t isSamplerValid(VernonRhiDevice handle, VernonRhiSampler sampler) {

    auto device = lookupDevice(handle);
    if (device.isErr())
        return 0;
    OpenGLDevice &state = device.value().device();
    return publicSlot(indexedSlot(state.samplers, sampler.index), sampler);
}

VernonRhiStatus getImageNativeHandle(VernonRhiDevice handle, VernonRhiImage image, uint64_t *output) {

    auto device = lookupDevice(handle);
    if (device.isErr() || !output)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    OpenGLDevice &state = device.value().device();
    ImageSlot *slot = indexedSlot(state.images, image.index);
    auto pin = pinSlot(slot, image);
    if (pin.isErr())
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(state.mutex);
    state.state.makeCurrent();
    *output = slot->image.name;
    return VERNON_RHI_STATUS_OK;
}

VernonRhiStatus createImageView(VernonRhiDevice handle, const VernonRhiImageViewDescriptor *descriptor,
                                VernonRhiImageView *output) {
    auto device = lookupDevice(handle);
    if (device.isErr() || !descriptor || descriptor->struct_size < sizeof(*descriptor) || !output ||
        !descriptor->mip_level_count || !descriptor->array_layer_count || !descriptor->aspects)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    *output = {static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
    OpenGLDevice &state = device.value().device();
    ImageSlot *parent = indexedSlot(state.images, descriptor->image.index);
    auto parentPin = pinSlot(parent, descriptor->image);
    if (parentPin.isErr())
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    FormatInfo format{};
    if (!parent || !formatInfo(descriptor->format, format))
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    if (!vernon::rhi::validImageViewDescriptor(parent->descriptor, *descriptor))
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    if (descriptor->dimension == VERNON_RHI_IMAGE_2D && descriptor->array_layer_count != 1)
        return VERNON_RHI_STATUS_UNSUPPORTED;
    auto parentLease = parent->lifecycle.retain({descriptor->image.index, descriptor->image.generation});
    if (parentLease.isErr())
        return statusFromError(parentLease.error());
    auto owner = device.value().retainOwner();
    if (owner.isErr())
        return statusFromError(owner.error());
    auto reservation = vernon::rhi::ResourceCreationReservation::create(owner.value());
    if (reservation.isErr())
        return statusFromError(reservation.error());
    std::lock_guard<std::mutex> allocationGuard(state.resourceAllocationMutex);
    auto available = availableSlot<ImageViewSlot, ImageViewLifecycle>(state.imageViews);
    if (available.isErr())
        return statusFromError(available.error());
    ImageViewSlot &slot = *available.value();
    std::unique_lock<std::mutex> guard(state.mutex);
    state.state.makeCurrent();
    const bool identityView =
        descriptor->format == parent->descriptor.format && descriptor->dimension == parent->descriptor.dimension &&
        descriptor->aspects == vernon::rhi::imageFormatAspects(parent->descriptor.format) &&
        descriptor->base_mip_level == 0 && descriptor->mip_level_count == parent->descriptor.mip_levels &&
        descriptor->base_array_layer == 0 && descriptor->array_layer_count == parent->descriptor.array_layers;
    slot.backing.emplace<IdentityImageViewBacking>(IdentityImageViewBacking{parent->image.name});
    if (!identityView) {
        if (state.state.embeddedProfile || !state.state.driver.textureView) {
            guard.unlock();
            return VERNON_RHI_STATUS_UNSUPPORTED;
        }
        auto &owned = slot.backing.emplace<OwnedTextureViewBacking>();
        state.state.driver.genTextures(1, &owned.image.name);
        if (!owned.image.name) {
            slot.backing.emplace<IdentityImageViewBacking>();
            guard.unlock();
            return VERNON_RHI_STATUS_INTERNAL_ERROR;
        }
        owned.image.imported = false;
        slot.target = imageTarget(descriptor->dimension);
        state.state.driver.textureView(owned.image.name, slot.target, parent->image.name, format.internal,
                                       descriptor->base_mip_level, descriptor->mip_level_count,
                                       descriptor->base_array_layer, descriptor->array_layer_count);
    }
    slot.target = imageTarget(descriptor->dimension);
    slot.descriptor = *descriptor;
    slot.imageDescriptor = parent->descriptor;
    slot.parentKey = resourceKey(descriptor->image);
    slot.parent.emplace(std::move(parentLease).value());
    auto published = slot.lifecycle.publish(std::move(reservation).value(), &state, teardownImageView);
    if (published.isErr()) {
        if (auto *owned = std::get_if<OwnedTextureViewBacking>(&slot.backing))
            state.state.destroyImage(owned->image);
        slot.backing.emplace<IdentityImageViewBacking>();
        slot.descriptor = {};
        slot.imageDescriptor = {};
        slot.parentKey = 0;
        slot.target = 0;
        guard.unlock();
        slot.parent.reset();
        return statusFromError(published.error());
    }
    *output = {published.value().index, published.value().generation};
    return VERNON_RHI_STATUS_OK;
}

VernonRhiStatus destroyImageView(VernonRhiDevice handle, VernonRhiImageView view) {
    auto device = lookupDevice(handle);
    if (device.isErr())
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    OpenGLDevice &state = device.value().device();
    ImageViewSlot *slot = indexedSlot(state.imageViews, view.index);
    if (!slot)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    auto destroyed = slot->lifecycle.destroyPublic({view.index, view.generation});
    return destroyed.isOk() ? VERNON_RHI_STATUS_OK : statusFromError(destroyed.error());
}

VernonRhiStatus getImageViewNativeHandle(VernonRhiDevice handle, VernonRhiImageView view, uint64_t *output) {
    auto device = lookupDevice(handle);
    if (device.isErr() || !output)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    OpenGLDevice &state = device.value().device();
    ImageViewSlot *slot = indexedSlot(state.imageViews, view.index);
    auto pin = pinSlot(slot, view);
    if (pin.isErr())
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(state.mutex);
    state.state.makeCurrent();
    *output = imageViewName(*slot);
    return VERNON_RHI_STATUS_OK;
}

VernonRhiStatus destroyBuffer(VernonRhiDevice handle, VernonRhiBuffer buffer) {

    auto device = lookupDevice(handle);
    if (device.isErr())
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    OpenGLDevice &state = device.value().device();
    BufferSlot *slot = indexedSlot(state.buffers, buffer.index);
    if (!slot)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    auto destroyed = slot->lifecycle.destroyPublic({buffer.index, buffer.generation});
    return destroyed.isOk() ? VERNON_RHI_STATUS_OK : statusFromError(destroyed.error());
}

VernonRhiStatus getBufferNativeHandle(VernonRhiDevice handle, VernonRhiBuffer buffer, void **output) {

    auto device = lookupDevice(handle);
    if (device.isErr() || !output)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    OpenGLDevice &state = device.value().device();
    BufferSlot *slot = indexedSlot(state.buffers, buffer.index);
    auto pin = pinSlot(slot, buffer);
    if (pin.isErr())
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(state.mutex);
    state.state.makeCurrent();
    *output = reinterpret_cast<void *>(static_cast<uintptr_t>(slot->buffer.name));
    return VERNON_RHI_STATUS_OK;
}

bool ownsDevice(VernonRhiDevice handle) { return lookupDevice(handle).isOk(); }

void *deviceStateForBackend(VernonRhiDevice handle) {
    auto device = lookupDevice(handle);
    return device.isOk() ? &device.value().device().state : nullptr;
}

bool beginCommands(VernonRhiDevice handle, uint64_t &native, VernonRhiBackend &backend) {
    native = 0;
    auto device = lookupDevice(handle);
    if (device.isErr())
        return false;
    OpenGLDevice &state = device.value().device();
    std::lock_guard<std::mutex> guard(state.mutex);
    state.state.makeCurrent();
    backend = state.state.embeddedProfile ? VERNON_RHI_BACKEND_OPENGL_ES : VERNON_RHI_BACKEND_OPENGL;
    native = reinterpret_cast<uintptr_t>(&state.state);
    return true;
}

bool submitCommands(VernonRhiDevice handle, uint64_t native, bool computeWrites, bool &completed,
                    bool &externalCompletion) {
    completed = false;
    externalCompletion = false;
    auto device = lookupDevice(handle);
    if (device.isErr())
        return false;
    OpenGLDevice &state = device.value().device();
    if (native != reinterpret_cast<uintptr_t>(&state.state))
        return false;
    std::lock_guard<std::mutex> guard(state.mutex);
    state.state.makeCurrent();
    if (computeWrites) {
        if (state.state.driver.memoryBarrier)
            state.state.driver.memoryBarrier(
                vernon::rhi::opengl::kShaderStorageBarrierBit | vernon::rhi::opengl::kVertexAttribArrayBarrierBit |
                vernon::rhi::opengl::kTextureFetchBarrierBit | vernon::rhi::opengl::kBufferUpdateBarrierBit);
        else
            state.state.driver.finish();
    }
    completed = true;
    return true;
}

bool completeBorrowedCommands(VernonRhiDevice handle, uint64_t native) {
    auto device = lookupDevice(handle);
    return device.isOk() && native == reinterpret_cast<uintptr_t>(&device.value().device().state);
}

void abandonCommands(VernonRhiDevice handle, uint64_t native) {
    auto device = lookupDevice(handle);
    if (device.isOk() && native == reinterpret_cast<uintptr_t>(&device.value().device().state))
        return;
}

bool recordBarriers(VernonRhiDevice handle, uint64_t encoderKey, uint64_t native, const VernonRhiBarrier *barriers,
                    size_t barrierCount) {
    auto device = lookupDevice(handle);
    if (device.isOk()) {
        OpenGLDevice &deviceState = device.value().device();
        std::lock_guard<std::mutex> guard(deviceState.mutex);
        if (native != reinterpret_cast<uintptr_t>(&deviceState.state))
            return false;
        uint32_t bits = 0;
        for (size_t index = 0; index < barrierCount; ++index) {
            const uint32_t access = barriers[index].destination_access;
            if (access) {
                if (access & VERNON_RHI_ACCESS_SHADER_READ)
                    bits |= barriers[index].is_image ? vernon::rhi::opengl::kTextureFetchBarrierBit
                                                     : vernon::rhi::opengl::kShaderStorageBarrierBit |
                                                           vernon::rhi::opengl::kUniformBarrierBit;
                if (access & VERNON_RHI_ACCESS_SHADER_WRITE)
                    bits |= barriers[index].is_image ? vernon::rhi::opengl::kShaderImageAccessBarrierBit
                                                     : vernon::rhi::opengl::kShaderStorageBarrierBit;
                if (access & VERNON_RHI_ACCESS_VERTEX_READ)
                    bits |= vernon::rhi::opengl::kVertexAttribArrayBarrierBit;
                if (access & VERNON_RHI_ACCESS_INDEX_READ)
                    bits |= vernon::rhi::opengl::kElementArrayBarrierBit;
                if (access & (VERNON_RHI_ACCESS_COLOR_READ | VERNON_RHI_ACCESS_COLOR_WRITE |
                              VERNON_RHI_ACCESS_DEPTH_STENCIL_READ | VERNON_RHI_ACCESS_DEPTH_STENCIL_WRITE))
                    bits |= vernon::rhi::opengl::kFramebufferBarrierBit;
                if (access & (VERNON_RHI_ACCESS_TRANSFER_READ | VERNON_RHI_ACCESS_TRANSFER_WRITE))
                    bits |= barriers[index].is_image ? vernon::rhi::opengl::kTextureUpdateBarrierBit
                                                     : vernon::rhi::opengl::kBufferUpdateBarrierBit;
                continue;
            }
            const auto state = static_cast<VernonRhiResourceState>(barriers[index].new_state);
            if (barriers[index].is_image) {
                if (state == VERNON_RHI_STATE_SHADER_READ)
                    bits |= vernon::rhi::opengl::kTextureFetchBarrierBit;
                else if (state == VERNON_RHI_STATE_SHADER_WRITE)
                    bits |= vernon::rhi::opengl::kShaderImageAccessBarrierBit;
                else if (state == VERNON_RHI_STATE_TRANSFER_SOURCE || state == VERNON_RHI_STATE_TRANSFER_DESTINATION)
                    bits |= vernon::rhi::opengl::kTextureUpdateBarrierBit;
                else if (state == VERNON_RHI_STATE_COLOR_ATTACHMENT ||
                         state == VERNON_RHI_STATE_DEPTH_STENCIL_ATTACHMENT)
                    bits |= vernon::rhi::opengl::kFramebufferBarrierBit;
            } else {
                if (state == VERNON_RHI_STATE_SHADER_READ)
                    bits |= vernon::rhi::opengl::kShaderStorageBarrierBit | vernon::rhi::opengl::kUniformBarrierBit |
                            vernon::rhi::opengl::kVertexAttribArrayBarrierBit |
                            vernon::rhi::opengl::kElementArrayBarrierBit;
                else if (state == VERNON_RHI_STATE_SHADER_WRITE)
                    bits |= vernon::rhi::opengl::kShaderStorageBarrierBit;
                else if (state == VERNON_RHI_STATE_TRANSFER_SOURCE || state == VERNON_RHI_STATE_TRANSFER_DESTINATION)
                    bits |= vernon::rhi::opengl::kBufferUpdateBarrierBit;
            }
        }
        deviceState.state.makeCurrent();
        if (bits) {
            if (deviceState.state.driver.memoryBarrier)
                deviceState.state.driver.memoryBarrier(bits);
            else
                // glMemoryBarrier is core only in OpenGL 4.2. Earlier
                // contexts still need a synchronization point between graph
                // scopes, for example when sampling a depth attachment.
                deviceState.state.driver.finish();
        }
        return true;
    }
    return false;
}

bool recordBufferCopy(VernonRhiDevice handle, uint64_t native, VernonRhiBuffer source, uint64_t sourceOffset,
                      VernonRhiBuffer destination, uint64_t destinationOffset, uint64_t size) {
    auto device = lookupDevice(handle);
    if (device.isErr() || !size)
        return false;
    OpenGLDevice &state = device.value().device();
    if (native != reinterpret_cast<uintptr_t>(&state.state))
        return false;
    BufferSlot *sourceSlot = indexedSlot(state.buffers, source.index);
    BufferSlot *destinationSlot = indexedSlot(state.buffers, destination.index);
    auto sourcePin = pinSlot(sourceSlot, source);
    auto destinationPin = pinSlot(destinationSlot, destination);
    if (sourcePin.isErr() || destinationPin.isErr())
        return false;
    std::lock_guard<std::mutex> guard(state.mutex);
    state.state.makeCurrent();
    if (sourceOffset > sourceSlot->descriptor.size || size > sourceSlot->descriptor.size - sourceOffset ||
        destinationOffset > destinationSlot->descriptor.size ||
        size > destinationSlot->descriptor.size - destinationOffset)
        return false;
    return state.state.copyBuffer(sourceSlot->buffer, static_cast<size_t>(sourceOffset), destinationSlot->buffer,
                                  static_cast<size_t>(destinationOffset), static_cast<size_t>(size), state.error);
}

bool supportsImageCopy(VernonRhiDevice handle) {
    auto device = lookupDevice(handle);
    return device.isOk() && device.value().device().state.driver.copyImageSubData;
}

bool recordImageCopy(VernonRhiDevice handle, uint64_t, uint64_t native, VernonRhiImage source,
                     VernonRhiImage destination, const VernonRhiImageCopyRegion *regions, size_t regionCount) {
    auto device = lookupDevice(handle);
    if (device.isErr() || !regions || !regionCount)
        return false;
    OpenGLDevice &state = device.value().device();
    if (native != reinterpret_cast<uintptr_t>(&state.state))
        return false;
    ImageSlot *sourceSlot = indexedSlot(state.images, source.index);
    ImageSlot *destinationSlot = indexedSlot(state.images, destination.index);
    auto sourcePin = pinSlot(sourceSlot, source);
    auto destinationPin = pinSlot(destinationSlot, destination);
    if (sourcePin.isErr() || destinationPin.isErr())
        return false;
    std::lock_guard<std::mutex> guard(state.mutex);
    state.state.makeCurrent();
    if (!state.state.driver.copyImageSubData)
        return false;
    for (size_t index = 0; index < regionCount; ++index) {
        const VernonRhiImageCopyRegion &region = regions[index];
        const Int sourceZ = sourceSlot->descriptor.dimension == VERNON_RHI_IMAGE_3D
                                ? static_cast<Int>(region.source_z)
                                : static_cast<Int>(region.source_array_layer);
        const Int destinationZ = destinationSlot->descriptor.dimension == VERNON_RHI_IMAGE_3D
                                     ? static_cast<Int>(region.destination_z)
                                     : static_cast<Int>(region.destination_array_layer);
        state.state.driver.copyImageSubData(
            sourceSlot->image.name, sourceSlot->target, static_cast<Int>(region.source_mip_level),
            static_cast<Int>(region.source_x), static_cast<Int>(region.source_y), sourceZ, destinationSlot->image.name,
            destinationSlot->target, static_cast<Int>(region.destination_mip_level),
            static_cast<Int>(region.destination_x), static_cast<Int>(region.destination_y), destinationZ,
            static_cast<Size>(region.width), static_cast<Size>(region.height), static_cast<Size>(region.depth));
    }
    return true;
}

bool endRendering(VernonRhiDevice handle, uint64_t native, VernonRhiBackend backend, uint32_t backendKind,
                  uint32_t colorDiscardMask, uint32_t depthStencilDiscard, const uint64_t *colorResources,
                  size_t colorCount, uint64_t depthResource, uint64_t) {

    if ((backend == VERNON_RHI_BACKEND_OPENGL || backend == VERNON_RHI_BACKEND_OPENGL_ES) &&
        backendKind == CommandRenderingStateless) {
        auto device = lookupDevice(handle);
        if (device.isErr())
            return false;
        OpenGLDevice &state = device.value().device();
        if (native != reinterpret_cast<uintptr_t>(&state.state))
            return false;
        std::array<vernon::rhi::opengl::Enum, 9> discarded{};
        size_t discardedCount = 0;
        for (size_t index = 0; index < colorCount; ++index)
            if (colorDiscardMask & (uint32_t{1} << index))
                discarded[discardedCount++] = vernon::rhi::opengl::kColorAttachment0 + static_cast<Enum>(index);
        if (depthStencilDiscard & VERNON_RHI_ATTACHMENT_DEPTH)
            discarded[discardedCount++] = vernon::rhi::opengl::kDepthAttachment;
        if (depthStencilDiscard & VERNON_RHI_ATTACHMENT_STENCIL)
            discarded[discardedCount++] = vernon::rhi::opengl::kStencilAttachment;
        std::lock_guard<std::mutex> guard(state.mutex);
        state.state.makeCurrent();
        if (discardedCount && state.state.driver.invalidateFramebuffer)
            state.state.driver.invalidateFramebuffer(vernon::rhi::opengl::kFramebuffer,
                                                     static_cast<Size>(discardedCount), discarded.data());
        return true;
    }
    return false;
}

bool clearColor(VernonRhiDevice handle, uint64_t native, VernonRhiBackend backend, uint32_t backendKind, uint64_t,
                int32_t x, int32_t y, uint32_t width, uint32_t height, uint32_t layers, uint64_t target,
                uint32_t location, const float color[4]) {

    if (backend == VERNON_RHI_BACKEND_OPENGL || backend == VERNON_RHI_BACKEND_OPENGL_ES) {
        auto device = lookupDevice(handle);
        if (device.isErr() || backendKind != CommandRenderingStateless)
            return false;
        OpenGLDevice &state = device.value().device();
        if (native != reinterpret_cast<uintptr_t>(&state.state))
            return false;
        std::lock_guard<std::mutex> guard(state.mutex);
        state.state.makeCurrent();
        state.state.driver.clearBufferfv(vernon::rhi::opengl::kColor, static_cast<Int>(location), color);
        return true;
    }
    return false;
}

bool clearDepthStencil(VernonRhiDevice handle, uint64_t native, VernonRhiBackend backend, uint32_t backendKind,
                       uint64_t, int32_t x, int32_t y, uint32_t width, uint32_t height, uint32_t layers,
                       uint64_t target, float depth, uint32_t stencil, uint32_t aspects) {

    if (backend == VERNON_RHI_BACKEND_OPENGL || backend == VERNON_RHI_BACKEND_OPENGL_ES) {
        auto device = lookupDevice(handle);
        if (device.isErr() || backendKind != CommandRenderingStateless ||
            (aspects & ~(VERNON_RHI_ATTACHMENT_DEPTH | VERNON_RHI_ATTACHMENT_STENCIL)) || !aspects)
            return false;
        OpenGLDevice &state = device.value().device();
        if (native != reinterpret_cast<uintptr_t>(&state.state))
            return false;
        std::lock_guard<std::mutex> guard(state.mutex);
        state.state.makeCurrent();
        if (aspects == (VERNON_RHI_ATTACHMENT_DEPTH | VERNON_RHI_ATTACHMENT_STENCIL))
            state.state.driver.clearBufferfi(vernon::rhi::opengl::kDepthStencil, 0, depth, static_cast<Int>(stencil));
        else if (aspects == VERNON_RHI_ATTACHMENT_DEPTH)
            state.state.driver.clearBufferfv(vernon::rhi::opengl::kDepth, 0, &depth);
        else {
            const Int value = static_cast<Int>(stencil);
            state.state.driver.clearBufferiv(vernon::rhi::opengl::kStencil, 0, &value);
        }
        return true;
    }
    return false;
}

uint64_t bufferResource(VernonRhiDevice handle, VernonRhiBuffer buffer) {
    auto device = lookupDevice(handle);
    return device.isOk() && publicSlot(indexedSlot(device.value().device().buffers, buffer.index), buffer)
               ? resourceKey(buffer)
               : 0;
}

uint64_t imageResource(VernonRhiDevice handle, VernonRhiImage image) {
    auto device = lookupDevice(handle);
    return device.isOk() && publicSlot(indexedSlot(device.value().device().images, image.index), image)
               ? resourceKey(image)
               : 0;
}

uint64_t imageViewResource(VernonRhiDevice handle, VernonRhiImageView view) {
    auto device = lookupDevice(handle);
    return device.isOk() && publicSlot(indexedSlot(device.value().device().imageViews, view.index), view)
               ? resourceKey(view)
               : 0;
}

uint64_t samplerResource(VernonRhiDevice handle, VernonRhiSampler sampler) {
    auto device = lookupDevice(handle);
    return device.isOk() && publicSlot(indexedSlot(device.value().device().samplers, sampler.index), sampler)
               ? resourceKey(sampler)
               : 0;
}

vernon::Result<vernon::rhi::RetainedRhiResourceLease, vernon::RhiError>
retainResource(VernonRhiDevice handle, ResourceKind kind, uint64_t key) noexcept {
    auto device = lookupDevice(handle);
    if (device.isErr())
        return vernon::Result<vernon::rhi::RetainedRhiResourceLease, vernon::RhiError>{
            vernon::err(std::move(device).error())};
    OpenGLDevice &state = device.value().device();
    if (kind == ResourceKind::Buffer) {
        BufferLifecycle::Handle resource{};
        BufferSlot *slot = decodeResourceKey(key, resource) ? indexedSlot(state.buffers, resource.index) : nullptr;
        if (slot) {
            auto retained = slot->lifecycle.retain(resource);
            if (retained.isOk())
                return vernon::Result<vernon::rhi::RetainedRhiResourceLease, vernon::RhiError>{
                    vernon::ok(vernon::rhi::RetainedRhiResourceLease{std::move(retained).value()})};
            return vernon::Result<vernon::rhi::RetainedRhiResourceLease, vernon::RhiError>{
                vernon::err(std::move(retained).error())};
        }
    }
    if (kind == ResourceKind::Image) {
        ImageLifecycle::Handle resource{};
        ImageSlot *slot = decodeResourceKey(key, resource) ? indexedSlot(state.images, resource.index) : nullptr;
        if (slot) {
            auto retained = slot->lifecycle.retain(resource);
            if (retained.isOk())
                return vernon::Result<vernon::rhi::RetainedRhiResourceLease, vernon::RhiError>{
                    vernon::ok(vernon::rhi::RetainedRhiResourceLease{std::move(retained).value()})};
            return vernon::Result<vernon::rhi::RetainedRhiResourceLease, vernon::RhiError>{
                vernon::err(std::move(retained).error())};
        }
    }
    if (kind == ResourceKind::ImageView) {
        ImageViewLifecycle::Handle resource{};
        ImageViewSlot *slot =
            decodeResourceKey(key, resource) ? indexedSlot(state.imageViews, resource.index) : nullptr;
        if (slot) {
            auto retained = slot->lifecycle.retain(resource);
            if (retained.isOk())
                return vernon::Result<vernon::rhi::RetainedRhiResourceLease, vernon::RhiError>{
                    vernon::ok(vernon::rhi::RetainedRhiResourceLease{std::move(retained).value()})};
            return vernon::Result<vernon::rhi::RetainedRhiResourceLease, vernon::RhiError>{
                vernon::err(std::move(retained).error())};
        }
    }
    if (kind == ResourceKind::Sampler) {
        SamplerLifecycle::Handle resource{};
        SamplerSlot *slot = decodeResourceKey(key, resource) ? indexedSlot(state.samplers, resource.index) : nullptr;
        if (slot) {
            auto retained = slot->lifecycle.retain(resource);
            if (retained.isOk())
                return vernon::Result<vernon::rhi::RetainedRhiResourceLease, vernon::RhiError>{
                    vernon::ok(vernon::rhi::RetainedRhiResourceLease{std::move(retained).value()})};
            return vernon::Result<vernon::rhi::RetainedRhiResourceLease, vernon::RhiError>{
                vernon::err(std::move(retained).error())};
        }
    }
    return vernon::Result<vernon::rhi::RetainedRhiResourceLease, vernon::RhiError>{vernon::err(
        vernon::RhiError{vernon::RhiErrorCode::InvalidArgument, {"retain_opengl_resource", key, uint32_t(kind)}})};
}

vernon::Result<uint64_t, vernon::RhiError> resolveResource(VernonRhiDevice handle, ResourceKind kind,
                                                           uint64_t key) noexcept {
    auto device = lookupDevice(handle);
    if (device.isErr())
        return vernon::Result<uint64_t, vernon::RhiError>{vernon::err(std::move(device).error())};
    OpenGLDevice &state = device.value().device();
    if (kind == ResourceKind::Buffer) {
        BufferLifecycle::Handle resource{};
        BufferSlot *slot = decodeResourceKey(key, resource) ? indexedSlot(state.buffers, resource.index) : nullptr;
        if (slot) {
            std::lock_guard<std::mutex> guard(state.mutex);
            state.state.makeCurrent();
            return vernon::Result<uint64_t, vernon::RhiError>{vernon::ok(uint64_t(slot->buffer.name))};
        }
    }
    if (kind == ResourceKind::Image) {
        ImageLifecycle::Handle resource{};
        ImageSlot *slot = decodeResourceKey(key, resource) ? indexedSlot(state.images, resource.index) : nullptr;
        if (slot) {
            std::lock_guard<std::mutex> guard(state.mutex);
            state.state.makeCurrent();
            return vernon::Result<uint64_t, vernon::RhiError>{vernon::ok(uint64_t(slot->image.name))};
        }
    }
    if (kind == ResourceKind::ImageView) {
        ImageViewLifecycle::Handle resource{};
        ImageViewSlot *slot =
            decodeResourceKey(key, resource) ? indexedSlot(state.imageViews, resource.index) : nullptr;
        if (slot) {
            std::lock_guard<std::mutex> guard(state.mutex);
            state.state.makeCurrent();
            return vernon::Result<uint64_t, vernon::RhiError>{vernon::ok(uint64_t(imageViewName(*slot)))};
        }
    }
    if (kind == ResourceKind::Sampler) {
        SamplerLifecycle::Handle resource{};
        SamplerSlot *slot = decodeResourceKey(key, resource) ? indexedSlot(state.samplers, resource.index) : nullptr;
        if (slot) {
            std::lock_guard<std::mutex> guard(state.mutex);
            state.state.makeCurrent();
            return vernon::Result<uint64_t, vernon::RhiError>{vernon::ok(uint64_t(slot->sampler.name))};
        }
    }
    return vernon::Result<uint64_t, vernon::RhiError>{vernon::err(
        vernon::RhiError{vernon::RhiErrorCode::InvalidArgument, {"resolve_opengl_resource", key, uint32_t(kind)}})};
}

vernon::Result<void, vernon::RhiError> describeImageResource(VernonRhiDevice handle, uint64_t key,
                                                             VernonRhiImageDescriptor *descriptor) noexcept {
    auto device = lookupDevice(handle);
    ImageLifecycle::Handle resource{};
    if (device.isErr() || !descriptor || !decodeResourceKey(key, resource))
        return vernon::Result<void, vernon::RhiError>{
            vernon::err(vernon::RhiError{vernon::RhiErrorCode::InvalidArgument, {"describe_opengl_image", key, 0}})};
    OpenGLDevice &state = device.value().device();
    ImageSlot *slot = indexedSlot(state.images, resource.index);
    if (!slot)
        return vernon::Result<void, vernon::RhiError>{
            vernon::err(vernon::RhiError{vernon::RhiErrorCode::InvalidArgument, {"describe_opengl_image", key, 0}})};
    std::lock_guard<std::mutex> guard(state.mutex);
    state.state.makeCurrent();
    *descriptor = slot->descriptor;
    return vernon::Result<void, vernon::RhiError>{vernon::ok()};
}

vernon::Result<void, vernon::RhiError> describeImageViewResource(VernonRhiDevice handle, uint64_t key,
                                                                 VernonRhiImageViewDescriptor *view,
                                                                 VernonRhiImageDescriptor *image,
                                                                 uint64_t *parentKey) noexcept {
    auto device = lookupDevice(handle);
    ImageViewLifecycle::Handle resource{};
    if (device.isErr() || !view || !image || !parentKey || !decodeResourceKey(key, resource))
        return vernon::Result<void, vernon::RhiError>{vernon::err(
            vernon::RhiError{vernon::RhiErrorCode::InvalidArgument, {"describe_opengl_image_view", key, 0}})};
    OpenGLDevice &state = device.value().device();
    ImageViewSlot *slot = indexedSlot(state.imageViews, resource.index);
    if (!slot)
        return vernon::Result<void, vernon::RhiError>{vernon::err(
            vernon::RhiError{vernon::RhiErrorCode::InvalidArgument, {"describe_opengl_image_view", key, 0}})};
    std::lock_guard<std::mutex> guard(state.mutex);
    state.state.makeCurrent();
    *view = slot->descriptor;
    *image = slot->imageDescriptor;
    *parentKey = slot->parentKey;
    return vernon::Result<void, vernon::RhiError>{vernon::ok()};
}

} // namespace vernon::rhi::opengl_api

const vernon::rhi::BackendDispatch &vernon::rhi::openGLBackendDispatch() {
    using namespace opengl_api;
    static const BackendDispatch dispatch{
        VERNON_RHI_BACKEND_OPENGL,
        0,
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
        downloadBuffer,
        destroyBuffer,
        isBufferValid,
        getBufferNativeHandle,
        createImage,
        setImageSampler,
        uploadImage,
        downloadImage,
        generateImageMipmaps,
        bindImage,
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
        nullptr,
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
