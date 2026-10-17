#include "backend_dispatch.h"
#include "image_data_layout.h"
#include "image_descriptor_validation.h"
#include "opengl_backend.h"
#include "public_c_boundary.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace {

using vernon::rhi::opengl::Buffer;
using vernon::rhi::opengl::DeviceState;
using vernon::rhi::opengl::Enum;
using vernon::rhi::opengl::Image;
using vernon::rhi::opengl::Int;
using vernon::rhi::opengl::Size;
using vernon::rhi::opengl::Uint;
namespace gl = vernon::rhi::opengl;

struct FormatInfo {
    Int internal{};
    Enum external{};
    Enum allocationType{};
};

using BufferLifecycle = vernon::rhi::ResourceLifecycleSlot<vernon::rhi::BufferResourceTag>;
using ImageLifecycle = vernon::rhi::ResourceLifecycleSlot<vernon::rhi::ImageResourceTag>;
using ImageViewLifecycle = vernon::rhi::ResourceLifecycleSlot<vernon::rhi::ImageViewResourceTag>;
using SamplerLifecycle = vernon::rhi::ResourceLifecycleSlot<vernon::rhi::SamplerResourceTag>;
using SurfaceLifecycle = vernon::rhi::ResourceLifecycleSlot<vernon::rhi::SurfaceResourceTag>;
using ImageLease = vernon::rhi::RetainedResourceLease<vernon::rhi::ImageResourceTag>;

constexpr std::size_t kMaximumOpenGLDevices = 256;

struct ImageSlot {
    explicit ImageSlot(ImageLifecycle lifecycle) noexcept : lifecycle(std::move(lifecycle)) {}
    ImageLifecycle lifecycle;
    Image image;
    VernonRhiImageDescriptor descriptor{};
    Enum target{};
    // A surface owns the image, so only configuring or destroying the surface destroys it.
    std::atomic<bool> surfaceOwned{};
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
    std::atomic<bool> surfaceOwned{};
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

struct SurfaceImage {
    VernonRhiImage image{};
    VernonRhiImageView view{};
};

struct SurfaceSlot {
    explicit SurfaceSlot(SurfaceLifecycle lifecycle) noexcept : lifecycle(std::move(lifecycle)) {}
    SurfaceLifecycle lifecycle;
    VernonOpenGLSurfaceCallbacks callbacks{};
    // Guards the fields below. It is taken before the device's mutexes.
    std::mutex mutex;
    VernonRhiSurfaceConfiguration configuration{};
    std::vector<SurfaceImage> images;
    size_t next{};
    bool acquired{};
};

struct OpenGLScopeAttachments {
    std::array<Uint, VERNON_RHI_MAX_COLOR_ATTACHMENTS> colors{};
    Uint depth{};
    bool stencil{};

    bool operator==(const OpenGLScopeAttachments &other) const {
        return colors == other.colors && depth == other.depth && stencil == other.stencil;
    }
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
    vernon::rhi::StableResourceSlotContainer<SurfaceSlot> surfaces;
    vernon::rhi::CommandDeviceStateRef commandState;
    std::string error;
    std::mutex resourceAllocationMutex;
    std::mutex mutex;
    // Every rendering scope renders into this framebuffer, created with the device so that encoding allocates no GL
    // names. Texture names are reusable once `state.framebufferGeneration` moves past `scopeGeneration`, so
    // `scopeAttachments` then no longer describes what is attached.
    Uint scopeFramebuffer{};
    OpenGLScopeAttachments scopeAttachments{};
    uint64_t scopeGeneration{};
    bool scopeAttachmentsValid{};
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
    slot->surfaceOwned.store(false, std::memory_order_relaxed);
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
        slot->surfaceOwned.store(false, std::memory_order_relaxed);
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

// Encoders that retained a surface image keep it alive until they complete.
vernon::Result<void, vernon::RhiError> destroySurfaceImages(OpenGLDevice &device,
                                                            std::vector<SurfaceImage> &images) noexcept {
    vernon::Option<vernon::RhiError> failure;
    const auto keep = [&](vernon::Result<void, vernon::RhiError> destroyed) {
        if (destroyed.isErr() && !failure)
            failure.emplace(std::move(destroyed).error());
    };
    for (const SurfaceImage &image : images) {
        if (ImageViewSlot *view = indexedSlot(device.imageViews, image.view.index))
            keep(view->lifecycle.destroyPublic({image.view.index, image.view.generation}));
        if (ImageSlot *slot = indexedSlot(device.images, image.image.index))
            keep(slot->lifecycle.destroyPublic({image.image.index, image.image.generation}));
    }
    images.clear();
    if (failure)
        return vernon::Result<void, vernon::RhiError>{vernon::err(std::move(failure).value())};
    return vernon::Result<void, vernon::RhiError>{vernon::ok()};
}

// Runs under the slot's mutex, which destroySurface holds so that no image is acquired meanwhile.
vernon::Result<void, vernon::RhiError> teardownSurface(void *context, SurfaceLifecycle::Handle handle) noexcept {
    auto &device = *static_cast<OpenGLDevice *>(context);
    SurfaceSlot *slot = indexedSlot(device.surfaces, handle.index);
    if (!slot)
        return vernon::Result<void, vernon::RhiError>{vernon::err(vernon::RhiError{
            vernon::RhiErrorCode::LifecycleFailure, {"teardown_opengl_surface", handle.generation, handle.index}})};
    auto destroyed = destroySurfaceImages(device, slot->images);
    slot->callbacks = {};
    slot->configuration = {};
    slot->next = 0;
    slot->acquired = false;
    return destroyed;
}

vernon::Result<void, vernon::RhiError> teardownDevice(OpenGLDevice &device) noexcept {
    std::lock_guard<std::mutex> guard(device.mutex);
    device.state.makeCurrent();
    device.state.destroyFramebuffer(device.scopeFramebuffer);
    device.scopeFramebuffer = 0;
    device.scopeAttachmentsValid = false;
    device.state.destroyDepthStencilCopyObjects();
    return vernon::Result<void, vernon::RhiError>{vernon::ok()};
}

bool formatInfo(VernonRhiFormat format, FormatInfo &result) {
    const vernon::formats::Format *entry = vernon::formats::find(format);
    if (!entry)
        return false;
    result = {static_cast<Int>(entry->glInternalFormat), entry->glFormat, entry->glType};
    return true;
}

// OpenGL addresses 3D slices and array layers (cube-array layer-faces included) as the z coordinate of the
// 3D texture entry points; cube maps instead use one 2D target per face.
bool volumeTarget(VernonRhiImageDimension dimension) {
    return dimension == VERNON_RHI_IMAGE_3D || dimension == VERNON_RHI_IMAGE_2D_ARRAY ||
           dimension == VERNON_RHI_IMAGE_CUBE_ARRAY;
}

uint32_t volumeDepth(const VernonRhiImageDescriptor &descriptor, uint32_t mip) {
    return descriptor.dimension == VERNON_RHI_IMAGE_3D ? vernon::rhi::imageMipExtent(descriptor.depth, mip)
                                                       : descriptor.array_layers;
}

Int samplerFilter(uint32_t value) {
    constexpr Int values[] = {0x2600, 0x2601, 0x2700, 0x2703, 0x2701, 0x2702};
    return value < sizeof(values) / sizeof(values[0]) ? values[value] : 0;
}

Int samplerAddress(uint32_t value) {
    constexpr Int values[] = {0x2901, 0x812F, 0x8370};
    return value < sizeof(values) / sizeof(values[0]) ? values[value] : 0;
}

struct TransferLayout {
    Enum format{};
    Enum type{};
};

// Callers validate the transfer with imageTransferAspectMatches first, so a transfer either moves whole texels in
// the table's layout or one aspect of a packed depth-stencil image.
TransferLayout transferLayout(VernonRhiFormat format, uint32_t aspect) {
    const vernon::formats::Format *entry = vernon::formats::find(format);
    if (!entry)
        return {};
    if (aspect == entry->aspects())
        return {entry->glFormat, entry->glType};
    if (aspect == VERNON_RHI_IMAGE_ASPECT_DEPTH)
        return {0x1902, 0x1406};
    if (aspect == VERNON_RHI_IMAGE_ASPECT_STENCIL)
        return {0x1901, 0x1401};
    return {};
}

vernon::RhiError fail(OpenGLDevice &device, std::string message,
                      vernon::RhiErrorCode code = vernon::RhiErrorCode::InvalidArgument) {
    device.error = std::move(message);
    return {code, {"opengl_backend", 0, 0}};
}

std::optional<vernon::RhiError> beginErrorScope(OpenGLDevice &device, const char *operation) {
    if (!device.state.driver.getError)
        return std::nullopt;
    const Enum error = device.state.driver.getError();
    if (error == 0)
        return std::nullopt;
    return fail(device,
                std::string("OpenGL context had error ") + std::to_string(error) +
                    " before entering Vernon operation '" + operation + "'",
                vernon::RhiErrorCode::BackendFailure);
}

std::optional<vernon::RhiError> endErrorScope(OpenGLDevice &device, const char *operation,
                                              vernon::RhiErrorCode code = vernon::RhiErrorCode::BackendFailure) {
    if (!device.state.driver.getError)
        return std::nullopt;
    const Enum error = device.state.driver.getError();
    if (error == 0)
        return std::nullopt;
    return fail(device, std::string("OpenGL operation '") + operation + "' failed with error " + std::to_string(error),
                code);
}

ImageViewSlot *imageViewSlot(OpenGLDevice &device, uint64_t key) noexcept {
    ImageViewLifecycle::Handle resource{};
    return decodeResourceKey(key, resource) ? indexedSlot(device.imageViews, resource.index) : nullptr;
}

// Attaches mip 0 and layer 0 of the view's texture: owned views start at the view's subresources, and identity views
// reach rendering only when they have one mip and render their first layer.
void attachOpenGLView(gl::Driver &driver, Enum attachment, const ImageViewSlot *view) {
    if (!view)
        driver.framebufferTexture2D(gl::kFramebuffer, attachment, gl::kTexture2D, 0, 0);
    else if (view->target == gl::kTexture2D)
        driver.framebufferTexture2D(gl::kFramebuffer, attachment, gl::kTexture2D, imageViewName(*view), 0);
    else if (view->target == gl::kTextureCubeMap)
        driver.framebufferTexture2D(gl::kFramebuffer, attachment, gl::kTextureCubeMapPositiveX, imageViewName(*view),
                                    0);
    else
        driver.framebufferTextureLayer(gl::kFramebuffer, attachment, imageViewName(*view), 0, 0);
}

// glInvalidateFramebuffer discards whole attachments, so a store discard applies only when the render area covers
// the attachment.
bool renderAreaCoversOpenGLView(OpenGLDevice &device, uint64_t key, const vernon::rhi::RenderingScope &scope) {
    const ImageViewSlot *view = imageViewSlot(device, key);
    if (!view)
        return false;
    const uint32_t mip = view->descriptor.base_mip_level;
    return scope.x == 0 && scope.y == 0 &&
           scope.width == vernon::rhi::imageMipExtent(view->imageDescriptor.width, mip) &&
           scope.height == vernon::rhi::imageMipExtent(view->imageDescriptor.height, mip);
}

// glClearBuffer honors the write masks and the scissor test, which hold the last draw's state.
void prepareOpenGLClears(gl::DeviceState &state, const vernon::rhi::RenderingScope &scope) {
    auto &driver = state.driver;
    driver.colorMask(1, 1, 1, 1);
    driver.depthMask(1);
    driver.stencilMaskSeparate(gl::kFront, UINT32_MAX);
    driver.stencilMaskSeparate(gl::kBack, UINT32_MAX);
    driver.enable(gl::kScissorTest);
    driver.scissor(static_cast<Int>(scope.x), static_cast<Int>(scope.y), static_cast<Size>(scope.width),
                   static_cast<Size>(scope.height));
    ++state.drawStateGeneration;
}

// glBlitFramebuffer honors the scissor test, which holds the last scope's or draw's state, and on desktop GL
// GL_FRAMEBUFFER_SRGB, which a rendering scope whose native begin failed leaves enabled.
void prepareOpenGLBlits(gl::DeviceState &state) {
    state.driver.disable(gl::kScissorTest);
    if (!state.embeddedProfile)
        state.driver.disable(gl::kFramebufferSrgb);
    ++state.drawStateGeneration;
}

void clearOpenGLDepthStencil(gl::Driver &driver, bool depth, bool stencil, float depthValue, uint32_t stencilValue) {
    if (depth && stencil) {
        driver.clearBufferfi(gl::kDepthStencil, 0, depthValue, static_cast<Int>(stencilValue));
        return;
    }
    if (depth)
        driver.clearBufferfv(gl::kDepth, 0, &depthValue);
    if (stencil) {
        const Int value = static_cast<Int>(stencilValue);
        driver.clearBufferiv(gl::kStencil, 0, &value);
    }
}

// glClearBufferfv on an integer color buffer is undefined, so each component class uses its own entry point.
void clearOpenGLColor(gl::Driver &driver, uint32_t location, VernonRhiFormat format, const VernonClearColor &color) {
    const vernon::formats::Format *entry = vernon::formats::find(format);
    const auto componentClass = entry ? entry->componentClass : vernon::formats::ComponentClass::Float;
    const Int buffer = static_cast<Int>(location);
    if (componentClass == vernon::formats::ComponentClass::Uint)
        driver.clearBufferuiv(gl::kColor, buffer, color.u32);
    else if (componentClass == vernon::formats::ComponentClass::Sint)
        driver.clearBufferiv(gl::kColor, buffer, color.i32);
    else
        driver.clearBufferfv(gl::kColor, buffer, color.f32);
}

vernon::Result<VernonRhiDevice, vernon::RhiError> createOpenGLDeviceImpl(const VernonOpenGLContextCallbacks *callbacks,
                                                                         bool embeddedProfile) {
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
    if (!device.state.initialize(*callbacks, embeddedProfile, device.error) ||
        !device.state.createFramebuffer(device.scopeFramebuffer, device.error))
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
    return publicHandleBoundary([&] { return createOpenGLDeviceImpl(callbacks, embeddedProfile != 0); },
                                invalidDevice());
}

Result<VernonRhiDevice, RhiError> createOwnedDevice(const VernonRhiOwnedDeviceDescriptor *descriptor) {
    if (!descriptor || descriptor->struct_size < sizeof(*descriptor))
        return Result<VernonRhiDevice, RhiError>{
            err(RhiError{RhiErrorCode::InvalidArgument, {"create_owned_opengl_device", 0, 0}})};
    if (descriptor->flags & ~static_cast<uint32_t>(VERNON_RHI_OWNED_DEVICE_FORCE_SOFTWARE))
        return Result<VernonRhiDevice, RhiError>{
            err(RhiError{RhiErrorCode::InvalidArgument, {"create_owned_opengl_device", descriptor->flags, 0}})};
    if (descriptor->backend != VERNON_RHI_BACKEND_DIRECTX12 && descriptor->flags)
        return Result<VernonRhiDevice, RhiError>{
            err(RhiError{RhiErrorCode::InvalidArgument, {"create_owned_opengl_device", descriptor->flags, 0}})};
    if (descriptor->backend == VERNON_RHI_BACKEND_OPENGL || descriptor->backend == VERNON_RHI_BACKEND_OPENGL_ES)
        return createOpenGLDeviceImpl(descriptor->opengl_callbacks,
                                      descriptor->backend == VERNON_RHI_BACKEND_OPENGL_ES);
    return Result<VernonRhiDevice, RhiError>{err(RhiError{
        RhiErrorCode::Unsupported, {"create_owned_opengl_device", static_cast<uint64_t>(descriptor->backend), 0}})};
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

Option<VernonStringView> lastError(VernonRhiDevice handle) {
    auto device = lookupDevice(handle);
    if (device.isErr())
        return Option<VernonStringView>{};
    OpenGLDevice &state = device.value().device();
    std::lock_guard<std::mutex> guard(state.mutex);
    return Option<VernonStringView>{some(VernonStringView{state.error.data(), state.error.size()})};
}

Result<VernonRhiBuffer, RhiError> createBuffer(VernonRhiDevice handle, const VernonRhiBufferDescriptor *descriptor) {
    auto device = lookupDevice(handle);
    if (device.isErr())
        return Result<VernonRhiBuffer, RhiError>{err(std::move(device).error())};
    if (!descriptor || descriptor->struct_size < sizeof(*descriptor) || descriptor->size == 0 ||
        descriptor->size > (std::numeric_limits<size_t>::max)() ||
        descriptor->memory_class > VERNON_RHI_MEMORY_READBACK)
        return Result<VernonRhiBuffer, RhiError>{
            err(RhiError{RhiErrorCode::InvalidArgument, {"create_opengl_buffer", 0, 0}})};
    auto owner = device.value().retainOwner();
    if (owner.isErr())
        return Result<VernonRhiBuffer, RhiError>{err(std::move(owner).error())};
    auto reservation = vernon::rhi::ResourceCreationReservation::create(owner.value());
    if (reservation.isErr())
        return Result<VernonRhiBuffer, RhiError>{err(std::move(reservation).error())};
    OpenGLDevice &state = device.value().device();
    std::lock_guard<std::mutex> allocationGuard(state.resourceAllocationMutex);
    auto available = availableSlot<BufferSlot, BufferLifecycle>(state.buffers);
    if (available.isErr())
        return Result<VernonRhiBuffer, RhiError>{err(std::move(available).error())};
    BufferSlot &slot = *available.value();
    std::lock_guard<std::mutex> guard(state.mutex);
    state.state.makeCurrent();
    if (!state.state.createBuffer(slot.buffer, static_cast<size_t>(descriptor->size), state.error))
        return Result<VernonRhiBuffer, RhiError>{
            err(RhiError{RhiErrorCode::BackendFailure, {"create_opengl_buffer", descriptor->size, 0}})};
    slot.descriptor = *descriptor;
    auto published = slot.lifecycle.publish(std::move(reservation).value(), &state, teardownBuffer);
    if (published.isErr()) {
        state.state.destroyBuffer(slot.buffer);
        slot.buffer = {};
        slot.descriptor = {};
        return Result<VernonRhiBuffer, RhiError>{err(std::move(published).error())};
    }
    return Result<VernonRhiBuffer, RhiError>{
        ok(VernonRhiBuffer{published.value().index, published.value().generation})};
}

Result<void, RhiError> uploadBuffer(VernonRhiDevice handle, VernonRhiBuffer buffer, uint64_t offset, const void *source,
                                    uint64_t size) {

    auto device = lookupDevice(handle);
    if (device.isErr())
        return Result<void, RhiError>{err(std::move(device).error())};
    if (!source)
        return Result<void, RhiError>{err(RhiError{RhiErrorCode::InvalidArgument, {"upload_opengl_buffer", 0, 0}})};
    OpenGLDevice &state = device.value().device();
    BufferSlot *slot = indexedSlot(state.buffers, buffer.index);
    auto pin = pinSlot(slot, buffer);
    if (pin.isErr())
        return Result<void, RhiError>{err(std::move(pin).error())};
    std::lock_guard<std::mutex> guard(state.mutex);
    state.state.makeCurrent();
    if (!slot || offset > slot->descriptor.size || size > slot->descriptor.size - offset)
        return Result<void, RhiError>{err(fail(state, "OpenGL RHI buffer upload range is invalid"))};
    if (!state.state.uploadBuffer(slot->buffer, static_cast<size_t>(offset), source, static_cast<size_t>(size),
                                  state.error))
        return Result<void, RhiError>{err(RhiError{RhiErrorCode::BackendFailure, {"upload_opengl_buffer", offset, 0}})};
    return Result<void, RhiError>{ok()};
}

Result<void, RhiError> uploadBufferRanges(VernonRhiDevice handle, VernonRhiBuffer buffer,
                                          const VernonRhiBufferUploadRange *ranges, size_t rangeCount) {
    auto device = lookupDevice(handle);
    if (device.isErr())
        return Result<void, RhiError>{err(std::move(device).error())};
    if (!ranges || rangeCount == 0)
        return Result<void, RhiError>{
            err(RhiError{RhiErrorCode::InvalidArgument, {"upload_opengl_buffer_ranges", rangeCount, 0}})};
    OpenGLDevice &state = device.value().device();
    BufferSlot *slot = indexedSlot(state.buffers, buffer.index);
    auto pin = pinSlot(slot, buffer);
    if (pin.isErr())
        return Result<void, RhiError>{err(std::move(pin).error())};
    std::lock_guard<std::mutex> guard(state.mutex);
    state.state.makeCurrent();
    for (size_t index = 0; index < rangeCount; ++index) {
        const VernonRhiBufferUploadRange &range = ranges[index];
        if (!range.source || range.size == 0 || range.size > (std::numeric_limits<size_t>::max)() ||
            range.offset > slot->descriptor.size || range.size > slot->descriptor.size - range.offset)
            return Result<void, RhiError>{err(fail(state, "OpenGL RHI buffer upload range is invalid"))};
    }
    for (size_t index = 0; index < rangeCount; ++index) {
        const VernonRhiBufferUploadRange &range = ranges[index];
        if (!state.state.uploadBuffer(slot->buffer, static_cast<size_t>(range.offset), range.source,
                                      static_cast<size_t>(range.size), state.error))
            return Result<void, RhiError>{
                err(RhiError{RhiErrorCode::BackendFailure, {"upload_opengl_buffer_ranges", index, 0}})};
    }
    return Result<void, RhiError>{ok()};
}

Result<void, RhiError> downloadBuffer(VernonRhiDevice handle, VernonRhiBuffer buffer, uint64_t offset,
                                      void *destination, uint64_t size) {

    auto device = lookupDevice(handle);
    if (device.isErr())
        return Result<void, RhiError>{err(std::move(device).error())};
    if (!destination)
        return Result<void, RhiError>{err(RhiError{RhiErrorCode::InvalidArgument, {"download_opengl_buffer", 0, 0}})};
    OpenGLDevice &state = device.value().device();
    BufferSlot *slot = indexedSlot(state.buffers, buffer.index);
    auto pin = pinSlot(slot, buffer);
    if (pin.isErr())
        return Result<void, RhiError>{err(std::move(pin).error())};
    std::lock_guard<std::mutex> guard(state.mutex);
    state.state.makeCurrent();
    if (!slot || offset > slot->descriptor.size || size > slot->descriptor.size - offset)
        return Result<void, RhiError>{err(fail(state, "OpenGL RHI buffer readback range is invalid"))};
    if (!state.state.downloadBuffer(slot->buffer, static_cast<size_t>(offset), destination, static_cast<size_t>(size),
                                    state.error))
        return Result<void, RhiError>{
            err(RhiError{RhiErrorCode::BackendFailure, {"download_opengl_buffer", offset, 0}})};
    return Result<void, RhiError>{ok()};
}

Result<void, RhiError> downloadBufferRanges(VernonRhiDevice handle, VernonRhiBuffer buffer,
                                            const VernonRhiBufferDownloadRange *ranges, size_t rangeCount) {
    auto device = lookupDevice(handle);
    if (device.isErr())
        return Result<void, RhiError>{err(std::move(device).error())};
    if (!ranges || rangeCount == 0)
        return Result<void, RhiError>{
            err(RhiError{RhiErrorCode::InvalidArgument, {"download_opengl_buffer_ranges", rangeCount, 0}})};
    OpenGLDevice &state = device.value().device();
    BufferSlot *slot = indexedSlot(state.buffers, buffer.index);
    auto pin = pinSlot(slot, buffer);
    if (pin.isErr())
        return Result<void, RhiError>{err(std::move(pin).error())};
    std::lock_guard<std::mutex> guard(state.mutex);
    if (!slot)
        return Result<void, RhiError>{
            err(RhiError{RhiErrorCode::InvalidArgument, {"download_opengl_buffer_ranges", 0, 0}})};
    size_t totalSize = 0;
    for (size_t index = 0; index < rangeCount; ++index) {
        const VernonRhiBufferDownloadRange &range = ranges[index];
        if (!range.destination || range.size == 0 || range.size > (std::numeric_limits<size_t>::max)() ||
            range.offset > slot->descriptor.size || range.size > slot->descriptor.size - range.offset)
            return Result<void, RhiError>{err(fail(state, "OpenGL RHI buffer readback range is invalid"))};
        if (range.size > (std::numeric_limits<size_t>::max)() - totalSize)
            return Result<void, RhiError>{err(fail(state, "OpenGL RHI batched buffer readback size overflows"))};
        totalSize += static_cast<size_t>(range.size);
    }
    Buffer staging;
    if (!state.state.createBuffer(staging, totalSize, state.error))
        return Result<void, RhiError>{
            err(RhiError{RhiErrorCode::BackendFailure, {"download_opengl_buffer_ranges", totalSize, 0}})};
    std::vector<size_t> stagingOffsets;
    stagingOffsets.reserve(rangeCount);
    size_t stagingOffset = 0;
    for (size_t index = 0; index < rangeCount; ++index) {
        stagingOffsets.push_back(stagingOffset);
        if (!state.state.copyBuffer(slot->buffer, static_cast<size_t>(ranges[index].offset), staging, stagingOffset,
                                    static_cast<size_t>(ranges[index].size), state.error)) {
            state.state.destroyBuffer(staging);
            return Result<void, RhiError>{
                err(RhiError{RhiErrorCode::BackendFailure, {"download_opengl_buffer_ranges", index, 0}})};
        }
        stagingOffset += static_cast<size_t>(ranges[index].size);
    }
    std::vector<uint8_t> packed(totalSize);
    if (!state.state.downloadBuffer(staging, 0, packed.data(), packed.size(), state.error)) {
        state.state.destroyBuffer(staging);
        return Result<void, RhiError>{
            err(RhiError{RhiErrorCode::BackendFailure, {"download_opengl_buffer_ranges", totalSize, 0}})};
    }
    state.state.destroyBuffer(staging);
    for (size_t index = 0; index < rangeCount; ++index)
        std::memcpy(ranges[index].destination, packed.data() + stagingOffsets[index],
                    static_cast<size_t>(ranges[index].size));
    return Result<void, RhiError>{ok()};
}

Result<bool, RhiError> isBufferValid(VernonRhiDevice handle, VernonRhiBuffer buffer) {

    auto device = lookupDevice(handle);
    if (device.isErr())
        return Result<bool, RhiError>{err(std::move(device).error())};
    OpenGLDevice &state = device.value().device();
    auto pin = pinSlot(indexedSlot(state.buffers, buffer.index), buffer);
    if (pin.isErr())
        return Result<bool, RhiError>{err(std::move(pin).error())};
    return Result<bool, RhiError>{ok(true)};
}

Result<VernonRhiImage, RhiError> createImage(VernonRhiDevice handle, const VernonRhiImageDescriptor *descriptor) {
    auto device = lookupDevice(handle);
    if (device.isErr())
        return Result<VernonRhiImage, RhiError>{err(std::move(device).error())};
    FormatInfo format;
    const Enum target = vernon::rhi::opengl::textureTarget(descriptor->dimension);
    OpenGLDevice &state = device.value().device();
    const bool requiresImmutableStorage = (descriptor->usage & VERNON_RHI_IMAGE_STORAGE) != 0;
    if (!target || !formatInfo(descriptor->format, format))
        return Result<VernonRhiImage, RhiError>{err(fail(state, "OpenGL RHI image descriptor is invalid"))};
    auto owner = device.value().retainOwner();
    if (owner.isErr())
        return Result<VernonRhiImage, RhiError>{err(std::move(owner).error())};
    auto reservation = vernon::rhi::ResourceCreationReservation::create(owner.value());
    if (reservation.isErr())
        return Result<VernonRhiImage, RhiError>{err(std::move(reservation).error())};
    std::lock_guard<std::mutex> allocationGuard(state.resourceAllocationMutex);
    auto available = availableSlot<ImageSlot, ImageLifecycle>(state.images);
    if (available.isErr())
        return Result<VernonRhiImage, RhiError>{err(std::move(available).error())};
    ImageSlot &slot = *available.value();
    std::lock_guard<std::mutex> guard(state.mutex);
    state.state.makeCurrent();
    state.state.driver.genTextures(1, &slot.image.name);
    if (!slot.image.name)
        return Result<VernonRhiImage, RhiError>{
            err(fail(state, "OpenGL RHI image allocation failed", RhiErrorCode::BackendFailure))};
    slot.image.imported = false;
    {
        auto &driver = state.state.driver;
        driver.bindTexture(target, slot.image.name);
        const bool immutableStorage =
            driver.texStorage2D && (!volumeTarget(descriptor->dimension) || driver.texStorage3D);
        if (requiresImmutableStorage && !immutableStorage) {
            driver.bindTexture(target, 0);
            state.state.destroyImage(slot.image);
            return Result<VernonRhiImage, RhiError>{
                err(fail(state, "OpenGL storage images require immutable texture storage", RhiErrorCode::Unsupported))};
        }
        if (immutableStorage) {
            if (volumeTarget(descriptor->dimension))
                driver.texStorage3D(target, static_cast<Size>(descriptor->mip_levels),
                                    static_cast<Enum>(format.internal), static_cast<Size>(descriptor->width),
                                    static_cast<Size>(descriptor->height),
                                    static_cast<Size>(volumeDepth(*descriptor, 0)));
            else
                driver.texStorage2D(target, static_cast<Size>(descriptor->mip_levels),
                                    static_cast<Enum>(format.internal), static_cast<Size>(descriptor->width),
                                    static_cast<Size>(descriptor->height));
        } else {
            for (uint32_t mip = 0; mip < descriptor->mip_levels; ++mip) {
                const Size mipWidth = static_cast<Size>(vernon::rhi::imageMipExtent(descriptor->width, mip));
                const Size mipHeight = static_cast<Size>(vernon::rhi::imageMipExtent(descriptor->height, mip));
                const Size mipDepth = static_cast<Size>(volumeDepth(*descriptor, mip));
                if (volumeTarget(descriptor->dimension)) {
                    driver.texImage3D(target, static_cast<Int>(mip), format.internal, mipWidth, mipHeight, mipDepth, 0,
                                      format.external, format.allocationType, nullptr);
                } else if (descriptor->dimension == VERNON_RHI_IMAGE_CUBE) {
                    for (uint32_t face = 0; face < 6; ++face)
                        driver.texImage2D(vernon::rhi::opengl::kTextureCubeMapPositiveX + face, static_cast<Int>(mip),
                                          format.internal, mipWidth, mipHeight, 0, format.external,
                                          format.allocationType, nullptr);
                } else {
                    driver.texImage2D(target, static_cast<Int>(mip), format.internal, mipWidth, mipHeight, 0,
                                      format.external, format.allocationType, nullptr);
                }
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
        return Result<VernonRhiImage, RhiError>{err(std::move(published).error())};
    }
    return Result<VernonRhiImage, RhiError>{ok(VernonRhiImage{published.value().index, published.value().generation})};
}

Result<void, RhiError> setImageSampler(VernonRhiDevice handle, VernonRhiImage image,
                                       const VernonRhiSamplerDescriptor *descriptor) {

    auto device = lookupDevice(handle);
    if (device.isErr())
        return Result<void, RhiError>{err(std::move(device).error())};
    if (!descriptor || descriptor->struct_size < sizeof(*descriptor))
        return Result<void, RhiError>{err(RhiError{RhiErrorCode::InvalidArgument, {"set_opengl_sampler", 0, 0}})};
    OpenGLDevice &state = device.value().device();
    ImageSlot *slot = indexedSlot(state.images, image.index);
    auto pin = pinSlot(slot, image);
    if (pin.isErr())
        return Result<void, RhiError>{err(std::move(pin).error())};
    const Int minFilter = samplerFilter(descriptor->min_filter);
    const Int magFilter = samplerFilter(descriptor->mag_filter);
    const Int addressU = samplerAddress(descriptor->address_u);
    const Int addressV = samplerAddress(descriptor->address_v);
    const Int addressW = samplerAddress(descriptor->address_w);
    std::lock_guard<std::mutex> guard(state.mutex);
    state.state.makeCurrent();
    if (!slot || !minFilter || (descriptor->mag_filter > VERNON_RHI_FILTER_LINEAR) || !magFilter || !addressU ||
        !addressV || !addressW)
        return Result<void, RhiError>{err(fail(state, "OpenGL RHI image sampler descriptor is invalid"))};
    auto &driver = state.state.driver;
    driver.bindTexture(slot->target, slot->image.name);
    driver.texParameteri(slot->target, 0x2802, addressU);
    driver.texParameteri(slot->target, 0x2803, addressV);
    if (slot->descriptor.dimension != VERNON_RHI_IMAGE_2D)
        driver.texParameteri(slot->target, 0x8072, addressW);
    driver.texParameteri(slot->target, 0x2801, minFilter);
    driver.texParameteri(slot->target, 0x2800, magFilter);
    driver.bindTexture(slot->target, 0);
    return Result<void, RhiError>{ok()};
}

Result<void, RhiError> uploadImage(VernonRhiDevice handle, VernonRhiImage image,
                                   const VernonRhiImageUploadDescriptor *uploads, size_t uploadCount) {

    auto device = lookupDevice(handle);
    if (device.isErr())
        return Result<void, RhiError>{err(std::move(device).error())};
    if (uploadCount != 0 && !uploads)
        return Result<void, RhiError>{err(RhiError{RhiErrorCode::InvalidArgument, {"upload_opengl_image", 0, 0}})};
    OpenGLDevice &state = device.value().device();
    ImageSlot *slot = indexedSlot(state.images, image.index);
    auto pin = pinSlot(slot, image);
    if (pin.isErr())
        return Result<void, RhiError>{err(std::move(pin).error())};
    std::lock_guard<std::mutex> guard(state.mutex);
    for (size_t index = 0; index < uploadCount; ++index) {
        const VernonRhiImageUploadDescriptor &upload = uploads[index];
        const bool validLayer = upload.array_layer < slot->descriptor.array_layers;
        const bool validMip = upload.mip_level < slot->descriptor.mip_levels;
        const uint32_t mipWidth = validMip ? vernon::rhi::imageMipExtent(slot->descriptor.width, upload.mip_level) : 0;
        const uint32_t mipHeight =
            validMip ? vernon::rhi::imageMipExtent(slot->descriptor.height, upload.mip_level) : 0;
        const uint32_t mipDepth = !validMip ? 0
                                  : slot->descriptor.dimension == VERNON_RHI_IMAGE_3D
                                      ? vernon::rhi::imageMipExtent(slot->descriptor.depth, upload.mip_level)
                                      : slot->descriptor.depth;
        if (upload.struct_size < sizeof(upload) || upload.mip_level >= slot->descriptor.mip_levels ||
            upload.width == 0 || upload.height == 0 || upload.depth == 0 || !validLayer ||
            upload.offset_x >= mipWidth || upload.width > mipWidth - upload.offset_x || upload.offset_y >= mipHeight ||
            upload.height > mipHeight - upload.offset_y || upload.offset_z >= mipDepth ||
            upload.depth > mipDepth - upload.offset_z ||
            !vernon::rhi::imageTransferAspectMatches(slot->descriptor.format, upload.aspect, upload.source_format,
                                                     upload.source_type))
            return Result<void, RhiError>{err(fail(state, "OpenGL RHI image upload descriptor is invalid"))};
        // glTexSubImage has no format that writes one aspect of a packed depth-stencil texture.
        if (vernon::formats::hasStencil(slot->descriptor.format) &&
            upload.aspect != (VERNON_RHI_IMAGE_ASPECT_DEPTH | VERNON_RHI_IMAGE_ASPECT_STENCIL))
            return Result<void, RhiError>{err(fail(
                state, "OpenGL uploads the depth and stencil of a packed image together", RhiErrorCode::Unsupported))};
    }
    state.state.makeCurrent();
    if (auto scopeError = beginErrorScope(state, "upload image"))
        return Result<void, RhiError>{err(std::move(*scopeError))};
    auto &driver = state.state.driver;
    driver.bindTexture(slot->target, slot->image.name);
    driver.pixelStorei(0x0CF5, 1);
    for (size_t index = 0; index < uploadCount; ++index) {
        const VernonRhiImageUploadDescriptor &upload = uploads[index];
        // The packed depth-stencil layout is GL_FLOAT_32_UNSIGNED_INT_24_8_REV: the depth float, then a word whose low
        // byte is the stencil.
        const TransferLayout layout = transferLayout(slot->descriptor.format, upload.aspect);
        if (volumeTarget(slot->descriptor.dimension))
            driver.texSubImage3D(slot->target, static_cast<Int>(upload.mip_level), static_cast<Int>(upload.offset_x),
                                 static_cast<Int>(upload.offset_y),
                                 static_cast<Int>(slot->descriptor.dimension == VERNON_RHI_IMAGE_3D
                                                      ? upload.offset_z
                                                      : upload.array_layer),
                                 static_cast<Size>(upload.width), static_cast<Size>(upload.height),
                                 static_cast<Size>(upload.depth), layout.format, layout.type, upload.data);
        else
            driver.texSubImage2D(slot->descriptor.dimension == VERNON_RHI_IMAGE_CUBE
                                     ? vernon::rhi::opengl::kTextureCubeMapPositiveX + upload.array_layer
                                     : slot->target,
                                 static_cast<Int>(upload.mip_level), static_cast<Int>(upload.offset_x),
                                 static_cast<Int>(upload.offset_y), static_cast<Size>(upload.width),
                                 static_cast<Size>(upload.height), layout.format, layout.type, upload.data);
    }
    driver.bindTexture(slot->target, 0);
    if (auto scopeError = endErrorScope(state, "upload image"))
        return Result<void, RhiError>{err(std::move(*scopeError))};
    return Result<void, RhiError>{ok()};
}

bool validImageDownload(const ImageSlot &slot, const VernonRhiImageDownloadDescriptor &download,
                        const void *destination, size_t size) {
    const auto expected = imageDownloadByteSize(slot.descriptor, download);
    return destination && expected && size == *expected;
}

Result<void, RhiError> readDepthStencilLocked(OpenGLDevice &state, const ImageSlot &slot,
                                              const VernonRhiImageDownloadDescriptor &download, void *destination) {
    using vernon::rhi::opengl::DepthStencilAspect;
    const bool readDepth = (download.aspect & VERNON_RHI_IMAGE_ASPECT_DEPTH) != 0;
    const bool readStencil = (download.aspect & VERNON_RHI_IMAGE_ASPECT_STENCIL) != 0;
    const size_t texels = static_cast<size_t>(download.width) * download.height;
    Buffer words;
    if (!state.state.createBuffer(words, texels * sizeof(uint32_t), state.error))
        return Result<void, RhiError>{err(RhiError{RhiErrorCode::BackendFailure, {"download_opengl_image", 0, 0}})};
    std::vector<uint32_t> depth(readDepth ? texels : 0);
    std::vector<uint32_t> stencil(readStencil ? texels : 0);
    const auto read = [&](DepthStencilAspect aspect, std::vector<uint32_t> &values) {
        return state.state.copyDepthStencilToBuffer(
                   slot.image, slot.target, aspect, static_cast<Int>(download.mip_level),
                   static_cast<Int>(download.array_layer), static_cast<Int>(download.offset_x),
                   static_cast<Int>(download.offset_y), static_cast<Size>(download.width),
                   static_cast<Size>(download.height), words, state.error) &&
               state.state.downloadBuffer(words, 0, values.data(), values.size() * sizeof(uint32_t), state.error);
    };
    const bool copied = (!readDepth || read(DepthStencilAspect::Depth, depth)) &&
                        (!readStencil || read(DepthStencilAspect::Stencil, stencil));
    state.state.destroyBuffer(words);
    if (!copied)
        return Result<void, RhiError>{
            err(RhiError{state.state.supportsCompute() ? RhiErrorCode::BackendFailure : RhiErrorCode::Unsupported,
                         {"download_opengl_image", 0, 0}})};
    auto *output = static_cast<uint8_t *>(destination);
    for (size_t index = 0; index < texels; ++index) {
        float value{};
        if (readDepth)
            std::memcpy(&value, &depth[index], sizeof(value));
        if (readDepth && readStencil)
            storePackedDepthStencil(output + index * packedDepthStencilPixelSize, value,
                                    static_cast<uint8_t>(stencil[index]));
        else if (readDepth)
            std::memcpy(output + index * sizeof(value), &value, sizeof(value));
        else
            output[index] = static_cast<uint8_t>(stencil[index]);
    }
    return Result<void, RhiError>{ok()};
}

Result<void, RhiError> readImageLocked(OpenGLDevice &state, const ImageSlot &slot,
                                       const VernonRhiImageDownloadDescriptor &download, void *destination) {
    // OpenGL ES reads neither depth nor stencil with glReadPixels.
    if (state.state.embeddedProfile && vernon::formats::hasDepth(slot.descriptor.format))
        return readDepthStencilLocked(state, slot, download, destination);
    const size_t expected = *imageDownloadByteSize(slot.descriptor, download);
    const TransferLayout layout = transferLayout(slot.descriptor.format, download.aspect);
    const Enum externalFormat = layout.format;
    const Enum allocationType = layout.type;
    const bool packedDepthStencil =
        download.aspect == (VERNON_RHI_IMAGE_ASPECT_DEPTH | VERNON_RHI_IMAGE_ASPECT_STENCIL);
    if (!externalFormat || !allocationType)
        return Result<void, RhiError>{err(RhiError{RhiErrorCode::InvalidArgument, {"download_opengl_image", 0, 0}})};
    if (slot.descriptor.dimension == VERNON_RHI_IMAGE_3D)
        return state.state.downloadImage3D(slot.image, static_cast<Int>(download.mip_level),
                                           static_cast<Int>(download.offset_x), static_cast<Int>(download.offset_y),
                                           static_cast<Int>(download.offset_z), static_cast<Size>(download.width),
                                           static_cast<Size>(download.height), static_cast<Size>(download.depth),
                                           externalFormat, allocationType, expected / download.depth, destination,
                                           state.error)
                   ? Result<void, RhiError>{ok()}
                   : Result<void, RhiError>{
                         err(RhiError{RhiErrorCode::BackendFailure, {"download_opengl_image", 0, 0}})};
    const Enum target = slot.descriptor.dimension == VERNON_RHI_IMAGE_CUBE
                            ? vernon::rhi::opengl::kTextureCubeMapPositiveX + download.array_layer
                            : slot.target;
    const Int layer = static_cast<Int>(download.array_layer);
    if (!packedDepthStencil)
        return state.state.downloadImage2D(slot.image, target, static_cast<Int>(download.mip_level), layer,
                                           static_cast<Int>(download.offset_x), static_cast<Int>(download.offset_y),
                                           static_cast<Size>(download.width), static_cast<Size>(download.height),
                                           externalFormat, allocationType, destination, state.error)
                   ? Result<void, RhiError>{ok()}
                   : Result<void, RhiError>{
                         err(RhiError{RhiErrorCode::BackendFailure, {"download_opengl_image", 0, 0}})};
    std::vector<uint8_t> native(expected);
    if (!state.state.downloadImage2D(slot.image, target, static_cast<Int>(download.mip_level), layer,
                                     static_cast<Int>(download.offset_x), static_cast<Int>(download.offset_y),
                                     static_cast<Size>(download.width), static_cast<Size>(download.height),
                                     externalFormat, allocationType, native.data(), state.error))
        return Result<void, RhiError>{err(RhiError{RhiErrorCode::BackendFailure, {"download_opengl_image", 0, 0}})};
    auto *output = static_cast<uint8_t *>(destination);
    const size_t pixels = static_cast<size_t>(download.width) * download.height;
    for (size_t index = 0; index < pixels; ++index) {
        float depth{};
        uint32_t stencilWord{};
        std::memcpy(&depth, native.data() + index * packedDepthStencilPixelSize, sizeof(depth));
        std::memcpy(&stencilWord, native.data() + index * packedDepthStencilPixelSize + sizeof(depth),
                    sizeof(stencilWord));
        storePackedDepthStencil(output + index * packedDepthStencilPixelSize, depth,
                                static_cast<uint8_t>(stencilWord & 0xff));
    }
    return Result<void, RhiError>{ok()};
}

// glReadPixels reports an unsupported format or type only through glGetError and leaves the destination untouched.
Result<void, RhiError> downloadImageLocked(OpenGLDevice &state, const ImageSlot &slot,
                                           const VernonRhiImageDownloadDescriptor &download, void *destination) {
    if (auto scopeError = beginErrorScope(state, "download image"))
        return Result<void, RhiError>{err(std::move(*scopeError))};
    auto read = readImageLocked(state, slot, download, destination);
    if (read.isErr())
        return read;
    if (auto scopeError = endErrorScope(state, "download image"))
        return Result<void, RhiError>{err(std::move(*scopeError))};
    return read;
}

Result<void, RhiError> downloadImage(VernonRhiDevice handle, VernonRhiImage image,
                                     const VernonRhiImageDownloadDescriptor *download, void *destination, size_t size) {
    if (!download || download->struct_size < sizeof(*download))
        return Result<void, RhiError>{err(RhiError{RhiErrorCode::InvalidArgument, {"download_opengl_image", 0, 0}})};
    auto device = lookupDevice(handle);
    if (device.isErr())
        return Result<void, RhiError>{err(std::move(device).error())};
    OpenGLDevice &state = device.value().device();
    ImageSlot *slot = indexedSlot(state.images, image.index);
    auto pin = pinSlot(slot, image);
    if (pin.isErr())
        return Result<void, RhiError>{err(std::move(pin).error())};
    std::lock_guard<std::mutex> guard(state.mutex);
    if (!slot || !validImageDownload(*slot, *download, destination, size))
        return Result<void, RhiError>{err(RhiError{RhiErrorCode::InvalidArgument, {"download_opengl_image", 0, 0}})};
    state.state.makeCurrent();
    return downloadImageLocked(state, *slot, *download, destination);
}

Result<void, RhiError> downloadImageBatch(VernonRhiDevice handle, VernonRhiImage image,
                                          const VernonRhiImageDownload *downloads, size_t downloadCount) {
    if (!downloads || downloadCount == 0)
        return Result<void, RhiError>{
            err(RhiError{RhiErrorCode::InvalidArgument, {"download_opengl_image_batch", downloadCount, 0}})};
    auto device = lookupDevice(handle);
    if (device.isErr())
        return Result<void, RhiError>{err(std::move(device).error())};
    OpenGLDevice &state = device.value().device();
    ImageSlot *slot = indexedSlot(state.images, image.index);
    auto pin = pinSlot(slot, image);
    if (pin.isErr())
        return Result<void, RhiError>{err(std::move(pin).error())};
    std::lock_guard<std::mutex> guard(state.mutex);
    if (!slot)
        return Result<void, RhiError>{
            err(RhiError{RhiErrorCode::InvalidArgument, {"download_opengl_image_batch", 0, 0}})};
    for (size_t index = 0; index < downloadCount; ++index)
        if (downloads[index].descriptor.struct_size < sizeof(downloads[index].descriptor) ||
            !validImageDownload(*slot, downloads[index].descriptor, downloads[index].destination,
                                downloads[index].size))
            return Result<void, RhiError>{
                err(RhiError{RhiErrorCode::InvalidArgument, {"download_opengl_image_batch", index, 0}})};
    state.state.makeCurrent();
    for (size_t index = 0; index < downloadCount; ++index) {
        auto downloaded = downloadImageLocked(state, *slot, downloads[index].descriptor, downloads[index].destination);
        if (downloaded.isErr())
            return downloaded;
    }
    return Result<void, RhiError>{ok()};
}

Result<void, RhiError> generateImageMipmaps(VernonRhiDevice handle, VernonRhiImage image) {

    auto device = lookupDevice(handle);
    if (device.isErr())
        return Result<void, RhiError>{err(std::move(device).error())};
    OpenGLDevice &state = device.value().device();
    ImageSlot *slot = indexedSlot(state.images, image.index);
    auto pin = pinSlot(slot, image);
    if (pin.isErr())
        return Result<void, RhiError>{err(std::move(pin).error())};
    std::lock_guard<std::mutex> guard(state.mutex);
    if (!slot || slot->descriptor.mip_levels < 2 || !(slot->descriptor.usage & VERNON_RHI_IMAGE_TRANSFER_SOURCE) ||
        !(slot->descriptor.usage & VERNON_RHI_IMAGE_TRANSFER_DESTINATION) ||
        vernon::formats::hasDepth(slot->descriptor.format))
        return Result<void, RhiError>{
            err(RhiError{RhiErrorCode::InvalidArgument, {"generate_opengl_mipmaps", image.generation, image.index}})};
    state.state.makeCurrent();
    if (auto scopeError = beginErrorScope(state, "generate image mipmaps"))
        return Result<void, RhiError>{err(std::move(*scopeError))};
    state.state.driver.bindTexture(slot->target, slot->image.name);
    state.state.driver.generateMipmap(slot->target);
    state.state.driver.bindTexture(slot->target, 0);
    constexpr Enum invalidOperation = 0x0502;
    if (state.state.driver.getError) {
        const Enum error = state.state.driver.getError();
        if (error != 0)
            return Result<void, RhiError>{err(fail(
                state,
                error == invalidOperation
                    ? "OpenGL image format does not support mipmap generation (error " + std::to_string(error) + ")"
                    : "OpenGL mipmap generation failed with error " + std::to_string(error),
                error == invalidOperation ? RhiErrorCode::Unsupported : RhiErrorCode::BackendFailure))};
    }
    return Result<void, RhiError>{ok()};
}

Result<void, RhiError> bindImage(VernonRhiDevice handle, VernonRhiImage image, uint32_t textureUnit) {

    auto device = lookupDevice(handle);
    if (device.isErr())
        return Result<void, RhiError>{err(std::move(device).error())};
    OpenGLDevice &state = device.value().device();
    ImageSlot *slot = indexedSlot(state.images, image.index);
    auto pin = pinSlot(slot, image);
    if (pin.isErr())
        return Result<void, RhiError>{err(std::move(pin).error())};
    std::lock_guard<std::mutex> guard(state.mutex);
    state.state.makeCurrent();
    state.state.driver.activeTexture(vernon::rhi::opengl::kTexture0 + textureUnit);
    state.state.driver.bindTexture(slot->target, slot->image.name);
    return Result<void, RhiError>{ok()};
}

Result<void, RhiError> destroyImage(VernonRhiDevice handle, VernonRhiImage image) {

    auto device = lookupDevice(handle);
    if (device.isErr())
        return Result<void, RhiError>{err(std::move(device).error())};
    OpenGLDevice &state = device.value().device();
    ImageSlot *slot = indexedSlot(state.images, image.index);
    if (!slot || slot->surfaceOwned.load(std::memory_order_relaxed))
        return Result<void, RhiError>{
            err(RhiError{RhiErrorCode::InvalidArgument, {"destroy_opengl_image", image.generation, image.index}})};
    return slot->lifecycle.destroyPublic({image.index, image.generation});
}

Result<bool, RhiError> isImageValid(VernonRhiDevice handle, VernonRhiImage image) {

    auto device = lookupDevice(handle);
    if (device.isErr())
        return Result<bool, RhiError>{err(std::move(device).error())};
    OpenGLDevice &state = device.value().device();
    auto pin = pinSlot(indexedSlot(state.images, image.index), image);
    if (pin.isErr())
        return Result<bool, RhiError>{err(std::move(pin).error())};
    return Result<bool, RhiError>{ok(true)};
}

Result<VernonRhiSampler, RhiError> createSampler(VernonRhiDevice handle, const VernonRhiSamplerDescriptor *descriptor) {

    auto device = lookupDevice(handle);
    if (device.isErr())
        return Result<VernonRhiSampler, RhiError>{err(std::move(device).error())};
    if (!descriptor || descriptor->struct_size < sizeof(*descriptor))
        return Result<VernonRhiSampler, RhiError>{
            err(RhiError{RhiErrorCode::InvalidArgument, {"create_opengl_sampler", 0, 0}})};
    const Int minFilter = samplerFilter(descriptor->min_filter);
    const Int magFilter = samplerFilter(descriptor->mag_filter);
    const Int addressU = samplerAddress(descriptor->address_u);
    const Int addressV = samplerAddress(descriptor->address_v);
    const Int addressW = samplerAddress(descriptor->address_w);
    if (!minFilter || descriptor->mag_filter > VERNON_RHI_FILTER_LINEAR || !magFilter || !addressU || !addressV ||
        !addressW)
        return Result<VernonRhiSampler, RhiError>{
            err(fail(device.value().device(), "OpenGL RHI sampler descriptor is invalid"))};
    auto owner = device.value().retainOwner();
    if (owner.isErr())
        return Result<VernonRhiSampler, RhiError>{err(std::move(owner).error())};
    auto reservation = vernon::rhi::ResourceCreationReservation::create(owner.value());
    if (reservation.isErr())
        return Result<VernonRhiSampler, RhiError>{err(std::move(reservation).error())};
    OpenGLDevice &state = device.value().device();
    std::lock_guard<std::mutex> allocationGuard(state.resourceAllocationMutex);
    auto available = availableSlot<SamplerSlot, SamplerLifecycle>(state.samplers);
    if (available.isErr())
        return Result<VernonRhiSampler, RhiError>{err(std::move(available).error())};
    SamplerSlot &slot = *available.value();
    std::lock_guard<std::mutex> guard(state.mutex);
    state.state.makeCurrent();
    if (!state.state.createSampler(slot.sampler, addressU, addressV, addressW, minFilter, magFilter, state.error))
        return Result<VernonRhiSampler, RhiError>{
            err(RhiError{RhiErrorCode::BackendFailure, {"create_opengl_sampler", 0, 0}})};
    auto published = slot.lifecycle.publish(std::move(reservation).value(), &state, teardownSampler);
    if (published.isErr()) {
        state.state.destroySampler(slot.sampler);
        slot.sampler = {};
        return Result<VernonRhiSampler, RhiError>{err(std::move(published).error())};
    }
    return Result<VernonRhiSampler, RhiError>{
        ok(VernonRhiSampler{published.value().index, published.value().generation})};
}

Result<void, RhiError> destroySampler(VernonRhiDevice handle, VernonRhiSampler sampler) {

    auto device = lookupDevice(handle);
    if (device.isErr())
        return Result<void, RhiError>{err(std::move(device).error())};
    OpenGLDevice &state = device.value().device();
    SamplerSlot *slot = indexedSlot(state.samplers, sampler.index);
    if (!slot)
        return Result<void, RhiError>{err(
            RhiError{RhiErrorCode::InvalidArgument, {"destroy_opengl_sampler", sampler.generation, sampler.index}})};
    return slot->lifecycle.destroyPublic({sampler.index, sampler.generation});
}

Result<bool, RhiError> isSamplerValid(VernonRhiDevice handle, VernonRhiSampler sampler) {

    auto device = lookupDevice(handle);
    if (device.isErr())
        return Result<bool, RhiError>{err(std::move(device).error())};
    OpenGLDevice &state = device.value().device();
    auto pin = pinSlot(indexedSlot(state.samplers, sampler.index), sampler);
    if (pin.isErr())
        return Result<bool, RhiError>{err(std::move(pin).error())};
    return Result<bool, RhiError>{ok(true)};
}

Result<uint64_t, RhiError> getImageNativeHandle(VernonRhiDevice handle, VernonRhiImage image) {

    auto device = lookupDevice(handle);
    if (device.isErr())
        return Result<uint64_t, RhiError>{err(std::move(device).error())};
    OpenGLDevice &state = device.value().device();
    ImageSlot *slot = indexedSlot(state.images, image.index);
    auto pin = pinSlot(slot, image);
    if (pin.isErr())
        return Result<uint64_t, RhiError>{err(std::move(pin).error())};
    std::lock_guard<std::mutex> guard(state.mutex);
    state.state.makeCurrent();
    return Result<uint64_t, RhiError>{ok(uint64_t(slot->image.name))};
}

// OpenGL ES differs from desktop OpenGL in four rules: float color rendering needs ES 3.2 or an extension, 32-bit float
// filtering and blending need extensions, depth textures are never linearly filtered, and image units accept only the
// formats listed in ES 3.1.
uint32_t embeddedFormatCapabilities(const gl::DeviceState &device, const vernon::formats::Format &entry) {
    uint32_t result = entry.baseline;
    const bool color = entry.aspects() == VERNON_RHI_IMAGE_ASPECT_COLOR;
    const bool floatColor = color && entry.componentClass == vernon::formats::ComponentClass::Float;
    const bool float32 = floatColor && entry.dataType == VERNON_RHI_IMAGE_DATA_FLOAT32;
    const bool float16 = floatColor && entry.dataType == VERNON_RHI_IMAGE_DATA_FLOAT16;
    if (floatColor && !device.apiVersionAtLeast(3, 2) && !device.hasExtension("GL_EXT_color_buffer_float") &&
        !(float16 && device.hasExtension("GL_EXT_color_buffer_half_float")))
        result &= ~(VERNON_RHI_FORMAT_RENDERABLE | VERNON_RHI_FORMAT_BLENDABLE | VERNON_RHI_FORMAT_MULTISAMPLE |
                    VERNON_RHI_FORMAT_RESOLVABLE);
    if (float32 && !device.hasExtension("GL_OES_texture_float_linear"))
        result &= ~VERNON_RHI_FORMAT_FILTERABLE;
    if (float32 && !device.hasExtension("GL_EXT_float_blend"))
        result &= ~VERNON_RHI_FORMAT_BLENDABLE;
    if (!color)
        result &= ~VERNON_RHI_FORMAT_FILTERABLE;
    switch (entry.rhi) {
    case VERNON_RHI_FORMAT_RGBA8_UNORM:
    case VERNON_RHI_FORMAT_RGBA16_FLOAT:
    case VERNON_RHI_FORMAT_R32_FLOAT:
    case VERNON_RHI_FORMAT_RGBA32_FLOAT:
    case VERNON_RHI_FORMAT_R32_UINT:
        break;
    default:
        result &= ~VERNON_RHI_FORMAT_STORAGE;
    }
    return result;
}

Result<uint32_t, RhiError> formatCapabilities(VernonRhiDevice handle, VernonRhiFormat format) {
    auto device = lookupDevice(handle);
    if (device.isErr())
        return Result<uint32_t, RhiError>{err(std::move(device).error())};
    const vernon::formats::Format *entry = vernon::formats::find(format);
    if (!entry)
        return Result<uint32_t, RhiError>{ok(0u)};
    OpenGLDevice &state = device.value().device();
    std::lock_guard<std::mutex> guard(state.mutex);
    state.state.makeCurrent();
    const gl::DeviceState &gpu = state.state;
    uint32_t result = gpu.embeddedProfile ? embeddedFormatCapabilities(gpu, *entry) : entry->baseline;
    if (!gpu.supportsCompute())
        result &= ~VERNON_RHI_FORMAT_STORAGE;
    // 2D copies blit through framebuffers, so they need a color-renderable format; OpenGL renders RGB8, which the
    // table baseline leaves unrenderable for the other backends.
    if (!(result & VERNON_RHI_FORMAT_RENDERABLE) && entry->rhi != VERNON_RHI_FORMAT_RGB8_UNORM)
        result &= ~VERNON_RHI_FORMAT_COPYABLE;
    gl::Int sampleCounts = 0;
    if ((result & VERNON_RHI_FORMAT_RENDERABLE) && gpu.driver.getInternalformativ)
        gpu.driver.getInternalformativ(gl::kRenderbuffer, static_cast<gl::Enum>(entry->glInternalFormat),
                                       gl::kNumSampleCounts, 1, &sampleCounts);
    if (sampleCounts <= 0)
        result &= ~(VERNON_RHI_FORMAT_MULTISAMPLE | VERNON_RHI_FORMAT_RESOLVABLE);
    return Result<uint32_t, RhiError>{ok(result)};
}

Result<VernonRhiImageView, RhiError> createImageView(VernonRhiDevice handle,
                                                     const VernonRhiImageViewDescriptor *descriptor) {
    auto device = lookupDevice(handle);
    if (device.isErr())
        return Result<VernonRhiImageView, RhiError>{err(std::move(device).error())};
    if (!descriptor || descriptor->struct_size < sizeof(*descriptor) || !descriptor->mip_level_count ||
        !descriptor->array_layer_count || !descriptor->aspects)
        return Result<VernonRhiImageView, RhiError>{
            err(RhiError{RhiErrorCode::InvalidArgument, {"create_opengl_image_view", 0, 0}})};
    OpenGLDevice &state = device.value().device();
    ImageSlot *parent = indexedSlot(state.images, descriptor->image.index);
    auto parentPin = pinSlot(parent, descriptor->image);
    if (parentPin.isErr())
        return Result<VernonRhiImageView, RhiError>{err(std::move(parentPin).error())};
    FormatInfo format{};
    if (!parent || !formatInfo(descriptor->format, format))
        return Result<VernonRhiImageView, RhiError>{
            err(RhiError{RhiErrorCode::InvalidArgument, {"create_opengl_image_view", 0, 0}})};
    if (!vernon::rhi::validImageViewDescriptor(parent->descriptor, *descriptor))
        return Result<VernonRhiImageView, RhiError>{
            err(RhiError{RhiErrorCode::InvalidArgument, {"create_opengl_image_view", 0, 0}})};
    auto parentLease = parent->lifecycle.retain({descriptor->image.index, descriptor->image.generation});
    if (parentLease.isErr())
        return Result<VernonRhiImageView, RhiError>{err(std::move(parentLease).error())};
    auto owner = device.value().retainOwner();
    if (owner.isErr())
        return Result<VernonRhiImageView, RhiError>{err(std::move(owner).error())};
    auto reservation = vernon::rhi::ResourceCreationReservation::create(owner.value());
    if (reservation.isErr())
        return Result<VernonRhiImageView, RhiError>{err(std::move(reservation).error())};
    std::lock_guard<std::mutex> allocationGuard(state.resourceAllocationMutex);
    auto available = availableSlot<ImageViewSlot, ImageViewLifecycle>(state.imageViews);
    if (available.isErr())
        return Result<VernonRhiImageView, RhiError>{err(std::move(available).error())};
    ImageViewSlot &slot = *available.value();
    std::unique_lock<std::mutex> guard(state.mutex);
    state.state.makeCurrent();
    const bool identityView =
        descriptor->format == parent->descriptor.format && descriptor->dimension == parent->descriptor.dimension &&
        descriptor->aspects == vernon::formats::aspects(parent->descriptor.format) && descriptor->base_mip_level == 0 &&
        descriptor->mip_level_count == parent->descriptor.mip_levels && descriptor->base_array_layer == 0 &&
        descriptor->array_layer_count == parent->descriptor.array_layers;
    slot.backing.emplace<IdentityImageViewBacking>(IdentityImageViewBacking{parent->image.name});
    if (!identityView) {
        if (state.state.embeddedProfile || !state.state.driver.textureView) {
            guard.unlock();
            return Result<VernonRhiImageView, RhiError>{
                err(RhiError{RhiErrorCode::Unsupported, {"create_opengl_image_view", 0, 0}})};
        }
        auto &owned = slot.backing.emplace<OwnedTextureViewBacking>();
        state.state.driver.genTextures(1, &owned.image.name);
        if (!owned.image.name) {
            slot.backing.emplace<IdentityImageViewBacking>();
            guard.unlock();
            return Result<VernonRhiImageView, RhiError>{
                err(RhiError{RhiErrorCode::BackendFailure, {"create_opengl_image_view", 0, 0}})};
        }
        owned.image.imported = false;
        slot.target = vernon::rhi::opengl::textureTarget(descriptor->dimension);
        state.state.driver.textureView(owned.image.name, slot.target, parent->image.name, format.internal,
                                       descriptor->base_mip_level, descriptor->mip_level_count,
                                       descriptor->base_array_layer, descriptor->array_layer_count);
    }
    slot.target = vernon::rhi::opengl::textureTarget(descriptor->dimension);
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
        return Result<VernonRhiImageView, RhiError>{err(std::move(published).error())};
    }
    return Result<VernonRhiImageView, RhiError>{
        ok(VernonRhiImageView{published.value().index, published.value().generation})};
}

Result<void, RhiError> destroyImageView(VernonRhiDevice handle, VernonRhiImageView view) {
    auto device = lookupDevice(handle);
    if (device.isErr())
        return Result<void, RhiError>{err(std::move(device).error())};
    OpenGLDevice &state = device.value().device();
    ImageViewSlot *slot = indexedSlot(state.imageViews, view.index);
    if (!slot || slot->surfaceOwned.load(std::memory_order_relaxed))
        return Result<void, RhiError>{
            err(RhiError{RhiErrorCode::InvalidArgument, {"destroy_opengl_image_view", view.generation, view.index}})};
    return slot->lifecycle.destroyPublic({view.index, view.generation});
}

Result<uint64_t, RhiError> getImageViewNativeHandle(VernonRhiDevice handle, VernonRhiImageView view) {
    auto device = lookupDevice(handle);
    if (device.isErr())
        return Result<uint64_t, RhiError>{err(std::move(device).error())};
    OpenGLDevice &state = device.value().device();
    ImageViewSlot *slot = indexedSlot(state.imageViews, view.index);
    auto pin = pinSlot(slot, view);
    if (pin.isErr())
        return Result<uint64_t, RhiError>{err(std::move(pin).error())};
    std::lock_guard<std::mutex> guard(state.mutex);
    state.state.makeCurrent();
    return Result<uint64_t, RhiError>{ok(uint64_t(imageViewName(*slot)))};
}

Result<void, RhiError> destroyBuffer(VernonRhiDevice handle, VernonRhiBuffer buffer) {

    auto device = lookupDevice(handle);
    if (device.isErr())
        return Result<void, RhiError>{err(std::move(device).error())};
    OpenGLDevice &state = device.value().device();
    BufferSlot *slot = indexedSlot(state.buffers, buffer.index);
    if (!slot)
        return Result<void, RhiError>{
            err(RhiError{RhiErrorCode::InvalidArgument, {"destroy_opengl_buffer", buffer.generation, buffer.index}})};
    return slot->lifecycle.destroyPublic({buffer.index, buffer.generation});
}

Result<void *, RhiError> getBufferNativeHandle(VernonRhiDevice handle, VernonRhiBuffer buffer) {

    auto device = lookupDevice(handle);
    if (device.isErr())
        return Result<void *, RhiError>{err(std::move(device).error())};
    OpenGLDevice &state = device.value().device();
    BufferSlot *slot = indexedSlot(state.buffers, buffer.index);
    auto pin = pinSlot(slot, buffer);
    if (pin.isErr())
        return Result<void *, RhiError>{err(std::move(pin).error())};
    std::lock_guard<std::mutex> guard(state.mutex);
    state.state.makeCurrent();
    return Result<void *, RhiError>{ok(reinterpret_cast<void *>(static_cast<uintptr_t>(slot->buffer.name)))};
}

bool ownsDevice(VernonRhiDevice handle) { return lookupDevice(handle).isOk(); }

Result<void *, RhiError> deviceStateForBackend(VernonRhiDevice handle) {
    auto device = lookupDevice(handle);
    if (device.isErr())
        return Result<void *, RhiError>{err(std::move(device).error())};
    return Result<void *, RhiError>{ok(static_cast<void *>(&device.value().device().state))};
}

Result<CommandRecording, RhiError> beginCommands(VernonRhiDevice handle) noexcept {
    auto device = lookupDevice(handle);
    if (device.isErr())
        return Result<CommandRecording, RhiError>{err(std::move(device).error())};
    OpenGLDevice &state = device.value().device();
    std::lock_guard<std::mutex> guard(state.mutex);
    state.state.makeCurrent();
    const VernonRhiBackend backend =
        state.state.embeddedProfile ? VERNON_RHI_BACKEND_OPENGL_ES : VERNON_RHI_BACKEND_OPENGL;
    return Result<CommandRecording, RhiError>{ok(CommandRecording{reinterpret_cast<uintptr_t>(&state.state), backend})};
}

Result<CommandSubmission, RhiError> submitCommands(VernonRhiDevice handle, uint64_t native,
                                                   bool computeWrites) noexcept {
    auto device = lookupDevice(handle);
    if (device.isErr())
        return Result<CommandSubmission, RhiError>{err(std::move(device).error())};
    OpenGLDevice &state = device.value().device();
    if (native != reinterpret_cast<uintptr_t>(&state.state))
        return Result<CommandSubmission, RhiError>{
            err(RhiError{RhiErrorCode::InvalidArgument, {"submit_opengl_commands", native, 0}})};
    std::lock_guard<std::mutex> guard(state.mutex);
    state.state.makeCurrent();
    if (computeWrites) {
        if (state.state.driver.memoryBarrier)
            state.state.driver.memoryBarrier(
                vernon::rhi::opengl::kShaderStorageBarrierBit | vernon::rhi::opengl::kVertexAttribArrayBarrierBit |
                vernon::rhi::opengl::kShaderImageAccessBarrierBit | vernon::rhi::opengl::kTextureFetchBarrierBit |
                vernon::rhi::opengl::kTextureUpdateBarrierBit | vernon::rhi::opengl::kFramebufferBarrierBit |
                vernon::rhi::opengl::kBufferUpdateBarrierBit);
        else
            state.state.driver.finish();
    }
    const vernon::rhi::opengl::Sync fence =
        state.state.driver.fenceSync(vernon::rhi::opengl::kSyncGpuCommandsComplete, 0);
    if (!fence)
        return Result<CommandSubmission, RhiError>{
            err(RhiError{RhiErrorCode::BackendFailure, {"submit_opengl_commands", native, 0}})};
    return Result<CommandSubmission, RhiError>{ok(CommandSubmission{false, false, reinterpret_cast<uintptr_t>(fence)})};
}

// The flush bit makes a zero-timeout poll progress: an unflushed fence never signals.
Result<bool, RhiError> waitFence(VernonRhiDevice handle, uint64_t retirement, vernon::rhi::opengl::Uint64 timeout,
                                 bool release) noexcept {
    auto device = lookupDevice(handle);
    if (device.isErr())
        return Result<bool, RhiError>{err(std::move(device).error())};
    OpenGLDevice &state = device.value().device();
    std::lock_guard<std::mutex> guard(state.mutex);
    state.state.makeCurrent();
    const auto fence = reinterpret_cast<vernon::rhi::opengl::Sync>(static_cast<uintptr_t>(retirement));
    const vernon::rhi::opengl::Enum status =
        state.state.driver.clientWaitSync(fence, vernon::rhi::opengl::kSyncFlushCommandsBit, timeout);
    if (release)
        state.state.driver.deleteSync(fence);
    if (status == vernon::rhi::opengl::kAlreadySignaled || status == vernon::rhi::opengl::kConditionSatisfied)
        return Result<bool, RhiError>{ok(true)};
    if (status == vernon::rhi::opengl::kTimeoutExpired)
        return Result<bool, RhiError>{ok(false)};
    return Result<bool, RhiError>{err(RhiError{RhiErrorCode::BackendFailure, {"wait_opengl_fence", retirement, 0}})};
}

Result<CommandPoll, RhiError> pollCommands(VernonRhiDevice handle, uint64_t retirement) noexcept {
    auto signaled = waitFence(handle, retirement, 0, false);
    if (signaled.isErr())
        return Result<CommandPoll, RhiError>{ok(CommandPoll{true, false})};
    return Result<CommandPoll, RhiError>{ok(CommandPoll{signaled.value(), signaled.value()})};
}

Result<void, RhiError> completeCommands(VernonRhiDevice handle, uint64_t retirement) noexcept {
    auto signaled = waitFence(handle, retirement, vernon::rhi::opengl::kTimeoutIgnored, true);
    if (signaled.isErr())
        return Result<void, RhiError>{err(std::move(signaled).error())};
    return Result<void, RhiError>{ok()};
}

void abandonCommands(VernonRhiDevice handle, uint64_t native) noexcept {
    auto device = lookupDevice(handle);
    if (device.isErr() || native != reinterpret_cast<uintptr_t>(&device.value().device().state))
        return;
    OpenGLDevice &state = device.value().device();
    std::lock_guard<std::mutex> guard(state.mutex);
    state.state.makeCurrent();
    // Debug groups live on the context, not in the abandoned recording, so they would outlive it.
    state.state.popDebugGroups();
}

Result<void, RhiError> recordBarriers(VernonRhiDevice handle, uint64_t encoderKey, uint64_t native,
                                      const VernonRhiBarrier *barriers, size_t barrierCount) noexcept {
    auto device = lookupDevice(handle);
    if (device.isOk()) {
        OpenGLDevice &deviceState = device.value().device();
        std::lock_guard<std::mutex> guard(deviceState.mutex);
        if (native != reinterpret_cast<uintptr_t>(&deviceState.state))
            return Result<void, RhiError>{
                err(RhiError{RhiErrorCode::InvalidArgument, {"recordBarriers_opengl", native, 0}})};
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
                if (access & VERNON_RHI_ACCESS_TRANSFER_READ)
                    bits |= barriers[index].is_image ? vernon::rhi::opengl::kTextureUpdateBarrierBit |
                                                           vernon::rhi::opengl::kFramebufferBarrierBit
                                                     : vernon::rhi::opengl::kBufferUpdateBarrierBit;
                if (access & VERNON_RHI_ACCESS_TRANSFER_WRITE)
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
                else if (state == VERNON_RHI_STATE_TRANSFER_SOURCE)
                    bits |= vernon::rhi::opengl::kTextureUpdateBarrierBit | vernon::rhi::opengl::kFramebufferBarrierBit;
                else if (state == VERNON_RHI_STATE_TRANSFER_DESTINATION)
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
        return Result<void, RhiError>{ok()};
    }
    return Result<void, RhiError>{err(std::move(device).error())};
}

Result<void, RhiError> recordBufferCopy(VernonRhiDevice handle, uint64_t native, VernonRhiBuffer source,
                                        uint64_t sourceOffset, VernonRhiBuffer destination, uint64_t destinationOffset,
                                        uint64_t size) noexcept {
    auto device = lookupDevice(handle);
    if (device.isErr() || !size)
        return device.isErr() ? Result<void, RhiError>{err(std::move(device).error())}
                              : Result<void, RhiError>{err(
                                    RhiError{RhiErrorCode::InvalidArgument, {"recordBufferCopy_opengl", native, 0}})};
    OpenGLDevice &state = device.value().device();
    if (native != reinterpret_cast<uintptr_t>(&state.state))
        return Result<void, RhiError>{
            err(RhiError{RhiErrorCode::InvalidArgument, {"recordBufferCopy_opengl", native, 0}})};
    BufferSlot *sourceSlot = indexedSlot(state.buffers, source.index);
    BufferSlot *destinationSlot = indexedSlot(state.buffers, destination.index);
    auto sourcePin = pinSlot(sourceSlot, source);
    auto destinationPin = pinSlot(destinationSlot, destination);
    if (sourcePin.isErr() || destinationPin.isErr())
        return sourcePin.isErr() ? Result<void, RhiError>{err(std::move(sourcePin).error())}
                                 : Result<void, RhiError>{err(std::move(destinationPin).error())};
    std::lock_guard<std::mutex> guard(state.mutex);
    state.state.makeCurrent();
    if (sourceOffset > sourceSlot->descriptor.size || size > sourceSlot->descriptor.size - sourceOffset ||
        destinationOffset > destinationSlot->descriptor.size ||
        size > destinationSlot->descriptor.size - destinationOffset)
        return Result<void, RhiError>{
            err(RhiError{RhiErrorCode::InvalidArgument, {"recordBufferCopy_opengl", native, 0}})};
    if (!state.state.copyBuffer(sourceSlot->buffer, static_cast<size_t>(sourceOffset), destinationSlot->buffer,
                                static_cast<size_t>(destinationOffset), static_cast<size_t>(size), state.error))
        return Result<void, RhiError>{
            err(RhiError{RhiErrorCode::BackendFailure, {"recordBufferCopy_opengl", native, 0}})};
    return Result<void, RhiError>{ok()};
}

Result<void, RhiError> recordImageCopy(VernonRhiDevice handle, uint64_t, uint64_t native, VernonRhiImage source,
                                       VernonRhiImage destination, const VernonRhiImageCopyRegion *regions,
                                       size_t regionCount) noexcept {
    auto device = lookupDevice(handle);
    if (device.isErr() || !regions || !regionCount)
        return device.isErr() ? Result<void, RhiError>{err(std::move(device).error())}
                              : Result<void, RhiError>{err(
                                    RhiError{RhiErrorCode::InvalidArgument, {"recordImageCopy_opengl", native, 0}})};
    OpenGLDevice &state = device.value().device();
    if (native != reinterpret_cast<uintptr_t>(&state.state))
        return Result<void, RhiError>{
            err(RhiError{RhiErrorCode::InvalidArgument, {"recordImageCopy_opengl", native, 0}})};
    ImageSlot *sourceSlot = indexedSlot(state.images, source.index);
    ImageSlot *destinationSlot = indexedSlot(state.images, destination.index);
    auto sourcePin = pinSlot(sourceSlot, source);
    auto destinationPin = pinSlot(destinationSlot, destination);
    if (sourcePin.isErr() || destinationPin.isErr())
        return sourcePin.isErr() ? Result<void, RhiError>{err(std::move(sourcePin).error())}
                                 : Result<void, RhiError>{err(std::move(destinationPin).error())};
    std::lock_guard<std::mutex> guard(state.mutex);
    state.state.makeCurrent();
    const bool framebufferCopy = sourceSlot->descriptor.dimension == VERNON_RHI_IMAGE_2D &&
                                 destinationSlot->descriptor.dimension == VERNON_RHI_IMAGE_2D;
    if (!framebufferCopy && !state.state.driver.copyImageSubData)
        return Result<void, RhiError>{err(RhiError{RhiErrorCode::Unsupported, {"recordImageCopy_opengl", native, 0}})};
    if (auto scopeError = beginErrorScope(state, "copy image"))
        return Result<void, RhiError>{err(std::move(*scopeError))};
    Uint copyFramebuffers[2]{};
    Int previousReadFramebuffer = 0;
    Int previousDrawFramebuffer = 0;
    if (framebufferCopy) {
        if (!state.state.driver.blitFramebuffer)
            return Result<void, RhiError>{
                err(RhiError{RhiErrorCode::Unsupported, {"recordImageCopy_opengl", native, 0}})};
        state.state.driver.getIntegerv(gl::kReadFramebufferBinding, &previousReadFramebuffer);
        state.state.driver.getIntegerv(gl::kDrawFramebufferBinding, &previousDrawFramebuffer);
        state.state.driver.genFramebuffers(2, copyFramebuffers);
        if (!copyFramebuffers[0] || !copyFramebuffers[1]) {
            if (copyFramebuffers[0] || copyFramebuffers[1])
                state.state.driver.deleteFramebuffers(2, copyFramebuffers);
            return Result<void, RhiError>{
                err(RhiError{RhiErrorCode::ResourceExhausted, {"recordImageCopy_opengl", native, 0}})};
        }
        prepareOpenGLBlits(state.state);
    }
    const auto releaseCopyFramebuffers = [&] {
        if (!framebufferCopy)
            return;
        state.state.driver.bindFramebuffer(gl::kReadFramebuffer, static_cast<Uint>(previousReadFramebuffer));
        state.state.driver.bindFramebuffer(gl::kDrawFramebuffer, static_cast<Uint>(previousDrawFramebuffer));
        state.state.driver.deleteFramebuffers(2, copyFramebuffers);
    };
    for (size_t index = 0; index < regionCount; ++index) {
        const VernonRhiImageCopyRegion &region = regions[index];
        if (framebufferCopy) {
            const bool color = (region.aspects & VERNON_RHI_IMAGE_ASPECT_COLOR) != 0;
            const bool depth = (region.aspects & VERNON_RHI_IMAGE_ASPECT_DEPTH) != 0;
            const bool stencil = (region.aspects & VERNON_RHI_IMAGE_ASPECT_STENCIL) != 0;
            const Enum attachment = color              ? gl::kColorAttachment0
                                    : depth && stencil ? gl::kDepthStencilAttachment
                                    : depth            ? gl::kDepthAttachment
                                                       : gl::kStencilAttachment;
            const gl::Bitfield mask = (color ? gl::kColorBufferBit : 0u) | (depth ? gl::kDepthBufferBit : 0u) |
                                      (stencil ? gl::kStencilBufferBit : 0u);
            if (!mask) {
                releaseCopyFramebuffers();
                return Result<void, RhiError>{err(RhiError{
                    RhiErrorCode::InvalidArgument, {"recordImageCopy_opengl", native, static_cast<uint32_t>(index)}})};
            }
            state.state.driver.bindFramebuffer(gl::kReadFramebuffer, copyFramebuffers[0]);
            state.state.driver.framebufferTexture2D(gl::kReadFramebuffer, attachment, sourceSlot->target,
                                                    sourceSlot->image.name, static_cast<Int>(region.source_mip_level));
            state.state.driver.readBuffer(color ? gl::kColorAttachment0 : gl::kNone);
            state.state.driver.bindFramebuffer(gl::kDrawFramebuffer, copyFramebuffers[1]);
            state.state.driver.framebufferTexture2D(gl::kDrawFramebuffer, attachment, destinationSlot->target,
                                                    destinationSlot->image.name,
                                                    static_cast<Int>(region.destination_mip_level));
            const Enum drawBuffer = color ? gl::kColorAttachment0 : gl::kNone;
            state.state.driver.drawBuffers(1, &drawBuffer);
            const Enum readStatus = state.state.driver.checkFramebufferStatus(gl::kReadFramebuffer);
            const Enum drawStatus = state.state.driver.checkFramebufferStatus(gl::kDrawFramebuffer);
            if (readStatus != gl::kFramebufferComplete || drawStatus != gl::kFramebufferComplete) {
                state.error = "OpenGL image copy framebuffer is incomplete (read=" + std::to_string(readStatus) +
                              ", draw=" + std::to_string(drawStatus) + ")";
                releaseCopyFramebuffers();
                return Result<void, RhiError>{err(RhiError{
                    RhiErrorCode::BackendFailure, {"recordImageCopy_opengl", native, static_cast<uint32_t>(index)}})};
            }
            state.state.driver.blitFramebuffer(
                static_cast<Int>(region.source_x), static_cast<Int>(region.source_y),
                static_cast<Int>(region.source_x + region.width), static_cast<Int>(region.source_y + region.height),
                static_cast<Int>(region.destination_x), static_cast<Int>(region.destination_y),
                static_cast<Int>(region.destination_x + region.width),
                static_cast<Int>(region.destination_y + region.height), mask, gl::kNearest);
            if (auto scopeError = endErrorScope(state, "copy image with framebuffer")) {
                releaseCopyFramebuffers();
                return Result<void, RhiError>{err(std::move(*scopeError))};
            }
            continue;
        }
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
        if (auto scopeError = endErrorScope(state, "copy image"))
            return Result<void, RhiError>{err(std::move(*scopeError))};
    }
    releaseCopyFramebuffers();
    if (auto scopeError = endErrorScope(state, "finish image copy"))
        return Result<void, RhiError>{err(std::move(*scopeError))};
    return Result<void, RhiError>{ok()};
}

Result<OpenedRendering, RhiError> beginRendering(VernonRhiDevice handle, uint64_t encoderKey, uint64_t native,
                                                 const RenderingScope &scope, const RenderingRead *, size_t) noexcept {
    auto device = lookupDevice(handle);
    if (device.isErr())
        return Result<OpenedRendering, RhiError>{err(std::move(device).error())};
    OpenGLDevice &state = device.value().device();
    if (native != reinterpret_cast<uintptr_t>(&state.state))
        return Result<OpenedRendering, RhiError>{
            err(RhiError{RhiErrorCode::InvalidArgument, {"beginRendering_opengl", native, 0}})};
    if (scope.layers != 1)
        return Result<OpenedRendering, RhiError>{
            err(RhiError{RhiErrorCode::Unsupported, {"beginRendering_opengl_layered", scope.layers, 0}})};
    for (size_t index = 0; index < scope.attachmentCount(); ++index) {
        auto retained = retainCommandResource(handle, encoderKey, ResourceKind::ImageView, scope.attachmentView(index));
        if (retained.isErr())
            return Result<OpenedRendering, RhiError>{err(std::move(retained).error())};
    }
    std::lock_guard<std::mutex> guard(state.mutex);
    if (!state.state.makeCurrent())
        return Result<OpenedRendering, RhiError>{
            err(RhiError{RhiErrorCode::BackendFailure, {"beginRendering_opengl_make_current", 0, 0}})};
    if (auto scopeError = beginErrorScope(state, "begin rendering"))
        return Result<OpenedRendering, RhiError>{err(std::move(*scopeError))};
    std::array<const ImageViewSlot *, VERNON_RHI_MAX_COLOR_ATTACHMENTS> colorViews{};
    OpenGLScopeAttachments attachments{};
    OpenedRendering opened{CommandRenderingStateless, state.scopeFramebuffer};
    uint32_t colorSlots = 0;
    for (size_t index = 0; index < scope.colorCount; ++index) {
        const uint32_t location = scope.colors[index].location;
        colorViews[location] = imageViewSlot(state, scope.colors[index].view);
        attachments.colors[location] = imageViewName(*colorViews[location]);
        opened.colorTargets[location] = attachments.colors[location];
        colorSlots = std::max(colorSlots, location + 1);
    }
    const ImageViewSlot *depthView = scope.hasDepth ? imageViewSlot(state, scope.depth.view) : nullptr;
    if (depthView) {
        attachments.depth = imageViewName(*depthView);
        attachments.stencil = scope.depth.stencil;
        opened.depthTarget = attachments.depth;
    }
    auto &driver = state.state.driver;
    driver.bindFramebuffer(gl::kFramebuffer, state.scopeFramebuffer);
    ++state.state.drawStateGeneration;
    if (!state.scopeAttachmentsValid || state.scopeGeneration != state.state.framebufferGeneration ||
        !(state.scopeAttachments == attachments)) {
        state.scopeAttachmentsValid = false;
        std::array<Enum, VERNON_RHI_MAX_COLOR_ATTACHMENTS> drawBuffers{};
        for (uint32_t location = 0; location < VERNON_RHI_MAX_COLOR_ATTACHMENTS; ++location) {
            attachOpenGLView(driver, gl::kColorAttachment0 + location, colorViews[location]);
            drawBuffers[location] = colorViews[location] ? gl::kColorAttachment0 + location : gl::kNone;
        }
        attachOpenGLView(driver, gl::kDepthStencilAttachment, nullptr);
        attachOpenGLView(driver, gl::kDepthAttachment, nullptr);
        if (depthView)
            attachOpenGLView(driver, attachments.stencil ? gl::kDepthStencilAttachment : gl::kDepthAttachment,
                             depthView);
        driver.drawBuffers(static_cast<Size>(std::max(1u, colorSlots)), drawBuffers.data());
        driver.readBuffer(colorSlots ? drawBuffers[0] : gl::kNone);
        if (driver.checkFramebufferStatus(gl::kFramebuffer) != gl::kFramebufferComplete)
            return Result<OpenedRendering, RhiError>{
                err(fail(state, "OpenGL rendering framebuffer is incomplete", RhiErrorCode::BackendFailure))};
        state.scopeAttachments = attachments;
        state.scopeGeneration = state.state.framebufferGeneration;
        state.scopeAttachmentsValid = true;
    }
    // Desktop GL encodes sRGB attachment writes and clears only while GL_FRAMEBUFFER_SRGB is enabled (OpenGL ES always
    // encodes). Image copies blit through framebuffers and must stay raw, so the state is scoped to rendering and
    // every blit disables it again.
    if (!state.state.embeddedProfile)
        driver.enable(gl::kFramebufferSrgb);
    const bool depthClear = scope.hasDepth && scope.depth.depthLoad == VERNON_RHI_LOAD_CLEAR;
    const bool stencilClear = scope.hasDepth && scope.depth.stencil && scope.depth.stencilLoad == VERNON_RHI_LOAD_CLEAR;
    bool colorClear = false;
    for (size_t index = 0; index < scope.colorCount; ++index)
        colorClear = colorClear || scope.colors[index].load == VERNON_RHI_LOAD_CLEAR;
    if (colorClear || depthClear || stencilClear) {
        prepareOpenGLClears(state.state, scope);
        for (size_t index = 0; index < scope.colorCount; ++index)
            if (scope.colors[index].load == VERNON_RHI_LOAD_CLEAR)
                clearOpenGLColor(driver, scope.colors[index].location, scope.colors[index].format,
                                 scope.colors[index].clear);
        clearOpenGLDepthStencil(driver, depthClear, stencilClear, scope.depth.clearDepth, scope.depth.clearStencil);
    }
    if (auto scopeError = endErrorScope(state, "begin rendering"))
        return Result<OpenedRendering, RhiError>{err(std::move(*scopeError))};
    return Result<OpenedRendering, RhiError>{ok(opened)};
}

Result<void, RhiError> endRendering(VernonRhiDevice handle, uint64_t native, const RenderingScope &scope,
                                    const OpenedRendering &opened) noexcept {
    auto device = lookupDevice(handle);
    if (device.isErr())
        return Result<void, RhiError>{err(std::move(device).error())};
    OpenGLDevice &state = device.value().device();
    if (native != reinterpret_cast<uintptr_t>(&state.state) || opened.kind != CommandRenderingStateless)
        return Result<void, RhiError>{
            err(RhiError{RhiErrorCode::InvalidArgument, {"endRendering_opengl", native, opened.kind}})};
    std::lock_guard<std::mutex> guard(state.mutex);
    state.state.makeCurrent();
    if (!state.state.embeddedProfile)
        state.state.driver.disable(gl::kFramebufferSrgb);
    if (!state.state.driver.invalidateFramebuffer)
        return Result<void, RhiError>{ok()};
    std::array<Enum, VERNON_RHI_MAX_COLOR_ATTACHMENTS + 2> discarded{};
    size_t discardedCount = 0;
    for (size_t index = 0; index < scope.colorCount; ++index)
        if (scope.colors[index].store == VERNON_RHI_STORE_DISCARD &&
            renderAreaCoversOpenGLView(state, scope.colors[index].view, scope))
            discarded[discardedCount++] = gl::kColorAttachment0 + scope.colors[index].location;
    if (scope.hasDepth && renderAreaCoversOpenGLView(state, scope.depth.view, scope)) {
        if (scope.depth.depthStore == VERNON_RHI_STORE_DISCARD)
            discarded[discardedCount++] = gl::kDepthAttachment;
        if (scope.depth.stencil && scope.depth.stencilStore == VERNON_RHI_STORE_DISCARD)
            discarded[discardedCount++] = gl::kStencilAttachment;
    }
    if (!discardedCount)
        return Result<void, RhiError>{ok()};
    if (auto scopeError = beginErrorScope(state, "end rendering"))
        return Result<void, RhiError>{err(std::move(*scopeError))};
    state.state.driver.bindFramebuffer(gl::kFramebuffer, state.scopeFramebuffer);
    state.state.driver.invalidateFramebuffer(gl::kFramebuffer, static_cast<Size>(discardedCount), discarded.data());
    if (auto scopeError = endErrorScope(state, "end rendering"))
        return Result<void, RhiError>{err(std::move(*scopeError))};
    return Result<void, RhiError>{ok()};
}

Result<void, RhiError> recordDebugCommand(VernonRhiDevice handle, uint64_t native, const OpenedRendering *,
                                          DebugCommand command, std::string_view label) noexcept {
    auto device = lookupDevice(handle);
    if (device.isErr())
        return Result<void, RhiError>{err(std::move(device).error())};
    OpenGLDevice &state = device.value().device();
    if (native != reinterpret_cast<uintptr_t>(&state.state))
        return Result<void, RhiError>{
            err(RhiError{RhiErrorCode::InvalidArgument, {"record_debug_command_opengl", native, 0}})};
    std::lock_guard<std::mutex> guard(state.mutex);
    state.state.makeCurrent();
    try {
        switch (command) {
        case DebugCommand::PushGroup:
            state.state.pushDebugGroup(label);
            break;
        case DebugCommand::PopGroup:
            state.state.popDebugGroup();
            break;
        case DebugCommand::InsertMarker:
            state.state.insertDebugMarker(label);
            break;
        }
    } catch (const std::bad_alloc &) {
        return Result<void, RhiError>{
            err(RhiError{RhiErrorCode::ResourceExhausted, {"record_debug_command_opengl", native, 0}})};
    }
    return Result<void, RhiError>{ok()};
}

uint32_t commandCapabilitiesForDevice(VernonRhiDevice handle) {
    auto device = lookupDevice(handle);
    return device.isOk() && device.value().device().state.driver.queryCounter ? BackendCommandTimers : 0;
}

// Timestamp query names, the begin of timer `t` at `2 * t` and its end at `2 * t + 1`.
struct OpenGLTimers {
    std::vector<Uint> queries;
};

Result<void, RhiError> recordTimer(VernonRhiDevice handle, uint64_t native, uint64_t &storage, uint32_t timer,
                                   TimerEdge edge) noexcept {
    auto device = lookupDevice(handle);
    if (device.isErr())
        return Result<void, RhiError>{err(std::move(device).error())};
    OpenGLDevice &state = device.value().device();
    if (native != reinterpret_cast<uintptr_t>(&state.state))
        return Result<void, RhiError>{err(RhiError{RhiErrorCode::InvalidArgument, {"record_opengl_timer", native, 0}})};
    std::lock_guard<std::mutex> guard(state.mutex);
    state.state.makeCurrent();
    try {
        if (!storage) {
            storage = reinterpret_cast<uintptr_t>(new OpenGLTimers{});
            // Reading the disjoint flag clears it, so that only disjoint events during these timers invalidate them.
            if (state.state.embeddedProfile) {
                Int disjoint = 0;
                state.state.driver.getIntegerv(gl::kGpuDisjoint, &disjoint);
            }
        }
        auto &timers = *reinterpret_cast<OpenGLTimers *>(static_cast<uintptr_t>(storage));
        const size_t query = size_t{timer} * 2 + (edge == TimerEdge::End ? 1 : 0);
        if (timers.queries.size() <= query)
            timers.queries.resize(size_t{timer} * 2 + 2);
        state.state.driver.genQueries(1, &timers.queries[query]);
        state.state.driver.queryCounter(timers.queries[query], gl::kTimestamp);
    } catch (const std::bad_alloc &) {
        return Result<void, RhiError>{
            err(RhiError{RhiErrorCode::ResourceExhausted, {"record_opengl_timer", native, timer}})};
    }
    return Result<void, RhiError>{ok()};
}

Result<void, RhiError> readTimers(VernonRhiDevice handle, uint64_t storage, uint32_t count,
                                  VernonRhiTimerSpan *spans) noexcept {
    auto device = lookupDevice(handle);
    if (device.isErr())
        return Result<void, RhiError>{err(std::move(device).error())};
    OpenGLDevice &state = device.value().device();
    const auto &timers = *reinterpret_cast<const OpenGLTimers *>(static_cast<uintptr_t>(storage));
    if (timers.queries.size() != size_t{count} * 2)
        return Result<void, RhiError>{err(RhiError{RhiErrorCode::InvalidArgument, {"read_opengl_timers", count, 0}})};
    std::lock_guard<std::mutex> guard(state.mutex);
    state.state.makeCurrent();
    for (uint32_t timer = 0; timer < count; ++timer) {
        state.state.driver.getQueryObjectui64v(timers.queries[size_t{timer} * 2], gl::kQueryResult,
                                               &spans[timer].begin_nanoseconds);
        state.state.driver.getQueryObjectui64v(timers.queries[size_t{timer} * 2 + 1], gl::kQueryResult,
                                               &spans[timer].end_nanoseconds);
    }
    Int disjoint = 0;
    if (state.state.embeddedProfile)
        state.state.driver.getIntegerv(gl::kGpuDisjoint, &disjoint);
    if (disjoint)
        return Result<void, RhiError>{err(RhiError{RhiErrorCode::Unsupported, {"read_opengl_timers_disjoint", 0, 0}})};
    return Result<void, RhiError>{ok()};
}

void releaseTimers(VernonRhiDevice handle, uint64_t storage) noexcept {
    std::unique_ptr<OpenGLTimers> timers(reinterpret_cast<OpenGLTimers *>(static_cast<uintptr_t>(storage)));
    auto device = lookupDevice(handle);
    if (device.isErr())
        return;
    OpenGLDevice &state = device.value().device();
    std::lock_guard<std::mutex> guard(state.mutex);
    state.state.makeCurrent();
    state.state.driver.deleteQueries(static_cast<Size>(timers->queries.size()), timers->queries.data());
}

Result<void, RhiError> clearColor(VernonRhiDevice handle, uint64_t native, const RenderingScope &scope,
                                  const OpenedRendering &opened, uint32_t location,
                                  const VernonClearColor &color) noexcept {
    auto device = lookupDevice(handle);
    if (device.isErr())
        return Result<void, RhiError>{err(std::move(device).error())};
    OpenGLDevice &state = device.value().device();
    const RenderingColorAttachment *attachment = scope.colorAt(location);
    if (native != reinterpret_cast<uintptr_t>(&state.state) || opened.kind != CommandRenderingStateless ||
        location >= VERNON_RHI_MAX_COLOR_ATTACHMENTS || !opened.colorTargets[location] || !attachment)
        return Result<void, RhiError>{
            err(RhiError{RhiErrorCode::InvalidArgument, {"clearColor_opengl", native, location}})};
    std::lock_guard<std::mutex> guard(state.mutex);
    state.state.makeCurrent();
    if (auto scopeError = beginErrorScope(state, "clear color"))
        return Result<void, RhiError>{err(std::move(*scopeError))};
    state.state.driver.bindFramebuffer(gl::kFramebuffer, state.scopeFramebuffer);
    prepareOpenGLClears(state.state, scope);
    clearOpenGLColor(state.state.driver, location, attachment->format, color);
    if (auto scopeError = endErrorScope(state, "clear color"))
        return Result<void, RhiError>{err(std::move(*scopeError))};
    return Result<void, RhiError>{ok()};
}

Result<void, RhiError> clearDepthStencil(VernonRhiDevice handle, uint64_t native, const RenderingScope &scope,
                                         const OpenedRendering &opened, float depth, uint32_t stencil,
                                         uint32_t aspects) noexcept {
    auto device = lookupDevice(handle);
    if (device.isErr())
        return Result<void, RhiError>{err(std::move(device).error())};
    OpenGLDevice &state = device.value().device();
    if (native != reinterpret_cast<uintptr_t>(&state.state) || opened.kind != CommandRenderingStateless ||
        !opened.depthTarget)
        return Result<void, RhiError>{
            err(RhiError{RhiErrorCode::InvalidArgument, {"clearDepthStencil_opengl", native, aspects}})};
    std::lock_guard<std::mutex> guard(state.mutex);
    state.state.makeCurrent();
    if (auto scopeError = beginErrorScope(state, "clear depth stencil"))
        return Result<void, RhiError>{err(std::move(*scopeError))};
    state.state.driver.bindFramebuffer(gl::kFramebuffer, state.scopeFramebuffer);
    prepareOpenGLClears(state.state, scope);
    clearOpenGLDepthStencil(state.state.driver, (aspects & VERNON_RHI_ATTACHMENT_DEPTH) != 0,
                            (aspects & VERNON_RHI_ATTACHMENT_STENCIL) != 0, depth, stencil);
    if (auto scopeError = endErrorScope(state, "clear depth stencil"))
        return Result<void, RhiError>{err(std::move(*scopeError))};
    return Result<void, RhiError>{ok()};
}

#define VERNON_OPENGL_RESOURCE_RESULT(name, Handle, slots)                                                             \
    Result<uint64_t, RhiError> name(VernonRhiDevice deviceHandle, Handle handle) noexcept {                            \
        auto device = lookupDevice(deviceHandle);                                                                      \
        if (device.isErr())                                                                                            \
            return Result<uint64_t, RhiError>{err(std::move(device).error())};                                         \
        auto pin = pinSlot(indexedSlot(device.value().device().slots, handle.index), handle);                          \
        if (pin.isErr())                                                                                               \
            return Result<uint64_t, RhiError>{err(std::move(pin).error())};                                            \
        return Result<uint64_t, RhiError>{ok(resourceKey(handle))};                                                    \
    }

VERNON_OPENGL_RESOURCE_RESULT(bufferResource, VernonRhiBuffer, buffers)
VERNON_OPENGL_RESOURCE_RESULT(imageResource, VernonRhiImage, images)
VERNON_OPENGL_RESOURCE_RESULT(imageViewResource, VernonRhiImageView, imageViews)
VERNON_OPENGL_RESOURCE_RESULT(samplerResource, VernonRhiSampler, samplers)

#undef VERNON_OPENGL_RESOURCE_RESULT

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

vernon::Result<ImageResourceDescription, vernon::RhiError> describeImageResource(VernonRhiDevice handle,
                                                                                 uint64_t key) noexcept {
    auto device = lookupDevice(handle);
    ImageLifecycle::Handle resource{};
    if (device.isErr() || !decodeResourceKey(key, resource))
        return vernon::Result<ImageResourceDescription, vernon::RhiError>{
            vernon::err(vernon::RhiError{vernon::RhiErrorCode::InvalidArgument, {"describe_opengl_image", key, 0}})};
    OpenGLDevice &state = device.value().device();
    ImageSlot *slot = indexedSlot(state.images, resource.index);
    if (!slot)
        return vernon::Result<ImageResourceDescription, vernon::RhiError>{
            vernon::err(vernon::RhiError{vernon::RhiErrorCode::InvalidArgument, {"describe_opengl_image", key, 0}})};
    std::lock_guard<std::mutex> guard(state.mutex);
    state.state.makeCurrent();
    return vernon::Result<ImageResourceDescription, vernon::RhiError>{
        vernon::ok(ImageResourceDescription{slot->descriptor})};
}

vernon::Result<ImageViewResourceDescription, vernon::RhiError> describeImageViewResource(VernonRhiDevice handle,
                                                                                         uint64_t key) noexcept {
    auto device = lookupDevice(handle);
    ImageViewLifecycle::Handle resource{};
    if (device.isErr() || !decodeResourceKey(key, resource))
        return vernon::Result<ImageViewResourceDescription, vernon::RhiError>{vernon::err(
            vernon::RhiError{vernon::RhiErrorCode::InvalidArgument, {"describe_opengl_image_view", key, 0}})};
    OpenGLDevice &state = device.value().device();
    ImageViewSlot *slot = indexedSlot(state.imageViews, resource.index);
    if (!slot)
        return vernon::Result<ImageViewResourceDescription, vernon::RhiError>{vernon::err(
            vernon::RhiError{vernon::RhiErrorCode::InvalidArgument, {"describe_opengl_image_view", key, 0}})};
    std::lock_guard<std::mutex> guard(state.mutex);
    state.state.makeCurrent();
    return vernon::Result<ImageViewResourceDescription, vernon::RhiError>{
        vernon::ok(ImageViewResourceDescription{slot->descriptor, slot->imageDescriptor, slot->parentKey})};
}

Result<VernonRhiSurface, RhiError> createSurface(VernonRhiDevice handle, const VernonRhiSurfaceDescriptor &descriptor) {
    auto device = lookupDevice(handle);
    if (device.isErr())
        return Result<VernonRhiSurface, RhiError>{err(std::move(device).error())};
    const VernonOpenGLSurfaceCallbacks *callbacks = descriptor.native.opengl;
    if (!callbacks || callbacks->struct_size < sizeof(*callbacks) || !callbacks->make_current ||
        !callbacks->swap_interval || !callbacks->swap_buffers || !callbacks->drawable_size)
        return Result<VernonRhiSurface, RhiError>{err(invalidArgument("create_opengl_surface"))};
    OpenGLDevice &state = device.value().device();
    if (!state.state.driver.blitFramebuffer)
        return Result<VernonRhiSurface, RhiError>{err(unsupported("create_opengl_surface"))};
    auto owner = device.value().retainOwner();
    if (owner.isErr())
        return Result<VernonRhiSurface, RhiError>{err(std::move(owner).error())};
    auto reservation = vernon::rhi::ResourceCreationReservation::create(owner.value());
    if (reservation.isErr())
        return Result<VernonRhiSurface, RhiError>{err(std::move(reservation).error())};
    std::lock_guard<std::mutex> allocationGuard(state.resourceAllocationMutex);
    auto available = availableSlot<SurfaceSlot, SurfaceLifecycle>(state.surfaces);
    if (available.isErr())
        return Result<VernonRhiSurface, RhiError>{err(std::move(available).error())};
    SurfaceSlot &slot = *available.value();
    slot.callbacks = *callbacks;
    auto published = slot.lifecycle.publish(std::move(reservation).value(), &state, teardownSurface);
    if (published.isErr()) {
        slot.callbacks = {};
        return Result<VernonRhiSurface, RhiError>{err(std::move(published).error())};
    }
    return Result<VernonRhiSurface, RhiError>{
        ok(VernonRhiSurface{published.value().index, published.value().generation})};
}

Result<VernonRhiSurfaceCapabilities, RhiError> surfaceCapabilities(VernonRhiDevice handle, VernonRhiSurface surface) {
    auto device = lookupDevice(handle);
    if (device.isErr())
        return Result<VernonRhiSurfaceCapabilities, RhiError>{err(std::move(device).error())};
    SurfaceSlot *slot = indexedSlot(device.value().device().surfaces, surface.index);
    auto pin = pinSlot(slot, surface);
    if (pin.isErr())
        return Result<VernonRhiSurfaceCapabilities, RhiError>{err(std::move(pin).error())};
    VernonRhiSurfaceCapabilities capabilities{};
    capabilities.struct_size = sizeof(capabilities);
    capabilities.formats[0] = VERNON_RHI_FORMAT_RGBA8_UNORM;
    capabilities.format_count = 1;
    capabilities.present_modes =
        (1u << VERNON_RHI_PRESENT_FIFO) | (1u << VERNON_RHI_PRESENT_MAILBOX) | (1u << VERNON_RHI_PRESENT_IMMEDIATE);
    // Without sampled or storage usage, an acquired image is read only by a preserving rendering scope, a copy or a
    // download, which are the reads the command layer rejects while its contents are undefined.
    capabilities.usage = VERNON_RHI_IMAGE_COLOR_ATTACHMENT | VERNON_RHI_IMAGE_TRANSFER_SOURCE;
    capabilities.min_image_count = 1;
    slot->callbacks.drawable_size(slot->callbacks.user_data, &capabilities.current_width, &capabilities.current_height);
    return Result<VernonRhiSurfaceCapabilities, RhiError>{ok(capabilities)};
}

Result<SurfaceImage, RhiError> createSurfaceImage(VernonRhiDevice handle, OpenGLDevice &state,
                                                  const VernonRhiImageDescriptor &descriptor) {
    auto image = createImage(handle, &descriptor);
    if (image.isErr())
        return Result<SurfaceImage, RhiError>{err(std::move(image).error())};
    VernonRhiImageViewDescriptor viewDescriptor{};
    viewDescriptor.struct_size = sizeof(viewDescriptor);
    viewDescriptor.image = image.value();
    viewDescriptor.dimension = VERNON_RHI_IMAGE_2D;
    viewDescriptor.format = descriptor.format;
    viewDescriptor.mip_level_count = 1;
    viewDescriptor.array_layer_count = 1;
    viewDescriptor.aspects = VERNON_RHI_IMAGE_ASPECT_COLOR;
    auto view = createImageView(handle, &viewDescriptor);
    if (view.isErr()) {
        if (destroyImage(handle, image.value()).isErr())
            vernon::resultContractViolation();
        return Result<SurfaceImage, RhiError>{err(std::move(view).error())};
    }
    indexedSlot(state.images, image.value().index)->surfaceOwned.store(true, std::memory_order_relaxed);
    indexedSlot(state.imageViews, view.value().index)->surfaceOwned.store(true, std::memory_order_relaxed);
    return Result<SurfaceImage, RhiError>{ok(SurfaceImage{image.value(), view.value()})};
}

// The new images replace the old ones only once all of them exist, so a failed configuration keeps the previous one.
Result<void, RhiError> configureSurface(VernonRhiDevice handle, VernonRhiSurface surface,
                                        const VernonRhiSurfaceConfiguration &configuration) {
    auto device = lookupDevice(handle);
    if (device.isErr())
        return Result<void, RhiError>{err(std::move(device).error())};
    OpenGLDevice &state = device.value().device();
    SurfaceSlot *slot = indexedSlot(state.surfaces, surface.index);
    auto pin = pinSlot(slot, surface);
    if (pin.isErr())
        return Result<void, RhiError>{err(std::move(pin).error())};
    std::lock_guard<std::mutex> guard(slot->mutex);
    if (slot->acquired)
        return invalidResult("configure_opengl_surface", surface.generation, surface.index);
    std::vector<SurfaceImage> images;
    try {
        images.reserve(configuration.image_count);
    } catch (const std::bad_alloc &) {
        return Result<void, RhiError>{
            err(RhiError{RhiErrorCode::ResourceExhausted, {"configure_opengl_surface", configuration.image_count, 0}})};
    }
    VernonRhiImageDescriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.dimension = VERNON_RHI_IMAGE_2D;
    descriptor.format = configuration.format;
    descriptor.width = configuration.width;
    descriptor.height = configuration.height;
    descriptor.depth = 1;
    descriptor.mip_levels = 1;
    descriptor.array_layers = 1;
    descriptor.sample_count = 1;
    descriptor.usage = configuration.usage;
    for (uint32_t index = 0; index < configuration.image_count; ++index) {
        auto created = createSurfaceImage(handle, state, descriptor);
        if (created.isErr()) {
            if (destroySurfaceImages(state, images).isErr())
                vernon::resultContractViolation();
            return Result<void, RhiError>{err(std::move(created).error())};
        }
        images.push_back(created.value());
    }
    auto destroyed = destroySurfaceImages(state, slot->images);
    slot->images = std::move(images);
    slot->configuration = configuration;
    slot->next = 0;
    return destroyed;
}

Result<VernonRhiSurfaceImage, RhiError> acquireSurfaceImage(VernonRhiDevice handle, VernonRhiSurface surface) {
    auto device = lookupDevice(handle);
    if (device.isErr())
        return Result<VernonRhiSurfaceImage, RhiError>{err(std::move(device).error())};
    SurfaceSlot *slot = indexedSlot(device.value().device().surfaces, surface.index);
    auto pin = pinSlot(slot, surface);
    if (pin.isErr())
        return Result<VernonRhiSurfaceImage, RhiError>{err(std::move(pin).error())};
    std::lock_guard<std::mutex> guard(slot->mutex);
    if (slot->acquired || slot->images.empty())
        return Result<VernonRhiSurfaceImage, RhiError>{
            err(invalidArgument("acquire_opengl_surface_image", surface.generation, surface.index))};
    uint32_t width = 0;
    uint32_t height = 0;
    slot->callbacks.drawable_size(slot->callbacks.user_data, &width, &height);
    if (width != slot->configuration.width || height != slot->configuration.height)
        return Result<VernonRhiSurfaceImage, RhiError>{
            err(RhiError{RhiErrorCode::SurfaceOutOfDate, {"acquire_opengl_surface_image", width, height}})};
    slot->acquired = true;
    VernonRhiSurfaceImage acquired{};
    acquired.image = slot->images[slot->next].image;
    acquired.view = slot->images[slot->next].view;
    return Result<VernonRhiSurfaceImage, RhiError>{ok(acquired)};
}

// Runs with the window's surface current. Surface images hold the image's top row at row 0, and the window's
// framebuffer holds its bottom row there, so the blit reverses the rows.
Result<void, RhiError> blitToWindow(OpenGLDevice &state, const ImageSlot &image, uint32_t width, uint32_t height) {
    auto &driver = state.state.driver;
    if (auto scopeError = beginErrorScope(state, "present surface"))
        return Result<void, RhiError>{err(std::move(*scopeError))};
    Uint framebuffer{};
    driver.genFramebuffers(1, &framebuffer);
    if (!framebuffer)
        return Result<void, RhiError>{err(RhiError{RhiErrorCode::ResourceExhausted, {"present_opengl_surface", 0, 0}})};
    Int previousReadFramebuffer = 0;
    Int previousDrawFramebuffer = 0;
    driver.getIntegerv(gl::kReadFramebufferBinding, &previousReadFramebuffer);
    driver.getIntegerv(gl::kDrawFramebufferBinding, &previousDrawFramebuffer);
    prepareOpenGLBlits(state.state);
    driver.bindFramebuffer(gl::kReadFramebuffer, framebuffer);
    driver.framebufferTexture2D(gl::kReadFramebuffer, gl::kColorAttachment0, gl::kTexture2D, image.image.name, 0);
    driver.readBuffer(gl::kColorAttachment0);
    driver.bindFramebuffer(gl::kDrawFramebuffer, 0);
    const Int right = static_cast<Int>(width);
    const Int top = static_cast<Int>(height);
    driver.blitFramebuffer(0, 0, right, top, 0, top, right, 0, gl::kColorBufferBit, gl::kNearest);
    driver.bindFramebuffer(gl::kReadFramebuffer, static_cast<Uint>(previousReadFramebuffer));
    driver.bindFramebuffer(gl::kDrawFramebuffer, static_cast<Uint>(previousDrawFramebuffer));
    driver.deleteFramebuffers(1, &framebuffer);
    if (auto scopeError = endErrorScope(state, "present surface"))
        return Result<void, RhiError>{err(std::move(*scopeError))};
    return Result<void, RhiError>{ok()};
}

Result<void, RhiError> presentSurface(VernonRhiDevice handle, VernonRhiSurface surface, VernonRhiImage &released) {
    auto device = lookupDevice(handle);
    if (device.isErr())
        return Result<void, RhiError>{err(std::move(device).error())};
    OpenGLDevice &state = device.value().device();
    SurfaceSlot *slot = indexedSlot(state.surfaces, surface.index);
    auto pin = pinSlot(slot, surface);
    if (pin.isErr())
        return Result<void, RhiError>{err(std::move(pin).error())};
    std::lock_guard<std::mutex> surfaceGuard(slot->mutex);
    if (!slot->acquired)
        return invalidResult("present_opengl_surface", surface.generation, surface.index);
    const SurfaceImage presented = slot->images[slot->next];
    released = presented.image;
    slot->acquired = false;
    slot->next = (slot->next + 1) % slot->images.size();
    const VernonOpenGLSurfaceCallbacks &callbacks = slot->callbacks;
    std::lock_guard<std::mutex> guard(state.mutex);
    callbacks.make_current(callbacks.user_data);
    callbacks.swap_interval(callbacks.user_data, slot->configuration.present_mode == VERNON_RHI_PRESENT_FIFO ? 1 : 0);
    auto blitted = blitToWindow(state, *indexedSlot(state.images, presented.image.index), slot->configuration.width,
                                slot->configuration.height);
    if (blitted.isOk())
        callbacks.swap_buffers(callbacks.user_data);
    state.state.makeCurrent();
    return blitted;
}

Result<void, RhiError> destroySurface(VernonRhiDevice handle, VernonRhiSurface surface) {
    auto device = lookupDevice(handle);
    if (device.isErr())
        return Result<void, RhiError>{err(std::move(device).error())};
    SurfaceSlot *slot = indexedSlot(device.value().device().surfaces, surface.index);
    if (!slot)
        return invalidResult("destroy_opengl_surface", surface.generation, surface.index);
    std::lock_guard<std::mutex> guard(slot->mutex);
    if (slot->acquired)
        return invalidResult("destroy_opengl_surface", surface.generation, surface.index);
    return slot->lifecycle.destroyPublic({surface.index, surface.generation});
}

} // namespace vernon::rhi::opengl_api

const vernon::rhi::BackendDispatch &vernon::rhi::openGLBackendDispatch() {
    using namespace opengl_api;
    static const BackendDispatch dispatch = [] {
        BackendDispatch result{};
        result.backend = VERNON_RHI_BACKEND_OPENGL;
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
        result.setImageSampler.emplace(setImageSampler);
        result.uploadImage.emplace(uploadImage);
        result.downloadImage.emplace(downloadImage);
        result.downloadImageBatch.emplace(downloadImageBatch);
        result.generateImageMipmaps.emplace(generateImageMipmaps);
        result.bindImage.emplace(bindImage);
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
        result.commandCapabilitiesForDevice.emplace(commandCapabilitiesForDevice);
        result.recordTimer.emplace(recordTimer);
        result.readTimers.emplace(readTimers);
        result.releaseTimers.emplace(releaseTimers);
        result.createImageView.emplace(createImageView);
        result.destroyImageView.emplace(destroyImageView);
        result.getImageViewNativeHandle.emplace(getImageViewNativeHandle);
        result.imageViewResource.emplace(imageViewResource);
        result.describeImageViewResource.emplace(describeImageViewResource);
        result.createSurface.emplace(createSurface);
        result.surfaceCapabilities.emplace(surfaceCapabilities);
        result.configureSurface.emplace(configureSurface);
        result.acquireSurfaceImage.emplace(acquireSurfaceImage);
        result.presentSurface.emplace(presentSurface);
        result.destroySurface.emplace(destroySurface);
        return result;
    }();
    return dispatch;
}
