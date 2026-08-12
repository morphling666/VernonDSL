#include "backend_dispatch.h"
#include "image_data_layout.h"
#include "image_descriptor_validation.h"
#include "logical_resource_record.h"
#include "opengl_backend.h"
#include "rhi_test_hooks.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstring>
#include <deque>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <string>
#include <type_traits>
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

// Public handles and prepared bindings share the same logical record. Destroying
// the public owner invalidates its handle immediately; the native object and slot
// remain occupied until the final prepared binding releases the record.
struct ImageSlot : vernon::rhi::LogicalResourceRecord {
    Image image;
    VernonRhiImageDescriptor descriptor{};
    Enum target{};
};

struct IdentityImageViewBacking {};
struct OwnedTextureViewBacking {
    Image image;
};
using ImageViewBacking = std::variant<IdentityImageViewBacking, OwnedTextureViewBacking>;

struct ImageViewSlot : vernon::rhi::LogicalResourceRecord {
    ImageViewBacking backing;
    VernonRhiImageViewDescriptor descriptor{};
    uint64_t imageResource{};
    Enum target{};
};

struct BufferSlot : vernon::rhi::LogicalResourceRecord {
    vernon::rhi::opengl::Buffer buffer;
    VernonRhiBufferDescriptor descriptor{};
};

struct SamplerSlot : vernon::rhi::LogicalResourceRecord {
    vernon::rhi::opengl::Sampler sampler;
};

struct OpenGLDevice {
    DeviceState state;
    std::vector<BufferSlot> buffers;
    std::vector<ImageSlot> images;
    std::vector<ImageViewSlot> imageViews;
    std::vector<SamplerSlot> samplers;
    std::string error;
    std::mutex mutex;
};

struct DeviceSlot {
    std::shared_ptr<OpenGLDevice> device;
    uint32_t generation{1};
};
std::mutex deviceMutex;
std::vector<DeviceSlot> devices;

VernonRhiDevice invalidDevice() { return {static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0}; }

template <typename Handle> uint64_t resourceKey(Handle handle) {
    return (static_cast<uint64_t>(handle.generation) << 32) | (static_cast<uint64_t>(handle.index) + 1);
}

bool decodeResourceKey(uint64_t key, uint32_t &index, uint32_t &generation) {
    const uint64_t encodedIndex = key & UINT32_MAX;
    generation = static_cast<uint32_t>(key >> 32);
    if (!encodedIndex || !generation)
        return false;
    index = static_cast<uint32_t>(encodedIndex - 1);
    return true;
}

template <typename Slots> auto *lookupResourceRecord(Slots &slots, uint64_t key) {
    uint32_t index = 0;
    uint32_t generation = 0;
    if (!decodeResourceKey(key, index, generation) || index >= slots.size())
        return static_cast<typename Slots::value_type *>(nullptr);
    auto &slot = slots[index];
    return slot.isRetained(generation) ? &slot : nullptr;
}

template <typename Slot, typename = void> struct HasPublicAlive : std::false_type {};
template <typename Slot>
struct HasPublicAlive<Slot, std::void_t<decltype(std::declval<Slot &>().publicAlive)>> : std::true_type {};

template <typename Slot> bool publicAlive(const Slot &slot) {
    if constexpr (HasPublicAlive<Slot>::value)
        return slot.publicAlive;
    return true;
}

std::shared_ptr<OpenGLDevice> lookupDevice(VernonRhiDevice handle) {
    std::lock_guard<std::mutex> guard(deviceMutex);
    if (handle.index >= devices.size())
        return {};
    DeviceSlot &slot = devices[handle.index];
    return slot.device && slot.generation == handle.generation ? slot.device : std::shared_ptr<OpenGLDevice>{};
}
ImageSlot *lookupImage(OpenGLDevice &device, VernonRhiImage handle) {
    if (handle.index >= device.images.size())
        return nullptr;
    ImageSlot &slot = device.images[handle.index];
    return slot.isPublic(handle.generation) ? &slot : nullptr;
}

ImageViewSlot *lookupImageView(OpenGLDevice &device, VernonRhiImageView handle) {
    if (handle.index >= device.imageViews.size())
        return nullptr;
    ImageViewSlot &slot = device.imageViews[handle.index];
    return slot.isPublic(handle.generation) ? &slot : nullptr;
}

ImageSlot *lookupImageViewParent(OpenGLDevice &device, const ImageViewSlot &view) {
    return lookupResourceRecord(device.images, view.imageResource);
}

Uint imageViewName(OpenGLDevice &device, const ImageViewSlot &view) {
    if (const auto *owned = std::get_if<OwnedTextureViewBacking>(&view.backing))
        return owned->image.name;
    const ImageSlot *parent = lookupImageViewParent(device, view);
    return parent ? parent->image.name : 0;
}

void recycleImageView(OpenGLDevice &device, ImageViewSlot &view) {
    const uint64_t parentKey = view.imageResource;
    if (auto *owned = std::get_if<OwnedTextureViewBacking>(&view.backing)) {
        device.state.destroyImage(owned->image);
        view.backing.emplace<IdentityImageViewBacking>();
    }
    view.recycle();
    ImageSlot *parent = lookupResourceRecord(device.images, parentKey);
    if (parent && parent->release()) {
        device.state.destroyImage(parent->image);
        parent->recycle();
    }
}

BufferSlot *lookupBuffer(OpenGLDevice &device, VernonRhiBuffer handle) {
    if (handle.index >= device.buffers.size())
        return nullptr;
    BufferSlot &slot = device.buffers[handle.index];
    return slot.isPublic(handle.generation) ? &slot : nullptr;
}

SamplerSlot *lookupSampler(OpenGLDevice &device, VernonRhiSampler handle) {
    if (handle.index >= device.samplers.size())
        return nullptr;
    SamplerSlot &slot = device.samplers[handle.index];
    return slot.isPublic(handle.generation) ? &slot : nullptr;
}

void validate(const OpenGLDevice &device) {
    for (const BufferSlot &slot : device.buffers) {
        slot.validate();
        assert(slot.occupied == (slot.buffer.name != 0));
    }
    for (const ImageSlot &slot : device.images) {
        slot.validate();
        assert(slot.occupied == (slot.image.name != 0));
    }
    for (const ImageViewSlot &slot : device.imageViews) {
        slot.validate();
        const auto *owned = std::get_if<OwnedTextureViewBacking>(&slot.backing);
        assert(slot.occupied || !owned);
        assert(!owned || owned->image.name != 0);
        assert(!slot.occupied || slot.imageResource != 0);
    }
    for (const SamplerSlot &slot : device.samplers) {
        slot.validate();
        assert(slot.occupied == (slot.sampler.name != 0));
    }
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
} // namespace

namespace vernon::rhi::opengl_api {

extern "C" VERNON_RHI_CAPI VernonRhiDevice vernonRhiCreateOpenGLDevice(const VernonOpenGLContextCallbacks *callbacks,
                                                                       uint32_t embeddedProfile) {
    if (!callbacks)
        return invalidDevice();
    auto device = std::shared_ptr<OpenGLDevice>(new (std::nothrow) OpenGLDevice());
    if (!device || !device->state.initialize(*callbacks, embeddedProfile != 0, device->error))
        return invalidDevice();
    std::lock_guard<std::mutex> guard(deviceMutex);
    uint32_t index = 0;
    while (index < devices.size() && devices[index].device)
        ++index;
    if (index == devices.size())
        devices.emplace_back();
    devices[index].device = std::move(device);
    return {index, devices[index].generation};
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

void destroyDevice(VernonRhiDevice handle) {
    std::shared_ptr<OpenGLDevice> device;
    {
        std::lock_guard<std::mutex> guard(deviceMutex);
        if (handle.index >= devices.size() || devices[handle.index].generation != handle.generation)
            return;
        DeviceSlot &slot = devices[handle.index];
        device = std::move(slot.device);
        ++slot.generation;
        if (slot.generation == 0)
            slot.generation = 1;
    }
    if (!device)
        return;
    std::lock_guard<std::mutex> guard(device->mutex);
    for (BufferSlot &buffer : device->buffers)
        if (buffer.occupied)
            device->state.destroyBuffer(buffer.buffer);
    for (ImageViewSlot &view : device->imageViews)
        if (view.occupied)
            if (auto *owned = std::get_if<OwnedTextureViewBacking>(&view.backing))
                device->state.destroyImage(owned->image);
    for (ImageSlot &image : device->images)
        if (image.occupied)
            device->state.destroyImage(image.image);
    for (SamplerSlot &sampler : device->samplers)
        if (sampler.occupied)
            device->state.destroySampler(sampler.sampler);
}

VernonStringView lastError(VernonRhiDevice handle) {

    auto device = lookupDevice(handle);
    return device ? VernonStringView{device->error.data(), device->error.size()} : VernonStringView{};
}

VernonRhiStatus synchronize(VernonRhiDevice handle) {

    auto device = lookupDevice(handle);
    if (!device)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(device->mutex);
    device->state.makeCurrent();
    device->state.driver.finish();
    return VERNON_RHI_STATUS_OK;
}

VernonRhiStatus createBuffer(VernonRhiDevice handle, const VernonRhiBufferDescriptor *descriptor,
                             VernonRhiBuffer *output) {

    auto device = lookupDevice(handle);
    if (!device || !descriptor || !output || descriptor->struct_size < sizeof(*descriptor) || descriptor->size == 0)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(device->mutex);
    uint32_t index = 0;
    while (index < device->buffers.size() && device->buffers[index].occupied)
        ++index;
    if (index == device->buffers.size())
        device->buffers.emplace_back();
    BufferSlot &slot = device->buffers[index];
    if (!device->state.createBuffer(slot.buffer, static_cast<size_t>(descriptor->size), device->error))
        return VERNON_RHI_STATUS_INTERNAL_ERROR;
    slot.descriptor = *descriptor;
    slot.publish();
    *output = {index, slot.generation};
    validate(*device);
    return VERNON_RHI_STATUS_OK;
}

VernonRhiStatus uploadBuffer(VernonRhiDevice handle, VernonRhiBuffer buffer, uint64_t offset, const void *source,
                             uint64_t size) {

    auto device = lookupDevice(handle);
    if (!device || !source)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(device->mutex);
    BufferSlot *slot = lookupBuffer(*device, buffer);
    if (!slot || offset > slot->descriptor.size || size > slot->descriptor.size - offset)
        return fail(*device, "OpenGL RHI buffer upload range is invalid");
    return device->state.uploadBuffer(slot->buffer, static_cast<size_t>(offset), source, static_cast<size_t>(size),
                                      device->error)
               ? VERNON_RHI_STATUS_OK
               : VERNON_RHI_STATUS_INTERNAL_ERROR;
}

VernonRhiStatus uploadBufferRanges(VernonRhiDevice handle, VernonRhiBuffer buffer,
                                   const VernonRhiBufferUploadRange *ranges, size_t rangeCount) {
    auto device = lookupDevice(handle);
    if (!device || !ranges || rangeCount == 0)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(device->mutex);
    BufferSlot *slot = lookupBuffer(*device, buffer);
    if (!slot)
        return fail(*device, "OpenGL RHI buffer upload handle is invalid");
    for (size_t index = 0; index < rangeCount; ++index) {
        const VernonRhiBufferUploadRange &range = ranges[index];
        if (!range.source || range.size == 0 || range.size > (std::numeric_limits<size_t>::max)() ||
            range.offset > slot->descriptor.size || range.size > slot->descriptor.size - range.offset)
            return fail(*device, "OpenGL RHI buffer upload range is invalid");
    }
    for (size_t index = 0; index < rangeCount; ++index) {
        const VernonRhiBufferUploadRange &range = ranges[index];
        if (!device->state.uploadBuffer(slot->buffer, static_cast<size_t>(range.offset), range.source,
                                        static_cast<size_t>(range.size), device->error))
            return VERNON_RHI_STATUS_INTERNAL_ERROR;
    }
    return VERNON_RHI_STATUS_OK;
}

VernonRhiStatus downloadBuffer(VernonRhiDevice handle, VernonRhiBuffer buffer, uint64_t offset, void *destination,
                               uint64_t size) {

    auto device = lookupDevice(handle);
    if (!device || !destination)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(device->mutex);
    BufferSlot *slot = lookupBuffer(*device, buffer);
    if (!slot || offset > slot->descriptor.size || size > slot->descriptor.size - offset)
        return fail(*device, "OpenGL RHI buffer readback range is invalid");
    return device->state.downloadBuffer(slot->buffer, static_cast<size_t>(offset), destination,
                                        static_cast<size_t>(size), device->error)
               ? VERNON_RHI_STATUS_OK
               : VERNON_RHI_STATUS_INTERNAL_ERROR;
}

uint32_t isBufferValid(VernonRhiDevice handle, VernonRhiBuffer buffer) {

    auto device = lookupDevice(handle);
    if (!device)
        return 0;
    std::lock_guard<std::mutex> guard(device->mutex);
    return lookupBuffer(*device, buffer) != nullptr;
}

VernonRhiStatus createImage(VernonRhiDevice handle, const VernonRhiImageDescriptor *descriptor,
                            VernonRhiImage *output) {

    auto device = lookupDevice(handle);
    if (!device || !descriptor || !output || descriptor->struct_size < sizeof(*descriptor))
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(device->mutex);
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
    if (!target || !formatInfo(descriptor->format, format) || descriptor->width == 0 || descriptor->height == 0 ||
        descriptor->depth == 0 || descriptor->mip_levels == 0 || descriptor->sample_count != 1 || !validCube ||
        !validDimension || (depthFormat && (descriptor->usage & VERNON_RHI_IMAGE_COLOR_ATTACHMENT)) ||
        (!depthFormat && (descriptor->usage & VERNON_RHI_IMAGE_DEPTH_STENCIL_ATTACHMENT)))
        return fail(*device, "OpenGL RHI image descriptor is invalid");
    uint32_t index = 0;
    while (index < device->images.size() && device->images[index].occupied)
        ++index;
    if (index == device->images.size()) {
        try {
            device->images.emplace_back();
        } catch (const std::bad_alloc &) {
            return fail(*device, "OpenGL RHI image slot allocation failed", VERNON_RHI_STATUS_INTERNAL_ERROR);
        }
    }
    ImageSlot &slot = device->images[index];
    device->state.makeCurrent();
    device->state.driver.genTextures(1, &slot.image.name);
    if (!slot.image.name)
        return fail(*device, "OpenGL RHI image allocation failed", VERNON_RHI_STATUS_INTERNAL_ERROR);
    {
        auto &driver = device->state.driver;
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
    slot.publish();
    validate(*device);
    *output = {index, slot.generation};
    return VERNON_RHI_STATUS_OK;
}

VernonRhiStatus setImageSampler(VernonRhiDevice handle, VernonRhiImage image,
                                const VernonRhiSamplerDescriptor *descriptor) {

    auto device = lookupDevice(handle);
    if (!device || !descriptor || descriptor->struct_size < sizeof(*descriptor))
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(device->mutex);
    ImageSlot *slot = lookupImage(*device, image);
    const Int minFilter = samplerFilter(descriptor->min_filter);
    const Int magFilter = samplerFilter(descriptor->mag_filter);
    const Int addressU = samplerAddress(descriptor->address_u);
    const Int addressV = samplerAddress(descriptor->address_v);
    const Int addressW = samplerAddress(descriptor->address_w);
    if (!slot || !minFilter || (descriptor->mag_filter > VERNON_RHI_FILTER_LINEAR) || !magFilter || !addressU ||
        !addressV || !addressW)
        return fail(*device, "OpenGL RHI image sampler descriptor is invalid");
    device->state.makeCurrent();
    auto &driver = device->state.driver;
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
    if (!device || (uploadCount != 0 && !uploads))
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(device->mutex);
    ImageSlot *slot = lookupImage(*device, image);
    if (!slot)
        return fail(*device, "OpenGL RHI image handle is stale");
    device->state.makeCurrent();
    auto &driver = device->state.driver;
    driver.bindTexture(slot->target, slot->image.name);
    driver.pixelStorei(0x0CF5, 1);
    FormatInfo storageFormat;
    if (!formatInfo(slot->descriptor.format, storageFormat)) {
        driver.bindTexture(slot->target, 0);
        return fail(*device, "OpenGL RHI image storage format is invalid");
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
            return fail(*device, "OpenGL RHI image upload descriptor is invalid");
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
    if (!device)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(device->mutex);
    ImageSlot *slot = lookupImage(*device, image);
    const auto expected = slot ? imageDownloadByteSize(slot->descriptor, *download) : std::nullopt;
    FormatInfo format;
    if (!slot || !expected || size != *expected || !formatInfo(slot->descriptor.format, format))
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    if (slot->descriptor.dimension == VERNON_RHI_IMAGE_3D)
        return device->state.downloadImage3D(slot->image, static_cast<Int>(download->mip_level),
                                             static_cast<Int>(download->offset_x), static_cast<Int>(download->offset_y),
                                             static_cast<Int>(download->offset_z), static_cast<Size>(download->width),
                                             static_cast<Size>(download->height), static_cast<Size>(download->depth),
                                             format.external, format.allocationType, *expected / download->depth,
                                             destination, device->error)
                   ? VERNON_RHI_STATUS_OK
                   : VERNON_RHI_STATUS_INTERNAL_ERROR;
    const Enum target = slot->descriptor.dimension == VERNON_RHI_IMAGE_CUBE
                            ? vernon::rhi::opengl::kTextureCubeMapPositiveX + download->array_layer
                            : slot->target;
    if (slot->descriptor.format != VERNON_RHI_FORMAT_D32_FLOAT_S8_UINT)
        return device->state.downloadImage2D(slot->image, target, static_cast<Int>(download->mip_level),
                                             static_cast<Int>(download->offset_x), static_cast<Int>(download->offset_y),
                                             static_cast<Size>(download->width), static_cast<Size>(download->height),
                                             format.external, format.allocationType, destination, device->error)
                   ? VERNON_RHI_STATUS_OK
                   : VERNON_RHI_STATUS_INTERNAL_ERROR;
    std::vector<uint8_t> native(*expected);
    if (!device->state.downloadImage2D(slot->image, target, static_cast<Int>(download->mip_level),
                                       static_cast<Int>(download->offset_x), static_cast<Int>(download->offset_y),
                                       static_cast<Size>(download->width), static_cast<Size>(download->height),
                                       format.external, format.allocationType, native.data(), device->error))
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
    if (!device)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(device->mutex);
    ImageSlot *slot = lookupImage(*device, image);
    if (!slot || slot->descriptor.mip_levels < 2 || !(slot->descriptor.usage & VERNON_RHI_IMAGE_TRANSFER_SOURCE) ||
        !(slot->descriptor.usage & VERNON_RHI_IMAGE_TRANSFER_DESTINATION) ||
        slot->descriptor.format == VERNON_RHI_FORMAT_D32_FLOAT ||
        slot->descriptor.format == VERNON_RHI_FORMAT_D32_FLOAT_S8_UINT)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    device->state.makeCurrent();
    device->state.driver.bindTexture(slot->target, slot->image.name);
    device->state.driver.generateMipmap(slot->target);
    device->state.driver.bindTexture(slot->target, 0);
    return VERNON_RHI_STATUS_OK;
}

VernonRhiStatus bindImage(VernonRhiDevice handle, VernonRhiImage image, uint32_t textureUnit) {

    auto device = lookupDevice(handle);
    if (!device)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(device->mutex);
    ImageSlot *slot = lookupImage(*device, image);
    if (!slot)
        return fail(*device, "OpenGL RHI image handle is stale");
    device->state.makeCurrent();
    device->state.driver.activeTexture(vernon::rhi::opengl::kTexture0 + textureUnit);
    device->state.driver.bindTexture(slot->target, slot->image.name);
    return VERNON_RHI_STATUS_OK;
}

VernonRhiStatus destroyImage(VernonRhiDevice handle, VernonRhiImage image) {

    auto device = lookupDevice(handle);
    if (!device)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(device->mutex);
    ImageSlot *slot = lookupImage(*device, image);
    if (!slot)
        return fail(*device, "OpenGL RHI image handle is stale");
    slot->destroyPublicOwner();
    if (slot->bindingReferences == 0) {
        device->state.destroyImage(slot->image);
        slot->recycle();
    }
    validate(*device);
    return VERNON_RHI_STATUS_OK;
}

uint32_t isImageValid(VernonRhiDevice handle, VernonRhiImage image) {

    auto device = lookupDevice(handle);
    if (!device)
        return 0;
    std::lock_guard<std::mutex> guard(device->mutex);
    return lookupImage(*device, image) != nullptr;
}

VernonRhiStatus createSampler(VernonRhiDevice handle, const VernonRhiSamplerDescriptor *descriptor,
                              VernonRhiSampler *output) {

    auto device = lookupDevice(handle);
    if (!device || !descriptor || !output || descriptor->struct_size < sizeof(*descriptor))
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    const Int minFilter = samplerFilter(descriptor->min_filter);
    const Int magFilter = samplerFilter(descriptor->mag_filter);
    const Int addressU = samplerAddress(descriptor->address_u);
    const Int addressV = samplerAddress(descriptor->address_v);
    const Int addressW = samplerAddress(descriptor->address_w);
    if (!minFilter || descriptor->mag_filter > VERNON_RHI_FILTER_LINEAR || !magFilter || !addressU || !addressV ||
        !addressW)
        return fail(*device, "OpenGL RHI sampler descriptor is invalid");
    std::lock_guard<std::mutex> guard(device->mutex);
    uint32_t index = 0;
    while (index < device->samplers.size() && device->samplers[index].occupied)
        ++index;
    if (index == device->samplers.size())
        device->samplers.emplace_back();
    SamplerSlot &slot = device->samplers[index];
    if (!device->state.createSampler(slot.sampler, addressU, addressV, addressW, minFilter, magFilter, device->error))
        return VERNON_RHI_STATUS_INTERNAL_ERROR;
    slot.publish();
    *output = {index, slot.generation};
    validate(*device);
    return VERNON_RHI_STATUS_OK;
}

VernonRhiStatus destroySampler(VernonRhiDevice handle, VernonRhiSampler sampler) {

    auto device = lookupDevice(handle);
    if (!device)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(device->mutex);
    SamplerSlot *slot = lookupSampler(*device, sampler);
    if (!slot)
        return fail(*device, "OpenGL RHI sampler handle is stale");
    slot->destroyPublicOwner();
    if (slot->bindingReferences == 0) {
        device->state.destroySampler(slot->sampler);
        slot->recycle();
    }
    validate(*device);
    return VERNON_RHI_STATUS_OK;
}

uint32_t isSamplerValid(VernonRhiDevice handle, VernonRhiSampler sampler) {

    auto device = lookupDevice(handle);
    if (!device)
        return 0;
    std::lock_guard<std::mutex> guard(device->mutex);
    return lookupSampler(*device, sampler) != nullptr;
}

VernonRhiStatus getImageNativeHandle(VernonRhiDevice handle, VernonRhiImage image, uint64_t *output) {

    auto device = lookupDevice(handle);
    if (!device || !output)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(device->mutex);
    ImageSlot *slot = lookupImage(*device, image);
    if (!slot)
        return fail(*device, "OpenGL RHI image handle is stale");
    *output = slot->image.name;
    return VERNON_RHI_STATUS_OK;
}

VernonRhiStatus createImageView(VernonRhiDevice handle, const VernonRhiImageViewDescriptor *descriptor,
                                VernonRhiImageView *output) {
    auto device = lookupDevice(handle);
    if (!device || !descriptor || descriptor->struct_size < sizeof(*descriptor) || !output ||
        !descriptor->mip_level_count || !descriptor->array_layer_count || !descriptor->aspects)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    *output = {static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
    std::lock_guard<std::mutex> guard(device->mutex);
    ImageSlot *parent = lookupImage(*device, descriptor->image);
    FormatInfo format{};
    if (!parent || !formatInfo(descriptor->format, format))
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    if (!vernon::rhi::validImageViewDescriptor(parent->descriptor, *descriptor))
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    if (descriptor->dimension == VERNON_RHI_IMAGE_2D && descriptor->array_layer_count != 1)
        return VERNON_RHI_STATUS_UNSUPPORTED;
    if (!parent->retain())
        return VERNON_RHI_STATUS_INTERNAL_ERROR;
    uint32_t index = 0;
    while (index < device->imageViews.size() && device->imageViews[index].occupied)
        ++index;
    if (index == device->imageViews.size()) {
        try {
            device->imageViews.emplace_back();
        } catch (const std::bad_alloc &) {
            parent->release();
            return VERNON_RHI_STATUS_INTERNAL_ERROR;
        }
    }
    ImageViewSlot &slot = device->imageViews[index];
    const bool identityView =
        descriptor->format == parent->descriptor.format && descriptor->dimension == parent->descriptor.dimension &&
        descriptor->aspects == vernon::rhi::imageFormatAspects(parent->descriptor.format) &&
        descriptor->base_mip_level == 0 && descriptor->mip_level_count == parent->descriptor.mip_levels &&
        descriptor->base_array_layer == 0 && descriptor->array_layer_count == parent->descriptor.array_layers;
    slot.backing.emplace<IdentityImageViewBacking>();
    if (!identityView) {
        if (device->state.embeddedProfile || !device->state.driver.textureView) {
            parent->release();
            return VERNON_RHI_STATUS_UNSUPPORTED;
        }
        device->state.makeCurrent();
        auto &owned = slot.backing.emplace<OwnedTextureViewBacking>();
        device->state.driver.genTextures(1, &owned.image.name);
        if (!owned.image.name) {
            slot.backing.emplace<IdentityImageViewBacking>();
            parent->release();
            return VERNON_RHI_STATUS_INTERNAL_ERROR;
        }
        owned.image.imported = false;
        slot.target = imageTarget(descriptor->dimension);
        device->state.driver.textureView(owned.image.name, slot.target, parent->image.name, format.internal,
                                         descriptor->base_mip_level, descriptor->mip_level_count,
                                         descriptor->base_array_layer, descriptor->array_layer_count);
    }
    slot.target = imageTarget(descriptor->dimension);
    slot.descriptor = *descriptor;
    slot.imageResource = resourceKey(descriptor->image);
    slot.publish();
    *output = {index, slot.generation};
    validate(*device);
    return VERNON_RHI_STATUS_OK;
}

VernonRhiStatus destroyImageView(VernonRhiDevice handle, VernonRhiImageView view) {
    auto device = lookupDevice(handle);
    if (!device)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(device->mutex);
    ImageViewSlot *slot = lookupImageView(*device, view);
    if (!slot)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    slot->destroyPublicOwner();
    if (slot->bindingReferences != 0)
        return VERNON_RHI_STATUS_OK;
    recycleImageView(*device, *slot);
    validate(*device);
    return VERNON_RHI_STATUS_OK;
}

VernonRhiStatus getImageViewNativeHandle(VernonRhiDevice handle, VernonRhiImageView view, uint64_t *output) {
    auto device = lookupDevice(handle);
    if (!device || !output)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(device->mutex);
    ImageViewSlot *slot = lookupImageView(*device, view);
    if (!slot)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    *output = imageViewName(*device, *slot);
    return VERNON_RHI_STATUS_OK;
}

VernonRhiStatus destroyBuffer(VernonRhiDevice handle, VernonRhiBuffer buffer) {

    auto device = lookupDevice(handle);
    if (!device)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(device->mutex);
    BufferSlot *slot = lookupBuffer(*device, buffer);
    if (!slot)
        return fail(*device, "OpenGL RHI buffer handle is stale");
    slot->destroyPublicOwner();
    if (slot->bindingReferences == 0) {
        device->state.destroyBuffer(slot->buffer);
        slot->recycle();
    }
    validate(*device);
    return VERNON_RHI_STATUS_OK;
}

VernonRhiStatus getBufferNativeHandle(VernonRhiDevice handle, VernonRhiBuffer buffer, void **output) {

    auto device = lookupDevice(handle);
    if (!device || !output)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(device->mutex);
    BufferSlot *slot = lookupBuffer(*device, buffer);
    if (!slot)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    *output = reinterpret_cast<void *>(static_cast<uintptr_t>(slot->buffer.name));
    return VERNON_RHI_STATUS_OK;
}

bool ownsDevice(VernonRhiDevice handle) { return static_cast<bool>(lookupDevice(handle)); }

void *deviceStateForBackend(VernonRhiDevice handle) {
    auto device = lookupDevice(handle);
    return device ? &device->state : nullptr;
}

bool beginCommands(VernonRhiDevice handle, uint64_t &native, VernonRhiBackend &backend) {

    native = 0;
    if (auto device = lookupDevice(handle)) {
        std::lock_guard<std::mutex> guard(device->mutex);
        device->state.makeCurrent();
        backend = device->state.embeddedProfile ? VERNON_RHI_BACKEND_OPENGL_ES : VERNON_RHI_BACKEND_OPENGL;
        native = reinterpret_cast<uintptr_t>(&device->state);
        return true;
    }
    return false;
}

bool submitCommands(VernonRhiDevice handle, uint64_t native, bool computeWrites, bool &completed,
                    bool &externalCompletion) {

    completed = false;
    externalCompletion = false;
    if (auto device = lookupDevice(handle)) {
        if (native != reinterpret_cast<uintptr_t>(&device->state))
            return false;
        std::lock_guard<std::mutex> guard(device->mutex);
        device->state.makeCurrent();
        if (computeWrites) {
            if (device->state.driver.memoryBarrier)
                device->state.driver.memoryBarrier(
                    vernon::rhi::opengl::kShaderStorageBarrierBit | vernon::rhi::opengl::kVertexAttribArrayBarrierBit |
                    vernon::rhi::opengl::kTextureFetchBarrierBit | vernon::rhi::opengl::kBufferUpdateBarrierBit);
            else
                device->state.driver.finish();
        }
        completed = true;
        return true;
    }
    return false;
}

void completeBorrowedCommands(VernonRhiDevice handle, uint64_t native) {

    (void)handle;
    (void)native;
}

void abandonCommands(VernonRhiDevice handle, uint64_t native) {

    (void)handle;
    (void)native;
}

bool recordBarriers(VernonRhiDevice handle, uint64_t encoderKey, uint64_t native, const VernonRhiBarrier *barriers,
                    size_t barrierCount) {

    if (auto device = lookupDevice(handle)) {
        std::lock_guard<std::mutex> guard(device->mutex);
        if (native != reinterpret_cast<uintptr_t>(&device->state))
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
        device->state.makeCurrent();
        if (bits) {
            if (device->state.driver.memoryBarrier)
                device->state.driver.memoryBarrier(bits);
            else
                // glMemoryBarrier is core only in OpenGL 4.2. Earlier
                // contexts still need a synchronization point between graph
                // scopes, for example when sampling a depth attachment.
                device->state.driver.finish();
        }
        return true;
    }
    return false;
}

bool endRendering(VernonRhiDevice handle, uint64_t native, VernonRhiBackend backend, uint32_t backendKind,
                  uint32_t colorDiscardMask, uint32_t depthStencilDiscard, const uint64_t *colorResources,
                  size_t colorCount, uint64_t depthResource, uint64_t) {

    if ((backend == VERNON_RHI_BACKEND_OPENGL || backend == VERNON_RHI_BACKEND_OPENGL_ES) &&
        backendKind == CommandRenderingStateless) {
        auto device = lookupDevice(handle);
        if (!device || native != reinterpret_cast<uintptr_t>(&device->state))
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
        std::lock_guard<std::mutex> guard(device->mutex);
        device->state.makeCurrent();
        if (discardedCount && device->state.driver.invalidateFramebuffer)
            device->state.driver.invalidateFramebuffer(vernon::rhi::opengl::kFramebuffer,
                                                       static_cast<Size>(discardedCount), discarded.data());
        return true;
    }
    return false;
}

bool clearColor(VernonRhiDevice handle, uint64_t native, VernonRhiBackend backend, uint32_t backendKind, int32_t x,
                int32_t y, uint32_t width, uint32_t height, uint32_t layers, uint64_t target, uint32_t location,
                const float color[4]) {

    if (backend == VERNON_RHI_BACKEND_OPENGL || backend == VERNON_RHI_BACKEND_OPENGL_ES) {
        auto device = lookupDevice(handle);
        if (!device || native != reinterpret_cast<uintptr_t>(&device->state) ||
            backendKind != CommandRenderingStateless)
            return false;
        std::lock_guard<std::mutex> guard(device->mutex);
        device->state.makeCurrent();
        device->state.driver.clearBufferfv(vernon::rhi::opengl::kColor, static_cast<Int>(location), color);
        return true;
    }
    return false;
}

bool clearDepthStencil(VernonRhiDevice handle, uint64_t native, VernonRhiBackend backend, uint32_t backendKind,
                       int32_t x, int32_t y, uint32_t width, uint32_t height, uint32_t layers, uint64_t target,
                       float depth, uint32_t stencil, uint32_t aspects) {

    if (backend == VERNON_RHI_BACKEND_OPENGL || backend == VERNON_RHI_BACKEND_OPENGL_ES) {
        auto device = lookupDevice(handle);
        if (!device || native != reinterpret_cast<uintptr_t>(&device->state) ||
            backendKind != CommandRenderingStateless ||
            (aspects & ~(VERNON_RHI_ATTACHMENT_DEPTH | VERNON_RHI_ATTACHMENT_STENCIL)) || !aspects)
            return false;
        std::lock_guard<std::mutex> guard(device->mutex);
        device->state.makeCurrent();
        if (aspects == (VERNON_RHI_ATTACHMENT_DEPTH | VERNON_RHI_ATTACHMENT_STENCIL))
            device->state.driver.clearBufferfi(vernon::rhi::opengl::kDepthStencil, 0, depth, static_cast<Int>(stencil));
        else if (aspects == VERNON_RHI_ATTACHMENT_DEPTH)
            device->state.driver.clearBufferfv(vernon::rhi::opengl::kDepth, 0, &depth);
        else {
            const Int value = static_cast<Int>(stencil);
            device->state.driver.clearBufferiv(vernon::rhi::opengl::kStencil, 0, &value);
        }
        return true;
    }
    return false;
}

uint64_t bufferResource(VernonRhiDevice handle, VernonRhiBuffer buffer) {

    if (auto device = lookupDevice(handle)) {
        std::lock_guard<std::mutex> guard(device->mutex);
        if (BufferSlot *slot = lookupBuffer(*device, buffer))
            return resourceKey(buffer);
    }
    return 0;
}

uint64_t imageResource(VernonRhiDevice handle, VernonRhiImage image) {

    if (auto device = lookupDevice(handle)) {
        std::lock_guard<std::mutex> guard(device->mutex);
        if (ImageSlot *slot = lookupImage(*device, image))
            return resourceKey(image);
    }
    return 0;
}

uint64_t imageViewResource(VernonRhiDevice handle, VernonRhiImageView view) {
    if (auto device = lookupDevice(handle)) {
        std::lock_guard<std::mutex> guard(device->mutex);
        if (lookupImageView(*device, view))
            return resourceKey(view);
    }
    return 0;
}

uint64_t samplerResource(VernonRhiDevice handle, VernonRhiSampler sampler) {

    if (auto device = lookupDevice(handle)) {
        std::lock_guard<std::mutex> guard(device->mutex);
        if (SamplerSlot *slot = lookupSampler(*device, sampler))
            return resourceKey(sampler);
    }
    return 0;
}

bool retainResource(VernonRhiDevice handle, ResourceKind kind, uint64_t key) {

    auto device = lookupDevice(handle);
    if (!device)
        return false;
    std::lock_guard<std::mutex> guard(device->mutex);
    if (kind == ResourceKind::Buffer) {
        BufferSlot *slot = lookupResourceRecord(device->buffers, key);
        return slot && slot->retain();
    }
    if (kind == ResourceKind::Image) {
        ImageSlot *slot = lookupResourceRecord(device->images, key);
        return slot && slot->retain();
    }
    if (kind == ResourceKind::ImageView) {
        ImageViewSlot *slot = lookupResourceRecord(device->imageViews, key);
        return slot && slot->retain();
    }
    SamplerSlot *slot = lookupResourceRecord(device->samplers, key);
    return slot && slot->retain();
}

uint64_t resolveResource(VernonRhiDevice handle, ResourceKind kind, uint64_t key) {

    auto device = lookupDevice(handle);
    if (!device)
        return 0;
    std::lock_guard<std::mutex> guard(device->mutex);
    if (kind == ResourceKind::Buffer) {
        auto *slot = lookupResourceRecord(device->buffers, key);
        return slot && slot->occupied ? slot->buffer.name : 0;
    }
    if (kind == ResourceKind::Image) {
        auto *slot = lookupResourceRecord(device->images, key);
        return slot && slot->occupied ? slot->image.name : 0;
    }
    if (kind == ResourceKind::ImageView) {
        auto *slot = lookupResourceRecord(device->imageViews, key);
        return slot ? imageViewName(*device, *slot) : 0;
    }
    auto *slot = lookupResourceRecord(device->samplers, key);
    return slot && slot->occupied ? slot->sampler.name : 0;
}

bool describeImageResource(VernonRhiDevice handle, uint64_t key, VernonRhiImageDescriptor *descriptor) {
    auto device = lookupDevice(handle);
    if (!device || !descriptor)
        return false;
    std::lock_guard<std::mutex> guard(device->mutex);
    const ImageSlot *slot = lookupResourceRecord(device->images, key);
    if (!slot || !slot->occupied)
        return false;
    *descriptor = slot->descriptor;
    return true;
}

bool describeImageViewResource(VernonRhiDevice handle, uint64_t key, VernonRhiImageViewDescriptor *view,
                               VernonRhiImageDescriptor *image, uint64_t *parentKey) {
    auto device = lookupDevice(handle);
    if (!device || !view || !image || !parentKey)
        return false;
    std::lock_guard<std::mutex> guard(device->mutex);
    const ImageViewSlot *slot = lookupResourceRecord(device->imageViews, key);
    if (!slot)
        return false;
    const ImageSlot *parent = lookupResourceRecord(device->images, slot->imageResource);
    if (!parent)
        return false;
    *view = slot->descriptor;
    *image = parent->descriptor;
    *parentKey = slot->imageResource;
    return true;
}

void releaseResource(VernonRhiDevice handle, ResourceKind kind, uint64_t key) {

    auto device = lookupDevice(handle);
    if (!device)
        return;
    std::lock_guard<std::mutex> guard(device->mutex);
    if (kind == ResourceKind::Buffer) {
        auto *slot = lookupResourceRecord(device->buffers, key);
        if (!slot || !slot->release())
            return;
        device->state.destroyBuffer(slot->buffer);
        slot->recycle();
        return;
    }
    if (kind == ResourceKind::Image) {
        auto *slot = lookupResourceRecord(device->images, key);
        if (!slot || !slot->release())
            return;
        device->state.destroyImage(slot->image);
        slot->recycle();
        return;
    }
    if (kind == ResourceKind::ImageView) {
        auto *slot = lookupResourceRecord(device->imageViews, key);
        if (!slot || !slot->release())
            return;
        recycleImageView(*device, *slot);
        return;
    }
    auto *slot = lookupResourceRecord(device->samplers, key);
    if (!slot || !slot->release())
        return;
    device->state.destroySampler(slot->sampler);
    slot->recycle();
}

} // namespace vernon::rhi::opengl_api

const vernon::rhi::BackendDispatch &vernon::rhi::openGLBackendDispatch() {
    using namespace opengl_api;
    static const BackendDispatch dispatch{
        VERNON_RHI_BACKEND_OPENGL,
        ownsDevice,
        createOwnedDevice,
        destroyDevice,
        lastError,
        synchronize,
        deviceStateForBackend,
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
        releaseResource,
        beginCommands,
        submitCommands,
        completeBorrowedCommands,
        abandonCommands,
        recordBarriers,
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
