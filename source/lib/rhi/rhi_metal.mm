#include "backend_dispatch.h"
#include "image_data_layout.h"
#include "image_descriptor_validation.h"
#include "logical_resource_record.h"
#include "metal_backend.h"
#include "sampler_filter.h"
#include "VernonTextureTypes.h"

#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace vernon::rhi {
uint32_t metalTexturePixelFormat(VernonTextureFormat format);
}

namespace {

template <typename Native> struct MetalResourceSlot : vernon::rhi::LogicalResourceRecord {
    Native native;
};

struct MetalBufferSlot : MetalResourceSlot<vernon::rhi::metal::Buffer> {
    VernonRhiBufferDescriptor descriptor{};
};

struct MetalImageSlot : MetalResourceSlot<vernon::rhi::metal::Image> {
    VernonRhiImageDescriptor descriptor{};
};

struct MetalImageViewSlot : MetalResourceSlot<vernon::rhi::metal::ImageView> {
    VernonRhiImageViewDescriptor descriptor{};
    uint64_t imageResource{};
};

using MetalSamplerSlot = MetalResourceSlot<vernon::rhi::metal::Sampler>;

struct MetalDevice {
    vernon::rhi::metal::DeviceState state;
    std::vector<MetalBufferSlot> buffers;
    std::vector<MetalImageSlot> images;
    std::vector<MetalImageViewSlot> imageViews;
    std::vector<MetalSamplerSlot> samplers;
    std::string error;
    std::mutex mutex;
};

struct MetalDeviceSlot {
    std::shared_ptr<MetalDevice> device;
    uint32_t generation{1};
};

constexpr uint32_t metalDeviceBit = uint32_t{1} << 28;
std::vector<MetalDeviceSlot> metalDevices;
std::mutex deviceMutex;
std::unordered_map<uint64_t, vernon::rhi::metal::RenderingState *> renderingStates;
std::mutex renderingStateMutex;

VernonRhiDevice invalidDevice() { return {static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0}; }

template <typename Handle> uint64_t resourceKey(Handle handle) {
    return vernon::rhi::encodeResourceKey(handle);
}

bool decodeResourceKey(uint64_t key, uint32_t &index, uint32_t &generation) {
    const uint64_t encodedIndex = key & UINT32_MAX;
    generation = static_cast<uint32_t>(key >> 32);
    if (!encodedIndex || !generation)
        return false;
    index = static_cast<uint32_t>(encodedIndex - 1);
    return true;
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

template <typename Slots> auto *lookupResourceRecord(Slots &slots, uint64_t key) {
    uint32_t index = 0;
    uint32_t generation = 0;
    if (!decodeResourceKey(key, index, generation) || index >= slots.size())
        return static_cast<typename Slots::value_type *>(nullptr);
    auto &slot = slots[index];
    return slot.isRetained(generation) ? &slot : nullptr;
}

template <typename Slots, typename Handle> auto *lookupPublicResource(Slots &slots, Handle handle) {
    if (handle.index >= slots.size())
        return static_cast<typename Slots::value_type *>(nullptr);
    auto &slot = slots[handle.index];
    return slot.isPublic(handle.generation) ? &slot : nullptr;
}

template <typename Slots> uint32_t allocateSlot(Slots &slots) {
    uint32_t index = 0;
    while (index < slots.size() && slots[index].occupied)
        ++index;
    if (index == slots.size())
        slots.emplace_back();
    return index;
}

std::shared_ptr<MetalDevice> lookupMetalDevice(VernonRhiDevice handle) {
    if ((handle.index & metalDeviceBit) == 0)
        return {};
    const uint32_t index = handle.index & ~metalDeviceBit;
    std::lock_guard<std::mutex> guard(deviceMutex);
    if (index >= metalDevices.size())
        return {};
    MetalDeviceSlot &slot = metalDevices[index];
    return slot.device && slot.generation == handle.generation ? slot.device : std::shared_ptr<MetalDevice>{};
}

} // namespace

void vernon::rhi::metal::registerRenderingState(uint64_t commandBuffer, RenderingState *rendering) {
    std::lock_guard<std::mutex> guard(renderingStateMutex);
    renderingStates[commandBuffer] = rendering;
}

vernon::rhi::metal::RenderingState *vernon::rhi::metal::findRenderingState(uint64_t commandBuffer) {
    std::lock_guard<std::mutex> guard(renderingStateMutex);
    const auto found = renderingStates.find(commandBuffer);
    return found == renderingStates.end() ? nullptr : found->second;
}

void vernon::rhi::metal::unregisterRenderingState(uint64_t commandBuffer, RenderingState *rendering) {
    std::lock_guard<std::mutex> guard(renderingStateMutex);
    const auto found = renderingStates.find(commandBuffer);
    if (found != renderingStates.end() && found->second == rendering)
        renderingStates.erase(found);
}

namespace vernon::rhi::metal_api {

VernonRhiDevice createOwnedDevice(const VernonRhiOwnedDeviceDescriptor *descriptor) {
    if (!descriptor)
        return invalidDevice();
    auto device = std::shared_ptr<MetalDevice>(new (std::nothrow) MetalDevice());
    if (!device) {
        setDeviceCreationError("Metal device state allocation failed");
        return invalidDevice();
    }
    if (!device->state.initialize(descriptor->device_index, device->error)) {
        setDeviceCreationError(device->error);
        return invalidDevice();
    }
    std::lock_guard<std::mutex> guard(deviceMutex);
    uint32_t index = 0;
    while (index < metalDevices.size() && metalDevices[index].device)
        ++index;
    if (index == metalDevices.size()) {
        if (index >= metalDeviceBit)
            return invalidDevice();
        metalDevices.emplace_back();
    }
    metalDevices[index].device = std::move(device);
    return {index | metalDeviceBit, metalDevices[index].generation};
}

void destroyDevice(VernonRhiDevice handle) {
    std::shared_ptr<MetalDevice> device;
    {
        const uint32_t index = handle.index & ~metalDeviceBit;
        std::lock_guard<std::mutex> guard(deviceMutex);
        if (index >= metalDevices.size() || metalDevices[index].generation != handle.generation)
            return;
        MetalDeviceSlot &slot = metalDevices[index];
        device = std::move(slot.device);
        if (++slot.generation == 0)
            slot.generation = 1;
    }
    if (!device)
        return;
    std::lock_guard<std::mutex> guard(device->mutex);
    for (MetalImageViewSlot &slot : device->imageViews)
        if (slot.occupied)
            device->state.destroyImageView(slot.native);
    for (MetalBufferSlot &slot : device->buffers)
        if (slot.occupied)
            device->state.destroyBuffer(slot.native);
    for (MetalImageSlot &slot : device->images)
        if (slot.occupied)
            device->state.destroyImage(slot.native);
    for (MetalSamplerSlot &slot : device->samplers)
        if (slot.occupied)
            device->state.destroySampler(slot.native);
    device->state.shutdown();
}

bool ownsDevice(VernonRhiDevice handle) { return static_cast<bool>(lookupMetalDevice(handle)); }

VernonStringView lastError(VernonRhiDevice handle) {
    auto device = lookupMetalDevice(handle);
    return device ? VernonStringView{device->error.data(), device->error.size()} : VernonStringView{};
}

VernonRhiStatus synchronize(VernonRhiDevice handle) {
    auto device = lookupMetalDevice(handle);
    if (!device)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(device->mutex);
    return device->state.synchronize(device->error) ? VERNON_RHI_STATUS_OK : VERNON_RHI_STATUS_INTERNAL_ERROR;
}

void *deviceStateForBackend(VernonRhiDevice handle) {
    auto device = lookupMetalDevice(handle);
    return device ? &device->state : nullptr;
}

VernonRhiStatus createBuffer(VernonRhiDevice handle, const VernonRhiBufferDescriptor *descriptor,
                             VernonRhiBuffer *output) {
    auto device = lookupMetalDevice(handle);
    if (!device)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    constexpr uint32_t allUsages =
        VERNON_RHI_BUFFER_TRANSFER_SOURCE | VERNON_RHI_BUFFER_TRANSFER_DESTINATION | VERNON_RHI_BUFFER_UNIFORM |
        VERNON_RHI_BUFFER_STORAGE | VERNON_RHI_BUFFER_VERTEX | VERNON_RHI_BUFFER_INDEX | VERNON_RHI_BUFFER_INDIRECT;
    if (!descriptor || !output || descriptor->struct_size < sizeof(*descriptor) || descriptor->size == 0 ||
        descriptor->size > (std::numeric_limits<size_t>::max)() ||
        descriptor->memory_class > VERNON_RHI_MEMORY_READBACK || (descriptor->usage & ~allUsages) != 0)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(device->mutex);
    const uint32_t index = allocateSlot(device->buffers);
    MetalBufferSlot &slot = device->buffers[index];
    if (!device->state.createBuffer(slot.native, *descriptor, device->error))
        return VERNON_RHI_STATUS_INTERNAL_ERROR;
    slot.descriptor = *descriptor;
    slot.publish();
    *output = {index, slot.generation};
    return VERNON_RHI_STATUS_OK;
}

VernonRhiStatus uploadBuffer(VernonRhiDevice handle, VernonRhiBuffer buffer, uint64_t offset, const void *source,
                             uint64_t size) {
    auto device = lookupMetalDevice(handle);
    if (!device)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    if (!source || size == 0 || size > (std::numeric_limits<size_t>::max)())
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(device->mutex);
    MetalBufferSlot *slot = lookupPublicResource(device->buffers, buffer);
    if (!slot || offset > slot->descriptor.size || size > slot->descriptor.size - offset)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    return device->state.uploadBuffer(slot->native, offset, source, size, device->error)
               ? VERNON_RHI_STATUS_OK
               : VERNON_RHI_STATUS_INTERNAL_ERROR;
}

VernonRhiStatus uploadBufferRanges(VernonRhiDevice handle, VernonRhiBuffer buffer,
                                   const VernonRhiBufferUploadRange *ranges, size_t rangeCount) {
    auto device = lookupMetalDevice(handle);
    if (!device)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    if (!ranges || rangeCount == 0)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(device->mutex);
    MetalBufferSlot *slot = lookupPublicResource(device->buffers, buffer);
    if (!slot)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    for (size_t index = 0; index < rangeCount; ++index) {
        const VernonRhiBufferUploadRange &range = ranges[index];
        if (!range.source || range.size == 0 || range.size > (std::numeric_limits<size_t>::max)() ||
            range.offset > slot->descriptor.size || range.size > slot->descriptor.size - range.offset)
            return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    }
    return device->state.uploadBufferRanges(slot->native, ranges, rangeCount, device->error)
               ? VERNON_RHI_STATUS_OK
               : VERNON_RHI_STATUS_INTERNAL_ERROR;
}

VernonRhiStatus downloadBuffer(VernonRhiDevice handle, VernonRhiBuffer buffer, uint64_t offset, void *destination,
                               uint64_t size) {
    auto device = lookupMetalDevice(handle);
    if (!device)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    if (!destination || size == 0 || size > (std::numeric_limits<size_t>::max)())
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(device->mutex);
    MetalBufferSlot *slot = lookupPublicResource(device->buffers, buffer);
    if (!slot || offset > slot->descriptor.size || size > slot->descriptor.size - offset)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    return device->state.downloadBuffer(slot->native, offset, destination, size, device->error)
               ? VERNON_RHI_STATUS_OK
               : VERNON_RHI_STATUS_INTERNAL_ERROR;
}

VernonRhiStatus destroyBuffer(VernonRhiDevice handle, VernonRhiBuffer buffer) {
    auto device = lookupMetalDevice(handle);
    if (!device)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(device->mutex);
    MetalBufferSlot *slot = lookupPublicResource(device->buffers, buffer);
    if (!slot)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    slot->destroyPublicOwner();
    if (slot->bindingReferences == 0) {
        device->state.destroyBuffer(slot->native);
        slot->recycle();
    }
    return VERNON_RHI_STATUS_OK;
}

uint32_t isBufferValid(VernonRhiDevice handle, VernonRhiBuffer buffer) {
    auto device = lookupMetalDevice(handle);
    if (!device)
        return 0;
    std::lock_guard<std::mutex> guard(device->mutex);
    return lookupPublicResource(device->buffers, buffer) != nullptr;
}

VernonRhiStatus getBufferNativeHandle(VernonRhiDevice handle, VernonRhiBuffer buffer, void **output) {
    auto device = lookupMetalDevice(handle);
    if (!device || !output)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(device->mutex);
    MetalBufferSlot *slot = lookupPublicResource(device->buffers, buffer);
    if (!slot)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    *output = (__bridge void *)slot->native.buffer;
    return VERNON_RHI_STATUS_OK;
}

VernonRhiStatus createImage(VernonRhiDevice handle, const VernonRhiImageDescriptor *descriptor,
                            VernonRhiImage *output) {
    auto device = lookupMetalDevice(handle);
    if (!device)
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
    std::lock_guard<std::mutex> guard(device->mutex);
    const uint32_t index = allocateSlot(device->images);
    MetalImageSlot &slot = device->images[index];
    if (!device->state.createImage(slot.native, *descriptor, device->error))
        return VERNON_RHI_STATUS_INTERNAL_ERROR;
    slot.descriptor = *descriptor;
    slot.publish();
    *output = {index, slot.generation};
    return VERNON_RHI_STATUS_OK;
}

VernonRhiStatus createImageView(VernonRhiDevice handle, const VernonRhiImageViewDescriptor *descriptor,
                                VernonRhiImageView *output) {
    auto device = lookupMetalDevice(handle);
    if (!device)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    if (!descriptor || !output || descriptor->struct_size < sizeof(*descriptor) ||
        vernon::rhi::metal::pixelFormat(descriptor->format) == MTLPixelFormatInvalid ||
        descriptor->mip_level_count == 0 || descriptor->array_layer_count == 0 || descriptor->aspects == 0)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(device->mutex);
    MetalImageSlot *image = lookupPublicResource(device->images, descriptor->image);
    if (!image || !vernon::rhi::validImageViewDescriptor(image->descriptor, *descriptor))
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    if (!image->retain())
        return VERNON_RHI_STATUS_INTERNAL_ERROR;
    const uint32_t index = allocateSlot(device->imageViews);
    MetalImageViewSlot &slot = device->imageViews[index];
    if (!device->state.createImageView(slot.native, image->native, *descriptor, device->error)) {
        image->release();
        return VERNON_RHI_STATUS_UNSUPPORTED;
    }
    slot.descriptor = *descriptor;
    slot.imageResource = resourceKey(descriptor->image);
    slot.publish();
    *output = {index, slot.generation};
    return VERNON_RHI_STATUS_OK;
}

VernonRhiStatus destroyImageView(VernonRhiDevice handle, VernonRhiImageView imageView) {
    auto device = lookupMetalDevice(handle);
    if (!device)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(device->mutex);
    MetalImageViewSlot *slot = lookupPublicResource(device->imageViews, imageView);
    if (!slot)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    const uint64_t imageResource = slot->imageResource;
    slot->destroyPublicOwner();
    if (slot->bindingReferences != 0)
        return VERNON_RHI_STATUS_OK;
    device->state.destroyImageView(slot->native);
    slot->recycle();
    MetalImageSlot *image = lookupResourceRecord(device->images, imageResource);
    if (image && image->release()) {
        device->state.destroyImage(image->native);
        image->recycle();
    }
    return VERNON_RHI_STATUS_OK;
}

VernonRhiStatus getImageViewNativeHandle(VernonRhiDevice handle, VernonRhiImageView imageView, uint64_t *output) {
    auto device = lookupMetalDevice(handle);
    if (!device || !output)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(device->mutex);
    MetalImageViewSlot *slot = lookupPublicResource(device->imageViews, imageView);
    if (!slot)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    *output = reinterpret_cast<uintptr_t>((__bridge void *)slot->native.texture);
    return VERNON_RHI_STATUS_OK;
}

VernonRhiStatus uploadImage(VernonRhiDevice handle, VernonRhiImage image,
                            const VernonRhiImageUploadDescriptor *uploads, size_t uploadCount) {
    auto device = lookupMetalDevice(handle);
    if (!device)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    if (!uploads || uploadCount == 0)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(device->mutex);
    MetalImageSlot *slot = lookupPublicResource(device->images, image);
    if (!slot)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    for (size_t index = 0; index < uploadCount; ++index) {
        const auto &upload = uploads[index];
        if (upload.struct_size < sizeof(upload) || !upload.data || upload.mip_level >= slot->descriptor.mip_levels)
            return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    }
    return device->state.uploadImage(slot->native, slot->descriptor, uploads, uploadCount, device->error)
               ? VERNON_RHI_STATUS_OK
               : VERNON_RHI_STATUS_UNSUPPORTED;
}

VernonRhiStatus downloadImage(VernonRhiDevice handle, VernonRhiImage image,
                              const VernonRhiImageDownloadDescriptor *download, void *destination, size_t size) {
    auto device = lookupMetalDevice(handle);
    if (!device || !download || download->struct_size < sizeof(*download) || !destination)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(device->mutex);
    MetalImageSlot *slot = lookupPublicResource(device->images, image);
    if (!slot)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    const auto required = vernon::rhi::imageDownloadByteSize(slot->descriptor, *download);
    if (!required || size != *required)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    return device->state.downloadImage(slot->native, slot->descriptor, *download, destination, size, device->error)
               ? VERNON_RHI_STATUS_OK
               : VERNON_RHI_STATUS_INTERNAL_ERROR;
}

VernonRhiStatus generateImageMipmaps(VernonRhiDevice handle, VernonRhiImage image) {
    auto device = lookupMetalDevice(handle);
    if (!device)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(device->mutex);
    MetalImageSlot *slot = lookupPublicResource(device->images, image);
    if (!slot || slot->descriptor.mip_levels < 2 ||
        !(slot->descriptor.usage & VERNON_RHI_IMAGE_TRANSFER_SOURCE) ||
        !(slot->descriptor.usage & VERNON_RHI_IMAGE_TRANSFER_DESTINATION) ||
        slot->descriptor.format == VERNON_RHI_FORMAT_D32_FLOAT ||
        slot->descriptor.format == VERNON_RHI_FORMAT_D32_FLOAT_S8_UINT)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    return device->state.generateImageMipmaps(slot->native, slot->descriptor.mip_levels, device->error)
               ? VERNON_RHI_STATUS_OK
               : VERNON_RHI_STATUS_INTERNAL_ERROR;
}

VernonRhiStatus destroyImage(VernonRhiDevice handle, VernonRhiImage image) {
    auto device = lookupMetalDevice(handle);
    if (!device)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(device->mutex);
    MetalImageSlot *slot = lookupPublicResource(device->images, image);
    if (!slot)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    slot->destroyPublicOwner();
    if (slot->bindingReferences == 0) {
        device->state.destroyImage(slot->native);
        slot->recycle();
    }
    return VERNON_RHI_STATUS_OK;
}

uint32_t isImageValid(VernonRhiDevice handle, VernonRhiImage image) {
    auto device = lookupMetalDevice(handle);
    if (!device)
        return 0;
    std::lock_guard<std::mutex> guard(device->mutex);
    return lookupPublicResource(device->images, image) != nullptr;
}

VernonRhiStatus getImageNativeHandle(VernonRhiDevice handle, VernonRhiImage image, uint64_t *output) {
    auto device = lookupMetalDevice(handle);
    if (!device || !output)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(device->mutex);
    MetalImageSlot *slot = lookupPublicResource(device->images, image);
    if (!slot)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    *output = reinterpret_cast<uintptr_t>((__bridge void *)slot->native.texture);
    return VERNON_RHI_STATUS_OK;
}

VernonRhiStatus createSampler(VernonRhiDevice handle, const VernonRhiSamplerDescriptor *descriptor,
                              VernonRhiSampler *output) {
    auto device = lookupMetalDevice(handle);
    if (!device)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    vernon::rhi::SamplerFilter filter;
    if (!descriptor || !output || descriptor->struct_size < sizeof(*descriptor) ||
        !vernon::rhi::decodeSamplerFilter(*descriptor, filter))
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(device->mutex);
    const uint32_t index = allocateSlot(device->samplers);
    MetalSamplerSlot &slot = device->samplers[index];
    if (!device->state.createSampler(slot.native, *descriptor, device->error))
        return VERNON_RHI_STATUS_INTERNAL_ERROR;
    slot.publish();
    *output = {index, slot.generation};
    return VERNON_RHI_STATUS_OK;
}

VernonRhiStatus destroySampler(VernonRhiDevice handle, VernonRhiSampler sampler) {
    auto device = lookupMetalDevice(handle);
    if (!device)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(device->mutex);
    MetalSamplerSlot *slot = lookupPublicResource(device->samplers, sampler);
    if (!slot)
        return VERNON_RHI_STATUS_INVALID_ARGUMENT;
    slot->destroyPublicOwner();
    if (slot->bindingReferences == 0) {
        device->state.destroySampler(slot->native);
        slot->recycle();
    }
    return VERNON_RHI_STATUS_OK;
}

uint32_t isSamplerValid(VernonRhiDevice handle, VernonRhiSampler sampler) {
    auto device = lookupMetalDevice(handle);
    if (!device)
        return 0;
    std::lock_guard<std::mutex> guard(device->mutex);
    return lookupPublicResource(device->samplers, sampler) != nullptr;
}

uint64_t bufferResource(VernonRhiDevice handle, VernonRhiBuffer buffer) {
    auto device = lookupMetalDevice(handle);
    if (!device)
        return 0;
    std::lock_guard<std::mutex> guard(device->mutex);
    return lookupPublicResource(device->buffers, buffer) ? resourceKey(buffer) : 0;
}

uint64_t imageResource(VernonRhiDevice handle, VernonRhiImage image) {
    auto device = lookupMetalDevice(handle);
    if (!device)
        return 0;
    std::lock_guard<std::mutex> guard(device->mutex);
    return lookupPublicResource(device->images, image) ? resourceKey(image) : 0;
}

uint64_t imageViewResource(VernonRhiDevice handle, VernonRhiImageView view) {
    auto device = lookupMetalDevice(handle);
    if (!device)
        return 0;
    std::lock_guard<std::mutex> guard(device->mutex);
    return lookupPublicResource(device->imageViews, view) ? resourceKey(view) : 0;
}

uint64_t samplerResource(VernonRhiDevice handle, VernonRhiSampler sampler) {
    auto device = lookupMetalDevice(handle);
    if (!device)
        return 0;
    std::lock_guard<std::mutex> guard(device->mutex);
    return lookupPublicResource(device->samplers, sampler) ? resourceKey(sampler) : 0;
}

bool retainResource(VernonRhiDevice handle, ResourceKind kind, uint64_t key) {
    auto device = lookupMetalDevice(handle);
    if (!device)
        return false;
    std::lock_guard<std::mutex> guard(device->mutex);
    if (kind == ResourceKind::Buffer) {
        MetalBufferSlot *slot = lookupResourceRecord(device->buffers, key);
        return slot && slot->retain();
    }
    if (kind == ResourceKind::Image) {
        MetalImageSlot *slot = lookupResourceRecord(device->images, key);
        return slot && slot->retain();
    }
    if (kind == ResourceKind::ImageView) {
        MetalImageViewSlot *slot = lookupResourceRecord(device->imageViews, key);
        return slot && slot->retain();
    }
    MetalSamplerSlot *slot = lookupResourceRecord(device->samplers, key);
    return slot && slot->retain();
}

uint64_t resolveResource(VernonRhiDevice handle, ResourceKind kind, uint64_t key) {
    auto device = lookupMetalDevice(handle);
    if (!device)
        return 0;
    std::lock_guard<std::mutex> guard(device->mutex);
    if (kind == ResourceKind::Buffer) {
        MetalBufferSlot *slot = lookupResourceRecord(device->buffers, key);
        return slot ? reinterpret_cast<uintptr_t>((__bridge void *)slot->native.buffer) : 0;
    }
    if (kind == ResourceKind::Image) {
        MetalImageSlot *slot = lookupResourceRecord(device->images, key);
        return slot ? reinterpret_cast<uintptr_t>((__bridge void *)slot->native.texture) : 0;
    }
    if (kind == ResourceKind::ImageView) {
        MetalImageViewSlot *slot = lookupResourceRecord(device->imageViews, key);
        return slot ? reinterpret_cast<uintptr_t>((__bridge void *)slot->native.texture) : 0;
    }
    MetalSamplerSlot *slot = lookupResourceRecord(device->samplers, key);
    return slot ? reinterpret_cast<uintptr_t>((__bridge void *)slot->native.sampler) : 0;
}

bool describeImageResource(VernonRhiDevice handle, uint64_t key, VernonRhiImageDescriptor *descriptor) {
    auto device = lookupMetalDevice(handle);
    if (!device || !descriptor)
        return false;
    std::lock_guard<std::mutex> guard(device->mutex);
    const MetalImageSlot *slot = lookupResourceRecord(device->images, key);
    if (!slot || !slot->occupied)
        return false;
    *descriptor = slot->descriptor;
    return true;
}

bool describeImageViewResource(VernonRhiDevice handle, uint64_t key, VernonRhiImageViewDescriptor *view,
                               VernonRhiImageDescriptor *image, uint64_t *parentKey) {
    auto device = lookupMetalDevice(handle);
    if (!device || !view || !image || !parentKey)
        return false;
    std::lock_guard<std::mutex> guard(device->mutex);
    const MetalImageViewSlot *slot = lookupResourceRecord(device->imageViews, key);
    if (!slot)
        return false;
    const MetalImageSlot *parent = lookupResourceRecord(device->images, slot->imageResource);
    if (!parent)
        return false;
    *view = slot->descriptor;
    *image = parent->descriptor;
    *parentKey = slot->imageResource;
    return true;
}

void releaseResource(VernonRhiDevice handle, ResourceKind kind, uint64_t key) {
    auto device = lookupMetalDevice(handle);
    if (!device)
        return;
    std::lock_guard<std::mutex> guard(device->mutex);
    if (kind == ResourceKind::Buffer) {
        MetalBufferSlot *slot = lookupResourceRecord(device->buffers, key);
        if (slot && slot->release()) {
            device->state.destroyBuffer(slot->native);
            slot->recycle();
        }
    } else if (kind == ResourceKind::Image) {
        MetalImageSlot *slot = lookupResourceRecord(device->images, key);
        if (slot && slot->release()) {
            device->state.destroyImage(slot->native);
            slot->recycle();
        }
    } else if (kind == ResourceKind::ImageView) {
        MetalImageViewSlot *slot = lookupResourceRecord(device->imageViews, key);
        if (slot && slot->release()) {
            const uint64_t parentKey = slot->imageResource;
            device->state.destroyImageView(slot->native);
            slot->recycle();
            MetalImageSlot *parent = lookupResourceRecord(device->images, parentKey);
            if (parent && parent->release()) {
                device->state.destroyImage(parent->native);
                parent->recycle();
            }
        }
    } else {
        MetalSamplerSlot *slot = lookupResourceRecord(device->samplers, key);
        if (slot && slot->release()) {
            device->state.destroySampler(slot->native);
            slot->recycle();
        }
    }
}

bool beginCommands(VernonRhiDevice handle, uint64_t &native, VernonRhiBackend &backend) {
    auto device = lookupMetalDevice(handle);
    if (!device)
        return false;
    std::lock_guard<std::mutex> guard(device->mutex);
    if (!device->state.beginCommands(native, device->error))
        return false;
    backend = VERNON_RHI_BACKEND_METAL;
    return true;
}

bool submitCommands(VernonRhiDevice handle, uint64_t native, bool, bool &completed, bool &externalCompletion) {
    externalCompletion = false;
    auto device = lookupMetalDevice(handle);
    if (!device)
        return false;
    std::lock_guard<std::mutex> guard(device->mutex);
    completed = device->state.submitCommands(native, device->error);
    return completed;
}

void completeBorrowedCommands(VernonRhiDevice handle, uint64_t native) {
    auto device = lookupMetalDevice(handle);
    if (device) {
        std::lock_guard<std::mutex> guard(device->mutex);
        device->state.completeCommands(native);
    }
}

void abandonCommands(VernonRhiDevice handle, uint64_t native) {
    auto device = lookupMetalDevice(handle);
    if (device) {
        std::lock_guard<std::mutex> guard(device->mutex);
        device->state.abandonCommands(native);
    }
}

bool recordBarriers(VernonRhiDevice handle, uint64_t, uint64_t native, const VernonRhiBarrier *barriers,
                    size_t barrierCount) {
    auto device = lookupMetalDevice(handle);
    if (!device || !native || (barrierCount && !barriers))
        return false;
    std::lock_guard<std::mutex> guard(device->mutex);
    for (size_t index = 0; index < barrierCount; ++index) {
        const VernonRhiBarrier &barrier = barriers[index];
        if (barrier.is_image) {
            MetalImageSlot *slot = lookupResourceRecord(device->images, resourceKey(barrier.image));
            if (!slot || !supportsImageState(slot->descriptor, barrier.old_state) ||
                !supportsImageState(slot->descriptor, barrier.new_state))
                return false;
        } else {
            MetalBufferSlot *slot = lookupResourceRecord(device->buffers, resourceKey(barrier.buffer));
            if (!slot || !supportsBufferState(slot->descriptor, barrier.old_state) ||
                !supportsBufferState(slot->descriptor, barrier.new_state))
                return false;
        }
    }
    // Vernon barriers occur between provider encoders. All owned Metal resources use
    // hazard-tracked storage, so Metal orders their reads and writes without an
    // explicit fence or resource-state transition.
    return true;
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
    if (!lookupMetalDevice(handle) || backend != VERNON_RHI_BACKEND_METAL ||
        backendKind != vernon::rhi::CommandRenderingDynamic || !renderingObject)
        return false;
    auto *rendering = reinterpret_cast<vernon::rhi::metal::RenderingState *>(
        static_cast<uintptr_t>(renderingObject));
    vernon::rhi::metal::unregisterRenderingState(native, rendering);
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

bool clearColor(VernonRhiDevice handle, uint64_t native, VernonRhiBackend backend, uint32_t backendKind, int32_t x,
                int32_t y, uint32_t width, uint32_t height, uint32_t layers, uint64_t target, uint32_t location,
                const float color[4]) {
    if (!lookupMetalDevice(handle) || backend != VERNON_RHI_BACKEND_METAL ||
        backendKind != vernon::rhi::CommandRenderingDynamic || !target || location >= 8 || !color)
        return false;
    id<MTLTexture> texture = (__bridge id<MTLTexture>)(reinterpret_cast<void *>(static_cast<uintptr_t>(target)));
    auto *rendering = vernon::rhi::metal::findRenderingState(native);
    return rendering && rendering->colorTextures[location] == texture &&
           restartRendering(native, *rendering, x, y, width, height, layers, static_cast<int32_t>(location), color,
                            1.0f, 0, 0);
}

bool clearDepthStencil(VernonRhiDevice handle, uint64_t native, VernonRhiBackend backend, uint32_t backendKind,
                       int32_t x, int32_t y, uint32_t width, uint32_t height, uint32_t layers, uint64_t target,
                       float depth, uint32_t stencil, uint32_t aspects) {
    if (!lookupMetalDevice(handle) || backend != VERNON_RHI_BACKEND_METAL ||
        backendKind != vernon::rhi::CommandRenderingDynamic || !target ||
        (aspects & ~(VERNON_RHI_ATTACHMENT_DEPTH | VERNON_RHI_ATTACHMENT_STENCIL)) || !aspects)
        return false;
    id<MTLTexture> texture = (__bridge id<MTLTexture>)(reinterpret_cast<void *>(static_cast<uintptr_t>(target)));
    auto *rendering = vernon::rhi::metal::findRenderingState(native);
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
        nullptr,
        uploadImage,
        downloadImage,
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
