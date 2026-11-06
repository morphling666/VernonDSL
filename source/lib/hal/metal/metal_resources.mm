#include "hal/metal/metal_resources.h"

#include "VernonFormats.hpp"
#include "hal/metal/metal_native.h"
#include "hal/metal/metal_state.h"
#include "hal/native_buffers.h"
#include "hal/native_images.h"
#include "hal/resources.h"

#import <Foundation/Foundation.h>
#import <TargetConditionals.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>
#include <new>
#include <string>
#include <utility>
#include <vector>

namespace vernon::hal {
namespace {

using metal::DeviceState;

Serials<metal::Buffer> &serials() {
    static Serials<metal::Buffer> instance;
    return instance;
}

template <typename T = void> Done<T> internal(const char *what, const std::string &detail) {
    return fail<T>(VERNON_HAL_INTERNAL_ERROR, detail.empty() ? std::string(what) : std::string(what) + ": " + detail);
}

uint32_t mipExtent(uint32_t size, uint32_t level) { return (std::max)(size >> level, 1u); }

size_t alignUp(size_t value, size_t alignment) { return (value + alignment - 1) & ~(alignment - 1); }

// Staged rows of image copies; Metal needs no more than 256 on any device.
constexpr size_t kRowAlignment = 256;

// macOS blits copy buffer ranges in multiples of 4 bytes.
bool copyable(uint64_t offset, uint64_t size) {
#if TARGET_OS_OSX
    return offset % 4 == 0 && size % 4 == 0;
#else
    (void)offset;
    (void)size;
    return true;
#endif
}

bool hostVisible(VernonHalMemory memory) { return memory != VERNON_HAL_MEMORY_DEVICE; }

MTLResourceOptions bufferOptions(VernonHalMemory memory) {
    switch (memory) {
    case VERNON_HAL_MEMORY_UPLOAD:
        return MTLResourceStorageModeShared | MTLResourceCPUCacheModeWriteCombined;
    case VERNON_HAL_MEMORY_READBACK:
        return MTLResourceStorageModeShared;
    case VERNON_HAL_MEMORY_DEVICE:
        break;
    }
    return MTLResourceStorageModePrivate;
}

bool createBuffer(DeviceState &state, metal::Buffer &buffer, uint64_t size, VernonHalMemory memory,
                  std::string &error) {
    if (size > state.native->device.maxBufferLength) {
        error = "the buffer exceeds the device's largest buffer";
        return false;
    }
    buffer.buffer =
        [state.native->device newBufferWithLength:(std::max<uint64_t>)(size, 1) options:bufferOptions(memory)];
    buffer.size = size;
    buffer.memory = memory;
    if (!buffer.buffer) {
        error = "Metal cannot allocate the buffer";
        return false;
    }
    return true;
}

// An upload buffer is written in place, as on every backend; other buffers through a staged blit, which the queue
// orders after every earlier submission.
Result<void, std::string> writeBuffer(DeviceState &state, metal::Buffer &buffer, const VernonHalBufferWrite *writes,
                                      size_t count) {
    using Written = Result<void, std::string>;
    if (buffer.memory == VERNON_HAL_MEMORY_UPLOAD) {
        auto *mapped = static_cast<uint8_t *>(buffer.buffer.contents);
        for (size_t index = 0; index < count; ++index)
            std::memcpy(mapped + writes[index].offset, writes[index].source, static_cast<size_t>(writes[index].size));
        return Written{ok()};
    }
    std::vector<size_t> offsets(count);
    size_t total = 0;
    for (size_t index = 0; index < count; ++index) {
        if (!copyable(writes[index].offset, writes[index].size))
            return Written{err(std::string("Metal on macOS writes buffers in multiples of 4 bytes"))};
        offsets[index] = alignUp(total, 16);
        total = offsets[index] + static_cast<size_t>(writes[index].size);
    }
    if (!total)
        return Written{ok()};
    @autoreleasepool {
        std::lock_guard<std::mutex> guard(state.mutex);
        metal::Native &native = *state.native;
        id<MTLBuffer> staging =
            [native.device newBufferWithLength:total
                                       options:MTLResourceStorageModeShared | MTLResourceCPUCacheModeWriteCombined];
        id<MTLCommandBuffer> commands = [native.queue commandBuffer];
        id<MTLBlitCommandEncoder> blit = [commands blitCommandEncoder];
        if (!staging || !blit)
            return Written{err(std::string("cannot stage the write"))};
        auto *mapped = static_cast<uint8_t *>(staging.contents);
        for (size_t index = 0; index < count; ++index) {
            std::memcpy(mapped + offsets[index], writes[index].source, static_cast<size_t>(writes[index].size));
            [blit copyFromBuffer:staging
                     sourceOffset:offsets[index]
                         toBuffer:buffer.buffer
                destinationOffset:writes[index].offset
                             size:writes[index].size];
        }
        [blit endEncoding];
        metal::commit(state, commands);
    }
    return Written{ok()};
}

Result<void, std::string> readBuffer(DeviceState &state, metal::Buffer &buffer, const VernonHalBufferRead *reads,
                                     size_t count) {
    using Read = Result<void, std::string>;
    std::lock_guard<std::mutex> guard(state.mutex);
    std::string error;
    // Once the queue drains, a host-visible buffer's contents hold what was written to it.
    if (hostVisible(buffer.memory)) {
        if (!state.synchronize(error))
            return Read{err(std::move(error))};
        const auto *mapped = static_cast<const uint8_t *>(buffer.buffer.contents);
        for (size_t index = 0; index < count; ++index)
            std::memcpy(reads[index].destination, mapped + reads[index].offset, static_cast<size_t>(reads[index].size));
        return Read{ok()};
    }
    std::vector<size_t> offsets(count);
    size_t total = 0;
    for (size_t index = 0; index < count; ++index) {
        if (!copyable(reads[index].offset, reads[index].size))
            return Read{err(std::string("Metal on macOS reads buffers in multiples of 4 bytes"))};
        offsets[index] = alignUp(total, 16);
        total = offsets[index] + static_cast<size_t>(reads[index].size);
    }
    if (!total)
        return Read{ok()};
    @autoreleasepool {
        metal::Native &native = *state.native;
        id<MTLBuffer> staging = [native.device newBufferWithLength:total options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> commands = [native.queue commandBuffer];
        id<MTLBlitCommandEncoder> blit = [commands blitCommandEncoder];
        if (!staging || !blit)
            return Read{err(std::string("cannot stage the read"))};
        for (size_t index = 0; index < count; ++index)
            [blit copyFromBuffer:buffer.buffer
                     sourceOffset:reads[index].offset
                         toBuffer:staging
                destinationOffset:offsets[index]
                             size:reads[index].size];
        [blit endEncoding];
        if (!state.waitForValue(metal::commit(state, commands), error))
            return Read{err(std::move(error))};
        const auto *mapped = static_cast<const uint8_t *>(staging.contents);
        for (size_t index = 0; index < count; ++index)
            std::memcpy(reads[index].destination, mapped + offsets[index], static_cast<size_t>(reads[index].size));
    }
    return Read{ok()};
}

// One plane of an image transfer, staged at `offset` in rows of `rowPitch` bytes. Plane 1 is the stencil of a
// depth-stencil image, which Metal copies apart from its depth.
struct PlaneCopy {
    size_t transfer{};
    uint32_t plane{};
    uint32_t texelBytes{};
    size_t offset{};
    size_t rowPitch{};
    size_t imagePitch{};
};

// Lays out the plane copies of `regions` from offset 0 of a staging buffer; false with `unsupported` when Metal cannot
// copy a region, else when a region's data layout does not match the image.
bool planCopies(const VernonImageDescriptor &descriptor, const VernonHalImageRegion *const *regions, size_t count,
                bool writes, std::vector<PlaneCopy> &copies, size_t &total, bool &unsupported) {
    unsupported = false;
    total = 0;
    const bool depth = formats::hasDepth(descriptor.format);
    for (size_t index = 0; index < count; ++index) {
        const VernonHalImageRegion &region = *regions[index];
        const auto bytes = regionBytes(descriptor, region);
        if (!bytes)
            return false;
        if (depth && writes &&
            (region.offset_x || region.offset_y ||
             region.width != mipExtent(descriptor.extent.width, region.mip_level) ||
             region.height != mipExtent(descriptor.extent.height, region.mip_level))) {
            unsupported = true;
            return false;
        }
        const size_t texels = size_t{region.width} * region.height * region.depth;
        uint32_t planes[2]{};
        uint32_t planeCount = 0;
        if (region.aspect & (VERNON_IMAGE_ASPECT_COLOR | VERNON_IMAGE_ASPECT_DEPTH))
            planes[planeCount++] = 0;
        if (region.aspect & VERNON_IMAGE_ASPECT_STENCIL)
            planes[planeCount++] = 1;
        for (uint32_t which = 0; which < planeCount; ++which) {
            PlaneCopy copy;
            copy.transfer = index;
            copy.plane = planes[which];
            copy.texelBytes = !depth ? static_cast<uint32_t>(*bytes / texels) : copy.plane == 0 ? 4u : 1u;
            copy.rowPitch = alignUp(size_t{region.width} * copy.texelBytes, kRowAlignment);
            copy.imagePitch = copy.rowPitch * region.height;
            total = alignUp(total, kRowAlignment);
            copy.offset = total;
            total += copy.imagePitch * region.depth;
            copies.push_back(copy);
        }
    }
    return true;
}

// Copies the region's texels of one plane between the host data, whose texels pack depth and stencil into
// kPackedDepthStencilBytes when the region transfers both, and the staged rows at `staged`.
void moveTexels(const PlaneCopy &copy, const VernonHalImageRegion &region, uint8_t *staged, uint8_t *host,
                bool toStaging) {
    const bool packed = region.aspect == (VERNON_IMAGE_ASPECT_DEPTH | VERNON_IMAGE_ASPECT_STENCIL);
    const size_t hostStride = packed ? kPackedDepthStencilBytes : copy.texelBytes;
    const size_t hostOffset = packed && copy.plane == 1 ? sizeof(float) : 0;
    for (uint32_t z = 0; z < region.depth; ++z)
        for (uint32_t y = 0; y < region.height; ++y) {
            uint8_t *row = staged + copy.offset + z * copy.imagePitch + y * copy.rowPitch;
            uint8_t *hostRow = host + ((size_t{z} * region.height + y) * region.width) * hostStride + hostOffset;
            if (hostStride == copy.texelBytes) {
                const size_t bytes = size_t{region.width} * copy.texelBytes;
                toStaging ? std::memcpy(row, hostRow, bytes) : std::memcpy(hostRow, row, bytes);
                continue;
            }
            for (uint32_t x = 0; x < region.width; ++x) {
                uint8_t *texel = row + size_t{x} * copy.texelBytes;
                uint8_t *hostTexel = hostRow + x * hostStride;
                toStaging ? std::memcpy(texel, hostTexel, copy.texelBytes)
                          : std::memcpy(hostTexel, texel, copy.texelBytes);
            }
        }
}

MTLBlitOption planeOption(const VernonImageDescriptor &descriptor, uint32_t plane) {
    if (!formats::hasStencil(descriptor.format))
        return MTLBlitOptionNone;
    return plane == 0 ? MTLBlitOptionDepthFromDepthStencil : MTLBlitOptionStencilFromDepthStencil;
}

MTLTextureType textureType(VernonTextureDimension dimension, uint32_t layers) {
    switch (dimension) {
    case VERNON_TEXTURE_3D:
        return MTLTextureType3D;
    case VERNON_TEXTURE_CUBE:
        return layers > 6 ? MTLTextureTypeCubeArray : MTLTextureTypeCube;
    case VERNON_TEXTURE_CUBE_ARRAY:
        return MTLTextureTypeCubeArray;
    case VERNON_TEXTURE_2D_ARRAY:
        return MTLTextureType2DArray;
    case VERNON_TEXTURE_2D:
        break;
    }
    return layers > 1 ? MTLTextureType2DArray : MTLTextureType2D;
}

// One Metal buffer, named by its serial.
class NativeBuffer {
public:
    NativeBuffer(std::shared_ptr<DeviceState> state, metal::Buffer buffer)
        : state_(std::move(state)), identity_(nativeBufferIdentity(state_.get())), buffer_(std::move(buffer)) {
        serial_ = serials().add(&buffer_);
    }
    NativeBuffer(const NativeBuffer &) = delete;
    NativeBuffer &operator=(const NativeBuffer &) = delete;
    ~NativeBuffer() {
        serials().remove(serial_);
        std::lock_guard<std::mutex> guard(state_->mutex);
        metal::retain(*state_, buffer_.buffer);
    }

    metal::Buffer &buffer() { return buffer_; }
    uint64_t size() const { return buffer_.size; }
    VernonRuntimeProviderResourceReference reference(uint64_t offset, uint64_t size) const {
        VernonRuntimeProviderResourceReference reference{};
        reference.identity = identity_;
        reference.resource.value = serial_;
        reference.offset = offset;
        reference.size = size;
        return reference;
    }

private:
    std::shared_ptr<DeviceState> state_;
    uint64_t identity_;
    metal::Buffer buffer_;
    uint64_t serial_{};
};

struct MetalHalBuffer final : VernonHalBuffer {
    MetalHalBuffer(std::shared_ptr<DeviceState> state, metal::Buffer buffer)
        : native(std::move(state), std::move(buffer)) {}

    NativeBuffer native;
};

struct MetalImage final : VernonHalImage {
    explicit MetalImage(std::shared_ptr<DeviceState> state) : state(std::move(state)) {}
    MetalImage(const MetalImage &) = delete;
    MetalImage &operator=(const MetalImage &) = delete;
    ~MetalImage() override {
        forgetUndefined(&serial, 1);
        std::lock_guard<std::mutex> guard(state->mutex);
        metal::retain(*state, native.texture);
    }

    std::shared_ptr<DeviceState> state;
    metal::Image native;
    uint64_t serial{nextImageSerial()};
};

struct MetalView final : VernonHalImageView {
    explicit MetalView(std::shared_ptr<DeviceState> state) : state(std::move(state)) {}
    MetalView(const MetalView &) = delete;
    MetalView &operator=(const MetalView &) = delete;
    ~MetalView() override {
        if (serial)
            nativeViews().remove(serial);
        std::lock_guard<std::mutex> guard(state->mutex);
        metal::retain(*state, binding.texture);
    }

    std::shared_ptr<DeviceState> state;
    metal::View binding;
    NativeView entry;
    uint64_t serial{};
};

struct MetalSampler final : VernonHalSampler {
    explicit MetalSampler(std::shared_ptr<DeviceState> state) : state(std::move(state)) {}
    MetalSampler(const MetalSampler &) = delete;
    MetalSampler &operator=(const MetalSampler &) = delete;
    ~MetalSampler() override {
        if (serial)
            nativeSamplers().remove(serial);
        std::lock_guard<std::mutex> guard(state->mutex);
        metal::retain(*state, sampler.sampler);
    }

    std::shared_ptr<DeviceState> state;
    metal::Sampler sampler;
    uint64_t serial{};
};

// A layer and the ring of images that passes render to and presents copy into the layer's drawables, whose formats
// are BGRA: the images are the surface's own, so a frame acquires one without waiting for a drawable.
struct MetalSurface final : VernonHalSurface {
    explicit MetalSurface(std::shared_ptr<DeviceState> state) : state(std::move(state)) {}
    MetalSurface(const MetalSurface &) = delete;
    MetalSurface &operator=(const MetalSurface &) = delete;
    ~MetalSurface() override {
        std::lock_guard<std::mutex> guard(state->mutex);
        std::string ignored;
        (void)state->synchronize(ignored);
    }

    std::shared_ptr<DeviceState> state;
    CAMetalLayer *layer{};
    VernonHalSurfaceConfiguration configuration{};
    // Index-aligned; each view covers its image.
    std::vector<std::unique_ptr<MetalImage>> images;
    std::vector<std::unique_ptr<MetalView>> views;
    size_t next{};
};

// The layer's extent in pixels; zero while it has no bounds, as before the host lays it out.
void layerExtent(CAMetalLayer *layer, uint32_t &width, uint32_t &height) {
    const CGSize size = layer.bounds.size;
    const CGFloat scale = layer.contentsScale;
    width = static_cast<uint32_t>(std::lround((std::max)(size.width * scale, 0.0)));
    height = static_cast<uint32_t>(std::lround((std::max)(size.height * scale, 0.0)));
}

class MetalDeviceBuffers final : public DeviceBuffers {
public:
    explicit MetalDeviceBuffers(std::shared_ptr<DeviceState> state)
        : state_(state), pool_(std::make_shared<BufferPool<NativeBuffer>>([state](uint64_t size) {
              using Created = Result<std::unique_ptr<NativeBuffer>, std::string>;
              metal::Buffer native;
              std::string error;
              {
                  std::lock_guard<std::mutex> guard(state->mutex);
                  if (!createBuffer(*state, native, size, VERNON_HAL_MEMORY_DEVICE, error))
                      return Created{err("cannot create a Metal buffer: " + error)};
              }
              return Created{ok(std::make_unique<NativeBuffer>(state, std::move(native)))};
          })) {}

    Result<std::unique_ptr<Buffer>, std::string> allocate(uint64_t size) override { return pool_->allocate(size); }

    Result<void, std::string> upload(Buffer &buffer, const void *bytes, size_t size) override {
        NativeBuffer &native = static_cast<BufferPool<NativeBuffer>::Pooled &>(buffer).native();
        if (size > native.size())
            return Result<void, std::string>{err(std::string("the upload overruns the buffer"))};
        const size_t whole = alignUp(size, 4);
        if (whole == size || whole > native.size()) {
            const VernonHalBufferWrite write{0, bytes, size};
            return writeBuffer(*state_, native.buffer(), &write, 1);
        }
        // macOS copies whole words; the buffer's bytes past the upload are undefined anyway.
        std::vector<uint8_t> padded(whole);
        std::memcpy(padded.data(), bytes, size);
        const VernonHalBufferWrite write{0, padded.data(), whole};
        return writeBuffer(*state_, native.buffer(), &write, 1);
    }

    Result<void, std::string> download(const Buffer &buffer, void *bytes, size_t size) override {
        NativeBuffer &native = static_cast<const BufferPool<NativeBuffer>::Pooled &>(buffer).native();
        if (size > native.size())
            return Result<void, std::string>{err(std::string("the download overruns the buffer"))};
        const size_t whole = alignUp(size, 4);
        if (whole == size || whole > native.size()) {
            const VernonHalBufferRead read{0, bytes, size};
            return readBuffer(*state_, native.buffer(), &read, 1);
        }
        std::vector<uint8_t> padded(whole);
        const VernonHalBufferRead read{0, padded.data(), whole};
        auto done = readBuffer(*state_, native.buffer(), &read, 1);
        if (done.isOk())
            std::memcpy(bytes, padded.data(), size);
        return done;
    }

private:
    std::shared_ptr<DeviceState> state_;
    std::shared_ptr<BufferPool<NativeBuffer>> pool_;
};

class MetalResources final : public Resources {
public:
    explicit MetalResources(std::shared_ptr<DeviceState> state) : owner_(std::move(state)), state_(*owner_) {}

    Done<uint32_t> formatCapabilities(VernonTextureFormat format) override {
        const formats::Format *entry = formats::find(format);
        if (!entry)
            return fail<uint32_t>(VERNON_HAL_INVALID_ARGUMENT, "the format is unknown");
        return Done<uint32_t>{ok(capabilities(format))};
    }

    Done<std::unique_ptr<VernonHalBuffer>> createBuffer(const VernonHalBufferDescriptor &descriptor) override {
        using Created = Done<std::unique_ptr<VernonHalBuffer>>;
        metal::Buffer native;
        std::string error;
        {
            std::lock_guard<std::mutex> guard(state_.mutex);
            if (!hal::createBuffer(state_, native, descriptor.size, descriptor.memory, error))
                return internal<std::unique_ptr<VernonHalBuffer>>("cannot create the buffer", error);
        }
        auto buffer = std::unique_ptr<MetalHalBuffer>(new (std::nothrow) MetalHalBuffer(owner_, std::move(native)));
        if (!buffer)
            return fail<std::unique_ptr<VernonHalBuffer>>(VERNON_HAL_RESOURCE_EXHAUSTED, "cannot allocate a buffer");
        return Created{ok(std::unique_ptr<VernonHalBuffer>(std::move(buffer)))};
    }

    Done<void> write(VernonHalBuffer &buffer, const VernonHalBufferWrite *writes, size_t count) override {
        auto written = writeBuffer(state_, static_cast<MetalHalBuffer &>(buffer).native.buffer(), writes, count);
        if (written.isErr())
            return internal("cannot write the buffer", written.error());
        return Done<void>{ok()};
    }

    Done<void> read(VernonHalBuffer &buffer, const VernonHalBufferRead *reads, size_t count) override {
        auto read = readBuffer(state_, static_cast<MetalHalBuffer &>(buffer).native.buffer(), reads, count);
        if (read.isErr())
            return internal("cannot read the buffer", read.error());
        return Done<void>{ok()};
    }

    Done<void *> address(const VernonHalBuffer &) override {
        return fail<void *>(VERNON_HAL_UNSUPPORTED, "only CPU and CUDA buffers have an address");
    }

    Done<std::unique_ptr<VernonHalImage>> createImage(const VernonImageDescriptor &descriptor) override {
        auto made = makeImage(descriptor, 0);
        if (made.isErr())
            return Done<std::unique_ptr<VernonHalImage>>{err(std::move(made).error())};
        return Done<std::unique_ptr<VernonHalImage>>{ok(std::unique_ptr<VernonHalImage>(std::move(made).value()))};
    }

    Done<void> write(VernonHalImage &image, const VernonHalImageWrite *writes, size_t count) override {
        if (!count)
            return Done<void>{ok()};
        auto &target = static_cast<MetalImage &>(image);
        std::vector<const VernonHalImageRegion *> regions(count);
        for (size_t index = 0; index < count; ++index)
            regions[index] = &writes[index].region;
        std::vector<PlaneCopy> copies;
        size_t total = 0;
        bool unsupported = false;
        if (!planCopies(image.descriptor, regions.data(), count, true, copies, total, unsupported))
            return unsupported ? fail(VERNON_HAL_UNSUPPORTED, "Metal writes whole depth-stencil mip levels")
                               : fail(VERNON_HAL_INVALID_ARGUMENT, "a write's data layout does not match the image");
        const bool volume = image.descriptor.dimension == VERNON_TEXTURE_3D;
        @autoreleasepool {
            std::lock_guard<std::mutex> guard(state_.mutex);
            metal::Native &native = *state_.native;
            id<MTLBuffer> staging =
                [native.device newBufferWithLength:(std::max<size_t>)(total, 1)
                                           options:MTLResourceStorageModeShared | MTLResourceCPUCacheModeWriteCombined];
            id<MTLCommandBuffer> commands = [native.queue commandBuffer];
            id<MTLBlitCommandEncoder> blit = [commands blitCommandEncoder];
            if (!staging || !blit)
                return internal("cannot stage the image's write", {});
            auto *mapped = static_cast<uint8_t *>(staging.contents);
            for (const PlaneCopy &copy : copies) {
                const VernonHalImageRegion &region = writes[copy.transfer].region;
                moveTexels(copy, region, mapped,
                           const_cast<uint8_t *>(static_cast<const uint8_t *>(writes[copy.transfer].source)), true);
                [blit copyFromBuffer:staging
                           sourceOffset:copy.offset
                      sourceBytesPerRow:copy.rowPitch
                    sourceBytesPerImage:copy.imagePitch
                             sourceSize:MTLSizeMake(region.width, region.height, region.depth)
                              toTexture:target.native.texture
                       destinationSlice:volume ? 0 : region.array_layer
                       destinationLevel:region.mip_level
                      destinationOrigin:MTLOriginMake(region.offset_x, region.offset_y, region.offset_z)
                                options:planeOption(image.descriptor, copy.plane)];
            }
            [blit endEncoding];
            metal::commit(state_, commands);
        }
        forgetUndefined(&target.serial, 1);
        return Done<void>{ok()};
    }

    Done<void> read(VernonHalImage &image, const VernonHalImageRead *reads, size_t count) override {
        if (!count)
            return Done<void>{ok()};
        auto &source = static_cast<MetalImage &>(image);
        if (anyUndefined(&source.serial, 1))
            return fail(VERNON_HAL_INVALID_ARGUMENT, "the image's contents are undefined");
        std::vector<const VernonHalImageRegion *> regions(count);
        for (size_t index = 0; index < count; ++index) {
            const auto bytes = regionBytes(image.descriptor, reads[index].region);
            if (!bytes || *bytes != reads[index].size)
                return fail(VERNON_HAL_INVALID_ARGUMENT, "a read's size or data layout does not match the image");
            regions[index] = &reads[index].region;
        }
        std::vector<PlaneCopy> copies;
        size_t total = 0;
        bool unsupported = false;
        if (!planCopies(image.descriptor, regions.data(), count, false, copies, total, unsupported))
            return fail(VERNON_HAL_INVALID_ARGUMENT, "a read's data layout does not match the image");
        const bool volume = image.descriptor.dimension == VERNON_TEXTURE_3D;
        @autoreleasepool {
            std::lock_guard<std::mutex> guard(state_.mutex);
            metal::Native &native = *state_.native;
            id<MTLBuffer> staging =
                [native.device newBufferWithLength:(std::max<size_t>)(total, 1) options:MTLResourceStorageModeShared];
            id<MTLCommandBuffer> commands = [native.queue commandBuffer];
            id<MTLBlitCommandEncoder> blit = [commands blitCommandEncoder];
            if (!staging || !blit)
                return internal("cannot stage the image's read", {});
            for (const PlaneCopy &copy : copies) {
                const VernonHalImageRegion &region = reads[copy.transfer].region;
                [blit copyFromTexture:source.native.texture
                                 sourceSlice:volume ? 0 : region.array_layer
                                 sourceLevel:region.mip_level
                                sourceOrigin:MTLOriginMake(region.offset_x, region.offset_y, region.offset_z)
                                  sourceSize:MTLSizeMake(region.width, region.height, region.depth)
                                    toBuffer:staging
                           destinationOffset:copy.offset
                      destinationBytesPerRow:copy.rowPitch
                    destinationBytesPerImage:copy.imagePitch
                                     options:planeOption(image.descriptor, copy.plane)];
            }
            [blit endEncoding];
            std::string error;
            // The host waits for this read only, not for later submissions.
            if (!state_.waitForValue(metal::commit(state_, commands), error))
                return internal("cannot read the image", error);
            for (size_t index = 0; index < count; ++index)
                if (reads[index].region.aspect == (VERNON_IMAGE_ASPECT_DEPTH | VERNON_IMAGE_ASPECT_STENCIL))
                    std::memset(reads[index].destination, 0, reads[index].size);
            auto *mapped = static_cast<uint8_t *>(staging.contents);
            for (const PlaneCopy &copy : copies)
                moveTexels(copy, reads[copy.transfer].region, mapped,
                           static_cast<uint8_t *>(reads[copy.transfer].destination), false);
        }
        return Done<void>{ok()};
    }

    Done<void> generateMipmaps(VernonHalImage &image) override {
        const VernonImageDescriptor &descriptor = image.descriptor;
        if (descriptor.mip_level_count < 2 || formats::hasDepth(descriptor.format) ||
            !(descriptor.usage & VERNON_IMAGE_TRANSFER_SOURCE) ||
            !(descriptor.usage & VERNON_IMAGE_TRANSFER_DESTINATION))
            return fail(VERNON_HAL_INVALID_ARGUMENT,
                        "mipmaps need a color image with several mips that transfers in both directions");
        // A blit filters only color-renderable, filterable formats.
        const uint32_t needed = VERNON_HAL_FORMAT_RENDERABLE | VERNON_HAL_FORMAT_FILTERABLE;
        if ((capabilities(descriptor.format) & needed) != needed)
            return fail(VERNON_HAL_UNSUPPORTED, "the device cannot filter the image's mips");
        @autoreleasepool {
            std::lock_guard<std::mutex> guard(state_.mutex);
            id<MTLCommandBuffer> commands = [state_.native->queue commandBuffer];
            id<MTLBlitCommandEncoder> blit = [commands blitCommandEncoder];
            if (!blit)
                return internal("cannot generate the image's mipmaps", {});
            [blit generateMipmapsForTexture:static_cast<MetalImage &>(image).native.texture];
            [blit endEncoding];
            metal::commit(state_, commands);
        }
        return Done<void>{ok()};
    }

    Done<std::unique_ptr<VernonHalImageView>> createView(VernonHalImage &image,
                                                         const VernonImageViewDescriptor &descriptor) override {
        auto made = makeView(static_cast<MetalImage &>(image), descriptor);
        if (made.isErr())
            return Done<std::unique_ptr<VernonHalImageView>>{err(std::move(made).error())};
        return Done<std::unique_ptr<VernonHalImageView>>{
            ok(std::unique_ptr<VernonHalImageView>(std::move(made).value()))};
    }

    Done<std::unique_ptr<VernonHalSampler>> createSampler(const VernonHalSamplerDescriptor &descriptor) override {
        using Created = Done<std::unique_ptr<VernonHalSampler>>;
        if (descriptor.min_filter > VERNON_HAL_FILTER_LINEAR || descriptor.mag_filter > VERNON_HAL_FILTER_LINEAR ||
            descriptor.mip_filter > VERNON_HAL_FILTER_LINEAR ||
            descriptor.address_u > VERNON_HAL_ADDRESS_MIRRORED_REPEAT ||
            descriptor.address_v > VERNON_HAL_ADDRESS_MIRRORED_REPEAT ||
            descriptor.address_w > VERNON_HAL_ADDRESS_MIRRORED_REPEAT || !(descriptor.max_anisotropy <= 16.0f))
            return fail<std::unique_ptr<VernonHalSampler>>(VERNON_HAL_INVALID_ARGUMENT,
                                                           "the sampler descriptor is invalid");
        constexpr MTLSamplerAddressMode kAddress[] = {MTLSamplerAddressModeRepeat, MTLSamplerAddressModeClampToEdge,
                                                      MTLSamplerAddressModeMirrorRepeat};
        const auto filter = [](VernonHalFilter value) {
            return value == VERNON_HAL_FILTER_LINEAR ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;
        };
        auto sampler = std::unique_ptr<MetalSampler>(new (std::nothrow) MetalSampler(owner_));
        if (!sampler)
            return fail<std::unique_ptr<VernonHalSampler>>(VERNON_HAL_RESOURCE_EXHAUSTED, "cannot allocate a sampler");
        @autoreleasepool {
            MTLSamplerDescriptor *native = [[MTLSamplerDescriptor alloc] init];
            native.minFilter = filter(descriptor.min_filter);
            native.magFilter = filter(descriptor.mag_filter);
            native.mipFilter = descriptor.mip_filter == VERNON_HAL_FILTER_LINEAR ? MTLSamplerMipFilterLinear
                                                                                 : MTLSamplerMipFilterNearest;
            native.sAddressMode = kAddress[descriptor.address_u];
            native.tAddressMode = kAddress[descriptor.address_v];
            native.rAddressMode = kAddress[descriptor.address_w];
            native.maxAnisotropy =
                descriptor.max_anisotropy > 1.0f ? static_cast<NSUInteger>(std::ceil(descriptor.max_anisotropy)) : 1;
            native.lodMinClamp = 0;
            native.lodMaxClamp = std::numeric_limits<float>::max();
            native.supportArgumentBuffers = YES;
            sampler->sampler.sampler = [state_.native->device newSamplerStateWithDescriptor:native];
        }
        if (!sampler->sampler.sampler)
            return internal<std::unique_ptr<VernonHalSampler>>("cannot create the sampler", {});
        sampler->serial = nativeSamplers().add(&sampler->sampler);
        return Created{ok(std::unique_ptr<VernonHalSampler>(std::move(sampler)))};
    }

    Done<std::unique_ptr<VernonHalSurface>> createSurface(const VernonHalSurfaceDescriptor &descriptor) override {
        using Created = Done<std::unique_ptr<VernonHalSurface>>;
        id layer = (__bridge id)descriptor.native.metal_layer;
        if (!layer || ![layer isKindOfClass:[CAMetalLayer class]])
            return fail<std::unique_ptr<VernonHalSurface>>(VERNON_HAL_INVALID_ARGUMENT, "the layer is no CAMetalLayer");
        auto surface = std::unique_ptr<MetalSurface>(new (std::nothrow) MetalSurface(owner_));
        if (!surface)
            return fail<std::unique_ptr<VernonHalSurface>>(VERNON_HAL_RESOURCE_EXHAUSTED, "cannot allocate a surface");
        surface->layer = static_cast<CAMetalLayer *>(layer);
        return Created{ok(std::unique_ptr<VernonHalSurface>(std::move(surface)))};
    }

    Done<VernonHalSurfaceCapabilities> capabilities(VernonHalSurface &surface) override {
        VernonHalSurfaceCapabilities capabilities{};
        capabilities.struct_size = sizeof(capabilities);
        layerExtent(static_cast<MetalSurface &>(surface).layer, capabilities.current_width,
                    capabilities.current_height);
        if (!capabilities.current_width || !capabilities.current_height)
            capabilities.current_width = capabilities.current_height = UINT32_MAX;
        capabilities.formats[0] = VERNON_TEXTURE_RGBA8_UNORM;
        capabilities.format_count = 1;
        // Only macOS layers present without waiting for vertical blank.
        capabilities.present_modes = 1u << VERNON_HAL_PRESENT_FIFO;
#if TARGET_OS_OSX
        capabilities.present_modes |= 1u << VERNON_HAL_PRESENT_IMMEDIATE;
#endif
        // Without sampled or storage usage, an acquired image is read only by a preserving render scope or a read,
        // which a submission rejects while its contents are undefined.
        capabilities.usage = VERNON_IMAGE_COLOR_ATTACHMENT | VERNON_IMAGE_TRANSFER_SOURCE;
        capabilities.min_image_count = 1;
        return Done<VernonHalSurfaceCapabilities>{ok(capabilities)};
    }

    // The new images replace the old ones only once all of them exist.
    Done<void> configure(VernonHalSurface &surface, const VernonHalSurfaceConfiguration &configuration) override {
        auto &target = static_cast<MetalSurface &>(surface);
        VernonImageDescriptor descriptor{VERNON_TEXTURE_2D,
                                         {configuration.width, configuration.height, 1},
                                         configuration.format,
                                         1,
                                         1,
                                         1,
                                         configuration.usage | VERNON_IMAGE_TRANSFER_SOURCE};
        const VernonImageViewDescriptor viewDescriptor{
            VERNON_TEXTURE_2D, configuration.format, {0, 1, 0, 1, VERNON_IMAGE_ASPECT_COLOR}};
        std::vector<std::unique_ptr<MetalImage>> images;
        std::vector<std::unique_ptr<MetalView>> views;
        for (uint32_t index = 0; index < configuration.image_count; ++index) {
            // The present copy reads the image in a shader.
            auto image = makeImage(descriptor, MTLTextureUsageShaderRead);
            if (image.isErr())
                return Done<void>{err(std::move(image).error())};
            std::unique_ptr<MetalImage> made = std::move(image).value();
            made->device = surface.device;
            made->descriptor = descriptor;
            made->surface = &surface;
            auto view = makeView(*made, viewDescriptor);
            if (view.isErr())
                return Done<void>{err(std::move(view).error())};
            std::unique_ptr<MetalView> madeView = std::move(view).value();
            madeView->image = made.get();
            madeView->descriptor = viewDescriptor;
            images.push_back(std::move(made));
            views.push_back(std::move(madeView));
        }
        CAMetalLayer *layer = target.layer;
        layer.device = state_.native->device;
        layer.pixelFormat = MTLPixelFormatBGRA8Unorm;
        layer.framebufferOnly = YES;
        layer.drawableSize = CGSizeMake(configuration.width, configuration.height);
        layer.maximumDrawableCount = std::clamp<NSUInteger>(configuration.image_count, 2, 3);
#if TARGET_OS_OSX
        layer.displaySyncEnabled = configuration.present_mode != VERNON_HAL_PRESENT_IMMEDIATE;
#endif
        // The old views go before their images, once the queue no longer reads them.
        std::vector<std::unique_ptr<MetalView>> oldViews = std::exchange(target.views, std::move(views));
        std::vector<std::unique_ptr<MetalImage>> oldImages = std::exchange(target.images, std::move(images));
        target.configuration = configuration;
        target.next = 0;
        oldViews.clear();
        oldImages.clear();
        return Done<void>{ok()};
    }

    // A layer whose extent is no longer the configured one reports out of date, so the host configures again.
    Done<VernonHalSurfaceImage> acquire(VernonHalSurface &surface) override {
        auto &source = static_cast<MetalSurface &>(surface);
        if (source.images.empty())
            return fail<VernonHalSurfaceImage>(VERNON_HAL_INVALID_ARGUMENT, "the surface is not configured");
        uint32_t width = 0;
        uint32_t height = 0;
        layerExtent(source.layer, width, height);
        if (width && height && (width != source.configuration.width || height != source.configuration.height))
            return fail<VernonHalSurfaceImage>(VERNON_HAL_SURFACE_OUT_OF_DATE, "the surface is out of date");
        MetalImage &image = *source.images[source.next];
        markUndefined(image.serial);
        return Done<VernonHalSurfaceImage>{ok(VernonHalSurfaceImage{&image, source.views[source.next].get(), 0})};
    }

    Done<void> present(VernonHalSurface &surface) override {
        auto &target = static_cast<MetalSurface &>(surface);
        MetalImage &presented = *target.images[target.next];
        target.next = (target.next + 1) % target.images.size();
        @autoreleasepool {
            // Waits for a free drawable outside the lock, which submissions on other threads take.
            id<CAMetalDrawable> drawable = [target.layer nextDrawable];
            if (!drawable)
                return internal("cannot present the surface", "the layer has no free drawable");
            std::lock_guard<std::mutex> guard(state_.mutex);
            metal::Native &native = *state_.native;
            id<MTLCommandBuffer> commands = [native.queue commandBuffer];
            MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
            pass.colorAttachments[0].texture = drawable.texture;
            pass.colorAttachments[0].loadAction = MTLLoadActionDontCare;
            pass.colorAttachments[0].storeAction = MTLStoreActionStore;
            id<MTLRenderCommandEncoder> encoder = [commands renderCommandEncoderWithDescriptor:pass];
            if (!encoder)
                return internal("cannot present the surface", "cannot encode the copy");
            [encoder setRenderPipelineState:native.present];
            [encoder setFragmentTexture:presented.native.texture atIndex:0];
            [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
            [encoder endEncoding];
            [commands presentDrawable:drawable];
            metal::commit(state_, commands);
        }
        return Done<void>{ok()};
    }

    Done<VernonRuntimeProviderResourceReference> reference(VernonRuntimeContext &, const VernonHalBuffer &buffer,
                                                           uint64_t offset, uint64_t size) override {
        return Done<VernonRuntimeProviderResourceReference>{
            ok(static_cast<const MetalHalBuffer &>(buffer).native.reference(offset, size))};
    }
    Done<VernonRuntimeProviderResourceReference> reference(VernonRuntimeContext &,
                                                           const VernonHalImageView &view) override {
        VernonRuntimeProviderResourceReference reference{};
        reference.identity = nativeIdentity(&state_, kNativeViewReference);
        reference.resource.value = static_cast<const MetalView &>(view).serial;
        return Done<VernonRuntimeProviderResourceReference>{ok(reference)};
    }
    Done<VernonRuntimeProviderResourceReference> reference(VernonRuntimeContext &,
                                                           const VernonHalSampler &sampler) override {
        VernonRuntimeProviderResourceReference reference{};
        reference.identity = nativeIdentity(&state_, kNativeSamplerReference);
        reference.resource.value = static_cast<const MetalSampler &>(sampler).serial;
        return Done<VernonRuntimeProviderResourceReference>{ok(reference)};
    }

private:
    // The baseline narrowed to what the device does: Apple GPUs filter 32-bit floats only where they say so, and
    // depth only in the Mac family.
    uint32_t capabilities(VernonTextureFormat format) const {
        const formats::Format *entry = formats::find(format);
        if (!entry || metal::pixelFormat(format) == MTLPixelFormatInvalid)
            return 0;
        uint32_t result = entry->baseline;
        const bool float32 = entry->dataType == VERNON_HAL_IMAGE_DATA_FLOAT32 && !formats::hasDepth(format);
        if (float32 && !state_.native->filters32BitFloat)
            result &= ~VERNON_HAL_FORMAT_FILTERABLE;
        if (formats::hasDepth(format) && ![state_.native->device supportsFamily:MTLGPUFamilyMac2])
            result &= ~VERNON_HAL_FORMAT_FILTERABLE;
        return result;
    }

    // `extraUsage` is native usage the device itself needs, beyond the descriptor's.
    Done<std::unique_ptr<MetalImage>> makeImage(const VernonImageDescriptor &descriptor, MTLTextureUsage extraUsage) {
        using Made = Done<std::unique_ptr<MetalImage>>;
        if (!validImage(descriptor))
            return fail<std::unique_ptr<MetalImage>>(VERNON_HAL_INVALID_ARGUMENT, "the image descriptor is invalid");
        const MTLPixelFormat format = metal::pixelFormat(descriptor.format);
        if (format == MTLPixelFormatInvalid || descriptor.sample_count > 1)
            return fail<std::unique_ptr<MetalImage>>(VERNON_HAL_UNSUPPORTED, "Metal cannot create the image");
        const uint32_t usage = descriptor.usage;
        const uint32_t supported = capabilities(descriptor.format);
        if (((usage & (VERNON_IMAGE_COLOR_ATTACHMENT | VERNON_IMAGE_DEPTH_STENCIL_ATTACHMENT)) &&
             !(supported & VERNON_HAL_FORMAT_RENDERABLE)) ||
            ((usage & VERNON_IMAGE_STORAGE) && !(supported & VERNON_HAL_FORMAT_STORAGE)))
            return fail<std::unique_ptr<MetalImage>>(VERNON_HAL_UNSUPPORTED,
                                                     "the image's format does not support its usage");
        const bool volume = descriptor.dimension == VERNON_TEXTURE_3D;
        const bool cube =
            descriptor.dimension == VERNON_TEXTURE_CUBE || descriptor.dimension == VERNON_TEXTURE_CUBE_ARRAY;
        MTLTextureUsage native = extraUsage;
        if (usage & VERNON_IMAGE_SAMPLED)
            native |= MTLTextureUsageShaderRead;
        if (usage & VERNON_IMAGE_STORAGE)
            native |= MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
        if (usage & (VERNON_IMAGE_COLOR_ATTACHMENT | VERNON_IMAGE_DEPTH_STENCIL_ATTACHMENT))
            native |= MTLTextureUsageRenderTarget;
        // Views reinterpret only 8-bit RGBA and BGRA between UNORM and sRGB, and a sampled depth-stencil image's
        // stencil. Apple GPUs do not compress such images, so the others stay without it.
        if (descriptor.format == VERNON_TEXTURE_RGBA8_UNORM || descriptor.format == VERNON_TEXTURE_RGBA8_SRGB ||
            descriptor.format == VERNON_TEXTURE_BGRA8_UNORM || descriptor.format == VERNON_TEXTURE_BGRA8_SRGB ||
            (formats::hasStencil(descriptor.format) && (usage & VERNON_IMAGE_SAMPLED)))
            native |= MTLTextureUsagePixelFormatView;
        auto image = std::unique_ptr<MetalImage>(new (std::nothrow) MetalImage(owner_));
        if (!image)
            return fail<std::unique_ptr<MetalImage>>(VERNON_HAL_RESOURCE_EXHAUSTED, "cannot allocate an image");
        @autoreleasepool {
            MTLTextureDescriptor *texture = [[MTLTextureDescriptor alloc] init];
            texture.textureType = textureType(descriptor.dimension, descriptor.array_layer_count);
            texture.pixelFormat = format;
            texture.width = descriptor.extent.width;
            texture.height = descriptor.extent.height;
            texture.depth = volume ? descriptor.extent.depth : 1;
            texture.mipmapLevelCount = descriptor.mip_level_count;
            texture.arrayLength = volume ? 1 : cube ? descriptor.array_layer_count / 6 : descriptor.array_layer_count;
            texture.sampleCount = 1;
            texture.storageMode = MTLStorageModePrivate;
            texture.usage = native;
            image->native.texture = [state_.native->device newTextureWithDescriptor:texture];
        }
        if (!image->native.texture)
            return internal<std::unique_ptr<MetalImage>>("cannot create the image", {});
        metal::Image &made = image->native;
        made.format = descriptor.format;
        made.dimension = descriptor.dimension;
        made.mipLevels = descriptor.mip_level_count;
        made.layers = volume ? 1 : descriptor.array_layer_count;
        return Made{ok(std::move(image))};
    }

    Done<std::unique_ptr<MetalView>> makeView(MetalImage &image, const VernonImageViewDescriptor &descriptor) {
        using Made = Done<std::unique_ptr<MetalView>>;
        if (!validView(image.descriptor, descriptor))
            return fail<std::unique_ptr<MetalView>>(VERNON_HAL_INVALID_ARGUMENT, "the view descriptor is invalid");
        MTLPixelFormat format = metal::pixelFormat(descriptor.format);
        if (format == MTLPixelFormatInvalid)
            return fail<std::unique_ptr<MetalView>>(VERNON_HAL_UNSUPPORTED, "Metal cannot view the format");
        const VernonImageSubresourceRange &range = descriptor.subresources;
        // A sampled stencil view reads the stencil plane; an attachment view keeps the whole format.
        if (formats::hasStencil(descriptor.format) && range.aspects == VERNON_IMAGE_ASPECT_STENCIL &&
            (image.descriptor.usage & VERNON_IMAGE_SAMPLED))
            format = MTLPixelFormatX32_Stencil8;
        auto view = std::unique_ptr<MetalView>(new (std::nothrow) MetalView(owner_));
        if (!view)
            return fail<std::unique_ptr<MetalView>>(VERNON_HAL_RESOURCE_EXHAUSTED, "cannot allocate a view");
        const bool volume = descriptor.dimension == VERNON_TEXTURE_3D;
        metal::View &binding = view->binding;
        binding.owner = &image.native;
        binding.format = descriptor.format;
        binding.dimension = descriptor.dimension;
        binding.usage = image.descriptor.usage;
        binding.aspects = range.aspects;
        const MTLTextureType type = descriptor.dimension == VERNON_TEXTURE_2D
                                        ? MTLTextureType2D
                                        : textureType(descriptor.dimension, range.array_layer_count);
        binding.texture = [image.native.texture
            newTextureViewWithPixelFormat:format
                              textureType:type
                                   levels:NSMakeRange(range.base_mip_level, range.mip_level_count)
                                   slices:volume ? NSMakeRange(0, 1)
                                                 : NSMakeRange(range.base_array_layer, range.array_layer_count)];
        if (!binding.texture)
            return internal<std::unique_ptr<MetalView>>("cannot create the view",
                                                        "Metal cannot view the format or subresources");
        view->entry = {image.descriptor, descriptor, image.serial, &binding};
        view->serial = nativeViews().add(&view->entry);
        return Made{ok(std::move(view))};
    }

    std::shared_ptr<DeviceState> owner_;
    DeviceState &state_;
};

} // namespace

metal::Buffer *metalBuffer(uint64_t serial) { return serials().find(serial); }

std::unique_ptr<Resources> createMetalResources(std::shared_ptr<metal::DeviceState> state) {
    if (!state)
        return nullptr;
    return std::unique_ptr<Resources>(new (std::nothrow) MetalResources(std::move(state)));
}

std::unique_ptr<DeviceBuffers> createMetalDeviceBuffers(std::shared_ptr<metal::DeviceState> state) {
    if (!state)
        return nullptr;
    return std::unique_ptr<DeviceBuffers>(new (std::nothrow) MetalDeviceBuffers(std::move(state)));
}

} // namespace vernon::hal
