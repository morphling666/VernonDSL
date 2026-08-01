#include "metal_backend.h"
#include "sampler_filter.h"

#import <Foundation/Foundation.h>
#import <TargetConditionals.h>

#include <algorithm>
#include <cstring>
#include <limits>

namespace vernon::rhi::metal {
namespace {

void setCommandError(std::string &error, id<MTLCommandBuffer> commandBuffer, const char *operation) {
    NSString *description = commandBuffer.error.localizedDescription;
    error = operation;
    if (description.length != 0) {
        error += ": ";
        error += description.UTF8String;
    }
}

bool runCopy(DeviceState &device, id<MTLBuffer> source, NSUInteger sourceOffset, id<MTLBuffer> destination,
             NSUInteger destinationOffset, NSUInteger size, std::string &error) {
    id<MTLCommandBuffer> commandBuffer = [device.queue commandBuffer];
    id<MTLBlitCommandEncoder> encoder = [commandBuffer blitCommandEncoder];
    if (!commandBuffer || !encoder) {
        error = "Metal failed to create a blit command encoder";
        return false;
    }
    [encoder copyFromBuffer:source
               sourceOffset:sourceOffset
                   toBuffer:destination
          destinationOffset:destinationOffset
                       size:size];
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    if (commandBuffer.status == MTLCommandBufferStatusCompleted)
        return true;
    setCommandError(error, commandBuffer, "Metal buffer copy failed");
    return false;
}

size_t alignedTextureRowSize(size_t size) { return (size + 255u) & ~size_t{255u}; }

} // namespace

MTLPixelFormat pixelFormat(VernonRhiFormat format) {
    switch (format) {
    case VERNON_RHI_FORMAT_R8_UNORM:
        return MTLPixelFormatR8Unorm;
    case VERNON_RHI_FORMAT_RG8_UNORM:
        return MTLPixelFormatRG8Unorm;
    case VERNON_RHI_FORMAT_RGBA8_UNORM:
        return MTLPixelFormatRGBA8Unorm;
    case VERNON_RHI_FORMAT_RGBA8_SRGB:
        return MTLPixelFormatRGBA8Unorm_sRGB;
    case VERNON_RHI_FORMAT_R16_FLOAT:
        return MTLPixelFormatR16Float;
    case VERNON_RHI_FORMAT_RGBA16_FLOAT:
        return MTLPixelFormatRGBA16Float;
    case VERNON_RHI_FORMAT_R32_FLOAT:
        return MTLPixelFormatR32Float;
    case VERNON_RHI_FORMAT_RG32_FLOAT:
        return MTLPixelFormatRG32Float;
    case VERNON_RHI_FORMAT_RGBA32_FLOAT:
        return MTLPixelFormatRGBA32Float;
    case VERNON_RHI_FORMAT_R11G11B10_FLOAT:
        return MTLPixelFormatRG11B10Float;
    case VERNON_RHI_FORMAT_D32_FLOAT:
        return MTLPixelFormatDepth32Float;
    default:
        return MTLPixelFormatInvalid;
    }
}

size_t bytesPerPixel(VernonRhiFormat format) {
    switch (format) {
    case VERNON_RHI_FORMAT_R8_UNORM:
        return 1;
    case VERNON_RHI_FORMAT_RG8_UNORM:
    case VERNON_RHI_FORMAT_R16_FLOAT:
        return 2;
    case VERNON_RHI_FORMAT_RGBA8_UNORM:
    case VERNON_RHI_FORMAT_RGBA8_SRGB:
    case VERNON_RHI_FORMAT_R32_FLOAT:
    case VERNON_RHI_FORMAT_R11G11B10_FLOAT:
    case VERNON_RHI_FORMAT_D32_FLOAT:
        return 4;
    case VERNON_RHI_FORMAT_RGBA16_FLOAT:
    case VERNON_RHI_FORMAT_RG32_FLOAT:
        return 8;
    case VERNON_RHI_FORMAT_RGBA32_FLOAT:
        return 16;
    default:
        return 0;
    }
}

bool uploadLayoutMatches(VernonRhiFormat destination, VernonRhiImageDataFormat sourceFormat,
                         VernonRhiImageDataType sourceType) {
    if (sourceType == VERNON_RHI_IMAGE_DATA_UINT8) {
        return (destination == VERNON_RHI_FORMAT_R8_UNORM && sourceFormat == VERNON_RHI_IMAGE_DATA_RED) ||
               (destination == VERNON_RHI_FORMAT_RG8_UNORM && sourceFormat == VERNON_RHI_IMAGE_DATA_RG) ||
               ((destination == VERNON_RHI_FORMAT_RGBA8_UNORM || destination == VERNON_RHI_FORMAT_RGBA8_SRGB) &&
                sourceFormat == VERNON_RHI_IMAGE_DATA_RGBA);
    }
    if (sourceType != VERNON_RHI_IMAGE_DATA_FLOAT32)
        return false;
    return (destination == VERNON_RHI_FORMAT_R32_FLOAT && sourceFormat == VERNON_RHI_IMAGE_DATA_RED) ||
           (destination == VERNON_RHI_FORMAT_RG32_FLOAT && sourceFormat == VERNON_RHI_IMAGE_DATA_RG) ||
           (destination == VERNON_RHI_FORMAT_RGBA32_FLOAT && sourceFormat == VERNON_RHI_IMAGE_DATA_RGBA) ||
           (destination == VERNON_RHI_FORMAT_D32_FLOAT && sourceFormat == VERNON_RHI_IMAGE_DATA_DEPTH);
}

bool DeviceState::initialize(uint32_t deviceIndex, std::string &error) {
#if TARGET_OS_OSX
    NSArray<id<MTLDevice>> *availableDevices = MTLCopyAllDevices();
    if (deviceIndex >= availableDevices.count) {
        error = "Metal device index is out of range";
        return false;
    }
    device = availableDevices[deviceIndex];
#else
    if (deviceIndex != 0) {
        error = "Metal device enumeration is unavailable on iOS; device index must be zero";
        return false;
    }
    device = MTLCreateSystemDefaultDevice();
#endif
    queue = [device newCommandQueue];
    if (device && queue) {
        const MTLSize maximum = device.maxThreadsPerThreadgroup;
        maxComputeWorkGroupSize[0] = static_cast<uint32_t>(maximum.width);
        maxComputeWorkGroupSize[1] = static_cast<uint32_t>(maximum.height);
        maxComputeWorkGroupSize[2] = static_cast<uint32_t>(maximum.depth);
        // Metal exposes the dimensional device limit and a pipeline-specific
        // total. Apple GPU families use the X limit as the device-wide total.
        maxComputeInvocations = maxComputeWorkGroupSize[0];
        const NSOperatingSystemVersion version = NSProcessInfo.processInfo.operatingSystemVersion;
        operatingSystemVersion[0] = static_cast<uint32_t>(version.majorVersion);
        operatingSystemVersion[1] = static_cast<uint32_t>(version.minorVersion);
        argumentBuffersTier = static_cast<uint32_t>(device.argumentBuffersSupport);
        return true;
    }
    error = "Metal default device or command queue creation failed";
    shutdown();
    return false;
}

void DeviceState::shutdown() {
    queue = nil;
    device = nil;
    maxComputeInvocations = 0;
    std::fill(std::begin(maxComputeWorkGroupSize), std::end(maxComputeWorkGroupSize), 0);
    std::fill(std::begin(operatingSystemVersion), std::end(operatingSystemVersion), 0);
    argumentBuffersTier = 0;
}

bool DeviceState::synchronize(std::string &error) {
    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    if (!commandBuffer) {
        error = "Metal synchronization command buffer creation failed";
        return false;
    }
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    if (commandBuffer.status == MTLCommandBufferStatusCompleted)
        return true;
    setCommandError(error, commandBuffer, "Metal synchronization failed");
    return false;
}

bool DeviceState::createBuffer(Buffer &buffer, const VernonRhiBufferDescriptor &descriptor, std::string &error) {
    const MTLResourceOptions options = descriptor.memory_class == VERNON_RHI_MEMORY_DEVICE
                                           ? MTLResourceStorageModePrivate
                                           : MTLResourceStorageModeShared;
    buffer.buffer = [device newBufferWithLength:static_cast<NSUInteger>(descriptor.size) options:options];
    if (buffer.buffer)
        return true;
    error = "Metal buffer allocation failed";
    return false;
}

void DeviceState::destroyBuffer(Buffer &buffer) { buffer.buffer = nil; }

bool DeviceState::uploadBuffer(const Buffer &buffer, uint64_t offset, const void *source, uint64_t size,
                               std::string &error) {
    if (buffer.buffer.storageMode == MTLStorageModeShared) {
        std::memcpy(static_cast<unsigned char *>(buffer.buffer.contents) + offset, source, static_cast<size_t>(size));
        return true;
    }
    id<MTLBuffer> staging =
        [device newBufferWithBytes:source length:static_cast<NSUInteger>(size) options:MTLResourceStorageModeShared];
    if (!staging) {
        error = "Metal upload staging buffer allocation failed";
        return false;
    }
    return runCopy(*this, staging, 0, buffer.buffer, static_cast<NSUInteger>(offset), static_cast<NSUInteger>(size),
                   error);
}

bool DeviceState::downloadBuffer(const Buffer &buffer, uint64_t offset, void *destination, uint64_t size,
                                 std::string &error) {
    if (buffer.buffer.storageMode == MTLStorageModeShared) {
        std::memcpy(destination, static_cast<unsigned char *>(buffer.buffer.contents) + offset,
                    static_cast<size_t>(size));
        return true;
    }
    id<MTLBuffer> staging =
        [device newBufferWithLength:static_cast<NSUInteger>(size) options:MTLResourceStorageModeShared];
    if (!staging) {
        error = "Metal readback staging buffer allocation failed";
        return false;
    }
    if (!runCopy(*this, buffer.buffer, static_cast<NSUInteger>(offset), staging, 0, static_cast<NSUInteger>(size),
                 error))
        return false;
    std::memcpy(destination, staging.contents, static_cast<size_t>(size));
    return true;
}

bool DeviceState::createImage(Image &image, const VernonRhiImageDescriptor &descriptor, std::string &error) {
    const MTLPixelFormat format = pixelFormat(descriptor.format);
    if (format == MTLPixelFormatInvalid) {
        error = "Metal image format is unsupported";
        return false;
    }
    MTLTextureDescriptor *native = [[MTLTextureDescriptor alloc] init];
    native.pixelFormat = format;
    native.width = descriptor.width;
    native.height = descriptor.height;
    native.depth = descriptor.dimension == VERNON_RHI_IMAGE_3D ? descriptor.depth : 1;
    native.mipmapLevelCount = descriptor.mip_levels;
    native.sampleCount = 1;
    native.storageMode = MTLStorageModePrivate;
    native.usage = MTLTextureUsagePixelFormatView;
    if (descriptor.usage & VERNON_RHI_IMAGE_SAMPLED)
        native.usage |= MTLTextureUsageShaderRead;
    if (descriptor.usage & VERNON_RHI_IMAGE_STORAGE)
        native.usage |= MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    if (descriptor.usage & (VERNON_RHI_IMAGE_COLOR_ATTACHMENT | VERNON_RHI_IMAGE_DEPTH_STENCIL_ATTACHMENT))
        native.usage |= MTLTextureUsageRenderTarget;
    if (descriptor.dimension == VERNON_RHI_IMAGE_3D) {
        native.textureType = MTLTextureType3D;
        native.arrayLength = 1;
    } else if (descriptor.dimension == VERNON_RHI_IMAGE_CUBE) {
        native.textureType = MTLTextureTypeCube;
        native.arrayLength = 1;
    } else if (descriptor.array_layers > 1) {
        native.textureType = MTLTextureType2DArray;
        native.arrayLength = descriptor.array_layers;
    } else {
        native.textureType = MTLTextureType2D;
        native.arrayLength = 1;
    }
    image.texture = [device newTextureWithDescriptor:native];
    if (image.texture)
        return true;
    error = "Metal texture allocation failed";
    return false;
}

void DeviceState::destroyImage(Image &image) { image.texture = nil; }

bool DeviceState::uploadImage(const Image &image, const VernonRhiImageDescriptor &descriptor,
                              const VernonRhiImageUploadDescriptor *uploads, size_t uploadCount, std::string &error) {
    const size_t pixelSize = bytesPerPixel(descriptor.format);
    for (size_t index = 0; index < uploadCount; ++index) {
        const VernonRhiImageUploadDescriptor &upload = uploads[index];
        const bool validLayer =
            descriptor.dimension == VERNON_RHI_IMAGE_3D ? upload.array_layer == 0
                                                        : upload.array_layer < descriptor.array_layers;
        const uint32_t mipWidth = std::max(descriptor.width >> upload.mip_level, 1u);
        const uint32_t mipHeight = std::max(descriptor.height >> upload.mip_level, 1u);
        const uint32_t mipDepth =
            descriptor.dimension == VERNON_RHI_IMAGE_3D ? std::max(descriptor.depth >> upload.mip_level, 1u) : 1u;
        if (upload.struct_size < sizeof(upload) || !upload.data || upload.mip_level >= descriptor.mip_levels ||
            !validLayer || upload.width != mipWidth || upload.height != mipHeight || upload.depth != mipDepth ||
            !uploadLayoutMatches(descriptor.format, upload.source_format, upload.source_type)) {
            error = "Metal image upload layout is unsupported";
            return false;
        }
        const size_t rowSize = pixelSize * upload.width;
        const size_t rowPitch = alignedTextureRowSize(rowSize);
        if (upload.height > (std::numeric_limits<size_t>::max)() / rowPitch) {
            error = "Metal image upload staging size overflows the host address range";
            return false;
        }
        const size_t imagePitch = rowPitch * upload.height;
        if (upload.depth > (std::numeric_limits<size_t>::max)() / imagePitch) {
            error = "Metal image upload staging size overflows the host address range";
            return false;
        }
        id<MTLBuffer> staging =
            [device newBufferWithLength:imagePitch * upload.depth options:MTLResourceStorageModeShared];
        if (!staging) {
            error = "Metal image upload staging allocation failed";
            return false;
        }
        const auto *source = static_cast<const uint8_t *>(upload.data);
        auto *destination = static_cast<uint8_t *>(staging.contents);
        for (uint32_t z = 0; z < upload.depth; ++z)
            for (uint32_t y = 0; y < upload.height; ++y)
                std::memcpy(destination + z * imagePitch + y * rowPitch,
                            source + (static_cast<size_t>(z) * upload.height + y) * rowSize, rowSize);
        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
        id<MTLBlitCommandEncoder> encoder = [commandBuffer blitCommandEncoder];
        if (!commandBuffer || !encoder) {
            error = "Metal image upload blit encoder creation failed";
            return false;
        }
        [encoder copyFromBuffer:staging
                  sourceOffset:0
             sourceBytesPerRow:rowPitch
           sourceBytesPerImage:imagePitch
                    sourceSize:MTLSizeMake(upload.width, upload.height, upload.depth)
                     toTexture:image.texture
              destinationSlice:descriptor.dimension == VERNON_RHI_IMAGE_3D ? 0 : upload.array_layer
              destinationLevel:upload.mip_level
             destinationOrigin:MTLOriginMake(0, 0, 0)];
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        if (commandBuffer.status != MTLCommandBufferStatusCompleted) {
            setCommandError(error, commandBuffer, "Metal image upload failed");
            return false;
        }
    }
    return true;
}

bool DeviceState::downloadImage(const Image &image, const VernonRhiImageDescriptor &descriptor, void *destination,
                                size_t size, std::string &error) {
    const size_t pixelSize = bytesPerPixel(descriptor.format);
    if (descriptor.width > (std::numeric_limits<size_t>::max)() / pixelSize ||
        descriptor.height > (std::numeric_limits<size_t>::max)() / (pixelSize * descriptor.width)) {
        error = "Metal image readback size overflows the host address range";
        return false;
    }
    const size_t layerSize = pixelSize * descriptor.width * descriptor.height * descriptor.depth;
    if (descriptor.array_layers > (std::numeric_limits<size_t>::max)() / layerSize ||
        size < layerSize * descriptor.array_layers) {
        error = "Metal image readback destination is too small";
        return false;
    }
    const size_t rowSize = pixelSize * descriptor.width;
    const size_t rowPitch = alignedTextureRowSize(rowSize);
    if (descriptor.height > (std::numeric_limits<size_t>::max)() / rowPitch) {
        error = "Metal image readback staging size overflows the host address range";
        return false;
    }
    const size_t imagePitch = rowPitch * descriptor.height;
    if (descriptor.depth > (std::numeric_limits<size_t>::max)() / imagePitch ||
        descriptor.array_layers > (std::numeric_limits<size_t>::max)() / (imagePitch * descriptor.depth)) {
        error = "Metal image readback staging size overflows the host address range";
        return false;
    }
    const size_t stagingLayerSize = imagePitch * descriptor.depth;
    id<MTLBuffer> staging =
        [device newBufferWithLength:stagingLayerSize * descriptor.array_layers
                            options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    id<MTLBlitCommandEncoder> encoder = [commandBuffer blitCommandEncoder];
    if (!staging || !commandBuffer || !encoder) {
        error = "Metal image readback staging or encoder creation failed";
        return false;
    }
    for (uint32_t layer = 0; layer < descriptor.array_layers; ++layer) {
        [encoder copyFromTexture:image.texture
                    sourceSlice:descriptor.dimension == VERNON_RHI_IMAGE_3D ? 0 : layer
                    sourceLevel:0
                   sourceOrigin:MTLOriginMake(0, 0, 0)
                     sourceSize:MTLSizeMake(descriptor.width, descriptor.height, descriptor.depth)
                       toBuffer:staging
              destinationOffset:layer * stagingLayerSize
         destinationBytesPerRow:rowPitch
       destinationBytesPerImage:imagePitch];
    }
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    if (commandBuffer.status != MTLCommandBufferStatusCompleted) {
        setCommandError(error, commandBuffer, "Metal image readback failed");
        return false;
    }
    auto *output = static_cast<uint8_t *>(destination);
    const auto *source = static_cast<const uint8_t *>(staging.contents);
    for (uint32_t layer = 0; layer < descriptor.array_layers; ++layer)
        for (uint32_t z = 0; z < descriptor.depth; ++z)
            for (uint32_t y = 0; y < descriptor.height; ++y)
                std::memcpy(output + layer * layerSize +
                                (static_cast<size_t>(z) * descriptor.height + y) * rowSize,
                            source + layer * stagingLayerSize + z * imagePitch + y * rowPitch, rowSize);
    return true;
}

bool DeviceState::generateImageMipmaps(const Image &image, uint32_t mipLevels, std::string &error) {
    if (mipLevels < 2)
        return true;
    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    id<MTLBlitCommandEncoder> encoder = [commandBuffer blitCommandEncoder];
    if (!commandBuffer || !encoder) {
        error = "Metal mipmap command encoder creation failed";
        return false;
    }
    [encoder generateMipmapsForTexture:image.texture];
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    if (commandBuffer.status == MTLCommandBufferStatusCompleted)
        return true;
    setCommandError(error, commandBuffer, "Metal mipmap generation failed");
    return false;
}

bool DeviceState::createImageView(ImageView &view, const Image &image,
                                  const VernonRhiImageViewDescriptor &descriptor, std::string &error) {
    const MTLPixelFormat format = pixelFormat(descriptor.format);
    view.texture = [image.texture newTextureViewWithPixelFormat:format
                                                   textureType:image.texture.textureType
                                                        levels:NSMakeRange(descriptor.base_mip_level,
                                                                           descriptor.mip_level_count)
                                                        slices:NSMakeRange(descriptor.base_array_layer,
                                                                           descriptor.array_layer_count)];
    if (view.texture)
        return true;
    error = "Metal image view format or subresource range is unsupported";
    return false;
}

void DeviceState::destroyImageView(ImageView &view) { view.texture = nil; }

bool DeviceState::createSampler(Sampler &sampler, const VernonRhiSamplerDescriptor &descriptor, std::string &error) {
    SamplerFilter filter;
    if (!decodeSamplerFilter(descriptor, filter)) {
        error = "Metal sampler descriptor is invalid";
        return false;
    }
    auto addressMode = [](uint32_t mode) {
        switch (mode) {
        case VERNON_RHI_ADDRESS_REPEAT:
            return MTLSamplerAddressModeRepeat;
        case VERNON_RHI_ADDRESS_MIRRORED_REPEAT:
            return MTLSamplerAddressModeMirrorRepeat;
        default:
            return MTLSamplerAddressModeClampToEdge;
        }
    };
    MTLSamplerDescriptor *native = [[MTLSamplerDescriptor alloc] init];
    native.minFilter = filter.minLinear ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;
    native.magFilter = filter.magLinear ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;
    native.mipFilter = filter.mipLinear ? MTLSamplerMipFilterLinear : MTLSamplerMipFilterNearest;
    native.sAddressMode = addressMode(descriptor.address_u);
    native.tAddressMode = addressMode(descriptor.address_v);
    native.rAddressMode = addressMode(descriptor.address_w);
    native.maxAnisotropy = static_cast<NSUInteger>(filter.maxAnisotropy);
    native.supportArgumentBuffers = YES;
    sampler.sampler = [device newSamplerStateWithDescriptor:native];
    if (sampler.sampler)
        return true;
    error = "Metal sampler creation failed";
    return false;
}

void DeviceState::destroySampler(Sampler &sampler) { sampler.sampler = nil; }

bool DeviceState::beginCommands(uint64_t &native, std::string &error) {
    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    if (!commandBuffer) {
        error = "Metal command buffer creation failed";
        return false;
    }
    native = reinterpret_cast<uintptr_t>(CFBridgingRetain(commandBuffer));
    return true;
}

bool DeviceState::submitCommands(uint64_t native, std::string &error) {
    if (!native) {
        error = "Metal command buffer is invalid";
        return false;
    }
    id<MTLCommandBuffer> commandBuffer = (__bridge id<MTLCommandBuffer>)(reinterpret_cast<void *>(native));
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    if (commandBuffer.status == MTLCommandBufferStatusCompleted) {
        CFBridgingRelease(reinterpret_cast<void *>(native));
        return true;
    }
    setCommandError(error, commandBuffer, "Metal command buffer failed");
    CFBridgingRelease(reinterpret_cast<void *>(native));
    return false;
}

void DeviceState::completeCommands(uint64_t native) {
    if (!native)
        return;
    id<MTLCommandBuffer> commandBuffer = (__bridge id<MTLCommandBuffer>)(reinterpret_cast<void *>(native));
    [commandBuffer waitUntilCompleted];
    CFBridgingRelease(reinterpret_cast<void *>(native));
}

void DeviceState::abandonCommands(uint64_t native) {
    if (!native)
        return;
    id<MTLCommandBuffer> commandBuffer = CFBridgingRelease(reinterpret_cast<void *>(native));
    (void)commandBuffer;
}

} // namespace vernon::rhi::metal
