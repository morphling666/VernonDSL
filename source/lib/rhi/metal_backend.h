#ifndef VERNON_RHI_METAL_BACKEND_H
#define VERNON_RHI_METAL_BACKEND_H

#include "VernonRHI.h"

#import <Metal/Metal.h>

#include <cstddef>
#include <cstdint>
#include <string>

namespace vernon::rhi::metal {

MTLPixelFormat pixelFormat(VernonRhiFormat format);
size_t bytesPerPixel(VernonRhiFormat format);
bool uploadLayoutMatches(VernonRhiFormat destination, VernonRhiImageDataFormat sourceFormat,
                         VernonRhiImageDataType sourceType);

struct Buffer {
    id<MTLBuffer> buffer;
};

struct Image {
    id<MTLTexture> texture;
};

struct Sampler {
    id<MTLSamplerState> sampler;
};

struct VERNON_RHI_CAPI DeviceState {
    bool initialize(uint32_t deviceIndex, std::string &error);
    void shutdown();
    bool synchronize(std::string &error);

    bool createBuffer(Buffer &buffer, const VernonRhiBufferDescriptor &descriptor, std::string &error);
    void destroyBuffer(Buffer &buffer);
    bool uploadBuffer(const Buffer &buffer, uint64_t offset, const void *source, uint64_t size, std::string &error);
    bool downloadBuffer(const Buffer &buffer, uint64_t offset, void *destination, uint64_t size, std::string &error);

    bool createImage(Image &image, const VernonRhiImageDescriptor &descriptor, std::string &error);
    void destroyImage(Image &image);
    bool uploadImage(const Image &image, const VernonRhiImageDescriptor &descriptor,
                     const VernonRhiImageUploadDescriptor *uploads, size_t uploadCount, std::string &error);
    bool downloadImage(const Image &image, const VernonRhiImageDescriptor &descriptor, void *destination, size_t size,
                       std::string &error);
    bool generateImageMipmaps(const Image &image, uint32_t mipLevels, std::string &error);

    bool createSampler(Sampler &sampler, const VernonRhiSamplerDescriptor &descriptor, std::string &error);
    void destroySampler(Sampler &sampler);

    bool beginCommands(uint64_t &native, std::string &error);
    bool submitCommands(uint64_t native, std::string &error);
    void abandonCommands(uint64_t native);

    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
};

} // namespace vernon::rhi::metal

#endif
