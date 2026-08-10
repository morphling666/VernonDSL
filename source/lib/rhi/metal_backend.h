#ifndef VERNON_RHI_METAL_BACKEND_H
#define VERNON_RHI_METAL_BACKEND_H

#include "VernonRHI.h"

#import <Metal/Metal.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

namespace vernon::rhi::metal {

uint32_t pixelFormat(VernonRhiFormat format);
size_t bytesPerPixel(VernonRhiFormat format);
bool uploadLayoutMatches(VernonRhiFormat destination, VernonRhiImageDataFormat sourceFormat,
                         VernonRhiImageDataType sourceType);

struct Buffer {
    id<MTLBuffer> buffer;
};

struct Image {
    id<MTLTexture> texture;
};

struct ImageView {
    id<MTLTexture> texture;
};

struct Sampler {
    id<MTLSamplerState> sampler;
};

struct RenderAttachmentSignature {
    uint64_t identity{};
    uint64_t resource{};
    uint32_t location{};
    uint32_t format{};
    uint32_t load{};
    uint32_t store{};
    uint32_t sampleCount{};

    bool operator==(const RenderAttachmentSignature &other) const {
        return identity == other.identity && resource == other.resource && location == other.location &&
               format == other.format && load == other.load && store == other.store && sampleCount == other.sampleCount;
    }
};

struct RenderingState {
    id<MTLRenderCommandEncoder> encoder;
    std::array<id<MTLTexture>, 8> colorTextures{};
    id<MTLTexture> depthStencilTexture;
    std::array<RenderAttachmentSignature, 8> colors{};
    RenderAttachmentSignature depth{};
    size_t colorCount{};
    bool hasDepth{};
};

void registerRenderingState(uint64_t commandBuffer, RenderingState *rendering);
RenderingState *findRenderingState(uint64_t commandBuffer);
void unregisterRenderingState(uint64_t commandBuffer, RenderingState *rendering);

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
    bool downloadImage(const Image &image, const VernonRhiImageDescriptor &descriptor,
                       const VernonRhiImageDownloadDescriptor &download, void *destination, size_t size,
                       std::string &error);
    bool generateImageMipmaps(const Image &image, uint32_t mipLevels, std::string &error);
    bool createImageView(ImageView &view, const Image &image, const VernonRhiImageViewDescriptor &descriptor,
                         std::string &error);
    void destroyImageView(ImageView &view);

    bool createSampler(Sampler &sampler, const VernonRhiSamplerDescriptor &descriptor, std::string &error);
    void destroySampler(Sampler &sampler);

    bool beginCommands(uint64_t &native, std::string &error);
    bool submitCommands(uint64_t native, std::string &error);
    void completeCommands(uint64_t native);
    void abandonCommands(uint64_t native);

    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    uint32_t maxComputeInvocations{};
    uint32_t maxComputeWorkGroupSize[3]{};
    uint32_t operatingSystemVersion[2]{};
    uint32_t argumentBuffersTier{};
    bool argumentBufferEncodingSupported{};
};

} // namespace vernon::rhi::metal

#endif
