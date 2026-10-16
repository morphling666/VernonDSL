#ifndef VERNON_RHI_OPENGL_BACKEND_H
#define VERNON_RHI_OPENGL_BACKEND_H

#include "VernonRHI.h"
#include "opengl_driver.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace vernon::rhi::opengl {

inline Enum textureTarget(VernonRhiImageDimension dimension) {
    switch (dimension) {
    case VERNON_RHI_IMAGE_2D:
        return kTexture2D;
    case VERNON_RHI_IMAGE_3D:
        return kTexture3D;
    case VERNON_RHI_IMAGE_CUBE:
        return kTextureCubeMap;
    case VERNON_RHI_IMAGE_2D_ARRAY:
        return kTexture2DArray;
    case VERNON_RHI_IMAGE_CUBE_ARRAY:
        return kTextureCubeMapArray;
    }
    return 0;
}

struct Buffer {
    Uint name{};
    bool imported{};
};

struct Image {
    Uint name{};
    bool imported{};
};

struct Sampler {
    Uint name{};
    bool imported{};
};

enum class DepthStencilAspect { Depth, Stencil };

struct VERNON_RHI_CAPI DeviceState {
    bool initialize(const VernonOpenGLContextCallbacks &callbacks, bool embeddedProfile, std::string &error);
    bool makeCurrent() const noexcept;
    bool supportsCompute() const;
    bool apiVersionAtLeast(uint32_t major, uint32_t minor) const;
    bool hasExtension(std::string_view name) const;

    bool createBuffer(Buffer &buffer, size_t size, std::string &error);
    void importBuffer(Buffer &buffer, Uint name);
    void destroyBuffer(Buffer &buffer);
    bool uploadBuffer(const Buffer &buffer, size_t offset, const void *source, size_t size, std::string &error);
    bool downloadBuffer(const Buffer &buffer, size_t offset, void *destination, size_t size, std::string &error);
    bool copyBuffer(const Buffer &source, size_t sourceOffset, const Buffer &destination, size_t destinationOffset,
                    size_t size, std::string &error);

    bool createImage2D(Image &image, Int internalFormat, Size width, Size height, Enum externalFormat, Enum type,
                       std::string &error);
    void importImage(Image &image, Uint name);
    void destroyImage(Image &image);
    bool uploadImage2D(const Image &image, Size width, Size height, Enum externalFormat, Enum type, const void *source);
    // `layer` selects the layer of 2D-array and cube-array targets; other targets ignore it.
    bool downloadImage2D(const Image &image, Enum target, Int mipLevel, Int layer, Int x, Int y, Size width,
                         Size height, Enum externalFormat, Enum type, void *destination, std::string &error);
    bool downloadImage3D(const Image &image, Int mipLevel, Int x, Int y, Int z, Size width, Size height, Size depth,
                         Enum externalFormat, Enum type, size_t layerSize, void *destination, std::string &error);
    // Writes one 32-bit word per texel of the region into `destination` from offset 0, row by row from (x, y): the
    // float bits of depth, or the stencil value. OpenGL ES reads neither aspect with glReadPixels, so this is a compute
    // pass and needs OpenGL ES 3.1. `target` is the texture's own target; `layer` selects the layer, or the layer-face
    // of cube targets.
    bool copyDepthStencilToBuffer(const Image &image, Enum target, DepthStencilAspect aspect, Int mipLevel, Int layer,
                                  Int x, Int y, Size width, Size height, const Buffer &destination, std::string &error);
    void destroyDepthStencilCopyObjects();

    bool createSampler(Sampler &sampler, Int wrapU, Int wrapV, Int wrapW, Int minFilter, Int magFilter,
                       std::string &error);
    void importSampler(Sampler &sampler, Uint name);
    void destroySampler(Sampler &sampler);

    Uint compileShader(Enum kind, const std::string &source, std::string &error);
    Uint linkProgram(const std::vector<Uint> &shaders, std::string &error);
    void destroyProgram(Uint program);
    bool createVertexArray(Uint &vertexArray, std::string &error);
    void destroyVertexArray(Uint vertexArray);
    bool createFramebuffer(Uint &framebuffer, std::string &error);
    void destroyFramebuffer(Uint framebuffer);

    // No-ops without debug labels. A push beyond the context's group stack limit is skipped, and so is its pop, so
    // that labels never raise a GL error. Labels are cut to the context's message length.
    void pushDebugGroup(std::string_view label);
    void popDebugGroup();
    void popDebugGroups();
    void insertDebugMarker(std::string_view label);

    VernonOpenGLContextCallbacks callbacks{};
    bool embeddedProfile{};
    std::vector<std::string> extensions;
    uint64_t framebufferGeneration{};
    // Advances whenever the RHI changes state that draws set (the bound framebuffer, write masks and scissor), so that
    // adapters caching draw state reapply it.
    uint64_t drawStateGeneration{};
    Driver driver{};
    // Created on first use; programs are indexed by aspect and by 2D, 2D-array, cube and cube-array target.
    Uint depthStencilCopyPrograms[2][4]{};
    Uint depthStencilCopySampler{};

private:
    Uint depthStencilCopyProgram(Enum target, DepthStencilAspect aspect, std::string &error);
    Size debugLabelLength(std::string_view label) const;

    Int maxDebugMessageLength{};
    Int maxDebugGroupDepth{};
    // One entry per open group, telling whether its push reached the context.
    std::vector<bool> debugGroupsPushed;
};

} // namespace vernon::rhi::opengl

#endif
