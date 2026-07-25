#ifndef VERNON_RHI_OPENGL_BACKEND_H
#define VERNON_RHI_OPENGL_BACKEND_H

#include "VernonRHI.h"
#include "opengl_driver.h"

#include <cstddef>
#include <string>
#include <vector>

namespace vernon::rhi::opengl {

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

struct VERNON_RHI_CAPI DeviceState {
    bool initialize(const VernonOpenGLContextCallbacks &callbacks, bool embeddedProfile, std::string &error);
    void makeCurrent() const;
    bool supportsCompute() const;

    bool createBuffer(Buffer &buffer, size_t size, std::string &error);
    void importBuffer(Buffer &buffer, Uint name);
    void destroyBuffer(Buffer &buffer);
    bool uploadBuffer(const Buffer &buffer, size_t offset, const void *source, size_t size, std::string &error);
    bool downloadBuffer(const Buffer &buffer, size_t offset, void *destination, size_t size, std::string &error);

    bool createImage2D(Image &image, Int internalFormat, Size width, Size height, Enum externalFormat, Enum type,
                       std::string &error);
    void importImage(Image &image, Uint name);
    void destroyImage(Image &image);
    bool uploadImage2D(const Image &image, Size width, Size height, Enum externalFormat, Enum type, const void *source);
    bool downloadImage2D(const Image &image, Size width, Size height, Enum externalFormat, Enum type, void *destination,
                         std::string &error);

    bool createSampler(Sampler &sampler, Int wrapU, Int wrapV, Int wrapW, Int minFilter, Int magFilter,
                       std::string &error);
    void importSampler(Sampler &sampler, Uint name);
    void destroySampler(Sampler &sampler);

    Uint compileShader(Enum kind, const std::string &source, std::string &error);
    Uint linkProgram(const std::vector<Uint> &shaders, std::string &error);
    void destroyProgram(Uint program);
    bool createGraphicsObjects(Uint &vertexArray, Uint &framebuffer, std::string &error);
    void destroyGraphicsObjects(Uint vertexArray, Uint framebuffer);

    VernonOpenGLContextCallbacks callbacks{};
    bool embeddedProfile{};
    Driver driver{};
};

} // namespace vernon::rhi::opengl

#endif
