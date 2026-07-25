#include "opengl_backend.h"

#include <algorithm>
#include <cstring>
#include <limits>

namespace vernon::rhi::opengl {
namespace {

std::string shaderLog(Driver &driver, Uint shader) {
    Int size = 0;
    driver.getShaderiv(shader, kInfoLogLength, &size);
    std::string result(static_cast<size_t>(std::max(size, 1)), '\0');
    Size written = 0;
    driver.getShaderInfoLog(shader, size, &written, result.data());
    result.resize(static_cast<size_t>(std::max(written, 0)));
    return result;
}

std::string programLog(Driver &driver, Uint program) {
    Int size = 0;
    driver.getProgramiv(program, kInfoLogLength, &size);
    std::string result(static_cast<size_t>(std::max(size, 1)), '\0');
    Size written = 0;
    driver.getProgramInfoLog(program, size, &written, result.data());
    result.resize(static_cast<size_t>(std::max(written, 0)));
    return result;
}

} // namespace

bool DeviceState::initialize(const VernonOpenGLContextCallbacks &contextCallbacks, bool isEmbeddedProfile,
                             std::string &error) {
    callbacks = contextCallbacks;
    embeddedProfile = isEmbeddedProfile;
    if (!loadDriver(callbacks, driver, error))
        return false;
    if (!supportsCompute() || (driver.dispatchCompute && driver.memoryBarrier))
        return true;
    error = "OpenGL context reports compute support but required compute functions are missing";
    return false;
}

void DeviceState::makeCurrent() const { callbacks.make_current(callbacks.user_data); }

bool DeviceState::supportsCompute() const {
    return embeddedProfile ? (callbacks.api_version_major > 3 ||
                              (callbacks.api_version_major == 3 && callbacks.api_version_minor >= 1))
                           : (callbacks.api_version_major > 4 ||
                              (callbacks.api_version_major == 4 && callbacks.api_version_minor >= 3));
}

bool DeviceState::createBuffer(Buffer &buffer, size_t size, std::string &error) {
    if (size == 0 || size > static_cast<size_t>(std::numeric_limits<SizePtr>::max())) {
        error = "OpenGL buffer size exceeds the backend address range";
        return false;
    }
    makeCurrent();
    driver.genBuffers(1, &buffer.name);
    if (!buffer.name) {
        error = "OpenGL buffer creation failed";
        return false;
    }
    driver.bindBuffer(kArrayBuffer, buffer.name);
    driver.bufferData(kArrayBuffer, static_cast<SizePtr>(size), nullptr, kDynamicCopy);
    return true;
}

void DeviceState::importBuffer(Buffer &buffer, Uint name) {
    buffer.name = name;
    buffer.imported = true;
}

void DeviceState::destroyBuffer(Buffer &buffer) {
    if (buffer.imported || !buffer.name)
        return;
    makeCurrent();
    driver.deleteBuffers(1, &buffer.name);
    buffer.name = 0;
}

bool DeviceState::uploadBuffer(const Buffer &buffer, size_t offset, const void *source, size_t size,
                               std::string &error) {
    if (!buffer.name || !source || offset > static_cast<size_t>(std::numeric_limits<IntPtr>::max()) ||
        size > static_cast<size_t>(std::numeric_limits<SizePtr>::max())) {
        error = "invalid OpenGL buffer upload";
        return false;
    }
    makeCurrent();
    driver.bindBuffer(kArrayBuffer, buffer.name);
    driver.bufferSubData(kArrayBuffer, static_cast<IntPtr>(offset), static_cast<SizePtr>(size), source);
    return true;
}

bool DeviceState::downloadBuffer(const Buffer &buffer, size_t offset, void *destination, size_t size,
                                 std::string &error) {
    if (!buffer.name || !destination || offset > static_cast<size_t>(std::numeric_limits<IntPtr>::max()) ||
        size > static_cast<size_t>(std::numeric_limits<SizePtr>::max())) {
        error = "invalid OpenGL buffer readback";
        return false;
    }
    makeCurrent();
    driver.bindBuffer(kArrayBuffer, buffer.name);
    void *mapped =
        driver.mapBufferRange(kArrayBuffer, static_cast<IntPtr>(offset), static_cast<SizePtr>(size), kMapReadBit);
    if (!mapped) {
        error = "OpenGL buffer readback mapping failed";
        return false;
    }
    std::memcpy(destination, mapped, size);
    if (!driver.unmapBuffer(kArrayBuffer)) {
        error = "OpenGL buffer readback mapping was invalidated";
        return false;
    }
    return true;
}

bool DeviceState::createImage2D(Image &image, Int internalFormat, Size width, Size height, Enum externalFormat,
                                Enum type, std::string &error) {
    if (width <= 0 || height <= 0) {
        error = "OpenGL image extent is invalid";
        return false;
    }
    makeCurrent();
    driver.genTextures(1, &image.name);
    if (!image.name) {
        error = "OpenGL image creation failed";
        return false;
    }
    driver.bindTexture(kTexture2D, image.name);
    driver.texImage2D(kTexture2D, 0, internalFormat, width, height, 0, externalFormat, type, nullptr);
    return true;
}

void DeviceState::importImage(Image &image, Uint name) {
    image.name = name;
    image.imported = true;
}

void DeviceState::destroyImage(Image &image) {
    if (image.imported || !image.name)
        return;
    makeCurrent();
    driver.deleteTextures(1, &image.name);
    image.name = 0;
}

bool DeviceState::uploadImage2D(const Image &image, Size width, Size height, Enum externalFormat, Enum type,
                                const void *source) {
    if (!image.name || !source)
        return false;
    makeCurrent();
    constexpr Enum unpackAlignment = 0x0CF5;
    driver.pixelStorei(unpackAlignment, 1);
    driver.bindTexture(kTexture2D, image.name);
    driver.texSubImage2D(kTexture2D, 0, 0, 0, width, height, externalFormat, type, source);
    return true;
}

bool DeviceState::downloadImage2D(const Image &image, Size width, Size height, Enum externalFormat, Enum type,
                                  void *destination, std::string &error) {
    if (!image.name || !destination)
        return false;
    makeCurrent();
    Uint framebuffer = 0;
    driver.genFramebuffers(1, &framebuffer);
    if (!framebuffer) {
        error = "OpenGL readback framebuffer creation failed";
        return false;
    }
    driver.bindFramebuffer(kFramebuffer, framebuffer);
    driver.framebufferTexture2D(kFramebuffer, kColorAttachment0, kTexture2D, image.name, 0);
    if (driver.checkFramebufferStatus(kFramebuffer) != kFramebufferComplete) {
        driver.deleteFramebuffers(1, &framebuffer);
        error = "OpenGL image is not readable as an attachment";
        return false;
    }
    constexpr Enum packAlignment = 0x0D05;
    driver.pixelStorei(packAlignment, 1);
    driver.readPixels(0, 0, width, height, externalFormat, type, destination);
    driver.deleteFramebuffers(1, &framebuffer);
    return true;
}

bool DeviceState::createSampler(Sampler &sampler, Int wrapU, Int wrapV, Int wrapW, Int minFilter, Int magFilter,
                                std::string &error) {
    makeCurrent();
    driver.genSamplers(1, &sampler.name);
    if (!sampler.name) {
        error = "OpenGL sampler creation failed";
        return false;
    }
    constexpr Enum textureMagFilter = 0x2800;
    constexpr Enum textureMinFilter = 0x2801;
    constexpr Enum textureWrapS = 0x2802;
    constexpr Enum textureWrapT = 0x2803;
    constexpr Enum textureWrapR = 0x8072;
    driver.samplerParameteri(sampler.name, textureWrapS, wrapU);
    driver.samplerParameteri(sampler.name, textureWrapT, wrapV);
    driver.samplerParameteri(sampler.name, textureWrapR, wrapW);
    driver.samplerParameteri(sampler.name, textureMinFilter, minFilter);
    driver.samplerParameteri(sampler.name, textureMagFilter, magFilter);
    return true;
}

void DeviceState::importSampler(Sampler &sampler, Uint name) {
    sampler.name = name;
    sampler.imported = true;
}

void DeviceState::destroySampler(Sampler &sampler) {
    if (sampler.imported || !sampler.name)
        return;
    makeCurrent();
    driver.deleteSamplers(1, &sampler.name);
    sampler.name = 0;
}

Uint DeviceState::compileShader(Enum kind, const std::string &source, std::string &error) {
    if (source.size() > static_cast<size_t>(std::numeric_limits<Int>::max())) {
        error = "OpenGL shader source is too large";
        return 0;
    }
    makeCurrent();
    const Uint shader = driver.createShader(kind);
    if (!shader) {
        error = "OpenGL shader creation failed";
        return 0;
    }
    const char *data = source.data();
    const Int size = static_cast<Int>(source.size());
    driver.shaderSource(shader, 1, &data, &size);
    driver.compileShader(shader);
    Int compiled = 0;
    driver.getShaderiv(shader, kCompileStatus, &compiled);
    if (compiled)
        return shader;
    error = "OpenGL shader compilation failed: " + shaderLog(driver, shader);
    driver.deleteShader(shader);
    return 0;
}

Uint DeviceState::linkProgram(const std::vector<Uint> &shaders, std::string &error) {
    makeCurrent();
    const Uint program = driver.createProgram();
    for (Uint shader : shaders)
        driver.attachShader(program, shader);
    driver.linkProgram(program);
    for (Uint shader : shaders)
        driver.deleteShader(shader);
    Int linked = 0;
    driver.getProgramiv(program, kLinkStatus, &linked);
    if (linked)
        return program;
    error = "OpenGL program link failed: " + programLog(driver, program);
    driver.deleteProgram(program);
    return 0;
}

void DeviceState::destroyProgram(Uint program) {
    if (!program)
        return;
    makeCurrent();
    driver.deleteProgram(program);
}

bool DeviceState::createGraphicsObjects(Uint &vertexArray, Uint &framebuffer, std::string &error) {
    makeCurrent();
    driver.genVertexArrays(1, &vertexArray);
    driver.genFramebuffers(1, &framebuffer);
    if (vertexArray && framebuffer)
        return true;
    error = "OpenGL graphics object creation failed";
    destroyGraphicsObjects(vertexArray, framebuffer);
    vertexArray = 0;
    framebuffer = 0;
    return false;
}

void DeviceState::destroyGraphicsObjects(Uint vertexArray, Uint framebuffer) {
    makeCurrent();
    if (framebuffer)
        driver.deleteFramebuffers(1, &framebuffer);
    if (vertexArray)
        driver.deleteVertexArrays(1, &vertexArray);
}

} // namespace vernon::rhi::opengl
