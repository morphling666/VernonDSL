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

int depthStencilCopyTargetIndex(Enum target) {
    switch (target) {
    case kTexture2D:
        return 0;
    case kTexture2DArray:
        return 1;
    case kTextureCubeMap:
        return 2;
    case kTextureCubeMapArray:
        return 3;
    default:
        return -1;
    }
}

// Cube targets have no texelFetch, so their copies sample the texel's center through a nearest-filtering sampler,
// which selects exactly that texel. cubeDirection inverts the face selection of the OpenGL specification's cube map
// table for face `layer % 6`.
std::string depthStencilCopySource(Enum target, DepthStencilAspect aspect, const char *version,
                                   bool cubeArrayExtension) {
    const bool stencil = aspect == DepthStencilAspect::Stencil;
    const char *samplerType = "sampler2D";
    const char *fetch = "texelFetch(source, texel, level).r";
    if (target == kTexture2DArray) {
        samplerType = "sampler2DArray";
        fetch = "texelFetch(source, ivec3(texel, layer), level).r";
    } else if (target == kTextureCubeMap) {
        samplerType = "samplerCube";
        fetch = "textureLod(source, cubeDirection(texel), float(level)).r";
    } else if (target == kTextureCubeMapArray) {
        samplerType = "samplerCubeArray";
        fetch = "textureLod(source, vec4(cubeDirection(texel), float(layer / 6)), float(level)).r";
    }
    const bool cube = target == kTextureCubeMap || target == kTextureCubeMapArray;
    std::string source = std::string("#version ") + version + "\n";
    if (target == kTextureCubeMapArray && cubeArrayExtension)
        source += "#extension GL_EXT_texture_cube_map_array : require\n";
    source += "precision highp float;\n"
              "precision highp int;\n"
              "layout(local_size_x = 8, local_size_y = 8) in;\n"
              "uniform highp ";
    source += std::string(stencil ? "u" : "") + samplerType + " source;\n";
    source += "uniform ivec4 region;\n"
              "uniform int level;\n"
              "uniform int layer;\n"
              "layout(std430, binding = 0) writeonly buffer Texels { uint texels[]; };\n";
    if (cube)
        source += "vec3 cubeDirection(ivec2 texel) {\n"
                  "    vec2 c = (vec2(texel) + 0.5) / vec2(textureSize(source, level).xy) * 2.0 - 1.0;\n"
                  "    switch (layer % 6) {\n"
                  "    case 0: return vec3(1.0, -c.y, -c.x);\n"
                  "    case 1: return vec3(-1.0, -c.y, c.x);\n"
                  "    case 2: return vec3(c.x, 1.0, c.y);\n"
                  "    case 3: return vec3(c.x, -1.0, -c.y);\n"
                  "    case 4: return vec3(c.x, -c.y, 1.0);\n"
                  "    default: return vec3(-c.x, -c.y, -1.0);\n"
                  "    }\n"
                  "}\n";
    source += "void main() {\n"
              "    ivec2 offset = ivec2(gl_GlobalInvocationID.xy);\n"
              "    if (offset.x >= region.z || offset.y >= region.w)\n"
              "        return;\n"
              "    ivec2 texel = region.xy + offset;\n"
              "    texels[offset.y * region.z + offset.x] = ";
    source += stencil ? std::string(fetch) : "floatBitsToUint(" + std::string(fetch) + ")";
    source += ";\n}\n";
    return source;
}

} // namespace

bool DeviceState::initialize(const VernonOpenGLContextCallbacks &contextCallbacks, bool isEmbeddedProfile,
                             std::string &error) {
    callbacks = contextCallbacks;
    embeddedProfile = isEmbeddedProfile;
    if (!loadDriver(callbacks, driver, error))
        return false;
    makeCurrent();
    Int extensionCount = 0;
    driver.getIntegerv(kNumExtensions, &extensionCount);
    for (Int index = 0; index < extensionCount; ++index)
        if (const unsigned char *name = driver.getStringi(kExtensions, static_cast<Uint>(index)))
            extensions.emplace_back(reinterpret_cast<const char *>(name));
    if (!embeddedProfile && apiVersionAtLeast(3, 2)) {
        // Desktop GL otherwise clamps linear filtering at each cube face,
        // exposing discontinuities that Vulkan and D3D12 filter across.
        driver.enable(kTextureCubeMapSeamless);
    }
    const bool coreDebug = embeddedProfile ? apiVersionAtLeast(3, 2) : apiVersionAtLeast(4, 3);
    if (coreDebug || hasExtension("GL_KHR_debug"))
        loadDebugLabelDriver(callbacks, driver, embeddedProfile && !coreDebug);
    if (driver.pushDebugGroup) {
        driver.getIntegerv(kMaxDebugMessageLength, &maxDebugMessageLength);
        driver.getIntegerv(kMaxDebugGroupStackDepth, &maxDebugGroupDepth);
    }
    if (embeddedProfile ? hasExtension("GL_EXT_disjoint_timer_query")
                        : apiVersionAtLeast(3, 3) || hasExtension("GL_ARB_timer_query"))
        loadTimerQueryDriver(callbacks, driver, embeddedProfile);
    if (!supportsCompute() || (driver.dispatchCompute && driver.memoryBarrier))
        return true;
    error = "OpenGL context reports compute support but required compute functions are missing";
    return false;
}

bool DeviceState::makeCurrent() const noexcept {
    try {
        callbacks.make_current(callbacks.user_data);
        return true;
    } catch (...) {
        return false;
    }
}

bool DeviceState::supportsCompute() const {
    return embeddedProfile ? apiVersionAtLeast(3, 1) : apiVersionAtLeast(4, 3);
}

bool DeviceState::apiVersionAtLeast(uint32_t major, uint32_t minor) const {
    return callbacks.api_version_major > major ||
           (callbacks.api_version_major == major && callbacks.api_version_minor >= minor);
}

bool DeviceState::hasExtension(std::string_view name) const {
    return std::find(extensions.begin(), extensions.end(), name) != extensions.end();
}

Size DeviceState::debugLabelLength(std::string_view label) const {
    // The limit counts the terminator; the cut backs off to a UTF-8 code point boundary.
    size_t length = std::min(label.size(), static_cast<size_t>(std::max<Int>(maxDebugMessageLength - 1, 0)));
    if (length < label.size())
        while (length && (static_cast<unsigned char>(label[length]) & 0xC0u) == 0x80u)
            --length;
    return static_cast<Size>(length);
}

void DeviceState::pushDebugGroup(std::string_view label) {
    if (!driver.pushDebugGroup)
        return;
    Int depth = 0;
    driver.getIntegerv(kDebugGroupStackDepth, &depth);
    const bool pushed = depth < maxDebugGroupDepth;
    debugGroupsPushed.push_back(pushed);
    if (pushed)
        driver.pushDebugGroup(kDebugSourceApplication, 0, debugLabelLength(label), label.data());
}

void DeviceState::popDebugGroup() {
    if (!driver.pushDebugGroup || debugGroupsPushed.empty())
        return;
    const bool pushed = debugGroupsPushed.back();
    debugGroupsPushed.pop_back();
    if (pushed)
        driver.popDebugGroup();
}

void DeviceState::popDebugGroups() {
    while (!debugGroupsPushed.empty())
        popDebugGroup();
}

void DeviceState::insertDebugMarker(std::string_view label) {
    if (driver.debugMessageInsert)
        driver.debugMessageInsert(kDebugSourceApplication, kDebugTypeMarker, 0, kDebugSeverityNotification,
                                  debugLabelLength(label), label.data());
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

bool DeviceState::copyBuffer(const Buffer &source, size_t sourceOffset, const Buffer &destination,
                             size_t destinationOffset, size_t size, std::string &error) {
    if (!source.name || !destination.name || !size || !driver.copyBufferSubData ||
        sourceOffset > static_cast<size_t>(std::numeric_limits<IntPtr>::max()) ||
        destinationOffset > static_cast<size_t>(std::numeric_limits<IntPtr>::max()) ||
        size > static_cast<size_t>(std::numeric_limits<SizePtr>::max())) {
        error = "invalid OpenGL buffer copy";
        return false;
    }
    makeCurrent();
    driver.bindBuffer(kCopyReadBuffer, source.name);
    driver.bindBuffer(kCopyWriteBuffer, destination.name);
    driver.copyBufferSubData(kCopyReadBuffer, kCopyWriteBuffer, static_cast<IntPtr>(sourceOffset),
                             static_cast<IntPtr>(destinationOffset), static_cast<SizePtr>(size));
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
    // GL may immediately reuse the numeric texture name. Cached framebuffer
    // signatures must not treat the replacement as the deleted attachment.
    ++framebufferGeneration;
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

bool DeviceState::downloadImage2D(const Image &image, Enum target, Int mipLevel, Int layer, Int x, Int y, Size width,
                                  Size height, Enum externalFormat, Enum type, void *destination, std::string &error) {
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
    ++framebufferGeneration;
    // Stencil is only ever read from a packed depth-stencil image, which must attach at the depth-stencil point.
    const bool depthStencil = externalFormat == 0x84F9 || externalFormat == 0x1901;
    const bool depthOnly = externalFormat == 0x1902;
    const Enum attachment = depthStencil ? kDepthStencilAttachment : depthOnly ? kDepthAttachment : kColorAttachment0;
    if (target == kTexture2DArray || target == kTextureCubeMapArray)
        driver.framebufferTextureLayer(kFramebuffer, attachment, image.name, mipLevel, layer);
    else
        driver.framebufferTexture2D(kFramebuffer, attachment, target, image.name, mipLevel);
    if (depthStencil || depthOnly) {
        const Enum none = kNone;
        driver.drawBuffers(1, &none);
        driver.readBuffer(kNone);
    }
    if (driver.checkFramebufferStatus(kFramebuffer) != kFramebufferComplete) {
        driver.deleteFramebuffers(1, &framebuffer);
        error = "OpenGL image is not readable as an attachment";
        return false;
    }
    constexpr Enum packAlignment = 0x0D05;
    driver.pixelStorei(packAlignment, 1);
    driver.readPixels(x, y, width, height, externalFormat, type, destination);
    driver.deleteFramebuffers(1, &framebuffer);
    return true;
}

bool DeviceState::downloadImage3D(const Image &image, Int mipLevel, Int x, Int y, Int z, Size width, Size height,
                                  Size depth, Enum externalFormat, Enum type, size_t layerSize, void *destination,
                                  std::string &error) {
    if (!image.name || width <= 0 || height <= 0 || depth <= 0 || !layerSize || !destination)
        return false;
    makeCurrent();
    Uint framebuffer = 0;
    driver.genFramebuffers(1, &framebuffer);
    if (!framebuffer) {
        error = "OpenGL 3D image readback framebuffer creation failed";
        return false;
    }
    driver.bindFramebuffer(kFramebuffer, framebuffer);
    ++framebufferGeneration;
    constexpr Enum packAlignment = 0x0D05;
    driver.pixelStorei(packAlignment, 1);
    auto *output = static_cast<unsigned char *>(destination);
    for (Int layer = 0; layer < depth; ++layer) {
        driver.framebufferTextureLayer(kFramebuffer, kColorAttachment0, image.name, mipLevel, z + layer);
        const Enum framebufferStatus = driver.checkFramebufferStatus(kFramebuffer);
        if (framebufferStatus != kFramebufferComplete) {
            driver.deleteFramebuffers(1, &framebuffer);
            error = "OpenGL 3D image layer is not readable as an attachment (framebuffer status " +
                    std::to_string(framebufferStatus) + ")";
            return false;
        }
        driver.readPixels(x, y, width, height, externalFormat, type, output + static_cast<size_t>(layer) * layerSize);
    }
    driver.deleteFramebuffers(1, &framebuffer);
    return true;
}

Uint DeviceState::depthStencilCopyProgram(Enum target, DepthStencilAspect aspect, std::string &error) {
    const int targetIndex = depthStencilCopyTargetIndex(target);
    if (targetIndex < 0) {
        error = "OpenGL depth and stencil copies read 2D, 2D-array, cube and cube-array textures";
        return 0;
    }
    if (!supportsCompute()) {
        error = embeddedProfile ? "OpenGL ES depth and stencil copies require OpenGL ES 3.1"
                                : "OpenGL depth and stencil copies require OpenGL 4.3";
        return 0;
    }
    Uint &program = depthStencilCopyPrograms[aspect == DepthStencilAspect::Stencil ? 1 : 0][targetIndex];
    if (program)
        return program;
    const bool embedded32 = embeddedProfile && apiVersionAtLeast(3, 2);
    const char *version = !embeddedProfile ? "430 core" : embedded32 ? "320 es" : "310 es";
    const Uint shader = compileShader(
        kComputeShader, depthStencilCopySource(target, aspect, version, embeddedProfile && !embedded32), error);
    if (shader)
        program = linkProgram({shader}, error);
    return program;
}

bool DeviceState::copyDepthStencilToBuffer(const Image &image, Enum target, DepthStencilAspect aspect, Int mipLevel,
                                           Int layer, Int x, Int y, Size width, Size height, const Buffer &destination,
                                           std::string &error) {
    if (!image.name || !destination.name || width <= 0 || height <= 0) {
        error = "invalid OpenGL depth or stencil copy";
        return false;
    }
    makeCurrent();
    const Uint program = depthStencilCopyProgram(target, aspect, error);
    if (!program)
        return false;
    constexpr Enum textureMagFilter = 0x2800;
    constexpr Enum textureMinFilter = 0x2801;
    constexpr Int nearest = 0x2600;
    // Without a mipmap filter, cube sampling reads the base level whatever textureLod requests.
    constexpr Int nearestMipmapNearest = 0x2700;
    if (!depthStencilCopySampler) {
        driver.genSamplers(1, &depthStencilCopySampler);
        if (!depthStencilCopySampler) {
            error = "OpenGL depth and stencil copy sampler creation failed";
            return false;
        }
        driver.samplerParameteri(depthStencilCopySampler, textureMinFilter, nearestMipmapNearest);
        driver.samplerParameteri(depthStencilCopySampler, textureMagFilter, nearest);
    }
    constexpr Enum depthStencilTextureMode = 0x90EA;
    constexpr Int stencilIndex = 0x1901;
    constexpr Int depthComponent = 0x1902;
    const Int region[] = {x, y, width, height};
    driver.useProgram(program);
    driver.uniform1i(driver.getUniformLocation(program, "source"), 0);
    driver.uniform4iv(driver.getUniformLocation(program, "region"), 1, region);
    driver.uniform1i(driver.getUniformLocation(program, "level"), mipLevel);
    driver.uniform1i(driver.getUniformLocation(program, "layer"), layer);
    driver.activeTexture(kTexture0);
    driver.bindTexture(target, image.name);
    driver.bindSampler(0, depthStencilCopySampler);
    if (aspect == DepthStencilAspect::Stencil)
        driver.texParameteri(target, depthStencilTextureMode, stencilIndex);
    driver.bindBufferBase(kShaderStorageBuffer, 0, destination.name);
    driver.dispatchCompute(static_cast<Uint>(width + 7) / 8, static_cast<Uint>(height + 7) / 8, 1);
    driver.memoryBarrier(kBufferUpdateBarrierBit);
    if (aspect == DepthStencilAspect::Stencil)
        driver.texParameteri(target, depthStencilTextureMode, depthComponent);
    driver.bindBufferBase(kShaderStorageBuffer, 0, 0);
    driver.bindSampler(0, 0);
    driver.bindTexture(target, 0);
    driver.useProgram(0);
    ++drawStateGeneration;
    return true;
}

void DeviceState::destroyDepthStencilCopyObjects() {
    makeCurrent();
    for (auto &programs : depthStencilCopyPrograms)
        for (Uint &program : programs) {
            destroyProgram(program);
            program = 0;
        }
    if (depthStencilCopySampler)
        driver.deleteSamplers(1, &depthStencilCopySampler);
    depthStencilCopySampler = 0;
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

bool DeviceState::createVertexArray(Uint &vertexArray, std::string &error) {
    makeCurrent();
    driver.genVertexArrays(1, &vertexArray);
    if (vertexArray)
        return true;
    error = "OpenGL vertex array creation failed";
    return false;
}

void DeviceState::destroyVertexArray(Uint vertexArray) {
    makeCurrent();
    if (vertexArray)
        driver.deleteVertexArrays(1, &vertexArray);
}

bool DeviceState::createFramebuffer(Uint &framebuffer, std::string &error) {
    makeCurrent();
    driver.genFramebuffers(1, &framebuffer);
    if (framebuffer)
        return true;
    error = "OpenGL framebuffer creation failed";
    return false;
}

void DeviceState::destroyFramebuffer(Uint framebuffer) {
    makeCurrent();
    if (framebuffer)
        driver.deleteFramebuffers(1, &framebuffer);
}

} // namespace vernon::rhi::opengl
