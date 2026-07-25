#include "backend_opengl.h"

#include "rhi_adapter/adapter_internal.h"
#include "runtime_state.h"
#include "tensor_bridge.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>

namespace vernon::runtime {

using GlEnum = rhi::opengl::Enum;
using GlInt = rhi::opengl::Int;
using GlSize = rhi::opengl::Size;
using GlUint = rhi::opengl::Uint;

namespace {

struct OpenGLTextureFormat {
    GlInt internal;
    GlEnum external;
    GlEnum type;
    size_t bytesPerPixel;
};

std::optional<OpenGLTextureFormat> openGLTextureFormat(VernonTextureFormat format) {
    constexpr GlEnum kUnsignedByte = 0x1401;
    constexpr GlEnum kHalfFloat = 0x140B;
    constexpr GlEnum kUnsignedInt10f11f11fRev = 0x8C3B;
    constexpr GlEnum kRed = 0x1903;
    constexpr GlEnum kRg = 0x8227;
    constexpr GlEnum kRgb = 0x1907;
    constexpr GlEnum kRgba = 0x1908;
    switch (format) {
    case VERNON_TEXTURE_RGBA8_UNORM:
        return OpenGLTextureFormat{0x8058, kRgba, kUnsignedByte, 4};
    case VERNON_TEXTURE_RGBA8_SRGB:
        return OpenGLTextureFormat{0x8C43, kRgba, kUnsignedByte, 4};
    case VERNON_TEXTURE_RGBA16_FLOAT:
        return OpenGLTextureFormat{0x881A, kRgba, kHalfFloat, 8};
    case VERNON_TEXTURE_RGBA32_FLOAT:
        return OpenGLTextureFormat{0x8814, kRgba, rhi::opengl::kFloat, 16};
    case VERNON_TEXTURE_R8_UNORM:
        return OpenGLTextureFormat{0x8229, kRed, kUnsignedByte, 1};
    case VERNON_TEXTURE_R16_FLOAT:
        return OpenGLTextureFormat{0x822D, kRed, kHalfFloat, 2};
    case VERNON_TEXTURE_R32_FLOAT:
        return OpenGLTextureFormat{0x822E, kRed, rhi::opengl::kFloat, 4};
    case VERNON_TEXTURE_RG8_UNORM:
        return OpenGLTextureFormat{0x822B, kRg, kUnsignedByte, 2};
    case VERNON_TEXTURE_RGB8_UNORM:
        return OpenGLTextureFormat{0x8051, kRgb, kUnsignedByte, 3};
    case VERNON_TEXTURE_R11G11B10_FLOAT:
        return OpenGLTextureFormat{0x8C3A, kRgb, kUnsignedInt10f11f11fRev, 4};
    }
    return std::nullopt;
}

std::optional<size_t> openGLTextureByteSize(const VernonDeviceTexture &texture) {
    const std::optional<OpenGLTextureFormat> mapping = openGLTextureFormat(texture.format);
    if (!mapping || texture.width > std::numeric_limits<size_t>::max() / texture.height ||
        static_cast<size_t>(texture.width) * texture.height >
            std::numeric_limits<size_t>::max() / mapping->bytesPerPixel)
        return std::nullopt;
    return static_cast<size_t>(texture.width) * texture.height * mapping->bytesPerPixel;
}

} // namespace

bool isOpenGL(const VernonRuntimeContext *context) {
    return context && (context->backend == VERNON_RUNTIME_OPENGL || context->backend == VERNON_RUNTIME_OPENGL_ES);
}

void makeCurrent(VernonRuntimeContext *context) { openGLState(*context).device.makeCurrent(); }

bool initializeOpenGLContext(VernonRuntimeContext &context, const VernonOpenGLContextCallbacks &callbacks) {
    auto state = std::make_unique<OpenGLContextState>();
    if (!state->device.initialize(callbacks, context.backend == VERNON_RUNTIME_OPENGL_ES, context.error))
        return false;
    state->adapter = createBorrowedOpenGLRhiAdapter(state->device);
    if (!state->adapter) {
        context.error = "failed to create borrowed OpenGL RHI adapter";
        return false;
    }
    installRuntimeBackendState(context, state.release());
    return true;
}

bool createOpenGLBuffer(VernonDeviceBuffer &buffer) {
    VernonRuntimeContext &context = *buffer.context;
    auto *state = new OpenGLBufferState();
    if (!openGLState(context).device.createBuffer(*state, buffer.size, context.error)) {
        delete state;
        return false;
    }
    installRuntimeBackendState(buffer, state);
    return true;
}

void importOpenGLBuffer(VernonDeviceBuffer &buffer, GlUint name) {
    auto *state = new OpenGLBufferState();
    openGLState(*buffer.context).device.importBuffer(*state, name);
    installRuntimeBackendState(buffer, state);
}

void destroyOpenGLBuffer(VernonDeviceBuffer &buffer) {
    openGLState(*buffer.context).device.destroyBuffer(openGLBufferState(buffer));
}

VernonStatus copyToOpenGLBuffer(VernonDeviceBuffer &buffer, size_t offset, const void *source, size_t size) {
    if (!source || offset > buffer.size || size > buffer.size - offset) {
        buffer.context->error = "invalid compute buffer upload";
        return VERNON_STATUS_INVALID_ARGUMENT;
    }
    return openGLState(*buffer.context)
                   .device.uploadBuffer(openGLBufferState(buffer), offset, source, size, buffer.context->error)
               ? VERNON_STATUS_OK
               : VERNON_STATUS_INVALID_ARGUMENT;
}

VernonStatus copyFromOpenGLBuffer(const VernonDeviceBuffer &buffer, size_t offset, void *destination, size_t size) {
    VernonRuntimeContext &context = *buffer.context;
    if (!destination || offset > buffer.size || size > buffer.size - offset) {
        context.error = "invalid compute buffer readback";
        return VERNON_STATUS_INVALID_ARGUMENT;
    }
    return openGLState(context).device.downloadBuffer(openGLBufferState(buffer), offset, destination, size,
                                                      context.error)
               ? VERNON_STATUS_OK
               : VERNON_STATUS_INVALID_ARGUMENT;
}

bool createOpenGLTexture(VernonDeviceTexture &texture) {
    VernonRuntimeContext &context = *texture.context;
    if (texture.dimension != VERNON_TEXTURE_2D || texture.depth != 1 || texture.mipLevels != 1) {
        context.error = "OpenGL owned textures currently require one-mip 2D";
        return false;
    }
    const std::optional<OpenGLTextureFormat> mapping = openGLTextureFormat(texture.format);
    if (!mapping) {
        context.error = "OpenGL texture format is unsupported";
        return false;
    }
    if (texture.width > static_cast<uint32_t>(std::numeric_limits<GlSize>::max()) ||
        texture.height > static_cast<uint32_t>(std::numeric_limits<GlSize>::max())) {
        context.error = "OpenGL texture extent is unsupported";
        return false;
    }
    auto *state = new OpenGLTextureState();
    if (!openGLState(context).device.createImage2D(*state, mapping->internal, static_cast<GlSize>(texture.width),
                                                   static_cast<GlSize>(texture.height), mapping->external,
                                                   mapping->type, context.error)) {
        delete state;
        return false;
    }
    installRuntimeBackendState(texture, state);
    return true;
}

void importOpenGLTexture(VernonDeviceTexture &texture, GlUint name) {
    auto *state = new OpenGLTextureState();
    openGLState(*texture.context).device.importImage(*state, name);
    installRuntimeBackendState(texture, state);
}

void destroyOpenGLTexture(VernonDeviceTexture &texture) {
    openGLState(*texture.context).device.destroyImage(openGLTextureState(texture));
}

VernonStatus copyToOpenGLTexture(VernonDeviceTexture &texture, const void *source, size_t size) {
    const std::optional<OpenGLTextureFormat> mapping = openGLTextureFormat(texture.format);
    const std::optional<size_t> expected = openGLTextureByteSize(texture);
    if (texture.dimension != VERNON_TEXTURE_2D || texture.mipLevels != 1 || !mapping || !expected || !source ||
        size != *expected) {
        texture.context->error = "OpenGL host upload requires a tightly packed one-mip 2D texture";
        return VERNON_STATUS_INVALID_ARGUMENT;
    }
    return openGLState(*texture.context)
                   .device.uploadImage2D(openGLTextureState(texture), static_cast<GlSize>(texture.width),
                                         static_cast<GlSize>(texture.height), mapping->external, mapping->type, source)
               ? VERNON_STATUS_OK
               : VERNON_STATUS_INVALID_ARGUMENT;
}

VernonStatus copyFromOpenGLTexture(const VernonDeviceTexture &texture, void *destination, size_t size) {
    VernonRuntimeContext &context = *texture.context;
    const std::optional<OpenGLTextureFormat> mapping = openGLTextureFormat(texture.format);
    const std::optional<size_t> expected = openGLTextureByteSize(texture);
    if (texture.dimension != VERNON_TEXTURE_2D || texture.mipLevels != 1 || !mapping || !expected || !destination ||
        size != *expected) {
        context.error = "OpenGL host readback requires a tightly packed one-mip 2D texture";
        return VERNON_STATUS_INVALID_ARGUMENT;
    }
    return openGLState(context).device.downloadImage2D(openGLTextureState(texture), static_cast<GlSize>(texture.width),
                                                       static_cast<GlSize>(texture.height), mapping->external,
                                                       mapping->type, destination, context.error)
               ? VERNON_STATUS_OK
               : VERNON_STATUS_INVALID_ARGUMENT;
}

bool createOpenGLSampler(VernonDeviceSampler &sampler) {
    VernonRuntimeContext &context = *sampler.context;
    auto wrap = [&context](VernonSamplerWrapMode value) -> std::optional<GlInt> {
        switch (value) {
        case VERNON_SAMPLER_REPEAT:
            return 0x2901;
        case VERNON_SAMPLER_MIRRORED_REPEAT:
            return 0x8370;
        case VERNON_SAMPLER_CLAMP_TO_EDGE:
            return 0x812F;
        case VERNON_SAMPLER_CLAMP_TO_BORDER:
            return context.backend == VERNON_RUNTIME_OPENGL ? std::optional<GlInt>(0x812D) : std::nullopt;
        }
        return std::nullopt;
    };
    auto filter = [](VernonSamplerFilter value) -> std::optional<GlInt> {
        switch (value) {
        case VERNON_SAMPLER_NEAREST:
            return 0x2600;
        case VERNON_SAMPLER_LINEAR:
            return 0x2601;
        }
        return std::nullopt;
    };
    const auto wrapU = wrap(sampler.descriptor.wrap_u);
    const auto wrapV = wrap(sampler.descriptor.wrap_v);
    const auto wrapW = wrap(sampler.descriptor.wrap_w);
    const auto minFilter = filter(sampler.descriptor.min_filter);
    const auto magFilter = filter(sampler.descriptor.mag_filter);
    if (!wrapU || !wrapV || !wrapW || !minFilter || !magFilter || !filter(sampler.descriptor.mip_filter)) {
        context.error = "OpenGL sampler descriptor contains an unsupported value";
        return false;
    }
    auto *state = new OpenGLSamplerState();
    if (!openGLState(context).device.createSampler(*state, *wrapU, *wrapV, *wrapW, *minFilter, *magFilter,
                                                   context.error)) {
        delete state;
        return false;
    }
    installRuntimeBackendState(sampler, state);
    return true;
}

void importOpenGLSampler(VernonDeviceSampler &sampler, GlUint name) {
    auto *state = new OpenGLSamplerState();
    openGLState(*sampler.context).device.importSampler(*state, name);
    installRuntimeBackendState(sampler, state);
}

void destroyOpenGLSampler(VernonDeviceSampler &sampler) {
    openGLState(*sampler.context).device.destroySampler(openGLSamplerState(sampler));
}

VernonStatus openGLComputeToGraphicsBarrier(VernonRuntimeContext &context) {
    rhi::opengl::Driver &gl = openGLState(context).driver;
    if (!gl.memoryBarrier)
        return VERNON_STATUS_UNSUPPORTED_TARGET;
    makeCurrent(&context);
    gl.memoryBarrier(rhi::opengl::kShaderStorageBarrierBit | rhi::opengl::kVertexAttribArrayBarrierBit);
    return VERNON_STATUS_OK;
}

VernonStatus synchronizeOpenGL(VernonRuntimeContext &context) {
    makeCurrent(&context);
    openGLState(context).driver.finish();
    return VERNON_STATUS_OK;
}

} // namespace vernon::runtime
