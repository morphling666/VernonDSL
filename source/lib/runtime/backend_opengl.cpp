#include "backend_opengl.h"

#include "runtime_state.h"
#include "tensor_bridge.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>

namespace vernon::runtime {
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
        return OpenGLTextureFormat{0x8814, kRgba, kFloat, 16};
    case VERNON_TEXTURE_R8_UNORM:
        return OpenGLTextureFormat{0x8229, kRed, kUnsignedByte, 1};
    case VERNON_TEXTURE_R16_FLOAT:
        return OpenGLTextureFormat{0x822D, kRed, kHalfFloat, 2};
    case VERNON_TEXTURE_R32_FLOAT:
        return OpenGLTextureFormat{0x822E, kRed, kFloat, 4};
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

std::string shaderLog(OpenGLDriver &gl, GlUint shader) {
    GlInt size = 0;
    gl.getShaderiv(shader, kInfoLogLength, &size);
    std::string result(static_cast<size_t>(std::max(size, 1)), '\0');
    GlSize written = 0;
    gl.getShaderInfoLog(shader, size, &written, result.data());
    result.resize(static_cast<size_t>(std::max(written, 0)));
    return result;
}

std::string programLog(OpenGLDriver &gl, GlUint program) {
    GlInt size = 0;
    gl.getProgramiv(program, kInfoLogLength, &size);
    std::string result(static_cast<size_t>(std::max(size, 1)), '\0');
    GlSize written = 0;
    gl.getProgramInfoLog(program, size, &written, result.data());
    result.resize(static_cast<size_t>(std::max(written, 0)));
    return result;
}

} // namespace

bool isOpenGL(const VernonRuntimeContext *context) {
    return context && (context->backend == VERNON_RUNTIME_OPENGL || context->backend == VERNON_RUNTIME_OPENGL_ES);
}

void makeCurrent(VernonRuntimeContext *context) {
    OpenGLContextState &state = openGLState(*context);
    state.external.make_current(state.external.user_data);
}

bool initializeOpenGLContext(VernonRuntimeContext &context, const VernonExternalOpenGLContext &external) {
    auto state = std::make_unique<OpenGLContextState>();
    state->external = external;
    installRuntimeBackendState(context, state.release());
    makeCurrent(&context);
    OpenGLContextState &installed = openGLState(context);
    if (!loadOpenGLDriver(installed.external, installed.driver, context.error)) {
        destroyRuntimeBackendState(context);
        return false;
    }
    const bool computeExpected =
        context.backend == VERNON_RUNTIME_OPENGL_ES
            ? (installed.external.api_version_major > 3 ||
               (installed.external.api_version_major == 3 && installed.external.api_version_minor >= 1))
            : (installed.external.api_version_major > 4 ||
               (installed.external.api_version_major == 4 && installed.external.api_version_minor >= 3));
    if (!computeExpected || (installed.driver.dispatchCompute && installed.driver.memoryBarrier))
        return true;
    destroyRuntimeBackendState(context);
    return false;
}

bool createOpenGLBuffer(VernonDeviceBuffer &buffer) {
    VernonRuntimeContext &context = *buffer.context;
    if (buffer.size > static_cast<size_t>(std::numeric_limits<GlSizePtr>::max())) {
        context.error = "OpenGL buffer size exceeds the backend address range";
        return false;
    }
    makeCurrent(&context);
    OpenGLDriver &gl = openGLState(context).driver;
    auto *state = new OpenGLBufferState();
    gl.genBuffers(1, &state->name);
    if (!state->name) {
        delete state;
        return false;
    }
    gl.bindBuffer(kArrayBuffer, state->name);
    gl.bufferData(kArrayBuffer, static_cast<GlSizePtr>(buffer.size), nullptr, kDynamicCopy);
    installRuntimeBackendState(buffer, state);
    return true;
}

void importOpenGLBuffer(VernonDeviceBuffer &buffer, GlUint name) {
    auto *state = new OpenGLBufferState();
    state->name = name;
    state->imported = true;
    installRuntimeBackendState(buffer, state);
}

void destroyOpenGLBuffer(VernonDeviceBuffer &buffer) {
    OpenGLBufferState &state = openGLBufferState(buffer);
    if (state.imported || !state.name)
        return;
    makeCurrent(buffer.context);
    openGLState(*buffer.context).driver.deleteBuffers(1, &state.name);
}

VernonStatus copyToOpenGLBuffer(VernonDeviceBuffer &buffer, size_t offset, const void *source, size_t size) {
    if (!source || offset > buffer.size || size > buffer.size - offset) {
        buffer.context->error = "invalid compute buffer upload";
        return VERNON_STATUS_INVALID_ARGUMENT;
    }
    makeCurrent(buffer.context);
    OpenGLDriver &gl = openGLState(*buffer.context).driver;
    gl.bindBuffer(kArrayBuffer, openGLBufferState(buffer).name);
    gl.bufferSubData(kArrayBuffer, static_cast<GlIntPtr>(offset), static_cast<GlSizePtr>(size), source);
    return VERNON_STATUS_OK;
}

VernonStatus copyFromOpenGLBuffer(const VernonDeviceBuffer &buffer, size_t offset, void *destination, size_t size) {
    VernonRuntimeContext &context = *buffer.context;
    if (!destination || offset > buffer.size || size > buffer.size - offset) {
        context.error = "invalid compute buffer readback";
        return VERNON_STATUS_INVALID_ARGUMENT;
    }
    makeCurrent(&context);
    OpenGLDriver &gl = openGLState(context).driver;
    gl.bindBuffer(kArrayBuffer, openGLBufferState(buffer).name);
    void *mapped =
        gl.mapBufferRange(kArrayBuffer, static_cast<GlIntPtr>(offset), static_cast<GlSizePtr>(size), kMapReadBit);
    if (!mapped) {
        context.error = "OpenGL buffer readback mapping failed";
        return VERNON_STATUS_INVALID_ARGUMENT;
    }
    std::memcpy(destination, mapped, size);
    if (!gl.unmapBuffer(kArrayBuffer)) {
        context.error = "OpenGL buffer readback mapping was invalidated";
        return VERNON_STATUS_INVALID_ARGUMENT;
    }
    return VERNON_STATUS_OK;
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
    makeCurrent(&context);
    OpenGLDriver &gl = openGLState(context).driver;
    auto *state = new OpenGLTextureState();
    gl.genTextures(1, &state->name);
    if (!state->name) {
        delete state;
        return false;
    }
    gl.bindTexture(kTexture2D, state->name);
    gl.texImage2D(kTexture2D, 0, mapping->internal, static_cast<GlSize>(texture.width),
                  static_cast<GlSize>(texture.height), 0, mapping->external, mapping->type, nullptr);
    installRuntimeBackendState(texture, state);
    return true;
}

void importOpenGLTexture(VernonDeviceTexture &texture, GlUint name) {
    auto *state = new OpenGLTextureState();
    state->name = name;
    state->imported = true;
    installRuntimeBackendState(texture, state);
}

void destroyOpenGLTexture(VernonDeviceTexture &texture) {
    OpenGLTextureState &state = openGLTextureState(texture);
    if (state.imported || !state.name)
        return;
    makeCurrent(texture.context);
    openGLState(*texture.context).driver.deleteTextures(1, &state.name);
}

VernonStatus copyToOpenGLTexture(VernonDeviceTexture &texture, const void *source, size_t size) {
    const std::optional<OpenGLTextureFormat> mapping = openGLTextureFormat(texture.format);
    const std::optional<size_t> expected = openGLTextureByteSize(texture);
    if (texture.dimension != VERNON_TEXTURE_2D || texture.mipLevels != 1 || !mapping || !expected || !source ||
        size != *expected) {
        texture.context->error = "OpenGL host upload requires a tightly packed one-mip 2D texture";
        return VERNON_STATUS_INVALID_ARGUMENT;
    }
    makeCurrent(texture.context);
    OpenGLDriver &gl = openGLState(*texture.context).driver;
    const OpenGLTextureState &state = openGLTextureState(texture);
    constexpr GlEnum kUnpackAlignment = 0x0CF5;
    gl.pixelStorei(kUnpackAlignment, 1);
    gl.bindTexture(kTexture2D, state.name);
    gl.texSubImage2D(kTexture2D, 0, 0, 0, static_cast<GlSize>(texture.width), static_cast<GlSize>(texture.height),
                     mapping->external, mapping->type, source);
    return VERNON_STATUS_OK;
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
    makeCurrent(&context);
    OpenGLDriver &gl = openGLState(context).driver;
    const OpenGLTextureState &state = openGLTextureState(texture);
    GlUint framebuffer = 0;
    gl.genFramebuffers(1, &framebuffer);
    if (!framebuffer) {
        context.error = "OpenGL readback framebuffer creation failed";
        return VERNON_STATUS_INVALID_ARGUMENT;
    }
    gl.bindFramebuffer(kFramebuffer, framebuffer);
    gl.framebufferTexture2D(kFramebuffer, kColorAttachment0, kTexture2D, state.name, 0);
    if (gl.checkFramebufferStatus(kFramebuffer) != kFramebufferComplete) {
        gl.deleteFramebuffers(1, &framebuffer);
        context.error = "OpenGL texture is not readable as an attachment";
        return VERNON_STATUS_INVALID_ARGUMENT;
    }
    constexpr GlEnum kPackAlignment = 0x0D05;
    gl.pixelStorei(kPackAlignment, 1);
    gl.readPixels(0, 0, static_cast<GlSize>(texture.width), static_cast<GlSize>(texture.height), mapping->external,
                  mapping->type, destination);
    gl.deleteFramebuffers(1, &framebuffer);
    return VERNON_STATUS_OK;
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
    makeCurrent(&context);
    OpenGLDriver &gl = openGLState(context).driver;
    auto *state = new OpenGLSamplerState();
    gl.genSamplers(1, &state->name);
    if (!state->name) {
        delete state;
        return false;
    }
    constexpr GlEnum kTextureMagFilter = 0x2800;
    constexpr GlEnum kTextureMinFilter = 0x2801;
    constexpr GlEnum kTextureWrapS = 0x2802;
    constexpr GlEnum kTextureWrapT = 0x2803;
    constexpr GlEnum kTextureWrapR = 0x8072;
    gl.samplerParameteri(state->name, kTextureWrapS, *wrapU);
    gl.samplerParameteri(state->name, kTextureWrapT, *wrapV);
    gl.samplerParameteri(state->name, kTextureWrapR, *wrapW);
    gl.samplerParameteri(state->name, kTextureMinFilter, *minFilter);
    gl.samplerParameteri(state->name, kTextureMagFilter, *magFilter);
    installRuntimeBackendState(sampler, state);
    return true;
}

void importOpenGLSampler(VernonDeviceSampler &sampler, GlUint name) {
    auto *state = new OpenGLSamplerState();
    state->name = name;
    state->imported = true;
    installRuntimeBackendState(sampler, state);
}

void destroyOpenGLSampler(VernonDeviceSampler &sampler) {
    OpenGLSamplerState &state = openGLSamplerState(sampler);
    if (state.imported || !state.name)
        return;
    makeCurrent(sampler.context);
    openGLState(*sampler.context).driver.deleteSamplers(1, &state.name);
}

GlUint compileShader(VernonRuntimeContext *context, GlEnum kind, const std::string &source) {
    if (source.size() > static_cast<size_t>(std::numeric_limits<GlInt>::max())) {
        context->error = "OpenGL shader source is too large";
        return 0;
    }
    OpenGLDriver &gl = openGLState(*context).driver;
    const GlUint shader = gl.createShader(kind);
    if (!shader) {
        context->error = "OpenGL shader creation failed";
        return 0;
    }
    const char *data = source.data();
    const GlInt size = static_cast<GlInt>(source.size());
    gl.shaderSource(shader, 1, &data, &size);
    gl.compileShader(shader);
    GlInt compiled = 0;
    gl.getShaderiv(shader, kCompileStatus, &compiled);
    if (compiled)
        return shader;
    context->error = "OpenGL shader compilation failed: " + shaderLog(gl, shader);
    gl.deleteShader(shader);
    return 0;
}

GlUint linkProgram(VernonRuntimeContext *context, const std::vector<GlUint> &shaders) {
    OpenGLDriver &gl = openGLState(*context).driver;
    const GlUint program = gl.createProgram();
    for (GlUint shader : shaders)
        gl.attachShader(program, shader);
    gl.linkProgram(program);
    for (GlUint shader : shaders)
        gl.deleteShader(shader);
    GlInt linked = 0;
    gl.getProgramiv(program, kLinkStatus, &linked);
    if (linked)
        return program;
    context->error = "OpenGL program link failed: " + programLog(gl, program);
    gl.deleteProgram(program);
    return 0;
}

void destroyOpenGLProgram(VernonRuntimeContext &context, GlUint program) {
    if (!program)
        return;
    makeCurrent(&context);
    openGLState(context).driver.deleteProgram(program);
}

VernonStatus launchOpenGLKernel(VernonRuntimeContext &context, GlUint program, const ReflectedEntry &reflection,
                                VernonLaunchSize globalSize, const VernonLaunchArgument *arguments,
                                size_t argumentCount) {
    makeCurrent(&context);
    OpenGLDriver &gl = openGLState(context).driver;
    gl.useProgram(program);
    std::vector<GlUint> scalarBuffers;
    scalarBuffers.reserve(argumentCount);
    size_t supplied = 0;
    for (const ReflectedArgument &reflected : reflection.arguments) {
        if (reflected.kind == "builtin")
            continue;
        const VernonLaunchArgument &argument = arguments[supplied++];
        GlUint name = 0;
        if (argument.kind == VERNON_LAUNCH_TENSOR) {
            name = openGLBufferState(*argument.buffer).name;
        } else {
            gl.genBuffers(1, &name);
            if (!name) {
                if (!scalarBuffers.empty())
                    gl.deleteBuffers(static_cast<GlSize>(scalarBuffers.size()), scalarBuffers.data());
                context.error = "OpenGL scalar buffer allocation failed";
                return VERNON_STATUS_INVALID_ARGUMENT;
            }
            scalarBuffers.push_back(name);
            gl.bindBuffer(kShaderStorageBuffer, name);
            gl.bufferData(kShaderStorageBuffer, static_cast<GlSizePtr>(argument.scalar_size), argument.scalar_data,
                          kDynamicCopy);
        }
        if (reflected.kind == "tensor" && !reflected.storageLeaves.empty()) {
            for (const ReflectedStorageLeaf &leaf : reflected.storageLeaves)
                gl.bindBufferBase(kShaderStorageBuffer, leaf.binding, name);
        } else {
            gl.bindBufferBase(kShaderStorageBuffer, reflected.binding, name);
        }
    }
    const uint32_t *workgroup = reflection.workgroup;
    gl.dispatchCompute((globalSize.x - 1) / workgroup[0] + 1, (globalSize.y - 1) / workgroup[1] + 1,
                       (globalSize.z - 1) / workgroup[2] + 1);
    gl.memoryBarrier(kShaderStorageBarrierBit);
    if (!scalarBuffers.empty())
        gl.deleteBuffers(static_cast<GlSize>(scalarBuffers.size()), scalarBuffers.data());
    return VERNON_STATUS_OK;
}

bool createOpenGLPipeline(VernonRuntimeContext &context, const Variant &variant,
                          const std::unordered_map<std::string, Stage> &stages, GlUint &computeProgram,
                          GlUint &graphicsProgram, GlUint &vertexArray, GlUint &framebuffer, uint32_t (&workgroup)[3]) {
    computeProgram = 0;
    graphicsProgram = 0;
    vertexArray = 0;
    framebuffer = 0;
    makeCurrent(&context);
    OpenGLDriver &gl = openGLState(context).driver;
    if (!variant.compute.empty()) {
        const Stage &stage = stages.at(variant.compute);
        std::copy(std::begin(stage.workgroup), std::end(stage.workgroup), std::begin(workgroup));
        const GlUint shader = compileShader(&context, kComputeShader, stage.source);
        if (!shader)
            return false;
        computeProgram = linkProgram(&context, {shader});
        if (!computeProgram)
            return false;
    }
    if (variant.vertex.empty())
        return true;

    const GlUint vertex = compileShader(&context, kVertexShader, stages.at(variant.vertex).source);
    const GlUint fragment = compileShader(&context, kFragmentShader, stages.at(variant.fragment).source);
    if (!vertex || !fragment) {
        if (vertex)
            gl.deleteShader(vertex);
        if (fragment)
            gl.deleteShader(fragment);
        destroyOpenGLPipeline(context, computeProgram, 0, 0, 0);
        computeProgram = 0;
        return false;
    }
    graphicsProgram = linkProgram(&context, {vertex, fragment});
    if (!graphicsProgram) {
        destroyOpenGLPipeline(context, computeProgram, 0, 0, 0);
        computeProgram = 0;
        return false;
    }
    gl.genVertexArrays(1, &vertexArray);
    gl.genFramebuffers(1, &framebuffer);
    if (!vertexArray || !framebuffer) {
        context.error = "OpenGL graphics object creation failed";
        destroyOpenGLPipeline(context, computeProgram, graphicsProgram, vertexArray, framebuffer);
        computeProgram = 0;
        graphicsProgram = 0;
        vertexArray = 0;
        framebuffer = 0;
        return false;
    }
    return true;
}

void destroyOpenGLPipeline(VernonRuntimeContext &context, GlUint computeProgram, GlUint graphicsProgram,
                           GlUint vertexArray, GlUint framebuffer) {
    makeCurrent(&context);
    OpenGLDriver &gl = openGLState(context).driver;
    if (framebuffer)
        gl.deleteFramebuffers(1, &framebuffer);
    if (vertexArray)
        gl.deleteVertexArrays(1, &vertexArray);
    if (graphicsProgram)
        gl.deleteProgram(graphicsProgram);
    if (computeProgram)
        gl.deleteProgram(computeProgram);
}

VernonStatus encodeAndSubmitOpenGLCompute(VernonRuntimeContext &context, GlUint program, const uint32_t (&workgroup)[3],
                                          const Variant &variant, const VernonPipelineInvocation &invocation,
                                          const PlannedGraphicsInvocation &plan, std::string &error) {
    makeCurrent(&context);
    OpenGLDriver &gl = openGLState(context).driver;
    gl.useProgram(program);
    std::vector<GlUint> temporaryBuffers;
    temporaryBuffers.reserve(variant.parameters.size());
    auto failCompute = [&](const char *message) {
        if (!temporaryBuffers.empty())
            gl.deleteBuffers(static_cast<GlSize>(temporaryBuffers.size()), temporaryBuffers.data());
        error = message;
        return VERNON_STATUS_INVALID_ARGUMENT;
    };

    for (const Parameter &parameter : variant.parameters) {
        const VernonPipelineArgument &argument = *plan.arguments.at(parameter.slot);
        for (const ParameterUse &use : parameter.uses) {
            if (use.stage != "compute" && use.stage != variant.compute)
                continue;
            if (use.binding == UINT32_MAX)
                return failCompute("compute parameter has no resource binding");
            GlUint buffer = 0;
            if (argument.tensor.storage == VERNON_TENSOR_DEVICE) {
                if (!argument.tensor.buffer || argument.tensor.buffer->context != &context ||
                    argument.tensor.byte_offset != 0 || !isRowMajorContiguous(argument.tensor))
                    return failCompute("compute Tensor view is invalid");
                buffer = openGLBufferState(*argument.tensor.buffer).name;
            } else {
                const std::optional<size_t> byteSize = tensorLogicalByteSize(argument.tensor);
                if (!byteSize || *byteSize > static_cast<size_t>(std::numeric_limits<GlSizePtr>::max()))
                    return failCompute("compute host Tensor is invalid");
                const void *source = hostTensorData(argument.tensor);
                std::optional<std::vector<uint8_t>> packed;
                if (!isRowMajorContiguous(argument.tensor)) {
                    packed = packTensorRowMajor(argument.tensor);
                    if (!packed)
                        return failCompute("compute host Tensor is invalid");
                    source = packed->data();
                }
                gl.genBuffers(1, &buffer);
                if (!buffer)
                    return failCompute("OpenGL compute buffer allocation failed");
                temporaryBuffers.push_back(buffer);
                gl.bindBuffer(kShaderStorageBuffer, buffer);
                gl.bufferData(kShaderStorageBuffer, static_cast<GlSizePtr>(*byteSize), source, kDynamicCopy);
            }
            gl.bindBufferBase(kShaderStorageBuffer, use.binding, buffer);
        }
    }

    VernonLaunchSize grid = invocation.compute_grid;
    if (!grid.x || !grid.y || !grid.z) {
        for (const Parameter &parameter : variant.parameters) {
            const auto argumentIt = plan.arguments.find(parameter.slot);
            if (argumentIt == plan.arguments.end())
                continue;
            const VernonPipelineArgument &argument = *argumentIt->second;
            if (argument.kind != VERNON_PIPELINE_TENSOR || !argument.tensor.rank || !argument.tensor.shape)
                continue;
            const uint32_t rank = argument.tensor.rank;
            const uint32_t dimensions = std::min(rank, uint32_t{3});
            for (uint32_t dimension = 0; dimension < dimensions; ++dimension)
                if (argument.tensor.shape[rank - 1 - dimension] > std::numeric_limits<uint32_t>::max())
                    return failCompute("inferred compute grid exceeds uint32 range");
            grid = {1, 1, 1};
            grid.x = static_cast<uint32_t>(argument.tensor.shape[rank - 1]);
            if (rank > 1)
                grid.y = static_cast<uint32_t>(argument.tensor.shape[rank - 2]);
            if (rank > 2)
                grid.z = static_cast<uint32_t>(argument.tensor.shape[rank - 3]);
            break;
        }
    }
    if (!grid.x || !grid.y || !grid.z)
        return failCompute("compute grid cannot be inferred");
    if (!workgroup[0] || !workgroup[1] || !workgroup[2])
        return failCompute("compute workgroup dimensions must be positive");
    gl.dispatchCompute((grid.x - 1) / workgroup[0] + 1, (grid.y - 1) / workgroup[1] + 1,
                       (grid.z - 1) / workgroup[2] + 1);
    gl.memoryBarrier(kShaderStorageBarrierBit);
    if (!temporaryBuffers.empty())
        gl.deleteBuffers(static_cast<GlSize>(temporaryBuffers.size()), temporaryBuffers.data());
    return VERNON_STATUS_OK;
}

VernonStatus openGLComputeToGraphicsBarrier(VernonRuntimeContext &context) {
    OpenGLDriver &gl = openGLState(context).driver;
    if (!gl.memoryBarrier)
        return VERNON_STATUS_UNSUPPORTED_TARGET;
    makeCurrent(&context);
    gl.memoryBarrier(kShaderStorageBarrierBit | kVertexAttribArrayBarrierBit);
    return VERNON_STATUS_OK;
}

VernonStatus synchronizeOpenGL(VernonRuntimeContext &context) {
    makeCurrent(&context);
    openGLState(context).driver.finish();
    return VERNON_STATUS_OK;
}

} // namespace vernon::runtime
