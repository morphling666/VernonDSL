#ifndef VERNON_RUNTIME_BACKEND_OPENGL_H
#define VERNON_RUNTIME_BACKEND_OPENGL_H

#include "VernonRuntime.h"
#include "backend_opengl_driver.h"
#include "graphics_invocation_planner.h"
#include "pipeline_bundle.h"
#include "pipeline_metadata.h"
#include "runtime_state.h"

#include <string>
#include <unordered_map>
#include <vector>

struct VernonDeviceBuffer;
struct VernonDeviceSampler;
struct VernonDeviceTexture;
namespace vernon::runtime {

struct OpenGLContextState {
    VernonExternalOpenGLContext external{};
    OpenGLDriver driver{};
};

struct OpenGLBufferState {
    GlUint name{};
    bool imported{};
};

struct OpenGLTextureState {
    GlUint name{};
    bool imported{};
};

struct OpenGLSamplerState {
    GlUint name{};
    bool imported{};
};

struct OpenGLKernelState {
    GlUint program{};
};

struct OpenGLPipelineCompute {
    GlUint program{};
    uint32_t workgroup[3]{1, 1, 1};
};

struct OpenGLPipelineState {
    std::unordered_map<std::string, OpenGLPipelineCompute> computePrograms;
    GlUint computeProgram{};
    GlUint graphicsProgram{};
    GlUint vertexArray{};
    GlUint framebuffer{};
    uint32_t workgroup[3]{1, 1, 1};
};

inline OpenGLContextState &openGLState(VernonRuntimeContext &context) {
    return runtimeBackendState<OpenGLContextState>(context);
}

inline const OpenGLContextState &openGLState(const VernonRuntimeContext &context) {
    return runtimeBackendState<OpenGLContextState>(context);
}

inline OpenGLBufferState &openGLBufferState(VernonDeviceBuffer &buffer) {
    return runtimeBackendState<OpenGLBufferState>(buffer);
}

inline const OpenGLBufferState &openGLBufferState(const VernonDeviceBuffer &buffer) {
    return runtimeBackendState<OpenGLBufferState>(buffer);
}

inline OpenGLTextureState &openGLTextureState(VernonDeviceTexture &texture) {
    return runtimeBackendState<OpenGLTextureState>(texture);
}

inline const OpenGLTextureState &openGLTextureState(const VernonDeviceTexture &texture) {
    return runtimeBackendState<OpenGLTextureState>(texture);
}

inline OpenGLSamplerState &openGLSamplerState(VernonDeviceSampler &sampler) {
    return runtimeBackendState<OpenGLSamplerState>(sampler);
}

inline const OpenGLSamplerState &openGLSamplerState(const VernonDeviceSampler &sampler) {
    return runtimeBackendState<OpenGLSamplerState>(sampler);
}

bool isOpenGL(const VernonRuntimeContext *context);
void makeCurrent(VernonRuntimeContext *context);

bool initializeOpenGLContext(VernonRuntimeContext &context, const VernonExternalOpenGLContext &external);

bool createOpenGLBuffer(VernonDeviceBuffer &buffer);
void importOpenGLBuffer(VernonDeviceBuffer &buffer, GlUint name);
void destroyOpenGLBuffer(VernonDeviceBuffer &buffer);
VernonStatus copyToOpenGLBuffer(VernonDeviceBuffer &buffer, size_t offset, const void *source, size_t size);
VernonStatus copyFromOpenGLBuffer(const VernonDeviceBuffer &buffer, size_t offset, void *destination, size_t size);

bool createOpenGLTexture(VernonDeviceTexture &texture);
void importOpenGLTexture(VernonDeviceTexture &texture, GlUint name);
void destroyOpenGLTexture(VernonDeviceTexture &texture);
VernonStatus copyToOpenGLTexture(VernonDeviceTexture &texture, const void *source, size_t size);
VernonStatus copyFromOpenGLTexture(const VernonDeviceTexture &texture, void *destination, size_t size);

bool createOpenGLSampler(VernonDeviceSampler &sampler);
void importOpenGLSampler(VernonDeviceSampler &sampler, GlUint name);
void destroyOpenGLSampler(VernonDeviceSampler &sampler);

GlUint compileShader(VernonRuntimeContext *context, GlEnum kind, const std::string &source);
GlUint linkProgram(VernonRuntimeContext *context, const std::vector<GlUint> &shaders);
void destroyOpenGLProgram(VernonRuntimeContext &context, GlUint program);

VernonStatus launchOpenGLKernel(VernonRuntimeContext &context, GlUint program, const ReflectedEntry &reflection,
                                VernonLaunchSize globalSize, const VernonLaunchArgument *arguments,
                                size_t argumentCount);

bool createOpenGLPipeline(VernonRuntimeContext &context, const Variant &variant,
                          const std::unordered_map<std::string, Stage> &stages, GlUint &computeProgram,
                          GlUint &graphicsProgram, GlUint &vertexArray, GlUint &framebuffer, uint32_t (&workgroup)[3]);

void destroyOpenGLPipeline(VernonRuntimeContext &context, GlUint computeProgram, GlUint graphicsProgram,
                           GlUint vertexArray, GlUint framebuffer);

VernonStatus encodeAndSubmitOpenGLCompute(VernonRuntimeContext &context, GlUint program, const uint32_t (&workgroup)[3],
                                          const Variant &variant, const VernonPipelineInvocation &invocation,
                                          const PlannedGraphicsInvocation &plan, std::string &error);

VernonStatus openGLComputeToGraphicsBarrier(VernonRuntimeContext &context);
VernonStatus synchronizeOpenGL(VernonRuntimeContext &context);

} // namespace vernon::runtime

#endif
