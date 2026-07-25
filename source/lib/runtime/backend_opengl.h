#ifndef VERNON_RUNTIME_BACKEND_OPENGL_H
#define VERNON_RUNTIME_BACKEND_OPENGL_H

#include "../rhi/opengl_backend.h"
#include "VernonRuntime.h"
#include "VernonRuntimeCore.h"
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
struct VernonRuntimeRhiAdapter;
namespace vernon::runtime {

struct OpenGLContextState {
    rhi::opengl::DeviceState device;
    rhi::opengl::Driver &driver{device.driver};
    VernonRuntimeRhiAdapter *adapter{};
};

using OpenGLBufferState = rhi::opengl::Buffer;
using OpenGLTextureState = rhi::opengl::Image;
using OpenGLSamplerState = rhi::opengl::Sampler;

struct OpenGLPipelineState {
    struct InlineBinding {
        enum Source {
            EXTERNAL_UNIFORM,
            EXTERNAL_VERTEX,
            EXTERNAL_TEXTURE,
            EXTERNAL_SAMPLER,
            EXTERNAL_STORAGE,
            COMPUTE_INLINE,
            IMPLICIT_SAMPLER,
            RESOLUTION
        };

        Source source{EXTERNAL_UNIFORM};
        uint32_t externalSlot{};
        std::vector<uint8_t> storage;
    };

    VernonRuntimeCorePipeline *rhiPipeline{};
    VernonRuntimeCoreBindings *rhiBindings{};
    std::vector<VernonRuntimeProviderBindingLayoutEntry> rhiLayout;
    std::vector<VernonRuntimeProviderBindingValue> rhiValues;
    std::vector<InlineBinding> rhiInlineBindings;
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

bool initializeOpenGLContext(VernonRuntimeContext &context, const VernonOpenGLContextCallbacks &callbacks);

bool createOpenGLBuffer(VernonDeviceBuffer &buffer);
void importOpenGLBuffer(VernonDeviceBuffer &buffer, rhi::opengl::Uint name);
void destroyOpenGLBuffer(VernonDeviceBuffer &buffer);
VernonStatus copyToOpenGLBuffer(VernonDeviceBuffer &buffer, size_t offset, const void *source, size_t size);
VernonStatus copyFromOpenGLBuffer(const VernonDeviceBuffer &buffer, size_t offset, void *destination, size_t size);

bool createOpenGLTexture(VernonDeviceTexture &texture);
void importOpenGLTexture(VernonDeviceTexture &texture, rhi::opengl::Uint name);
void destroyOpenGLTexture(VernonDeviceTexture &texture);
VernonStatus copyToOpenGLTexture(VernonDeviceTexture &texture, const void *source, size_t size);
VernonStatus copyFromOpenGLTexture(const VernonDeviceTexture &texture, void *destination, size_t size);

bool createOpenGLSampler(VernonDeviceSampler &sampler);
void importOpenGLSampler(VernonDeviceSampler &sampler, rhi::opengl::Uint name);
void destroyOpenGLSampler(VernonDeviceSampler &sampler);

VernonStatus openGLComputeToGraphicsBarrier(VernonRuntimeContext &context);
VernonStatus synchronizeOpenGL(VernonRuntimeContext &context);

} // namespace vernon::runtime

#endif
