#ifndef VERNON_RUNTIME_BACKEND_OPENGL_H
#define VERNON_RUNTIME_BACKEND_OPENGL_H

#include "VernonRuntime.h"
#include "VernonRuntimeCore.h"
#include "rhi/opengl_backend.h"
#include "rhi_provider_pipeline.h"
#include "runtime_state.h"

struct VernonRuntimeRhiAdapter;
namespace vernon::runtime {

struct OpenGLContextState {
    rhi::opengl::DeviceState device;
    rhi::opengl::Driver &driver{device.driver};
    VernonRuntimeRhiAdapter *adapter{};
};

struct OpenGLPipelineState {
    RhiComputePipeline rhiCompute;
    RhiGraphicsPipeline rhiGraphics;
};

inline OpenGLContextState &openGLState(VernonRuntimeContext &context) {
    return runtimeBackendState<OpenGLContextState>(context);
}

inline const OpenGLContextState &openGLState(const VernonRuntimeContext &context) {
    return runtimeBackendState<OpenGLContextState>(context);
}

bool isOpenGL(const VernonRuntimeContext *context);
void makeCurrent(VernonRuntimeContext *context);

VernonStatus synchronizeOpenGL(VernonRuntimeContext &context);

} // namespace vernon::runtime

#endif
