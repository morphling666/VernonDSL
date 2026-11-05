#ifndef VERNON_RUNTIME_BACKEND_OPENGL_H
#define VERNON_RUNTIME_BACKEND_OPENGL_H

#include "VernonRuntime.h"
#include "VernonRuntimeCore.h"
#include "prepared_binding_plan.h"
#include "rhi/opengl_backend.h"
#include "runtime_state.h"

#include <memory>
#include <mutex>
#include <string>

struct VernonRuntimeRhiAdapter;
namespace vernon::runtime {

struct OpenGLContextState {
    rhi::opengl::DeviceState device;
    rhi::opengl::Driver &driver{device.driver};
    VernonRuntimeRhiAdapter *adapter{};
};

// A graphics Stage has vertex and fragment GLSL and `graphicsPlan`; a compute Stage has compute GLSL and `computePlan`.
struct OpenGLPipelineState {
    PreparedComputeBindingPlan computePlan;
    PreparedGraphicsBindingPlan graphicsPlan;
    // The GLSL of each shader stage, which the HAL device links its own program from.
    std::string compute;
    std::string vertex;
    std::string fragment;
    // The HAL device's program of the Stage, made on first use under `halMutex` (hal/opengl/opengl_device.cpp).
    std::mutex halMutex;
    std::shared_ptr<void> hal;
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
