#ifndef VERNON_RUNTIME_BACKEND_OPENGL_H
#define VERNON_RUNTIME_BACKEND_OPENGL_H

#include "VernonRuntime.h"
#include "prepared_binding_plan.h"
#include "runtime_state.h"

#include <memory>
#include <mutex>
#include <string>

namespace vernon::hal::opengl {
struct DeviceState;
} // namespace vernon::hal::opengl

namespace vernon::runtime {

// The OpenGL state of the context's HAL device (hal/opengl/opengl_state.h).
struct OpenGLContextState {
    std::shared_ptr<hal::opengl::DeviceState> device;
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

} // namespace vernon::runtime

#endif
