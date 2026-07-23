#ifndef VERNON_RUNTIME_GRAPHICS_OPENGL_ENCODER_H
#define VERNON_RUNTIME_GRAPHICS_OPENGL_ENCODER_H

#include "backend_opengl_driver.h"
#include "graphics_invocation_planner.h"

#include <string>

struct VernonRuntimeContext;

namespace vernon::runtime {

struct OpenGLGraphicsState {
    VernonRuntimeContext *context{};
    GlUint program{};
    GlUint vertexArray{};
    GlUint framebuffer{};
};

VernonStatus encodeAndSubmitOpenGLGraphics(const OpenGLGraphicsState &state, const Variant &variant,
                                           const VernonPipelineInvocation &invocation,
                                           const PlannedGraphicsInvocation &plan, std::string &error);

} // namespace vernon::runtime

#endif
