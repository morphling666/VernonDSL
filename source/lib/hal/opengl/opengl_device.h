#ifndef VERNON_HAL_OPENGL_DEVICE_H
#define VERNON_HAL_OPENGL_DEVICE_H

// The OpenGL device of linked schedules (specs/runtime/architecture_v2.md section 16.3): it records each command
// buffer's passes once into a tape of OpenGL calls, with the names resolved and redundant state dropped, and a call
// runs the tape after it wrote the bytes that vary.

#include "hal/hal.h"

#include <memory>

struct VernonRuntimeContext;

namespace vernon::hal {

// The device over the context's OpenGL state (runtime/backend_opengl.h); null for a context of another backend.
std::unique_ptr<Device> createOpenGLDevice(VernonRuntimeContext &context);

} // namespace vernon::hal

#endif
