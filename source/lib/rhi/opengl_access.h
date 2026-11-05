#ifndef VERNON_RHI_OPENGL_ACCESS_H
#define VERNON_RHI_OPENGL_ACCESS_H

// What the OpenGL device of linked schedules (hal/opengl/opengl_device.cpp) reaches of an RHI OpenGL device, which
// issues its own OpenGL calls on the device's context.

#include "backend_dispatch.h"
#include "opengl_backend.h"

#include <mutex>

namespace vernon::rhi::opengl {

// Every OpenGL call on the device's context holds `mutex`, which RHI calls also take, so a holder calls no RHI function
// that takes it.
struct DeviceAccess {
    DeviceState *state{};
    std::mutex *mutex{};
};

VERNON_RHI_CAPI Result<DeviceAccess, RhiError> deviceAccess(VernonRhiDevice device) noexcept;
// The complete framebuffer that attaches the views of `scope`, as beginRendering binds it; made on first use and
// destroyed with the first texture it attaches. Takes the device's mutex.
VERNON_RHI_CAPI Result<Uint, RhiError> scopeFramebuffer(VernonRhiDevice device, const RenderingScope &scope) noexcept;
// Whether the render area of `scope` covers the whole of view `view`, as glInvalidateFramebuffer discards it.
VERNON_RHI_CAPI bool renderAreaCovers(VernonRhiDevice device, uint64_t view, const RenderingScope &scope) noexcept;

} // namespace vernon::rhi::opengl

#endif
