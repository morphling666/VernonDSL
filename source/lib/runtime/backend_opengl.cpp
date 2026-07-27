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

bool isOpenGL(const VernonRuntimeContext *context) {
    return context && (context->backend == VERNON_RUNTIME_OPENGL || context->backend == VERNON_RUNTIME_OPENGL_ES);
}

void makeCurrent(VernonRuntimeContext *context) { openGLState(*context).device.makeCurrent(); }

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
