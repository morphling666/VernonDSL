#include "backend_opengl.h"

#include "backend_stage_pipeline.h"
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

VernonStatus synchronizeOpenGL(VernonRuntimeContext &context) {
    makeCurrent(&context);
    openGLState(context).driver.finish();
    return VERNON_STATUS_OK;
}

} // namespace vernon::runtime
