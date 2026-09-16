#ifndef VERNON_RUNTIME_BACKEND_OPENGL_H
#define VERNON_RUNTIME_BACKEND_OPENGL_H

#include "VernonRuntime.h"
#include "VernonRuntimeCore.h"
#include "compute_launch_planner.h"
#include "graphics_invocation_planner.h"
#include "pipeline_metadata.h"
#include "prepared_binding_plan.h"
#include "rhi/opengl_backend.h"
#include "runtime_state.h"
#include "stage_artifact.h"

#include <string>
#include <unordered_map>
#include <vector>

struct VernonRuntimeRhiAdapter;
namespace vernon::runtime {

struct OpenGLContextState {
    rhi::opengl::DeviceState device;
    rhi::opengl::Driver &driver{device.driver};
    VernonRuntimeRhiAdapter *adapter{};
};

struct OpenGLPipelineState {
    VernonRuntimeCorePipeline *rhiPipeline{};
    VernonRuntimeCoreBindings *rhiBindings{};
    PreparedGraphicsVariant rhiGraphicsVariant;
    PreparedComputeBindingPlan rhiComputeBindingPlan;
    PreparedGraphicsBindingPlan rhiGraphicsBindingPlan;
    std::vector<VernonRuntimeProviderBindingValue> rhiValues;
    std::vector<std::vector<uint8_t>> rhiComputeBindingStorage;
    uint32_t workgroup[3]{1, 1, 1};
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
