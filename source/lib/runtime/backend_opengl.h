#ifndef VERNON_RUNTIME_BACKEND_OPENGL_H
#define VERNON_RUNTIME_BACKEND_OPENGL_H

#include "../rhi/opengl_backend.h"
#include "VernonRuntime.h"
#include "VernonRuntimeCore.h"
#include "compute_launch_planner.h"
#include "graphics_invocation_planner.h"
#include "pipeline_bundle.h"
#include "pipeline_metadata.h"
#include "runtime_state.h"
#include "tensor_bridge.h"

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
        TensorCopyPlan packing;
        std::vector<uint8_t> storage;
    };

    VernonRuntimeCorePipeline *rhiPipeline{};
    VernonRuntimeCoreBindings *rhiBindings{};
    PreparedGraphicsVariant rhiGraphicsVariant;
    std::vector<VernonRuntimeProviderBindingLayoutEntry> rhiLayout;
    std::vector<VernonRuntimeProviderVertexAttribute> rhiVertexAttributes;
    std::vector<VernonRuntimeProviderBindingValue> rhiValues;
    std::vector<InlineBinding> rhiInlineBindings;
    std::vector<ComputeBindingSource> rhiComputeBindingSources;
    std::vector<int64_t> rhiComputeDescriptorValues;
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
