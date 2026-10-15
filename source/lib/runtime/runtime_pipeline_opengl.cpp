#include "runtime_pipeline_backend.h"

#include "VernonFormats.hpp"
#include "backend_opengl.h"

#include <memory>
#include <optional>

namespace vernon::runtime {

namespace {

// The OpenGL provider validates depth-stencil formats as GL internal formats but reads only the count of color
// formats, so color formats stay portable.
std::optional<uint32_t> openGLColorFormat(VernonTextureFormat format) { return static_cast<uint32_t>(format); }

std::optional<uint32_t> openGLDepthFormat(VernonTextureFormat format) {
    const vernon::formats::Format *found = vernon::formats::find(format);
    return found ? std::optional<uint32_t>(found->glInternalFormat) : std::nullopt;
}

constexpr RhiComputeTarget kOpenGLCompute{"OpenGL", "glsl"};
constexpr RhiGraphicsTarget kOpenGLGraphics{"OpenGL", "glsl", openGLColorFormat, openGLDepthFormat};

} // namespace

BackendPipelineResult resolveOpenGLPipeline(BackendStageBuildInputs &inputs, const StageBindingPlan &plan,
                                            VernonStageExecutable &pipeline) {
    const auto resolve = [&]() {
        VernonRuntimeContext &context = *inputs.context;
        auto state = std::make_unique<OpenGLPipelineState>();
        bool prepared = false;
        if (!plan.compute.empty()) {
            prepared = prepareRhiComputePipeline(context, openGLState(context).adapter, kOpenGLCompute,
                                                 inputs.artifacts.at(plan.compute), plan, state->rhiCompute);
        } else if (!plan.vertex.empty() && !plan.fragment.empty()) {
            prepared = prepareRhiGraphicsPipeline(context, openGLState(context).adapter, kOpenGLGraphics,
                                                  inputs.artifacts.at(plan.vertex), inputs.artifacts.at(plan.fragment),
                                                  plan, {context.backend}, state->rhiGraphics);
        } else {
            invocationDiagnostic(context) = "OpenGL ProgramAsset is not representable by the RuntimeCore provider";
        }
        if (!prepared)
            return false;
        installRuntimeBackendState(pipeline, state.release());
        return true;
    };
    return resolve() ? BackendPipelineResult{vernon::ok()} : backendPipelineResolutionFailure(inputs);
}

VernonStatus invokeOpenGLGraphicsPipeline(VernonStageExecutable &pipeline,
                                          const VernonStageInvocationDescriptor &invocation,
                                          const PlannedGraphicsInvocation &plan) {
    return invokeRhiGraphicsPipeline(*pipeline.context, openGLState(*pipeline.context).adapter, kOpenGLGraphics,
                                     invocation, plan, runtimeBackendState<OpenGLPipelineState>(pipeline).rhiGraphics);
}

VernonStatus invokeOpenGLComputePipeline(VernonStageExecutable &pipeline, const PlannedComputeLaunch &launch) {
    return invokeRhiComputePipeline(*pipeline.context, openGLState(*pipeline.context).adapter, kOpenGLCompute,
                                    pipeline.bindingProjection, launch,
                                    runtimeBackendState<OpenGLPipelineState>(pipeline).rhiCompute);
}

} // namespace vernon::runtime
