#include "runtime_pipeline_backend.h"

#include "backend_opengl.h"

#include <memory>

namespace vernon::runtime {

BackendPipelineResult resolveOpenGLPipeline(BackendStageBuildInputs &inputs, const StageBindingPlan &plan,
                                            VernonStageExecutable &pipeline) {
    const auto resolve = [&]() {
        VernonRuntimeContext &context = *inputs.context;
        auto state = std::make_unique<OpenGLPipelineState>();
        if (!plan.compute.empty()) {
            if (!prepareComputeBindingPlan(context, inputs.artifacts.at(plan.compute), plan, state->computePlan))
                return false;
            state->compute = inputs.artifacts.at(plan.compute).source;
        } else if (!plan.vertex.empty() && !plan.fragment.empty()) {
            if (!buildPreparedGraphicsBindingPlan(plan, {context.backend}, state->graphicsPlan,
                                                  invocationDiagnostic(context)))
                return false;
            state->vertex = inputs.artifacts.at(plan.vertex).source;
            state->fragment = inputs.artifacts.at(plan.fragment).source;
        } else {
            invocationDiagnostic(context) = "OpenGL Stage has neither a compute nor a vertex and fragment shader";
            return false;
        }
        installRuntimeBackendState(pipeline, state.release());
        return true;
    };
    return resolve() ? BackendPipelineResult{vernon::ok()} : backendPipelineResolutionFailure(inputs);
}

} // namespace vernon::runtime
