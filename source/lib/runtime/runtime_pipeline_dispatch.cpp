#include "runtime_dispatch.h"

#include "backend_opengl.h"
#include "runtime_pipeline_backend.h"

namespace vernon::runtime {

BackendPipelineResult resolveBackendPipeline(BackendStageBuildInputs &inputs, const StageBindingPlan &plan,
                                             VernonStageExecutable &pipeline) {
    std::string error;
    if (!inputs.context || !validateStageBindingPlan(plan, error)) {
        return BackendPipelineResult{
            vernon::err(BackendPipelineFailure{BackendPipelineError::InvalidBindingPlan, std::move(error)})};
    }
    if (inputs.context->backend == VERNON_RUNTIME_CUDA)
        return resolveCudaPipeline(inputs, plan, pipeline);
    else if (inputs.context->backend == VERNON_RUNTIME_VULKAN)
        return resolveVulkanPipeline(inputs, plan, pipeline);
    else if (isOpenGLBackend(inputs.context->backend))
        return resolveOpenGLPipeline(inputs, plan, pipeline);
    return BackendPipelineResult{vernon::err(BackendPipelineError::UnsupportedBackend)};
}

std::string renderBackendPipelineError(const BackendPipelineFailure &error) {
    if (!error.diagnostic.empty())
        return error.diagnostic;
    switch (error.code) {
    case BackendPipelineError::InvalidBindingPlan:
        return "backend pipeline binding plan is invalid";
    case BackendPipelineError::UnsupportedBackend:
        return "unsupported runtime pipeline backend";
    case BackendPipelineError::BackendResolutionFailed:
        return "backend pipeline resolution failed";
    case BackendPipelineError::LifecycleUnavailable:
        return "runtime context cannot admit or publish the backend Stage";
    case BackendPipelineError::InvalidArtifact:
        return "backend Stage artifact is absent";
    }
    return "unknown backend pipeline error";
}

void destroyBackendPipeline(VernonStageExecutable &pipeline) { destroyRuntimeBackendState(pipeline); }

} // namespace vernon::runtime
