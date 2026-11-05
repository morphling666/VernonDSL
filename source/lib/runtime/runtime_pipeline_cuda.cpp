#include "runtime_pipeline_backend.h"

namespace vernon::runtime {

// The CUDA device links Stages from their kernel fragments, so resolution only validates the binding plan.
BackendPipelineResult resolveCudaPipeline(BackendStageBuildInputs &inputs, const StageBindingPlan &plan,
                                          VernonStageExecutable &) {
    PreparedComputeBindingPlan validated;
    return prepareComputeBindingPlan(*inputs.context, inputs.artifacts.at(plan.compute), plan, validated)
               ? BackendPipelineResult{vernon::ok()}
               : backendPipelineResolutionFailure(inputs);
}

} // namespace vernon::runtime
