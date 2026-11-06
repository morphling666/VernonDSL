#include "runtime_pipeline_backend.h"

#if defined(VERNON_HAS_DIRECTX12_RUNTIME)
#include "backend_directx12.h"

#include <memory>
#endif

namespace vernon::runtime {

BackendPipelineResult resolveDirectX12Pipeline(BackendStageBuildInputs &inputs, const StageBindingPlan &plan,
                                               VernonStageExecutable &pipeline) {
    const auto resolve = [&]() {
#if defined(VERNON_HAS_DIRECTX12_RUNTIME)
        VernonRuntimeContext &context = *inputs.context;
        auto state = std::make_unique<DirectX12PipelineState>();
        if (!plan.compute.empty() &&
            !prepareComputeBindingPlan(context, inputs.artifacts.at(plan.compute), plan, state->computePlan))
            return false;
        if (!plan.vertex.empty() &&
            !buildPreparedGraphicsBindingPlan(plan, {VERNON_RUNTIME_DIRECTX12}, state->graphicsPlan,
                                              invocationDiagnostic(context)))
            return false;
        const auto code = [&](const std::string &name, DirectX12ShaderCode &output) {
            if (name.empty())
                return true;
            const LoadedStageArtifact &artifact = inputs.artifacts.at(name);
            if (artifact.binary.empty()) {
                invocationDiagnostic(context) = "DirectX 12 stage artifact has no DXIL";
                return false;
            }
            output.bytecode.assign(artifact.binary.begin(), artifact.binary.end());
            output.entry = artifact.entry;
            return true;
        };
        if (!code(plan.compute, state->compute) || !code(plan.vertex, state->vertex) ||
            !code(plan.fragment, state->fragment))
            return false;
        installRuntimeBackendState(pipeline, state.release());
        return true;
#else
        (void)inputs;
        (void)plan;
        (void)pipeline;
        return false;
#endif
    };
    return resolve() ? BackendPipelineResult{vernon::ok()} : backendPipelineResolutionFailure(inputs);
}

} // namespace vernon::runtime
