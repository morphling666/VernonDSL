#include "runtime_pipeline_backend.h"

#if defined(VERNON_HAS_VULKAN_RUNTIME)
#include "backend_vulkan.h"

#include <cstring>
#include <memory>
#endif

namespace vernon::runtime {

BackendPipelineResult resolveVulkanPipeline(BackendStageBuildInputs &inputs, const StageBindingPlan &plan,
                                            VernonStageExecutable &pipeline) {
    const auto resolve = [&]() {
#if defined(VERNON_HAS_VULKAN_RUNTIME)
        VernonRuntimeContext &context = *inputs.context;
        auto state = std::make_unique<VulkanPipelineState>();
        if (!plan.compute.empty() &&
            !prepareComputeBindingPlan(context, inputs.artifacts.at(plan.compute), plan, state->computePlan))
            return false;
        if (!plan.vertex.empty() &&
            !buildPreparedGraphicsBindingPlan(plan, {VERNON_RUNTIME_VULKAN}, state->graphicsPlan,
                                              invocationDiagnostic(context)))
            return false;
        const auto code = [&](const std::string &name, VulkanShaderCode &output) {
            if (name.empty())
                return true;
            const LoadedStageArtifact &artifact = inputs.artifacts.at(name);
            if (artifact.binary.empty() || artifact.binary.size() % sizeof(uint32_t) != 0) {
                invocationDiagnostic(context) = "Vulkan stage artifact is not SPIR-V words";
                return false;
            }
            output.words.resize(artifact.binary.size() / sizeof(uint32_t));
            std::memcpy(output.words.data(), artifact.binary.data(), artifact.binary.size());
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
