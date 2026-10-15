#include "runtime_pipeline_backend.h"

#if defined(VERNON_HAS_VULKAN_RUNTIME)
#include "backend_vulkan.h"

#include <memory>
#include <optional>
#endif

namespace vernon::runtime {

#if defined(VERNON_HAS_VULKAN_RUNTIME)
namespace {

std::optional<uint32_t> vulkanAttachmentFormat(VernonTextureFormat format) {
    switch (format) {
#define VERNON_FORMAT(name, NAME, rhi, texture, cls, bytes, dataFormat, dataType, caps, glInternal, glFormat, glType,  \
                      vulkan, ...)                                                                                     \
    case VERNON_TEXTURE_##NAME:                                                                                        \
        return vulkan == VK_FORMAT_UNDEFINED ? std::nullopt : std::optional<uint32_t>(vulkan);
#include "VernonFormats.def"
    }
    return std::nullopt;
}

constexpr RhiComputeTarget kVulkanCompute{"Vulkan", "spirv"};
constexpr RhiGraphicsTarget kVulkanGraphics{"Vulkan", "spirv", vulkanAttachmentFormat, vulkanAttachmentFormat};

} // namespace
#endif

BackendPipelineResult resolveVulkanPipeline(BackendStageBuildInputs &inputs, const StageBindingPlan &plan,
                                            VernonStageExecutable &pipeline) {
    const auto resolve = [&]() {
#if defined(VERNON_HAS_VULKAN_RUNTIME)
        VernonRuntimeContext &context = *inputs.context;
        auto state = std::make_unique<VulkanPipelineState>();
        if (!plan.compute.empty() &&
            !prepareRhiComputePipeline(context, vulkanState(context).adapter, kVulkanCompute,
                                       inputs.artifacts.at(plan.compute), plan, state->rhiCompute))
            return false;
        if (!plan.vertex.empty() &&
            !prepareRhiGraphicsPipeline(context, vulkanState(context).adapter, kVulkanGraphics,
                                        inputs.artifacts.at(plan.vertex), inputs.artifacts.at(plan.fragment), plan,
                                        {VERNON_RUNTIME_VULKAN}, state->rhiGraphics))
            return false;
        installRuntimeBackendState(pipeline, state.release());
        return true;
#else
        return false;
#endif
    };
    return resolve() ? BackendPipelineResult{vernon::ok()} : backendPipelineResolutionFailure(inputs);
}

VernonStatus invokeVulkanGraphicsPipeline(VernonStageExecutable &pipeline,
                                          const VernonStageInvocationDescriptor &invocation,
                                          const PlannedGraphicsInvocation &plan) {
#if defined(VERNON_HAS_VULKAN_RUNTIME)
    return invokeRhiGraphicsPipeline(*pipeline.context, vulkanState(*pipeline.context).adapter, kVulkanGraphics,
                                     invocation, plan, runtimeBackendState<VulkanPipelineState>(pipeline).rhiGraphics);
#else
    (void)pipeline;
    (void)invocation;
    (void)plan;
    return VERNON_STATUS_UNSUPPORTED_TARGET;
#endif
}

VernonStatus invokeVulkanComputePipeline(VernonStageExecutable &pipeline, const PlannedComputeLaunch &launch) {
#if defined(VERNON_HAS_VULKAN_RUNTIME)
    return invokeRhiComputePipeline(*pipeline.context, vulkanState(*pipeline.context).adapter, kVulkanCompute,
                                    pipeline.bindingProjection, launch,
                                    runtimeBackendState<VulkanPipelineState>(pipeline).rhiCompute);
#else
    (void)pipeline;
    (void)launch;
    return VERNON_STATUS_UNSUPPORTED_TARGET;
#endif
}

} // namespace vernon::runtime
