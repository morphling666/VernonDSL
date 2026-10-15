#include "runtime_pipeline_backend.h"

#if defined(VERNON_HAS_DIRECTX12_RUNTIME)
#include "backend_directx12.h"

#include <memory>
#include <optional>
#endif

namespace vernon::runtime {

#if defined(VERNON_HAS_DIRECTX12_RUNTIME)
namespace {

std::optional<uint32_t> directX12AttachmentFormat(VernonTextureFormat format) {
    switch (format) {
#define VERNON_FORMAT(name, NAME, rhi, texture, cls, bytes, dataFormat, dataType, caps, glInternal, glFormat, glType,  \
                      vulkan, dxgi, ...)                                                                               \
    case VERNON_TEXTURE_##NAME:                                                                                        \
        return dxgi == DXGI_FORMAT_UNKNOWN ? std::nullopt : std::optional<uint32_t>(dxgi);
#include "VernonFormats.def"
    }
    return std::nullopt;
}

constexpr RhiComputeTarget kDirectX12Compute{"D3D12", "dxil"};
constexpr RhiGraphicsTarget kDirectX12Graphics{"D3D12", "dxil", directX12AttachmentFormat, directX12AttachmentFormat};

} // namespace
#endif

BackendPipelineResult resolveDirectX12Pipeline(BackendStageBuildInputs &inputs, const StageBindingPlan &plan,
                                               VernonStageExecutable &pipeline) {
    const auto resolve = [&]() {
#if defined(VERNON_HAS_DIRECTX12_RUNTIME)
        VernonRuntimeContext &context = *inputs.context;
        auto state = std::make_unique<DirectX12PipelineState>();
        const bool prepared =
            plan.compute.empty()
                ? prepareRhiGraphicsPipeline(context, directX12State(context).adapter, kDirectX12Graphics,
                                             inputs.artifacts.at(plan.vertex), inputs.artifacts.at(plan.fragment), plan,
                                             {VERNON_RUNTIME_DIRECTX12}, state->rhiGraphics)
                : prepareRhiComputePipeline(context, directX12State(context).adapter, kDirectX12Compute,
                                            inputs.artifacts.at(plan.compute), plan, state->rhiCompute);
        if (!prepared)
            return false;
        installRuntimeBackendState(pipeline, state.release());
        return true;
#else
        return false;
#endif
    };
    return resolve() ? BackendPipelineResult{vernon::ok()} : backendPipelineResolutionFailure(inputs);
}

VernonStatus invokeDirectX12GraphicsPipeline(VernonStageExecutable &pipeline,
                                             const VernonStageInvocationDescriptor &invocation,
                                             const PlannedGraphicsInvocation &plan) {
#if defined(VERNON_HAS_DIRECTX12_RUNTIME)
    return invokeRhiGraphicsPipeline(*pipeline.context, directX12State(*pipeline.context).adapter, kDirectX12Graphics,
                                     invocation, plan,
                                     runtimeBackendState<DirectX12PipelineState>(pipeline).rhiGraphics);
#else
    (void)pipeline;
    (void)invocation;
    (void)plan;
    return VERNON_STATUS_UNSUPPORTED_TARGET;
#endif
}

VernonStatus invokeDirectX12ComputePipeline(VernonStageExecutable &pipeline, const PlannedComputeLaunch &launch) {
#if defined(VERNON_HAS_DIRECTX12_RUNTIME)
    return invokeRhiComputePipeline(*pipeline.context, directX12State(*pipeline.context).adapter, kDirectX12Compute,
                                    pipeline.bindingProjection, launch,
                                    runtimeBackendState<DirectX12PipelineState>(pipeline).rhiCompute);
#else
    (void)pipeline;
    (void)launch;
    return VERNON_STATUS_UNSUPPORTED_TARGET;
#endif
}

} // namespace vernon::runtime
