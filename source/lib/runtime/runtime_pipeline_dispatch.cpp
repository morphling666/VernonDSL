#include "runtime_dispatch.h"

#include "backend_opengl.h"
#include "resolved_execution_plan.h"
#include "runtime_pipeline_backend.h"

#include <algorithm>

#if defined(VERNON_HAS_DIRECTX12_RUNTIME)
#include "backend_directx12.h"
#endif

#if defined(VERNON_HAS_CUDA_RUNTIME)
#include "backend_cuda.h"
#endif

#if defined(VERNON_HAS_VULKAN_RUNTIME)
#include "backend_vulkan.h"
#endif

#if defined(VERNON_HAS_METAL_RUNTIME)
#include "backend_metal.h"
#endif

#if defined(VERNON_RUNTIME_TESTING)
#include "rhi/rhi_test_hooks.h"
#include "rhi_adapter/adapter_test_hooks.h"
#if defined(VERNON_HAS_DIRECTX12_RUNTIME)
#include "rhi_adapter/adapter_directx12_test_hooks.h"
#endif
#include "runtime_test_hooks.h"
#endif

namespace vernon::runtime {

bool resolveBackendPipeline(BackendStageBuildInputs &inputs, const StageBindingPlan &plan,
                            VernonStageExecutable &pipeline) {
    std::string error;
    if (!inputs.context || !validateStageBindingPlan(plan, error)) {
        if (inputs.context)
            invocationDiagnostic(*inputs.context) = std::move(error);
        return false;
    }
    if (inputs.context->backend == VERNON_RUNTIME_CPU)
        return resolveCpuPipeline(inputs, plan, pipeline);
    if (inputs.context->backend == VERNON_RUNTIME_CUDA)
        return resolveCudaPipeline(inputs, plan, pipeline);
    if (inputs.context->backend == VERNON_RUNTIME_VULKAN)
        return resolveVulkanPipeline(inputs, plan, pipeline);
    if (inputs.context->backend == VERNON_RUNTIME_DIRECTX12)
        return resolveDirectX12Pipeline(inputs, plan, pipeline);
    if (inputs.context->backend == VERNON_RUNTIME_METAL)
        return resolveMetalPipeline(inputs, plan, pipeline);
    if (isOpenGLBackend(inputs.context->backend))
        return resolveOpenGLPipeline(inputs, plan, pipeline);
    invocationDiagnostic(*inputs.context) = "unsupported runtime pipeline backend";
    return false;
}

void destroyBackendPipeline(VernonStageExecutable &pipeline) {
    if (pipeline.context->backend == VERNON_RUNTIME_CPU) {
        destroyCpuPipeline(pipeline);
    } else if (pipeline.context->backend == VERNON_RUNTIME_CUDA) {
        destroyCudaPipeline(pipeline);
    } else if (pipeline.context->backend == VERNON_RUNTIME_DIRECTX12) {
        destroyDirectX12Pipeline(pipeline);
    } else if (pipeline.context->backend == VERNON_RUNTIME_VULKAN) {
        destroyVulkanPipeline(pipeline);
    } else if (pipeline.context->backend == VERNON_RUNTIME_METAL) {
        destroyMetalPipeline(pipeline);
    } else {
        destroyOpenGLPipeline(pipeline);
    }
    destroyRuntimeBackendState(pipeline);
}

VernonStatus invokeBackendPipeline(VernonStageExecutable &pipeline, const VernonStageInvocationDescriptor &invocation,
                                   const PlannedGraphicsInvocation &plan) {
    if (pipeline.context->backend == VERNON_RUNTIME_CPU || pipeline.context->backend == VERNON_RUNTIME_CUDA)
        return VERNON_STATUS_OK;
    if (pipeline.context->backend == VERNON_RUNTIME_VULKAN)
        return invokeVulkanGraphicsPipeline(pipeline, invocation, plan);
    if (pipeline.context->backend == VERNON_RUNTIME_DIRECTX12)
        return invokeDirectX12GraphicsPipeline(pipeline, invocation, plan);
    if (pipeline.context->backend == VERNON_RUNTIME_METAL)
        return invokeMetalGraphicsPipeline(pipeline, invocation, plan);
    return invokeOpenGLGraphicsPipeline(pipeline, invocation, plan);
}

VernonStatus invokeBackendComputePipeline(VernonStageExecutable &pipeline, const PlannedComputeLaunch &plan) {
    if (pipeline.context->backend == VERNON_RUNTIME_CPU)
        return invokeCpuComputePipeline(pipeline, plan);
    if (pipeline.context->backend == VERNON_RUNTIME_CUDA)
        return invokeCudaComputePipeline(pipeline, plan);
    if (pipeline.context->backend == VERNON_RUNTIME_VULKAN)
        return invokeVulkanComputePipeline(pipeline, plan);
    if (pipeline.context->backend == VERNON_RUNTIME_DIRECTX12)
        return invokeDirectX12ComputePipeline(pipeline, plan);
    if (pipeline.context->backend == VERNON_RUNTIME_METAL)
        return invokeMetalComputePipeline(pipeline, plan);
    if (!isOpenGLBackend(pipeline.context->backend))
        return VERNON_STATUS_UNSUPPORTED_TARGET;
    return invokeOpenGLComputePipeline(pipeline, plan);
}

#if defined(VERNON_RUNTIME_TESTING)
const VernonStageExecutable *resolvedGraphicsImplementation(const VernonProgramExecutable *pipeline) {
    if (!pipeline)
        return nullptr;
    const program::ResolvedExecutionPlan &execution = *pipeline->executionPlan;
    const auto stage = std::find_if(execution.stageCache.begin(), execution.stageCache.end(),
                                    [](const std::shared_ptr<VernonStageExecutable> &candidate) {
                                        return candidate && candidate->backendState &&
                                               (!candidate->bindingProjection.vertex.empty() ||
                                                !candidate->bindingProjection.fragment.empty());
                                    });
    return stage == execution.stageCache.end() ? nullptr : stage->get();
}

VulkanGraphicsCacheStats getVulkanGraphicsCacheStats(const VernonRuntimeContext *context,
                                                     const VernonProgramExecutable *pipeline) {
    VulkanGraphicsCacheStats result;
#if defined(VERNON_HAS_VULKAN_RUNTIME)
    if (!context || !pipeline || pipeline->context != context || context->backend != VERNON_RUNTIME_VULKAN)
        return result;
    const VernonStageExecutable *stage = resolvedGraphicsImplementation(pipeline);
    if (!stage)
        return result;
    const vernon::rhi::VulkanCacheStats rhiStats = vernon::rhi::getVulkanCacheStats(context->rhiDevice);
    result.defaultImplicitSamplerCreations = rhiStats.defaultImplicitSamplerCreations;
    const VulkanPipelineState &state = runtimeBackendState<VulkanPipelineState>(*stage);
    result.descriptorSetLayoutCreations = state.rhiGraphicsPipeline ? 1 : 0;
    result.pipelineLayoutCreations = state.rhiGraphicsPipeline ? 1 : 0;
    result.graphicsPipelineCreations = state.rhiGraphicsVariant.handle ? 1 : 0;
    const RhiAdapterPreparationStats adapterStats = getRhiAdapterPreparationStats(*vulkanState(*context).adapter);
    result.bindingSnapshotCreations = adapterStats.bindingSnapshotCreations;
    result.commandBufferAllocations = rhiStats.commandBufferAllocations;
    result.descriptorPoolCreations = rhiStats.descriptorPoolCreations;
    result.stagingBufferAllocations = rhiStats.stagingBufferAllocations;
    result.renderPassCreations = state.rhiGraphicsVariant.handle && !rhiStats.dynamicRendering ? 1 : 0;
    result.lastStencilReference = adapterStats.lastStencilReference;
    result.lastDrawIndexed = adapterStats.lastDrawIndexed;
    result.dynamicRendering = rhiStats.dynamicRendering;
#else
    (void)context;
    (void)pipeline;
#endif
    return result;
}

size_t getDirectX12GraphicsPipelineCreationCount(const VernonProgramExecutable *pipeline) {
#if defined(VERNON_HAS_DIRECTX12_RUNTIME)
    const VernonStageExecutable *stage = resolvedGraphicsImplementation(pipeline);
    if (stage && stage->context && stage->context->backend == VERNON_RUNTIME_DIRECTX12)
        return runtimeBackendState<DirectX12PipelineState>(*stage).rhiGraphicsVariant.handle ? 1 : 0;
#else
    (void)pipeline;
#endif
    return 0;
}

size_t getDirectX12GraphicsRootSignatureCreationCount(const VernonProgramExecutable *pipeline) {
#if defined(VERNON_HAS_DIRECTX12_RUNTIME)
    const VernonStageExecutable *stage = resolvedGraphicsImplementation(pipeline);
    if (stage && stage->context && stage->context->backend == VERNON_RUNTIME_DIRECTX12)
        return runtimeBackendState<DirectX12PipelineState>(*stage).rhiGraphicsVariant.handle ? 1 : 0;
#else
    (void)pipeline;
#endif
    return 0;
}

uint32_t getDirectX12LastStencilReference(const VernonRuntimeContext *context) {
#if defined(VERNON_HAS_DIRECTX12_RUNTIME)
    if (context && context->backend == VERNON_RUNTIME_DIRECTX12) {
        const VernonRuntimeRhiAdapter *adapter = runtimeBackendState<DirectX12ContextState>(*context).adapter;
        return adapter ? getRhiAdapterPreparationStats(*adapter).lastStencilReference : 0;
    }
#else
    (void)context;
#endif
    return 0;
}

DirectX12DepthStencilStateStats getDirectX12DepthStencilStateStats(const VernonRuntimeContext *context) {
    DirectX12DepthStencilStateStats result;
#if defined(VERNON_HAS_DIRECTX12_RUNTIME)
    if (context && context->backend == VERNON_RUNTIME_DIRECTX12) {
        const VernonRuntimeRhiAdapter *adapter = runtimeBackendState<DirectX12ContextState>(*context).adapter;
        if (adapter) {
            const DirectX12AdapterDepthStencilStats stats = getDirectX12AdapterDepthStencilStats(*adapter);
            result = {stats.depthEnable,          stats.depthWriteMask,
                      stats.depthFunction,        stats.stencilEnable,
                      stats.stencilReadMask,      stats.stencilWriteMask,
                      stats.frontStencilFunction, stats.frontStencilPassOperation,
                      stats.backStencilFunction,  stats.backStencilPassOperation};
        }
    }
#else
    (void)context;
#endif
    return result;
}

size_t getRhiAdapterLivePreparedPipelineCount(const VernonRuntimeContext *context) {
    if (!context)
        return 0;
    const VernonRuntimeRhiAdapter *adapter = nullptr;
    if (context->backend == VERNON_RUNTIME_OPENGL || context->backend == VERNON_RUNTIME_OPENGL_ES)
        adapter = runtimeBackendState<OpenGLContextState>(*context).adapter;
#if defined(VERNON_HAS_VULKAN_RUNTIME)
    else if (context->backend == VERNON_RUNTIME_VULKAN)
        adapter = runtimeBackendState<VulkanContextState>(*context).adapter;
#endif
#if defined(VERNON_HAS_METAL_RUNTIME)
    else if (context->backend == VERNON_RUNTIME_METAL)
        adapter = runtimeBackendState<MetalContextState>(*context).adapter;
#endif
#if defined(VERNON_HAS_DIRECTX12_RUNTIME)
    else if (context->backend == VERNON_RUNTIME_DIRECTX12)
        adapter = runtimeBackendState<DirectX12ContextState>(*context).adapter;
#endif
    return adapter ? getRhiAdapterPreparationStats(*adapter).livePreparedPipelines : 0;
}

size_t getRhiAdapterRecordedCommandCount(const VernonRuntimeContext *context) {
    if (!context)
        return 0;
    const VernonRuntimeRhiAdapter *adapter = nullptr;
    if (context->backend == VERNON_RUNTIME_OPENGL || context->backend == VERNON_RUNTIME_OPENGL_ES)
        adapter = runtimeBackendState<OpenGLContextState>(*context).adapter;
#if defined(VERNON_HAS_VULKAN_RUNTIME)
    else if (context->backend == VERNON_RUNTIME_VULKAN)
        adapter = runtimeBackendState<VulkanContextState>(*context).adapter;
#endif
#if defined(VERNON_HAS_DIRECTX12_RUNTIME)
    else if (context->backend == VERNON_RUNTIME_DIRECTX12)
        adapter = runtimeBackendState<DirectX12ContextState>(*context).adapter;
#endif
#if defined(VERNON_HAS_CUDA_RUNTIME)
    else if (context->backend == VERNON_RUNTIME_CUDA)
        adapter = runtimeBackendState<CudaContextState>(*context).adapter;
#endif
    return adapter ? getRhiAdapterPreparationStats(*adapter).dispatches : 0;
}
#endif

} // namespace vernon::runtime
