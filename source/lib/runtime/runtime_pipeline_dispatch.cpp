#include "runtime_dispatch.h"

#include "backend_opengl.h"
#include "runtime_pipeline_backend.h"

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

bool resolveBackendPipeline(VernonProgramBundle &bundle, const Variant &variant, VernonProgramExecutable &pipeline) {
    if (bundle.context->backend == VERNON_RUNTIME_CPU)
        return resolveCpuPipeline(bundle, variant, pipeline);
    if (bundle.context->backend == VERNON_RUNTIME_CUDA)
        return resolveCudaPipeline(bundle, variant, pipeline);
    if (bundle.context->backend == VERNON_RUNTIME_VULKAN)
        return resolveVulkanPipeline(bundle, variant, pipeline);
    if (bundle.context->backend == VERNON_RUNTIME_DIRECTX12)
        return resolveDirectX12Pipeline(bundle, variant, pipeline);
    if (bundle.context->backend == VERNON_RUNTIME_METAL)
        return resolveMetalPipeline(bundle, variant, pipeline);
    if (isOpenGLBackend(bundle.context->backend))
        return resolveOpenGLPipeline(bundle, variant, pipeline);
    invocationDiagnostic(*bundle.context) = "unsupported runtime pipeline backend";
    return false;
}

void destroyBackendPipeline(VernonProgramExecutable &pipeline) {
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

VernonStatus invokeBackendPipeline(VernonProgramExecutable &pipeline, const VernonProgramSubmitDescriptor &invocation,
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

VernonStatus invokeBackendComputePipeline(VernonProgramExecutable &pipeline, const PlannedComputeLaunch &plan) {
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
VulkanGraphicsCacheStats getVulkanGraphicsCacheStats(const VernonRuntimeContext *context,
                                                     const VernonProgramExecutable *pipeline) {
    VulkanGraphicsCacheStats result;
#if defined(VERNON_HAS_VULKAN_RUNTIME)
    if (!context || !pipeline || pipeline->context != context || context->backend != VERNON_RUNTIME_VULKAN)
        return result;
    const vernon::rhi::VulkanCacheStats rhiStats = vernon::rhi::getVulkanCacheStats(context->rhiDevice);
    result.defaultImplicitSamplerCreations = rhiStats.defaultImplicitSamplerCreations;
    const VulkanPipelineState &state = runtimeBackendState<VulkanPipelineState>(*pipeline);
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
    if (pipeline && pipeline->context && pipeline->context->backend == VERNON_RUNTIME_DIRECTX12)
        return runtimeBackendState<DirectX12PipelineState>(*pipeline).rhiGraphicsVariant.handle ? 1 : 0;
#else
    (void)pipeline;
#endif
    return 0;
}

size_t getDirectX12GraphicsRootSignatureCreationCount(const VernonProgramExecutable *pipeline) {
#if defined(VERNON_HAS_DIRECTX12_RUNTIME)
    if (pipeline && pipeline->context && pipeline->context->backend == VERNON_RUNTIME_DIRECTX12)
        return runtimeBackendState<DirectX12PipelineState>(*pipeline).rhiGraphicsVariant.handle ? 1 : 0;
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
