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

#if defined(VERNON_RUNTIME_TESTING)
#include "rhi/rhi_test_hooks.h"
#include "runtime_test_hooks.h"
#endif

namespace vernon::runtime {

bool resolveBackendPipeline(VernonPipelineBundle &bundle, const Variant &variant, VernonLoadedPipeline &pipeline) {
    if (bundle.context->backend == VERNON_RUNTIME_CPU)
        return resolveCpuPipeline(bundle, variant, pipeline);
    if (bundle.context->backend == VERNON_RUNTIME_CUDA)
        return resolveCudaPipeline(bundle, variant, pipeline);
    if (bundle.context->backend == VERNON_RUNTIME_VULKAN)
        return resolveVulkanPipeline(bundle, variant, pipeline);
    if (bundle.context->backend == VERNON_RUNTIME_DIRECTX12)
        return resolveDirectX12Pipeline(bundle, variant, pipeline);
    if (isOpenGLBackend(bundle.context->backend))
        return resolveOpenGLPipeline(bundle, variant, pipeline);
    bundle.context->error = "unsupported runtime pipeline backend";
    return false;
}

void destroyBackendPipeline(VernonLoadedPipeline &pipeline) {
    if (pipeline.context->backend == VERNON_RUNTIME_CPU) {
        destroyCpuPipeline(pipeline);
    } else if (pipeline.context->backend == VERNON_RUNTIME_CUDA) {
        destroyCudaPipeline(pipeline);
    } else if (pipeline.context->backend == VERNON_RUNTIME_DIRECTX12) {
        destroyDirectX12Pipeline(pipeline);
    } else if (pipeline.context->backend == VERNON_RUNTIME_VULKAN) {
        destroyVulkanPipeline(pipeline);
    } else {
        destroyOpenGLPipeline(pipeline);
    }
    destroyRuntimeBackendState(pipeline);
}

VernonStatus invokeBackendPipeline(VernonLoadedPipeline &pipeline, const VernonPipelineInvocation &invocation,
                                   const PlannedGraphicsInvocation &plan) {
    if (pipeline.context->backend == VERNON_RUNTIME_CPU || pipeline.context->backend == VERNON_RUNTIME_CUDA)
        return VERNON_STATUS_OK;
    if (pipeline.context->backend == VERNON_RUNTIME_VULKAN)
        return invokeVulkanGraphicsPipeline(pipeline, invocation, plan);
    if (pipeline.context->backend == VERNON_RUNTIME_DIRECTX12)
        return invokeDirectX12GraphicsPipeline(pipeline, invocation, plan);
    return invokeOpenGLGraphicsPipeline(pipeline, invocation, plan);
}

VernonStatus invokeBackendComputePipeline(VernonLoadedPipeline &pipeline, const PlannedComputeLaunch &plan) {
    if (pipeline.context->backend == VERNON_RUNTIME_CPU)
        return invokeCpuComputePipeline(pipeline, plan);
    if (pipeline.context->backend == VERNON_RUNTIME_CUDA)
        return invokeCudaComputePipeline(pipeline, plan);
    if (pipeline.context->backend == VERNON_RUNTIME_VULKAN)
        return invokeVulkanComputePipeline(pipeline, plan);
    if (pipeline.context->backend == VERNON_RUNTIME_DIRECTX12)
        return invokeDirectX12ComputePipeline(pipeline, plan);
    if (!isOpenGLBackend(pipeline.context->backend))
        return VERNON_STATUS_UNSUPPORTED_TARGET;
    return invokeOpenGLComputePipeline(pipeline, plan);
}

VernonStatus synchronizeBackend(VernonRuntimeContext &context) {
    if (context.backend == VERNON_RUNTIME_CPU)
        return VERNON_STATUS_OK;
    switch (vernonRhiDeviceSynchronize(context.rhiDevice)) {
    case VERNON_RHI_STATUS_OK:
        return VERNON_STATUS_OK;
    case VERNON_RHI_STATUS_INVALID_ARGUMENT:
        return VERNON_STATUS_INVALID_ARGUMENT;
    case VERNON_RHI_STATUS_UNSUPPORTED:
        return VERNON_STATUS_UNSUPPORTED_TARGET;
    case VERNON_RHI_STATUS_INTERNAL_ERROR:
        return VERNON_STATUS_INTERNAL_ERROR;
    }
    return VERNON_STATUS_INTERNAL_ERROR;
}

#if defined(VERNON_RUNTIME_TESTING)
VulkanGraphicsCacheStats getVulkanGraphicsCacheStats(const VernonRuntimeContext *context,
                                                     const VernonLoadedPipeline *pipeline) {
    VulkanGraphicsCacheStats result;
#if defined(VERNON_HAS_VULKAN_RUNTIME)
    if (!context || !pipeline || pipeline->context != context || context->backend != VERNON_RUNTIME_VULKAN)
        return result;
    const vernon::rhi::VulkanCacheStats rhiStats = vernon::rhi::getVulkanCacheStats(context->rhiDevice);
    result.defaultImplicitSamplerCreations = rhiStats.defaultImplicitSamplerCreations;
    const VulkanPipelineState &state = runtimeBackendState<VulkanPipelineState>(*pipeline);
    result.descriptorSetLayoutCreations = state.rhiGraphicsPipeline ? 1 : 0;
    result.pipelineLayoutCreations = state.rhiGraphicsPipeline ? 1 : 0;
    result.graphicsPipelineCreations = state.rhiGraphicsVariant ? 1 : 0;
    result.commandBufferAllocations = rhiStats.commandBufferAllocations;
    result.descriptorPoolCreations = rhiStats.descriptorPoolCreations;
    result.stagingBufferAllocations = rhiStats.stagingBufferAllocations;
    result.renderPassCreations = state.rhiGraphicsVariant && !rhiStats.dynamicRendering ? 1 : 0;
    result.dynamicRendering = rhiStats.dynamicRendering;
#else
    (void)context;
    (void)pipeline;
#endif
    return result;
}

size_t getDirectX12GraphicsPipelineCreationCount(const VernonLoadedPipeline *pipeline) {
#if defined(VERNON_HAS_DIRECTX12_RUNTIME)
    if (pipeline && pipeline->context && pipeline->context->backend == VERNON_RUNTIME_DIRECTX12)
        return runtimeBackendState<DirectX12PipelineState>(*pipeline).rhiGraphicsVariant ? 1 : 0;
#else
    (void)pipeline;
#endif
    return 0;
}

size_t getDirectX12GraphicsRootSignatureCreationCount(const VernonLoadedPipeline *pipeline) {
#if defined(VERNON_HAS_DIRECTX12_RUNTIME)
    if (pipeline && pipeline->context && pipeline->context->backend == VERNON_RUNTIME_DIRECTX12)
        return runtimeBackendState<DirectX12PipelineState>(*pipeline).rhiGraphicsVariant ? 1 : 0;
#else
    (void)pipeline;
#endif
    return 0;
}
#endif

} // namespace vernon::runtime
