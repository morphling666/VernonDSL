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

VernonStatus backendComputeToGraphicsBarrier(VernonRuntimeContext &context) {
    return isOpenGLBackend(context.backend) ? openGLComputeToGraphicsBarrier(context)
                                            : VERNON_STATUS_UNSUPPORTED_TARGET;
}

VernonStatus synchronizeBackend(VernonRuntimeContext &context) {
    switch (context.backend) {
    case VERNON_RUNTIME_CPU:
        return VERNON_STATUS_OK;
    case VERNON_RUNTIME_CUDA:
#if defined(VERNON_HAS_CUDA_RUNTIME)
        return synchronizeCuda(context);
#else
        return VERNON_STATUS_UNSUPPORTED_TARGET;
#endif
    case VERNON_RUNTIME_VULKAN:
#if defined(VERNON_HAS_VULKAN_RUNTIME)
        return synchronizeVulkan(context);
#else
        return VERNON_STATUS_UNSUPPORTED_TARGET;
#endif
    case VERNON_RUNTIME_DIRECTX12:
#if defined(VERNON_HAS_DIRECTX12_RUNTIME)
        return synchronizeDirectX12(context);
#else
        return VERNON_STATUS_UNSUPPORTED_TARGET;
#endif
    case VERNON_RUNTIME_OPENGL:
    case VERNON_RUNTIME_OPENGL_ES:
        return synchronizeOpenGL(context);
    default:
        return VERNON_STATUS_UNSUPPORTED_TARGET;
    }
}

#if defined(VERNON_RUNTIME_TESTING)
VulkanGraphicsCacheStats getVulkanGraphicsCacheStats(const VernonRuntimeContext *context,
                                                     const VernonLoadedPipeline *pipeline) {
    VulkanGraphicsCacheStats result;
#if defined(VERNON_HAS_VULKAN_RUNTIME)
    if (!context || !pipeline || pipeline->context != context || context->backend != VERNON_RUNTIME_VULKAN)
        return result;
    result.defaultImplicitSamplerCreations = vulkanState(*context).defaultImplicitSamplerCreations;
    const VulkanPipelineState &state = runtimeBackendState<VulkanPipelineState>(*pipeline);
    result.descriptorSetLayoutCreations = state.rhiGraphicsPipeline ? 1 : 0;
    result.pipelineLayoutCreations = state.rhiGraphicsPipeline ? 1 : 0;
    result.graphicsPipelineCreations = state.rhiGraphicsVariant ? 1 : 0;
    result.commandBufferAllocations = vulkanState(*context).commandBufferAllocations;
    result.descriptorPoolCreations = vulkanState(*context).descriptorPoolCreations;
    result.stagingBufferAllocations = vulkanState(*context).stagingBufferAllocations;
    result.renderPassCreations = state.rhiGraphicsVariant && !vulkanState(*context).dynamicRendering ? 1 : 0;
    result.dynamicRendering = vulkanState(*context).dynamicRendering;
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
