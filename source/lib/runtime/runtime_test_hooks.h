#ifndef VERNON_RUNTIME_RUNTIME_TEST_HOOKS_H
#define VERNON_RUNTIME_RUNTIME_TEST_HOOKS_H

#include "VernonRuntime.h"

#include <cstddef>

namespace vernon::runtime {

// Internal counters for runtime integration tests. This header intentionally
// has no Vulkan dependency so non-Vulkan builds remain compile-safe.
struct VulkanGraphicsCacheStats {
    size_t defaultImplicitSamplerCreations{};
    size_t descriptorSetLayoutCreations{};
    size_t pipelineLayoutCreations{};
    size_t graphicsPipelineCreations{};
    size_t commandBufferAllocations{};
    size_t descriptorPoolCreations{};
    size_t stagingBufferAllocations{};
    size_t renderPassCreations{};
    bool dynamicRendering{};
};

VERNON_RUNTIME_CAPI VulkanGraphicsCacheStats getVulkanGraphicsCacheStats(const VernonRuntimeContext *context,
                                                                         const VernonLoadedPipeline *pipeline);
VERNON_RUNTIME_CAPI void setDirectX12WarpForTests(bool enabled);
VERNON_RUNTIME_CAPI size_t getDirectX12GraphicsPipelineCreationCount(const VernonLoadedPipeline *pipeline);
VERNON_RUNTIME_CAPI size_t getDirectX12GraphicsRootSignatureCreationCount(const VernonLoadedPipeline *pipeline);

} // namespace vernon::runtime

#endif
