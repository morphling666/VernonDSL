#ifndef VERNON_RUNTIME_RUNTIME_TEST_HOOKS_H
#define VERNON_RUNTIME_RUNTIME_TEST_HOOKS_H

#include "VernonRuntime.h"

#include <cstddef>
#include <cstdint>

namespace vernon::runtime {

// Internal counters for runtime integration tests. This header intentionally
// has no Vulkan dependency so non-Vulkan builds remain compile-safe.
struct VulkanGraphicsCacheStats {
    size_t defaultImplicitSamplerCreations{};
    size_t descriptorSetLayoutCreations{};
    size_t pipelineLayoutCreations{};
    size_t graphicsPipelineCreations{};
    size_t bindingSnapshotCreations{};
    size_t commandBufferAllocations{};
    size_t descriptorPoolCreations{};
    size_t stagingBufferAllocations{};
    size_t renderPassCreations{};
    uint32_t lastStencilReference{};
    bool lastDrawIndexed{};
    bool dynamicRendering{};
};

struct DirectX12DepthStencilStateStats {
    uint32_t depthEnable{};
    uint32_t depthWriteMask{};
    uint32_t depthFunction{};
    uint32_t stencilEnable{};
    uint32_t stencilReadMask{};
    uint32_t stencilWriteMask{};
    uint32_t frontStencilFunction{};
    uint32_t frontStencilPassOperation{};
    uint32_t backStencilFunction{};
    uint32_t backStencilPassOperation{};
};

VulkanGraphicsCacheStats getVulkanGraphicsCacheStats(const VernonRuntimeContext *context,
                                                     const VernonProgramExecutable *pipeline);
size_t getDirectX12GraphicsPipelineCreationCount(const VernonProgramExecutable *pipeline);
size_t getDirectX12GraphicsRootSignatureCreationCount(const VernonProgramExecutable *pipeline);
uint32_t getDirectX12LastStencilReference(const VernonRuntimeContext *context);
DirectX12DepthStencilStateStats getDirectX12DepthStencilStateStats(const VernonRuntimeContext *context);
size_t getRhiAdapterLivePreparedPipelineCount(const VernonRuntimeContext *context);
size_t getRhiAdapterRecordedCommandCount(const VernonRuntimeContext *context);
bool validateMetalArgumentBufferLimitsForTesting(uint64_t buffers, uint64_t textures, uint64_t samplers,
                                                 bool writableTexture, uint32_t deviceTier);
uint64_t diagnosticClockForTesting() noexcept;
void setDiagnosticClockForTesting(uint64_t clock) noexcept;
void failNextDiagnosticOverflowAllocationForTesting() noexcept;
uint64_t diagnosticGenerationCounterForTesting() noexcept;
void setDiagnosticGenerationCounterForTesting(uint64_t generation) noexcept;

} // namespace vernon::runtime

#endif
