#ifndef VERNON_RHI_TEST_HOOKS_H
#define VERNON_RHI_TEST_HOOKS_H

#include "VernonRHI.h"

#include <cstddef>

namespace vernon::rhi {

struct VulkanCacheStats {
    size_t defaultImplicitSamplerCreations{};
    size_t commandBufferAllocations{};
    size_t descriptorPoolCreations{};
    size_t stagingBufferAllocations{};
    bool dynamicRendering{};
};

// Driver calls that wait for a threaded driver or create objects, which a steady-state frame should not make.
struct OpenGLCallStats {
    size_t errorReads{};
    size_t queryCreations{};
    size_t debugGroupPushes{};
};

// The provider bookkeeping calls an encoder received, one per packet batch, and its recorded debug commands.
struct EncoderCallStats {
    size_t providerRecords{};
    size_t debugCommands{};
};

VERNON_RHI_CAPI VulkanCacheStats getVulkanCacheStats(VernonRhiDevice device);
VERNON_RHI_CAPI OpenGLCallStats getOpenGLCallStats(VernonRhiDevice device);
// Raises GL_INVALID_ENUM on the device's context, as a failing operation would.
VERNON_RHI_CAPI void raiseOpenGLError(VernonRhiDevice device);
// Committed buffers the device has created, staging chunks included.
VERNON_RHI_CAPI size_t getDirectX12BufferCreationCount(VernonRhiDevice device);
// Device allocations the device has made, transient chunks included.
VERNON_RHI_CAPI size_t getCudaAllocationCount(VernonRhiDevice device);
VERNON_RHI_CAPI uint64_t getTrackedBufferState(VernonRhiDevice device, VernonRhiBuffer buffer);
VERNON_RHI_CAPI EncoderCallStats getEncoderCallStats(VernonRhiDevice device, VernonRhiCommandEncoder encoder);

} // namespace vernon::rhi

#endif
