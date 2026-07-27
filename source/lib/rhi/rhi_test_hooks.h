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

VERNON_RHI_CAPI VulkanCacheStats getVulkanCacheStats(VernonRhiDevice device);
VERNON_RHI_CAPI uint64_t getTrackedBufferState(VernonRhiDevice device, VernonRhiBuffer buffer);

} // namespace vernon::rhi

#endif
