#ifndef VERNON_RHI_METAL_BACKEND_FWD_H
#define VERNON_RHI_METAL_BACKEND_FWD_H

#include <cstdint>

namespace vernon::rhi::metal {

struct DeviceState;
struct DeviceCapabilities {
    uint32_t maxComputeInvocations{};
    uint32_t maxComputeWorkGroupSize[3]{};
    uint32_t operatingSystemVersion[2]{};
    uint32_t argumentBuffersTier{};
};

} // namespace vernon::rhi::metal

#endif
