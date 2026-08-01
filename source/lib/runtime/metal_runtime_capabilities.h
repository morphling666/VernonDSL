#ifndef VERNON_RUNTIME_METAL_RUNTIME_CAPABILITIES_H
#define VERNON_RUNTIME_METAL_RUNTIME_CAPABILITIES_H

#include <cstdint>

struct VernonRuntimeRhiAdapter;

namespace vernon::runtime {

struct MetalRuntimeDeviceCapabilities {
    uint32_t maxComputeInvocations{};
    uint32_t maxComputeWorkGroupSize[3]{};
    uint32_t operatingSystemVersion[2]{};
    uint32_t argumentBuffersTier{};
    bool argumentBufferEncodingSupported{};
};

MetalRuntimeDeviceCapabilities metalRhiAdapterDeviceCapabilities(const VernonRuntimeRhiAdapter &adapter);

} // namespace vernon::runtime

#endif
