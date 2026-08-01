#ifndef VERNON_RUNTIME_RHI_ADAPTER_DIRECTX12_TEST_HOOKS_H
#define VERNON_RUNTIME_RHI_ADAPTER_DIRECTX12_TEST_HOOKS_H

#include "VernonRuntimeRHIAdapter.h"

#include <cstdint>

namespace vernon::runtime {

struct DirectX12AdapterDepthStencilStats {
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

DirectX12AdapterDepthStencilStats getDirectX12AdapterDepthStencilStats(const VernonRuntimeRhiAdapter &adapter) noexcept;

} // namespace vernon::runtime

#endif
