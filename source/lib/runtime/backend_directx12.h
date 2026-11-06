#ifndef VERNON_RUNTIME_BACKEND_DIRECTX12_H
#define VERNON_RUNTIME_BACKEND_DIRECTX12_H

#include "VernonRuntime.h"
#include "prepared_binding_plan.h"
#include "runtime_state.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace vernon::hal::d3d12 {
struct DeviceState;
} // namespace vernon::hal::d3d12

namespace vernon::runtime {

#if defined(VERNON_HAS_DIRECTX12_RUNTIME)
// The D3D12 state of the context's HAL device (hal/d3d12/d3d12_state.h).
struct DirectX12ContextState {
    std::shared_ptr<hal::d3d12::DeviceState> device;
};

inline DirectX12ContextState &directX12State(VernonRuntimeContext &context) {
    return runtimeBackendState<DirectX12ContextState>(context);
}

inline const DirectX12ContextState &directX12State(const VernonRuntimeContext &context) {
    return runtimeBackendState<DirectX12ContextState>(context);
}

// The DXIL of one shader stage, which the HAL device creates its own pipelines from.
struct DirectX12ShaderCode {
    std::vector<uint8_t> bytecode;
    std::string entry;
};

// A graphics Stage has vertex and fragment code and `graphicsPlan`; a compute Stage has compute code and `computePlan`.
struct DirectX12PipelineState {
    PreparedComputeBindingPlan computePlan;
    PreparedGraphicsBindingPlan graphicsPlan;
    DirectX12ShaderCode compute;
    DirectX12ShaderCode vertex;
    DirectX12ShaderCode fragment;
    // The HAL device's root signatures and pipelines of the Stage, made on first use under `halMutex`
    // (hal/d3d12/d3d12_device.cpp).
    std::mutex halMutex;
    std::shared_ptr<void> hal;
};
#endif

} // namespace vernon::runtime

#endif
