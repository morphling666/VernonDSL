#ifndef VERNON_RUNTIME_BACKEND_METAL_H
#define VERNON_RUNTIME_BACKEND_METAL_H

#include "VernonRuntime.h"
#include "prepared_binding_plan.h"
#include "runtime_state.h"

#include <array>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace vernon::hal::metal {
struct DeviceState;
} // namespace vernon::hal::metal

namespace vernon::runtime {

#if defined(VERNON_HAS_METAL_RUNTIME)
// The buffer index of a stage's push constants in the MSL the compiler emits (compiler_spirv_cross.cpp).
inline constexpr uint32_t kMetalPushConstantBuffer = 15;

// The Metal state of the context's HAL device (hal/metal/metal_state.h).
struct MetalContextState {
    std::shared_ptr<hal::metal::DeviceState> device;
};

inline MetalContextState &metalState(VernonRuntimeContext &context) {
    return runtimeBackendState<MetalContextState>(context);
}

inline const MetalContextState &metalState(const VernonRuntimeContext &context) {
    return runtimeBackendState<MetalContextState>(context);
}

// The MSL of one shader stage, which the HAL device compiles its own pipelines from.
struct MetalShaderCode {
    std::string source;
    std::string entry;
};

// Where one layout of a Stage's binding plan binds in its stage's MSL: member `member` of argument buffer `set`, or
// direct buffer `directBuffer`. A binding the stage's MSL does not read, which SPIRV-Cross drops, binds at neither.
struct MetalBindingSlot {
    uint32_t member{UINT32_MAX};
    uint32_t directBuffer{UINT32_MAX};
};

// The stage roles of `MetalPipelineState::argumentBuffers`.
enum MetalStageRole : uint32_t { kMetalVertex = 0, kMetalFragment = 1, kMetalCompute = 2 };

// A graphics Stage has vertex and fragment code and `graphicsPlan`; a compute Stage has compute code and `computePlan`.
struct MetalPipelineState {
    PreparedComputeBindingPlan computePlan;
    PreparedGraphicsBindingPlan graphicsPlan;
    MetalShaderCode compute;
    MetalShaderCode vertex;
    MetalShaderCode fragment;
    // Indexed like the layouts of the Stage's plan.
    std::vector<MetalBindingSlot> slots;
    // Per stage role, bit g is set when the stage's MSL declares descriptor set g as argument buffer g.
    std::array<uint32_t, 3> argumentBuffers{};
    // The HAL device's libraries and pipelines of the Stage, made on first use under `halMutex`
    // (hal/metal/metal_device.mm).
    std::mutex halMutex;
    std::shared_ptr<void> hal;
};
#endif

} // namespace vernon::runtime

#endif
