#pragma once

#include "mlir/Pass/Pass.h"

namespace mlir::vernon {

enum class SpirvInterfaceAbi {
    Vulkan,
    // Inline uniforms share the Vulkan push-constant block, which SPIRV-Cross emits as the root-constant cbuffer.
    DirectX,
    // Inline uniforms stay loose UniformConstant variables, which OpenGL and Metal bind natively.
    NativeUniforms,
};

std::unique_ptr<Pass> createVernonToSPIRVPass(SpirvInterfaceAbi abi = SpirvInterfaceAbi::Vulkan);

void registerVernonToSPIRVPass();

} // namespace mlir::vernon
