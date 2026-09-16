#pragma once

#include "mlir/Pass/Pass.h"

namespace mlir::vernon {

std::unique_ptr<Pass> createVernonToGPUPass(bool useSpirvStorage = false, bool useSpirvWorkgroupReduction = false);
void registerVernonToGPUPass();

} // namespace mlir::vernon
