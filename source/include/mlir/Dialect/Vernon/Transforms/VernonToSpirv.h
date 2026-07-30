#pragma once

#include "mlir/Pass/Pass.h"

namespace mlir::vernon {

std::unique_ptr<Pass> createVernonToSPIRVPass(bool aggregatePushConstants = true);

void registerVernonToSPIRVPass();

} // namespace mlir::vernon