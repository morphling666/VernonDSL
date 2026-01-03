#pragma once

#include "mlir/Pass/Pass.h"

namespace mlir::vernon {

std::unique_ptr<Pass> createVernonToSPIRVPass();

void registerVernonToSPIRVPass();

} // namespace mlir::vernon