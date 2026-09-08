#pragma once

#include "mlir/Pass/Pass.h"

#include <memory>

namespace mlir::vernon::program {

std::unique_ptr<Pass> createVernonProgramSelectImplementationsPass();
void registerVernonProgramSelectImplementationsPass();

} // namespace mlir::vernon::program
