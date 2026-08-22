#pragma once

#include "mlir/Pass/Pass.h"

#include <memory>

namespace mlir::vernon::program {

std::unique_ptr<Pass> createVernonProgramBuildExecutablePass();
void registerVernonProgramBuildExecutablePass();

} // namespace mlir::vernon::program
