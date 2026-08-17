#pragma once

#include "mlir/Support/LogicalResult.h"

namespace mlir {
class ModuleOp;

namespace vernon {

/// Converts target-neutral TensorView inputs in generated autodiff profiles
/// into GPU resource bindings. This runs only on a target-prepared clone; the
/// logical VJP module remains free of descriptor-set and binding assignments.
LogicalResult materializeGPUAutodiffProfileBindings(ModuleOp module);

} // namespace vernon
} // namespace mlir
