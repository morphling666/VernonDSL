#pragma once

#include <memory>

namespace mlir {
class Pass;

namespace vernon {

/// Lowers logical autodiff Tape operations in GPU compute profiles to two
/// ordinary reflected device resources: a byte-addressed Tape and bounded
/// replay segment metadata.
std::unique_ptr<Pass> createVernonLowerGPUAutodiffPass();
void registerVernonLowerGPUAutodiffPass();

} // namespace vernon
} // namespace mlir
