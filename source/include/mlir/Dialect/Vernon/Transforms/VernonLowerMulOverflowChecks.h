#ifndef MLIR_DIALECT_VERNON_TRANSFORMS_VERNONLOWERMULOVERFLOWCHECKS_H
#define MLIR_DIALECT_VERNON_TRANSFORMS_VERNONLOWERMULOVERFLOWCHECKS_H

#include <memory>

namespace mlir {
class Pass;

namespace vernon {

std::unique_ptr<Pass> createVernonLowerMulOverflowChecksPass();
void registerVernonLowerMulOverflowChecksPass();

} // namespace vernon
} // namespace mlir

#endif
