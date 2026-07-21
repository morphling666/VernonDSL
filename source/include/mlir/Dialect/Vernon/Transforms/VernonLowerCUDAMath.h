#ifndef MLIR_DIALECT_VERNON_TRANSFORMS_VERNONLOWERCUDAMATH_H
#define MLIR_DIALECT_VERNON_TRANSFORMS_VERNONLOWERCUDAMATH_H

#include <memory>

namespace mlir {
class Pass;

namespace vernon {

std::unique_ptr<Pass> createVernonLowerCUDAMathPass();
void registerVernonLowerCUDAMathPass();

} // namespace vernon
} // namespace mlir

#endif
