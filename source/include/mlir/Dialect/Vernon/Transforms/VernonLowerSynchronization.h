#ifndef MLIR_DIALECT_VERNON_TRANSFORMS_VERNONLOWERSYNCHRONIZATION_H_
#define MLIR_DIALECT_VERNON_TRANSFORMS_VERNONLOWERSYNCHRONIZATION_H_

#include <memory>

namespace mlir {
class Pass;

namespace vernon {

std::unique_ptr<Pass> createVernonLowerCPUSynchronizationPass();
std::unique_ptr<Pass> createVernonLowerGPUSynchronizationPass(bool spirv = false);

} // namespace vernon
} // namespace mlir

#endif
