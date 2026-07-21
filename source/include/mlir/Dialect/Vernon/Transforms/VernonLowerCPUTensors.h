#ifndef MLIR_DIALECT_VERNON_TRANSFORMS_VERNONLOWERCPUTENSORS_H
#define MLIR_DIALECT_VERNON_TRANSFORMS_VERNONLOWERCPUTENSORS_H

#include <memory>

namespace mlir {
class Pass;

namespace vernon {

std::unique_ptr<Pass> createVernonLowerCPUTensorsPass();
void registerVernonLowerCPUTensorsPass();

} // namespace vernon
} // namespace mlir

#endif
