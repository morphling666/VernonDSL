#ifndef MLIR_DIALECT_VERNON_TRANSFORMS_VERNONLOWERGPUTENSORS_H
#define MLIR_DIALECT_VERNON_TRANSFORMS_VERNONLOWERGPUTENSORS_H

#include <memory>

namespace mlir {
class Pass;

namespace vernon {

std::unique_ptr<Pass> createVernonLowerGPUTensorsPass(bool useSpirvTupleAbi = false);
void registerVernonLowerGPUTensorsPass();

} // namespace vernon
} // namespace mlir

#endif
