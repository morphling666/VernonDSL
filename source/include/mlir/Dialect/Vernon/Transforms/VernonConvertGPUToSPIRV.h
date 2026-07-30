#ifndef MLIR_DIALECT_VERNON_TRANSFORMS_VERNONCONVERTGPUTOSPIRV_H_
#define MLIR_DIALECT_VERNON_TRANSFORMS_VERNONCONVERTGPUTOSPIRV_H_

#include <memory>

namespace mlir {
class Pass;

namespace vernon {

std::unique_ptr<Pass> createVernonConvertGPUToSPIRVPass();
void registerVernonConvertGPUToSPIRVPass();

} // namespace vernon
} // namespace mlir

#endif
