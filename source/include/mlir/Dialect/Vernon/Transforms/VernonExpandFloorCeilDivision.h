#ifndef MLIR_DIALECT_VERNON_TRANSFORMS_VERNONEXPANDFLOORCEILDIVISION_H
#define MLIR_DIALECT_VERNON_TRANSFORMS_VERNONEXPANDFLOORCEILDIVISION_H

#include <memory>

namespace mlir {
class Pass;

namespace vernon {

std::unique_ptr<Pass> createVernonExpandFloorCeilDivisionPass();
void registerVernonExpandFloorCeilDivisionPass();

} // namespace vernon
} // namespace mlir

#endif
