#ifndef MLIR_DIALECT_VERNON_TRANSFORMS_VERNONLINEHELPERS_H
#define MLIR_DIALECT_VERNON_TRANSFORMS_VERNONLINEHELPERS_H

#include <memory>

namespace mlir {
class Pass;

namespace vernon {

std::unique_ptr<Pass> createVernonInlineHelpersPass();
void registerVernonInlineHelpersPass();

} // namespace vernon
} // namespace mlir

#endif
