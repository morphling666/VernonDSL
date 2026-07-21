#ifndef MLIR_DIALECT_VERNON_TRANSFORMS_VERNONLOWERCPURESOURCES_H
#define MLIR_DIALECT_VERNON_TRANSFORMS_VERNONLOWERCPURESOURCES_H

#include <memory>

namespace mlir {
class Pass;

namespace vernon {

std::unique_ptr<Pass> createVernonLowerCPUResourcesPass();
void registerVernonLowerCPUResourcesPass();

} // namespace vernon
} // namespace mlir

#endif
