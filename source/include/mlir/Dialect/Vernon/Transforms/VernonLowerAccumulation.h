#ifndef MLIR_DIALECT_VERNON_TRANSFORMS_VERNONLOWERACCUMULATION_H
#define MLIR_DIALECT_VERNON_TRANSFORMS_VERNONLOWERACCUMULATION_H

#include <memory>

namespace mlir {
class Pass;

namespace vernon {

struct AccumulationTargetCapabilities {
    bool supportsF32AtomicAdd{false};
    bool supportsF64AtomicAdd{false};
};

std::unique_ptr<Pass> createVernonLowerAccumulationPass(AccumulationTargetCapabilities capabilities = {});
void registerVernonLowerAccumulationPass();

} // namespace vernon
} // namespace mlir

#endif
