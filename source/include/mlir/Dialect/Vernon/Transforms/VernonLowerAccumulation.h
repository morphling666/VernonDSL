#ifndef MLIR_DIALECT_VERNON_TRANSFORMS_VERNONLOWERACCUMULATION_H
#define MLIR_DIALECT_VERNON_TRANSFORMS_VERNONLOWERACCUMULATION_H

#include <memory>

namespace mlir {
class Pass;

namespace vernon {

enum class AggregateGradientStorage {
    Shared,
    InvocationPrivateStaging,
};

struct AccumulationTargetCapabilities {
    bool supportsF32AtomicAdd{false};
    bool supportsF64AtomicAdd{false};
    AggregateGradientStorage aggregateGradientStorage{AggregateGradientStorage::Shared};
};

inline constexpr const char kAccumulationOwnershipAttrName[] = "vernon.accumulation_ownership";
inline constexpr const char kInvocationPrivateAccumulationOwnership[] = "invocation_private";

std::unique_ptr<Pass> createVernonLowerAccumulationPass(AccumulationTargetCapabilities capabilities = {});
void registerVernonLowerAccumulationPass();

} // namespace vernon
} // namespace mlir

#endif
