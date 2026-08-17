#ifndef MLIR_DIALECT_VERNON_TRANSFORMS_VERNONLOWERACCUMULATION_H
#define MLIR_DIALECT_VERNON_TRANSFORMS_VERNONLOWERACCUMULATION_H

#include <cstdint>
#include <memory>

namespace mlir {
class Pass;

namespace vernon {

enum class AggregateGradientStorage {
    Shared,
    InvocationPrivateStaging,
};

enum class AtomicAddImplementation {
    Unsupported,
    Native,
    IntegerCompareExchange,
};

struct AtomicScopeCapabilities {
    AtomicAddImplementation f32{AtomicAddImplementation::Unsupported};
    AtomicAddImplementation f64{AtomicAddImplementation::Unsupported};
};

struct AccumulationCostModel {
    // Measured crossover expressed as contenders for one destination within a
    // workgroup. Keeping this in the profile makes policy selection independent
    // of backend names and allows benchmark evidence to replace these values.
    // The checked-in benchmark currently establishes publication wins at 64
    // contenders, its smallest measured workgroup. Stay conservative until a
    // finer-grained crossover sweep replaces that evidence.
    uint32_t nativeAtomicReductionCrossover{64};
    uint32_t integerCasReductionCrossover{64};
};

struct AccumulationTargetCapabilities {
    AtomicScopeCapabilities device;
    AtomicScopeCapabilities workgroup;
    AggregateGradientStorage aggregateGradientStorage{AggregateGradientStorage::Shared};
    bool supportsWorkgroupReduction{false};
    AccumulationCostModel costModel;
};

inline constexpr const char kAccumulationOwnershipAttrName[] = "vernon.accumulation_ownership";
inline constexpr const char kInvocationPrivateAccumulationOwnership[] = "invocation_private";
inline constexpr const char kAccumulationStrategyAttrName[] = "vernon.accumulation_strategy";
inline constexpr const char kExclusiveStoreAccumulationStrategy[] = "exclusive_store";
inline constexpr const char kInvocationPrivateAccumulationStrategy[] = "invocation_private";
inline constexpr const char kWorkgroupReductionAccumulationStrategy[] = "workgroup_reduction";
inline constexpr const char kAtomicAccumulationStrategy[] = "atomic";
inline constexpr const char kAtomicImplementationAttrName[] = "vernon.atomic_implementation";
inline constexpr const char kNativeAtomicImplementation[] = "native";
inline constexpr const char kIntegerCasAtomicImplementation[] = "integer_cas";

std::unique_ptr<Pass> createVernonLowerAccumulationPass(AccumulationTargetCapabilities capabilities = {});
std::unique_ptr<Pass> createVernonVerifyGeneratedAccumulationPass(AccumulationTargetCapabilities capabilities = {});
void registerVernonLowerAccumulationPass();

} // namespace vernon
} // namespace mlir

#endif
