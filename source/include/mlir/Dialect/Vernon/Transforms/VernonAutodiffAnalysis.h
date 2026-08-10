#pragma once

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vernon/IR/VernonValueAbi.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseSet.h"

#include <cstdint>
#include <optional>
#include <string>

namespace mlir::vernon {

class AutodiffAnalysisBuilder;
class VernonAutodiffRuleRegistry;

enum class AutodiffEffectKind {
    Pure,
    StorageRead,
    StorageWrite,
    Atomic,
    Barrier,
    ExternallyVisible,
    Unsupported,
};

struct AutodiffLeaf {
    Value value;
    unsigned abiLeafIndex{};
    std::string path;
    Type primalType;
    Type derivativeType;
    std::string dtype;
    SmallVector<uint64_t> shape;
};

struct AutodiffValueActivity {
    Value value;
    SmallVector<unsigned> activeAbiLeaves;
};

struct AutodiffValueAbi {
    Value value;
    ValueAbiLayout layout;
};

struct AutodiffOperationActivity {
    Operation *operation{};
    AutodiffEffectKind effect{AutodiffEffectKind::Pure};
    bool active{};
};

struct AutodiffRegion {
    Operation *operation{};
    unsigned ordinal{};
    std::optional<unsigned> parentOrdinal;
    SmallVector<unsigned> childOrdinals;
};

enum class StorageVersionKind {
    Entry,
    Write,
    AdditiveWrite,
    IfMerge,
    WhilePhi,
    WhileExit,
};

enum class AutodiffInternalAdjointOwnership {
    None,
    LanePrivate,
    WorkgroupCoupled,
};

enum class AutodiffExternalGradientOwnership {
    None,
    InvocationPrivate,
    AtomicShared,
    WorkgroupShared,
};

struct AutodiffStorageIdentity {
    unsigned id{};
    Value binding;
    AutodiffInternalAdjointOwnership internalAdjointOwnership{AutodiffInternalAdjointOwnership::None};
    AutodiffExternalGradientOwnership externalGradientOwnership{AutodiffExternalGradientOwnership::None};
    bool externalGradientDestination{};
};

struct AutodiffStorageVersion {
    unsigned id{};
    unsigned identity{};
    StorageVersionKind kind{StorageVersionKind::Entry};
    Operation *operation{};
    SmallVector<unsigned> incomingVersions;
};

struct AutodiffStorageEffect {
    unsigned id{};
    unsigned identity{};
    Operation *operation{};
    unsigned versionBefore{};
    std::optional<unsigned> versionAfter;
};

/// Immutable, mode-independent facts shared by semantic JVP and VJP
/// transforms. The pointed-to IR must outlive this result.
class VernonAutodiffAnalysisResult {
public:
    ArrayRef<AutodiffLeaf> getWrtLeaves() const { return wrtLeaves; }
    ArrayRef<AutodiffLeaf> getActiveResultLeaves() const { return activeResultLeaves; }
    ArrayRef<AutodiffValueActivity> getActiveValues() const { return activeValues; }
    ArrayRef<AutodiffOperationActivity> getOperations() const { return operations; }
    ArrayRef<AutodiffRegion> getRegions() const { return regions; }
    ArrayRef<AutodiffStorageIdentity> getStorageIdentities() const { return storageIdentities; }
    ArrayRef<AutodiffStorageVersion> getStorageVersions() const { return storageVersions; }
    ArrayRef<AutodiffStorageEffect> getStorageEffects() const { return storageEffects; }

    const ValueAbiLayout *getValueAbi(Value value) const;
    const AutodiffStorageIdentity *getStorageIdentity(Value binding) const;
    const AutodiffStorageEffect *getStorageEffect(Operation *operation) const;
    bool isActive(Value value, unsigned abiLeafIndex) const;
    bool isActive(Operation *operation) const;
    bool isActiveStorageVersion(unsigned version, unsigned abiLeafIndex) const;

private:
    friend class AutodiffAnalysisBuilder;
    friend FailureOr<VernonAutodiffAnalysisResult> analyzeAutodiffFunction(func::FuncOp, ArrayRef<StringRef>);

    SmallVector<AutodiffLeaf> wrtLeaves;
    SmallVector<AutodiffLeaf> activeResultLeaves;
    SmallVector<AutodiffValueActivity> activeValues;
    SmallVector<AutodiffValueAbi, 0> valueAbis;
    SmallVector<AutodiffOperationActivity> operations;
    SmallVector<AutodiffRegion> regions;
    SmallVector<AutodiffStorageIdentity> storageIdentities;
    SmallVector<AutodiffStorageVersion> storageVersions;
    SmallVector<AutodiffStorageEffect> storageEffects;
    SmallVector<SmallVector<unsigned>> storageVersionNodes;
    SmallVector<SmallVector<unsigned>> activeStorageVersionLeaves;
    DenseMap<Value, unsigned> valueAbiIndices;
    DenseMap<Value, unsigned> storageIdentityIndices;
    DenseMap<Operation *, unsigned> storageEffectIndices;
    DenseSet<Operation *> activeOperationSet;
};

/// Analyze a structured primal function. `wrtPaths` use frontend source names
/// followed by canonical Struct field or Tuple index components.
FailureOr<VernonAutodiffAnalysisResult> analyzeAutodiffFunction(func::FuncOp function, ArrayRef<StringRef> wrtPaths);
FailureOr<VernonAutodiffAnalysisResult> analyzeAutodiffFunction(func::FuncOp function, ArrayRef<StringRef> wrtPaths,
                                                                const VernonAutodiffRuleRegistry &registry);
FailureOr<VernonAutodiffAnalysisResult> analyzeAutodiffFunction(func::FuncOp function, ArrayRef<StringRef> wrtPaths,
                                                                ArrayRef<StringRef> outputPaths,
                                                                const VernonAutodiffRuleRegistry &registry);

bool isDifferentiableAutodiffLeaf(Type scalarType, StringRef logicalDtype = {});
FailureOr<Type> getAutodiffDerivativeType(Type scalarType);
AutodiffEffectKind classifyAutodiffEffect(Operation *operation);
StringRef stringifyAutodiffEffect(AutodiffEffectKind effect);

} // namespace mlir::vernon
