#pragma once

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vernon/IR/VernonValueAbi.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

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

/// Immutable, mode-independent facts shared by semantic JVP and VJP
/// transforms. The pointed-to IR must outlive this result.
class VernonAutodiffAnalysisResult {
public:
    ArrayRef<AutodiffLeaf> getWrtLeaves() const { return wrtLeaves; }
    ArrayRef<AutodiffLeaf> getActiveResultLeaves() const { return activeResultLeaves; }
    ArrayRef<AutodiffValueActivity> getActiveValues() const { return activeValues; }
    ArrayRef<AutodiffOperationActivity> getOperations() const { return operations; }
    ArrayRef<AutodiffRegion> getRegions() const { return regions; }

    const ValueAbiLayout *getValueAbi(Value value) const;
    bool isActive(Value value, unsigned abiLeafIndex) const;
    bool isActive(Operation *operation) const;

private:
    friend class AutodiffAnalysisBuilder;
    friend FailureOr<VernonAutodiffAnalysisResult> analyzeAutodiffFunction(func::FuncOp, ArrayRef<StringRef>);

    SmallVector<AutodiffLeaf> wrtLeaves;
    SmallVector<AutodiffLeaf> activeResultLeaves;
    SmallVector<AutodiffValueActivity> activeValues;
    SmallVector<AutodiffValueAbi, 0> valueAbis;
    SmallVector<AutodiffOperationActivity> operations;
    SmallVector<AutodiffRegion> regions;
};

/// Analyze a structured primal function. `wrtPaths` use frontend source names
/// followed by canonical Struct field or Tuple index components.
FailureOr<VernonAutodiffAnalysisResult> analyzeAutodiffFunction(func::FuncOp function, ArrayRef<StringRef> wrtPaths);
FailureOr<VernonAutodiffAnalysisResult> analyzeAutodiffFunction(func::FuncOp function, ArrayRef<StringRef> wrtPaths,
                                                                const VernonAutodiffRuleRegistry &registry);

bool isDifferentiableAutodiffLeaf(Type scalarType, StringRef logicalDtype = {});
FailureOr<Type> getAutodiffDerivativeType(Type scalarType);
AutodiffEffectKind classifyAutodiffEffect(Operation *operation);
StringRef stringifyAutodiffEffect(AutodiffEffectKind effect);

} // namespace mlir::vernon
