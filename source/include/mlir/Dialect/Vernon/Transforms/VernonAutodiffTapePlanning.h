#pragma once

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vernon/Transforms/VernonAutodiffRules.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

#include <cstdint>
#include <optional>
#include <string>

namespace mlir::vernon {

class VernonAutodiffAnalysisResult;

/// A physical slot in a logical tape record. This low-level description is
/// public so all offset, size, and alignment arithmetic can be tested without
/// constructing impractically large MLIR values.
struct AutodiffTapeSlot {
    uint64_t size{};
    uint64_t alignment{};
};

struct AutodiffTapeLayout {
    SmallVector<uint64_t> offsets;
    uint64_t size{};
    uint64_t stride{};
    uint64_t alignment{1};
};

/// Lay out slots after an optional fixed prefix. Every operation is checked;
/// zero and non-power-of-two alignments are rejected.
FailureOr<AutodiffTapeLayout> planAutodiffTapeLayout(ArrayRef<AutodiffTapeSlot> slots, uint64_t prefixSize = 0,
                                                     uint64_t prefixAlignment = 1);

enum class AutodiffTapeFieldKind {
    InvocationRecordIdentity,
    RootRegionHandle,
    ParentRecordIdentity,
    ChildRegionOrdinal,
    LastRecordOffset,
    ExecutedCount,
    ExitKind,
    RecordIdentity,
    PreviousRecordOffset,
    Predicate,
    ChildRegionHandle,
};

struct AutodiffTapeField {
    AutodiffTapeFieldKind kind;
    std::optional<unsigned> regionOrdinal;
    uint64_t offset{};
    uint64_t size{};
    uint64_t alignment{};
};

struct AutodiffTapeHeaderSchema {
    SmallVector<AutodiffTapeField> fields;
    uint64_t size{};
    uint64_t alignment{1};
};

struct AutodiffTapeLeaf {
    Value value;
    unsigned abiLeafIndex{};
    std::string path;
    Type scalarType;
    std::string dtype;
    uint64_t scalarCount{};
    uint64_t offset{};
    uint64_t size{};
    uint64_t alignment{};
};

struct AutodiffTapeRecord {
    AutodiffTapeHeaderSchema prefix;
    SmallVector<AutodiffTapeLeaf> leaves;
    uint64_t size{};
    uint64_t stride{};
    uint64_t alignment{1};
};

enum class AutodiffParentRecordKind {
    Invocation,
    DynamicRegion,
};

struct AutodiffParentRecordIdentity {
    AutodiffParentRecordKind kind{AutodiffParentRecordKind::Invocation};
    std::optional<unsigned> regionOrdinal;
};

struct AutodiffReverseControlRequirements {
    bool predicate{};
    bool executedCount{};
    bool exitKind{};
};

struct AutodiffTapeRegion {
    Operation *operation{};
    unsigned ordinal{};
    AutodiffParentRecordIdentity parentRecord;
    unsigned childOrdinal{};
    SmallVector<unsigned> childRegionOrdinals;
    AutodiffReverseControlRequirements control;
    AutodiffTapeHeaderSchema header;
    AutodiffTapeRecord record;
};

class VernonAutodiffTapePlan {
public:
    const AutodiffTapeHeaderSchema &getInvocationHeader() const { return invocationHeader; }
    const AutodiffTapeRecord &getInvocationRecord() const { return invocationRecord; }
    ArrayRef<AutodiffTapeRegion> getRegions() const { return regions; }

    /// Static layout/statistics hint: invocation storage plus one sample record
    /// and header per dynamic region. It is never a capacity or iteration cap.
    uint64_t getStaticTapeBytesHint() const { return staticTapeBytesHint; }

private:
    friend FailureOr<VernonAutodiffTapePlan> planAutodiffTape(func::FuncOp, const VernonAutodiffAnalysisResult &,
                                                              const VernonAutodiffRuleRegistry &);

    AutodiffTapeHeaderSchema invocationHeader;
    AutodiffTapeRecord invocationRecord;
    SmallVector<AutodiffTapeRegion, 0> regions;
    uint64_t staticTapeBytesHint{};
};

FailureOr<VernonAutodiffTapePlan> planAutodiffTape(func::FuncOp function, const VernonAutodiffAnalysisResult &analysis,
                                                   const VernonAutodiffRuleRegistry &registry);

} // namespace mlir::vernon
