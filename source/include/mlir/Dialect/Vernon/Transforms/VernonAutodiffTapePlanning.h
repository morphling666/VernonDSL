#pragma once

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vernon/Transforms/VernonAutodiffRules.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>

namespace mlir::vernon {

class VernonAutodiffAnalysisResult;
class VernonAutodiffTapePlan;

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

enum class AdMemoryDomain {
    PersistentResidual,
    TransientGradient,
    GraphCheckpoint,
    ForwardEffectShadow,
};
inline constexpr size_t kAdMemoryDomainCount = 4;

struct AdResidualInterval {
    Value value;
    unsigned abiLeafIndex{};
    AdMemoryDomain domain{AdMemoryDomain::PersistentResidual};
    uint64_t byteSize{};
    uint64_t alignment{1};
    uint64_t lifetimeBegin{};
    uint64_t lifetimeEnd{};
    bool rematerialized{};
    uint64_t recomputationCost{};
};

struct AdRematerializationRecipe {
    Value value;
    SmallVector<Operation *> operations;
    uint64_t estimatedCost{};
};

struct AdPhysicalBuffer {
    AdMemoryDomain domain{AdMemoryDomain::PersistentResidual};
    uint64_t offset{};
    uint64_t byteSize{};
    uint64_t alignment{1};
};

struct AdBufferSlice {
    unsigned physicalBuffer{std::numeric_limits<unsigned>::max()};
    uint64_t offset{};
    uint64_t byteSize{};
};

struct AdBufferAssignment {
    SmallVector<AdPhysicalBuffer> physicalBuffers;
    SmallVector<AdBufferSlice> residualSlices;
    std::array<uint64_t, kAdMemoryDomainCount> peakBytesByDomain{};
    uint64_t peakBytes{};
};

FailureOr<AdBufferAssignment> assignAdMemoryBuffers(ArrayRef<AdResidualInterval> residuals);

struct AdRematerializationCandidate {
    SmallVector<unsigned> residualIndices;
    uint64_t recomputationCost{};
};

struct AdBudgetedBufferAssignment {
    AdBufferAssignment buffers;
    SmallVector<bool> selectedCandidates;
    uint64_t recomputationCost{};
};

FailureOr<AdBudgetedBufferAssignment>
assignAdMemoryBuffersWithinBudget(ArrayRef<AdResidualInterval> residuals,
                                  ArrayRef<AdRematerializationCandidate> candidates, uint64_t budgetBytes);

class AdMemoryPlan {
public:
    ArrayRef<AdResidualInterval> getResiduals() const { return residuals; }
    ArrayRef<AdRematerializationRecipe> getRematerializations() const { return rematerializations; }
    const AdBufferAssignment &getBufferAssignment() const { return bufferAssignment; }
    uint64_t getEstimatedPersistentBytes() const { return estimatedPersistentBytes; }

private:
    friend FailureOr<VernonAutodiffTapePlan> planAutodiffTape(func::FuncOp, const VernonAutodiffAnalysisResult &,
                                                              const VernonAutodiffRuleRegistry &);

    SmallVector<AdResidualInterval> residuals;
    SmallVector<AdRematerializationRecipe> rematerializations;
    AdBufferAssignment bufferAssignment;
    uint64_t estimatedPersistentBytes{};
};

class VernonAutodiffTapePlan {
public:
    const AutodiffTapeHeaderSchema &getInvocationHeader() const { return invocationHeader; }
    const AutodiffTapeRecord &getInvocationRecord() const { return invocationRecord; }
    ArrayRef<AutodiffTapeRegion> getRegions() const { return regions; }
    const AdMemoryPlan &getMemoryPlan() const { return memoryPlan; }

    /// Static layout/statistics hint: invocation storage plus one sample record
    /// and header per dynamic region. It is never a capacity or iteration cap.
    uint64_t getStaticTapeBytesHint() const { return staticTapeBytesHint; }

private:
    friend FailureOr<VernonAutodiffTapePlan> planAutodiffTape(func::FuncOp, const VernonAutodiffAnalysisResult &,
                                                              const VernonAutodiffRuleRegistry &);

    AutodiffTapeHeaderSchema invocationHeader;
    AutodiffTapeRecord invocationRecord;
    SmallVector<AutodiffTapeRegion, 0> regions;
    AdMemoryPlan memoryPlan;
    uint64_t staticTapeBytesHint{};
};

FailureOr<VernonAutodiffTapePlan> planAutodiffTape(func::FuncOp function, const VernonAutodiffAnalysisResult &analysis,
                                                   const VernonAutodiffRuleRegistry &registry);

} // namespace mlir::vernon
