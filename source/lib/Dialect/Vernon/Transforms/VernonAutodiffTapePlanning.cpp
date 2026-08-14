#include "mlir/Dialect/Vernon/Transforms/VernonAutodiffTapePlanning.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vernon/IR/VernonAttrs.h"
#include "mlir/Dialect/Vernon/IR/VernonValueAbi.h"
#include "mlir/Dialect/Vernon/Transforms/VernonAutodiffAnalysis.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"

#include <limits>

namespace mlir::vernon {
namespace {

constexpr uint64_t kMaximumRematerializationCost = 8;

FailureOr<uint64_t> checkedAdd(uint64_t left, uint64_t right) {
    if (left > std::numeric_limits<uint64_t>::max() - right)
        return failure();
    return left + right;
}

FailureOr<uint64_t> checkedMultiply(uint64_t left, uint64_t right) {
    if (right != 0 && left > std::numeric_limits<uint64_t>::max() / right)
        return failure();
    return left * right;
}

FailureOr<unsigned> checkedUnsigned(size_t value) {
    if (value > std::numeric_limits<unsigned>::max())
        return failure();
    return static_cast<unsigned>(value);
}

FailureOr<uint64_t> checkedAlign(uint64_t value, uint64_t alignment) {
    if (alignment == 0 || !llvm::isPowerOf2_64(alignment))
        return failure();
    FailureOr<uint64_t> biased = checkedAdd(value, alignment - 1);
    if (failed(biased))
        return failure();
    return *biased & ~(alignment - 1);
}

std::string stringifyPath(ArrayRef<ValueAbiPathComponent> path) {
    std::string result;
    for (const ValueAbiPathComponent &component : path) {
        if (!result.empty())
            result.push_back('.');
        if (component.field)
            result.append(*component.field);
        else
            result.append(std::to_string(component.index));
    }
    return result;
}

AutodiffTapeSlot getFieldSlot(AutodiffTapeFieldKind kind) {
    switch (kind) {
    case AutodiffTapeFieldKind::ChildRegionOrdinal:
    case AutodiffTapeFieldKind::ExitKind:
        return {4, 4};
    case AutodiffTapeFieldKind::Predicate:
        return {1, 1};
    case AutodiffTapeFieldKind::InvocationRecordIdentity:
    case AutodiffTapeFieldKind::RootRegionHandle:
    case AutodiffTapeFieldKind::ParentRecordIdentity:
    case AutodiffTapeFieldKind::LastRecordOffset:
    case AutodiffTapeFieldKind::ExecutedCount:
    case AutodiffTapeFieldKind::RecordIdentity:
    case AutodiffTapeFieldKind::PreviousRecordOffset:
    case AutodiffTapeFieldKind::ChildRegionHandle:
        return {8, 8};
    }
    llvm_unreachable("unknown autodiff tape field");
}

struct FieldSpec {
    AutodiffTapeFieldKind kind;
    std::optional<unsigned> regionOrdinal;
};

FailureOr<AutodiffTapeHeaderSchema> buildSchema(ArrayRef<FieldSpec> specs) {
    SmallVector<AutodiffTapeSlot> slots;
    slots.reserve(specs.size());
    for (const FieldSpec &spec : specs)
        slots.push_back(getFieldSlot(spec.kind));
    FailureOr<AutodiffTapeLayout> layout = planAutodiffTapeLayout(slots);
    if (failed(layout))
        return failure();

    AutodiffTapeHeaderSchema schema;
    schema.size = layout->stride;
    schema.alignment = layout->alignment;
    for (auto [index, spec] : llvm::enumerate(specs)) {
        const AutodiffTapeSlot slot = slots[index];
        schema.fields.push_back(
            AutodiffTapeField{spec.kind, spec.regionOrdinal, layout->offsets[index], slot.size, slot.alignment});
    }
    return schema;
}

struct RecordBuilder {
    AutodiffTapeHeaderSchema prefix;
    SmallVector<AutodiffTapeLeaf> leaves;
    DenseSet<Value> values;

    LogicalResult add(Value value, const ValueAbiLayout &layout) {
        if (values.contains(value))
            return success();
        values.insert(value);

        for (auto [index, leaf] : llvm::enumerate(layout.leaves)) {
            if (!leaf.scalarType.isIntOrFloat() || leaf.scalarCount == 0)
                return failure();
            FailureOr<unsigned> abiLeafIndex = checkedUnsigned(index);
            if (failed(abiLeafIndex))
                return failure();
            const uint64_t scalarSize = std::max<uint64_t>(leaf.scalarType.getIntOrFloatBitWidth() / 8, 1);
            FailureOr<uint64_t> size = checkedMultiply(scalarSize, leaf.scalarCount);
            if (failed(size))
                return failure();
            leaves.push_back(AutodiffTapeLeaf{value, *abiLeafIndex, stringifyPath(leaf.path), leaf.scalarType,
                                              leaf.dtype, leaf.scalarCount, 0, *size, scalarSize});
        }
        return success();
    }

    FailureOr<AutodiffTapeRecord> finish() && {
        SmallVector<AutodiffTapeSlot> slots;
        slots.reserve(leaves.size());
        for (const AutodiffTapeLeaf &leaf : leaves)
            slots.push_back({leaf.size, leaf.alignment});
        FailureOr<AutodiffTapeLayout> layout = planAutodiffTapeLayout(slots, prefix.size, prefix.alignment);
        if (failed(layout))
            return failure();
        for (auto [index, offset] : llvm::enumerate(layout->offsets))
            leaves[index].offset = offset;
        return AutodiffTapeRecord{std::move(prefix), std::move(leaves), layout->size, layout->stride,
                                  layout->alignment};
    }
};

uint64_t estimateRematerializationCost(Operation *operation) {
    if (operation->hasTrait<OpTrait::ConstantLike>())
        return 0;
    const StringRef name = operation->getName().getStringRef();
    if (name == "arith.divf" || name == "arith.divsi" || name == "arith.divui" || name == "arith.remf" ||
        name == "arith.remsi" || name == "arith.remui")
        return 4;
    return 1;
}

bool isRematerializableDialect(Operation *operation) {
    const StringRef dialect = operation->getName().getDialectNamespace();
    return dialect == "arith" || dialect == "index" || dialect == "tensor" ||
           isa<TupleCreateOp, TupleGetOp, StructCreateOp, StructGetOp>(operation);
}

FailureOr<AdRematerializationRecipe> buildRematerializationRecipe(Value value, func::FuncOp function) {
    AdRematerializationRecipe recipe;
    recipe.value = value;
    DenseSet<Operation *> planned;
    DenseSet<Value> visiting;
    auto visit = [&](Value current, auto &self) -> LogicalResult {
        if (auto argument = dyn_cast<BlockArgument>(current)) {
            auto owner = dyn_cast_or_null<func::FuncOp>(argument.getOwner()->getParentOp());
            return success(owner && owner == function && owner.getArgAttr(argument.getArgNumber(), kBuiltinAttrName));
        }
        Operation *defining = current.getDefiningOp();
        if (!defining || defining->getNumRegions() != 0 ||
            classifyAutodiffEffect(defining) != AutodiffEffectKind::Pure || !isRematerializableDialect(defining))
            return failure();
        if (!visiting.insert(current).second)
            return failure();
        for (Value operand : defining->getOperands())
            if (failed(self(operand, self)))
                return failure();
        visiting.erase(current);
        if (!planned.insert(defining).second)
            return success();
        FailureOr<uint64_t> cost = checkedAdd(recipe.estimatedCost, estimateRematerializationCost(defining));
        if (failed(cost) || *cost > kMaximumRematerializationCost)
            return failure();
        recipe.estimatedCost = *cost;
        recipe.operations.push_back(defining);
        return success();
    };
    return succeeded(visit(value, visit)) ? FailureOr<AdRematerializationRecipe>(std::move(recipe))
                                          : FailureOr<AdRematerializationRecipe>(failure());
}

} // namespace

FailureOr<AutodiffTapeLayout> planAutodiffTapeLayout(ArrayRef<AutodiffTapeSlot> slots, uint64_t prefixSize,
                                                     uint64_t prefixAlignment) {
    if (prefixAlignment == 0 || !llvm::isPowerOf2_64(prefixAlignment))
        return failure();
    AutodiffTapeLayout result;
    result.alignment = prefixAlignment;
    uint64_t offset = prefixSize;
    result.offsets.reserve(slots.size());
    for (const AutodiffTapeSlot &slot : slots) {
        if (slot.size == 0 || slot.alignment == 0 || !llvm::isPowerOf2_64(slot.alignment))
            return failure();
        FailureOr<uint64_t> aligned = checkedAlign(offset, slot.alignment);
        if (failed(aligned))
            return failure();
        result.offsets.push_back(*aligned);
        FailureOr<uint64_t> end = checkedAdd(*aligned, slot.size);
        if (failed(end))
            return failure();
        offset = *end;
        result.alignment = std::max(result.alignment, slot.alignment);
    }
    FailureOr<uint64_t> stride = checkedAlign(offset, result.alignment);
    if (failed(stride))
        return failure();
    result.size = offset;
    result.stride = *stride;
    return result;
}

FailureOr<AdBufferAssignment> assignAdMemoryBuffers(ArrayRef<AdResidualInterval> residuals) {
    struct ReusableBuffer {
        unsigned physicalBuffer{};
        uint64_t availableAfter{};
    };
    AdBufferAssignment result;
    result.residualSlices.resize(residuals.size());
    SmallVector<unsigned> order;
    order.reserve(residuals.size());
    for (auto [index, residual] : llvm::enumerate(residuals)) {
        if (residual.rematerialized)
            continue;
        if (!residual.byteSize || !residual.alignment || !llvm::isPowerOf2_64(residual.alignment) ||
            residual.lifetimeEnd < residual.lifetimeBegin)
            return failure();
        FailureOr<unsigned> residualIndex = checkedUnsigned(index);
        if (failed(residualIndex))
            return failure();
        order.push_back(*residualIndex);
    }
    llvm::sort(order, [&](unsigned left, unsigned right) {
        if (residuals[left].lifetimeBegin != residuals[right].lifetimeBegin)
            return residuals[left].lifetimeBegin < residuals[right].lifetimeBegin;
        if (residuals[left].byteSize != residuals[right].byteSize)
            return residuals[left].byteSize > residuals[right].byteSize;
        return left < right;
    });

    SmallVector<ReusableBuffer> reusable;
    for (unsigned residualIndex : order) {
        const AdResidualInterval &residual = residuals[residualIndex];
        ReusableBuffer *selected = nullptr;
        for (ReusableBuffer &candidate : reusable) {
            AdPhysicalBuffer &physical = result.physicalBuffers[candidate.physicalBuffer];
            if (candidate.availableAfter > residual.lifetimeBegin || physical.domain != residual.domain ||
                physical.byteSize < residual.byteSize)
                continue;
            if (!selected || physical.byteSize < result.physicalBuffers[selected->physicalBuffer].byteSize)
                selected = &candidate;
        }
        if (!selected) {
            FailureOr<unsigned> physicalBuffer = checkedUnsigned(result.physicalBuffers.size());
            if (failed(physicalBuffer))
                return failure();
            result.physicalBuffers.push_back({residual.domain, 0, residual.byteSize, residual.alignment});
            reusable.push_back({*physicalBuffer, residual.lifetimeEnd});
            selected = &reusable.back();
        } else {
            selected->availableAfter = residual.lifetimeEnd;
            result.physicalBuffers[selected->physicalBuffer].alignment =
                std::max(result.physicalBuffers[selected->physicalBuffer].alignment, residual.alignment);
        }
        result.residualSlices[residualIndex] = {selected->physicalBuffer, 0, residual.byteSize};
    }

    auto domainIndex = [](AdMemoryDomain domain) -> size_t {
        switch (domain) {
        case AdMemoryDomain::PersistentResidual:
            return 0;
        case AdMemoryDomain::TransientGradient:
            return 1;
        case AdMemoryDomain::GraphCheckpoint:
            return 2;
        case AdMemoryDomain::ForwardEffectShadow:
            return 3;
        }
        llvm_unreachable("unknown autodiff memory domain");
    };
    for (AdPhysicalBuffer &physical : result.physicalBuffers) {
        uint64_t &cursor = result.peakBytesByDomain[domainIndex(physical.domain)];
        FailureOr<uint64_t> offset = checkedAlign(cursor, physical.alignment);
        if (failed(offset))
            return failure();
        FailureOr<uint64_t> end = checkedAdd(*offset, physical.byteSize);
        if (failed(end))
            return failure();
        physical.offset = *offset;
        cursor = *end;
    }
    for (AdBufferSlice &slice : result.residualSlices)
        if (slice.physicalBuffer != std::numeric_limits<unsigned>::max())
            slice.offset = result.physicalBuffers[slice.physicalBuffer].offset;
    for (uint64_t bytes : result.peakBytesByDomain) {
        FailureOr<uint64_t> peak = checkedAdd(result.peakBytes, bytes);
        if (failed(peak))
            return failure();
        result.peakBytes = *peak;
    }
    return result;
}

FailureOr<AdBudgetedBufferAssignment>
assignAdMemoryBuffersWithinBudget(ArrayRef<AdResidualInterval> residuals,
                                  ArrayRef<AdRematerializationCandidate> candidates, uint64_t budgetBytes) {
    struct ScoredCandidate {
        unsigned index{};
        uint64_t savedBytes{};
        uint64_t cost{};
    };

    SmallVector<AdResidualInterval> working(residuals);
    FailureOr<AdBufferAssignment> initial = assignAdMemoryBuffers(working);
    if (failed(initial))
        return failure();

    AdBudgetedBufferAssignment result;
    result.buffers = std::move(*initial);
    result.selectedCandidates.resize(candidates.size(), false);
    SmallVector<ScoredCandidate> order;
    order.reserve(candidates.size());
    DenseSet<unsigned> candidateOwners;
    for (auto [candidateIndex, candidate] : llvm::enumerate(candidates)) {
        FailureOr<unsigned> checkedIndex = checkedUnsigned(candidateIndex);
        if (failed(checkedIndex) || candidate.residualIndices.empty())
            return failure();
        DenseSet<unsigned> unique;
        uint64_t savedBytes = 0;
        for (unsigned residualIndex : candidate.residualIndices) {
            if (residualIndex >= working.size() || !unique.insert(residualIndex).second ||
                !candidateOwners.insert(residualIndex).second)
                return failure();
            if (working[residualIndex].rematerialized)
                continue;
            FailureOr<uint64_t> total = checkedAdd(savedBytes, working[residualIndex].byteSize);
            if (failed(total))
                return failure();
            savedBytes = *total;
        }
        order.push_back({*checkedIndex, savedBytes, candidate.recomputationCost});
    }
    llvm::stable_sort(order, [](const ScoredCandidate &left, const ScoredCandidate &right) {
        const long double leftScore = static_cast<long double>(left.savedBytes) / std::max<uint64_t>(left.cost, 1);
        const long double rightScore = static_cast<long double>(right.savedBytes) / std::max<uint64_t>(right.cost, 1);
        if (leftScore != rightScore)
            return leftScore > rightScore;
        if (left.cost != right.cost)
            return left.cost < right.cost;
        return left.index < right.index;
    });

    for (const ScoredCandidate &scored : order) {
        if (result.buffers.peakBytes <= budgetBytes)
            break;
        const AdRematerializationCandidate &candidate = candidates[scored.index];
        SmallVector<unsigned> changed;
        for (unsigned residualIndex : candidate.residualIndices)
            if (!working[residualIndex].rematerialized) {
                working[residualIndex].rematerialized = true;
                working[residualIndex].recomputationCost = candidate.recomputationCost;
                changed.push_back(residualIndex);
            }
        if (changed.empty())
            continue;
        FailureOr<AdBufferAssignment> assignment = assignAdMemoryBuffers(working);
        if (failed(assignment))
            return failure();
        if (assignment->peakBytes >= result.buffers.peakBytes) {
            for (unsigned residualIndex : changed) {
                working[residualIndex].rematerialized = false;
                working[residualIndex].recomputationCost = 0;
            }
            continue;
        }
        FailureOr<uint64_t> cost = checkedAdd(result.recomputationCost, candidate.recomputationCost);
        if (failed(cost))
            return failure();
        result.recomputationCost = *cost;
        result.selectedCandidates[scored.index] = true;
        result.buffers = std::move(*assignment);
    }
    return result.buffers.peakBytes <= budgetBytes ? FailureOr<AdBudgetedBufferAssignment>(std::move(result))
                                                   : FailureOr<AdBudgetedBufferAssignment>(failure());
}

FailureOr<VernonAutodiffTapePlan> planAutodiffTape(func::FuncOp function, const VernonAutodiffAnalysisResult &analysis,
                                                   const VernonAutodiffRuleRegistry &registry) {
    ModuleOp module = function->getParentOfType<ModuleOp>();
    if (!module) {
        function.emitError("autodiff tape planning requires a parent module");
        return failure();
    }
    for (const AutodiffOperationActivity &activity : analysis.getOperations()) {
        if (activity.operation->getParentOfType<func::FuncOp>() != function) {
            function.emitError("autodiff tape analysis belongs to a different function");
            return failure();
        }
    }
    if (failed(verifyAutodiffRuleCoverage(analysis, registry)))
        return failure();

    VernonAutodiffTapePlan plan;
    SmallVector<AdResidualInterval> residuals;
    SmallVector<AdRematerializationRecipe> rematerializations;
    DenseMap<Operation *, uint64_t> operationOrder;
    for (auto [index, activity] : llvm::enumerate(analysis.getOperations()))
        operationOrder.try_emplace(activity.operation, static_cast<uint64_t>(index));
    const uint64_t scheduleLength = static_cast<uint64_t>(analysis.getOperations().size());
    FailureOr<uint64_t> reverseScheduleEnd = checkedMultiply(scheduleLength, 2);
    if (failed(reverseScheduleEnd)) {
        function.emitError("autodiff residual schedule length overflows");
        return failure();
    }
    DenseMap<Operation *, unsigned> regionPlanIndices;
    DenseMap<unsigned, unsigned> sourceToPlanIndex;
    for (const AutodiffRegion &source : analysis.getRegions()) {
        if (!analysis.isActive(source.operation))
            continue;
        FailureOr<unsigned> planIndex = checkedUnsigned(plan.regions.size());
        if (failed(planIndex)) {
            function.emitError("autodiff region count exceeds the planner representation");
            return failure();
        }
        regionPlanIndices.try_emplace(source.operation, *planIndex);
        sourceToPlanIndex.try_emplace(source.ordinal, *planIndex);
        AutodiffTapeRegion region;
        region.operation = source.operation;
        region.ordinal = source.ordinal;
        plan.regions.push_back(std::move(region));
    }

    SmallVector<unsigned> rootRegions;
    for (const AutodiffRegion &source : analysis.getRegions()) {
        auto current = sourceToPlanIndex.find(source.ordinal);
        if (current == sourceToPlanIndex.end())
            continue;
        AutodiffTapeRegion &region = plan.regions[current->second];
        if (source.parentOrdinal) {
            auto parent = sourceToPlanIndex.find(*source.parentOrdinal);
            if (parent == sourceToPlanIndex.end()) {
                function.emitError("active autodiff region has no active parent region");
                return failure();
            }
            AutodiffTapeRegion &parentRegion = plan.regions[parent->second];
            region.parentRecord = {AutodiffParentRecordKind::DynamicRegion, parentRegion.ordinal};
            FailureOr<unsigned> childOrdinal = checkedUnsigned(parentRegion.childRegionOrdinals.size());
            if (failed(childOrdinal)) {
                region.operation->emitError("autodiff child-region count exceeds the planner representation");
                return failure();
            }
            region.childOrdinal = *childOrdinal;
            parentRegion.childRegionOrdinals.push_back(region.ordinal);
        } else {
            region.parentRecord = {AutodiffParentRecordKind::Invocation, std::nullopt};
            FailureOr<unsigned> childOrdinal = checkedUnsigned(rootRegions.size());
            if (failed(childOrdinal)) {
                region.operation->emitError("autodiff root-region count exceeds the planner representation");
                return failure();
            }
            region.childOrdinal = *childOrdinal;
            rootRegions.push_back(region.ordinal);
        }
        if (isa<scf::IfOp>(source.operation))
            region.control.predicate = true;
        else if (isa<scf::WhileOp>(source.operation)) {
            region.control.executedCount = true;
            region.control.exitKind = true;
        } else {
            source.operation->emitError("unsupported structured operation reached autodiff tape planning");
            return failure();
        }
    }

    SmallVector<FieldSpec> invocationFields = {{AutodiffTapeFieldKind::InvocationRecordIdentity, std::nullopt}};
    for (unsigned ordinal : rootRegions)
        invocationFields.push_back({AutodiffTapeFieldKind::RootRegionHandle, ordinal});
    FailureOr<AutodiffTapeHeaderSchema> invocationHeader = buildSchema(invocationFields);
    if (failed(invocationHeader)) {
        function.emitError("autodiff invocation header layout overflow");
        return failure();
    }
    plan.invocationHeader = std::move(*invocationHeader);

    SmallVector<RecordBuilder, 0> regionBuilders(plan.regions.size());
    for (auto [index, region] : llvm::enumerate(plan.regions)) {
        SmallVector<FieldSpec> headerFields = {
            {AutodiffTapeFieldKind::ParentRecordIdentity, region.parentRecord.regionOrdinal},
            {AutodiffTapeFieldKind::ChildRegionOrdinal, region.ordinal},
            {AutodiffTapeFieldKind::LastRecordOffset, region.ordinal},
        };
        if (region.control.executedCount)
            headerFields.push_back({AutodiffTapeFieldKind::ExecutedCount, region.ordinal});
        if (region.control.exitKind)
            headerFields.push_back({AutodiffTapeFieldKind::ExitKind, region.ordinal});
        FailureOr<AutodiffTapeHeaderSchema> header = buildSchema(headerFields);
        if (failed(header)) {
            region.operation->emitError("autodiff region header layout overflow");
            return failure();
        }
        region.header = std::move(*header);

        SmallVector<FieldSpec> prefixFields = {
            {AutodiffTapeFieldKind::RecordIdentity, region.ordinal},
            {AutodiffTapeFieldKind::PreviousRecordOffset, region.ordinal},
        };
        if (region.control.predicate)
            prefixFields.push_back({AutodiffTapeFieldKind::Predicate, region.ordinal});
        for (unsigned child : region.childRegionOrdinals)
            prefixFields.push_back({AutodiffTapeFieldKind::ChildRegionHandle, child});
        FailureOr<AutodiffTapeHeaderSchema> prefix = buildSchema(prefixFields);
        if (failed(prefix)) {
            region.operation->emitError("autodiff dynamic record prefix layout overflow");
            return failure();
        }
        regionBuilders[index].prefix = std::move(*prefix);
    }

    auto owningRegion = [&](Value value) -> std::optional<unsigned> {
        Region *valueRegion = value.getParentRegion();
        for (Operation *ancestor = valueRegion ? valueRegion->getParentOp() : nullptr;
             ancestor && ancestor != function.getOperation(); ancestor = ancestor->getParentOp()) {
            auto found = regionPlanIndices.find(ancestor);
            if (found != regionPlanIndices.end())
                return found->second;
        }
        return std::nullopt;
    };

    RecordBuilder invocationBuilder;
    DenseSet<Value> plannedRequirements;
    DenseSet<Value> plannedResidualValues;
    auto appendResidualIntervals = [&](Value value, const ValueAbiLayout &layout, uint64_t lifetimeEnd,
                                       bool rematerialized, uint64_t recomputationCost,
                                       Operation *diagnostic) -> LogicalResult {
        if (!plannedResidualValues.insert(value).second) {
            for (AdResidualInterval &residual : residuals) {
                if (residual.value != value)
                    continue;
                residual.lifetimeEnd = std::max(residual.lifetimeEnd, lifetimeEnd);
                if (!rematerialized) {
                    residual.rematerialized = false;
                    residual.recomputationCost = 0;
                }
            }
            if (!rematerialized)
                llvm::erase_if(rematerializations,
                               [&](const AdRematerializationRecipe &recipe) { return recipe.value == value; });
            return success();
        }
        const uint64_t lifetimeBegin =
            value.getDefiningOp() ? operationOrder.lookup(value.getDefiningOp()) : uint64_t{0};
        for (auto [leafIndex, leaf] : llvm::enumerate(layout.leaves)) {
            FailureOr<unsigned> abiLeafIndex = checkedUnsigned(leafIndex);
            const uint64_t scalarSize = std::max<uint64_t>(leaf.scalarType.getIntOrFloatBitWidth() / 8, 1);
            FailureOr<uint64_t> byteSize = checkedMultiply(scalarSize, leaf.scalarCount);
            if (failed(abiLeafIndex) || failed(byteSize)) {
                diagnostic->emitError("autodiff residual layout metadata overflows");
                return failure();
            }
            residuals.push_back(AdResidualInterval{value, *abiLeafIndex, AdMemoryDomain::PersistentResidual, *byteSize,
                                                   scalarSize, lifetimeBegin, lifetimeEnd, rematerialized,
                                                   rematerialized ? recomputationCost : uint64_t{0}});
        }
        return success();
    };
    for (const AutodiffOperationActivity &activity : analysis.getOperations()) {
        Operation *operation = activity.operation;
        if (!activity.active)
            continue;
        const DifferentiationRule *rule = registry.lookup(operation);
        if (!rule)
            continue;
        SmallVector<unsigned> activeOperands;
        for (auto [operandIndex, operand] : llvm::enumerate(operation->getOperands()))
            if (analysis.isActive(operand, 0))
                activeOperands.push_back(static_cast<unsigned>(operandIndex));
        for (const AutodiffPrimalRequirement &requirement : rule->getVjpPrimalRequirements()) {
            if (!requirement.isRequiredFor(activeOperands))
                continue;
            Value value = requirement.kind == AutodiffPrimalKind::Operand ? operation->getOperand(requirement.index)
                                                                          : operation->getResult(requirement.index);
            if (!plannedRequirements.insert(value).second)
                continue;
            std::optional<unsigned> owner = owningRegion(value);
            RecordBuilder &builder = owner ? regionBuilders[*owner] : invocationBuilder;
            const ValueAbiLayout *layout = analysis.getValueAbi(value);
            if (!layout) {
                operation->emitError("cannot lay out a rule-required primal value on the autodiff tape");
                return failure();
            }
            FailureOr<AdRematerializationRecipe> recipe = buildRematerializationRecipe(value, function);
            const bool rematerialized = succeeded(recipe);
            if (!rematerialized && failed(builder.add(value, *layout))) {
                operation->emitError("cannot lay out a rule-required primal value on the autodiff tape");
                return failure();
            }
            const uint64_t lifetimeEnd =
                *reverseScheduleEnd - std::min(scheduleLength, operationOrder.lookup(operation));
            if (failed(appendResidualIntervals(value, *layout, lifetimeEnd, rematerialized,
                                               rematerialized ? recipe->estimatedCost : uint64_t{0}, operation)))
                return failure();
            if (rematerialized)
                rematerializations.push_back(std::move(*recipe));
        }
    }
    auto saveIndexSource = [&](Value index, RecordBuilder &builder, uint64_t lifetimeEnd, Operation *diagnostic,
                               auto &self) -> LogicalResult {
        if (index.getDefiningOp<arith::ConstantOp>() || index.getDefiningOp<arith::ConstantIndexOp>())
            return success();
        if (auto argument = dyn_cast<BlockArgument>(index)) {
            auto owner = dyn_cast_or_null<func::FuncOp>(argument.getOwner()->getParentOp());
            if (owner && owner.getArgAttr(argument.getArgNumber(), kBuiltinAttrName))
                return success();
        }
        if (index.getType().isIntOrFloat()) {
            const ValueAbiLayout *layout = analysis.getValueAbi(index);
            if (layout) {
                if (failed(builder.add(index, *layout)))
                    return failure();
                return appendResidualIntervals(index, *layout, lifetimeEnd, false, 0, diagnostic);
            }
            FailureOr<ValueAbiLayout> canonical =
                getValueAbiLayout(index.getType(), function->getParentOfType<ModuleOp>());
            if (failed(canonical) || failed(builder.add(index, *canonical)))
                return failure();
            return appendResidualIntervals(index, *canonical, lifetimeEnd, false, 0, diagnostic);
        }
        Operation *defining = index.getDefiningOp();
        if (!defining || defining->getNumRegions() != 0)
            return failure();
        for (Value operand : defining->getOperands())
            if (failed(self(operand, builder, lifetimeEnd, diagnostic, self)))
                return failure();
        return success();
    };
    for (const AutodiffOperationActivity &activity : analysis.getOperations()) {
        if (!activity.active)
            continue;
        ValueRange indices;
        if (auto load = dyn_cast<LoadOp>(activity.operation))
            indices = load.getIndices();
        else if (auto store = dyn_cast<StoreOp>(activity.operation))
            indices = store.getIndices();
        else if (auto extract = dyn_cast<tensor::ExtractOp>(activity.operation))
            indices = extract.getIndices();
        else
            continue;
        std::optional<unsigned> owner = indices.empty() ? std::nullopt : owningRegion(indices.front());
        RecordBuilder &builder = owner ? regionBuilders[*owner] : invocationBuilder;
        const uint64_t lifetimeEnd =
            *reverseScheduleEnd - std::min(scheduleLength, operationOrder.lookup(activity.operation));
        for (Value index : indices)
            if (failed(saveIndexSource(index, builder, lifetimeEnd, activity.operation, saveIndexSource))) {
                activity.operation->emitError("cannot save a dynamic index in the canonical tape layout");
                return failure();
            }
    }

    FailureOr<AutodiffTapeRecord> invocationRecord = std::move(invocationBuilder).finish();
    if (failed(invocationRecord)) {
        function.emitError("autodiff invocation record layout overflow");
        return failure();
    }
    plan.invocationRecord = std::move(*invocationRecord);
    for (auto [index, builder] : llvm::enumerate(regionBuilders)) {
        FailureOr<AutodiffTapeRecord> record = std::move(builder).finish();
        if (failed(record)) {
            plan.regions[index].operation->emitError("autodiff dynamic record layout overflow");
            return failure();
        }
        plan.regions[index].record = std::move(*record);
    }

    FailureOr<uint64_t> hint = checkedAdd(plan.invocationHeader.size, plan.invocationRecord.stride);
    for (const AutodiffTapeRegion &region : plan.regions) {
        if (succeeded(hint))
            hint = checkedAdd(*hint, region.header.size);
        if (succeeded(hint))
            hint = checkedAdd(*hint, region.record.stride);
    }
    if (failed(hint)) {
        function.emitError("autodiff static tape size hint overflow");
        return failure();
    }
    plan.staticTapeBytesHint = *hint;
    plan.memoryPlan.residuals = std::move(residuals);
    plan.memoryPlan.rematerializations = std::move(rematerializations);
    FailureOr<AdBufferAssignment> assignment = assignAdMemoryBuffers(plan.memoryPlan.residuals);
    if (failed(assignment)) {
        function.emitError("autodiff residual buffer assignment failed");
        return failure();
    }
    plan.memoryPlan.bufferAssignment = std::move(*assignment);
    plan.memoryPlan.estimatedPersistentBytes = *hint;
    return plan;
}

} // namespace mlir::vernon
