#include "mlir/Dialect/Vernon/Transforms/VernonAutodiffTapePlanning.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vernon/IR/VernonValueAbi.h"
#include "mlir/Dialect/Vernon/Transforms/VernonAutodiffAnalysis.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"

#include <limits>

namespace mlir::vernon {
namespace {

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
    // The planner's current storage policy is intentionally conservative:
    // save every semantic requirement declared by a rule. A future
    // recomputation policy belongs here, but must also produce a materialization
    // recipe instead of silently dropping a required primal.
    for (const AutodiffOperationActivity &activity : analysis.getOperations()) {
        Operation *operation = activity.operation;
        if (!activity.active)
            continue;
        const DifferentiationRule *rule = registry.lookup(operation);
        if (!rule)
            continue;
        for (const AutodiffPrimalRequirement &requirement : rule->getVjpPrimalRequirements()) {
            Value value = requirement.kind == AutodiffPrimalKind::Operand ? operation->getOperand(requirement.index)
                                                                          : operation->getResult(requirement.index);
            std::optional<unsigned> owner = owningRegion(value);
            RecordBuilder &builder = owner ? regionBuilders[*owner] : invocationBuilder;
            const ValueAbiLayout *layout = analysis.getValueAbi(value);
            if (!layout || failed(builder.add(value, *layout))) {
                operation->emitError("cannot lay out a rule-required primal value on the autodiff tape");
                return failure();
            }
        }
    }
    auto saveIndexSource = [&](Value index, RecordBuilder &builder, auto &self) -> LogicalResult {
        if (index.getDefiningOp<arith::ConstantOp>() || index.getDefiningOp<arith::ConstantIndexOp>())
            return success();
        if (index.getType().isIntOrFloat()) {
            const ValueAbiLayout *layout = analysis.getValueAbi(index);
            if (layout)
                return builder.add(index, *layout);
            FailureOr<ValueAbiLayout> canonical =
                getValueAbiLayout(index.getType(), function->getParentOfType<ModuleOp>());
            return succeeded(canonical) ? builder.add(index, *canonical) : failure();
        }
        Operation *defining = index.getDefiningOp();
        if (!defining || defining->getNumRegions() != 0)
            return failure();
        for (Value operand : defining->getOperands())
            if (failed(self(operand, builder, self)))
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
        for (Value index : indices)
            if (failed(saveIndexSource(index, builder, saveIndexSource))) {
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
    return plan;
}

} // namespace mlir::vernon
