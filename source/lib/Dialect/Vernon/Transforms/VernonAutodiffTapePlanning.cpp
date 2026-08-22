#include "mlir/Dialect/Vernon/Transforms/VernonAutodiffTapePlanning.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vernon/IR/VernonAttrs.h"
#include "mlir/Dialect/Vernon/IR/VernonValueAbi.h"
#include "mlir/Dialect/Vernon/Transforms/VernonAutodiffAnalysis.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"

#include <functional>
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

FailureOr<AdRematerializationRecipe> buildRematerializationRecipeImpl(Value value, func::FuncOp function,
                                                                      llvm::function_ref<bool(Value)> isAvailableRoot) {
    AdRematerializationRecipe recipe;
    recipe.value = value;
    DenseSet<Operation *> planned;
    DenseSet<Value> visiting;
    auto visit = [&](Value current, auto &self) -> LogicalResult {
        if (isAvailableRoot(current))
            return success();
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

FailureOr<AdRematerializationRecipe> buildAutodiffRematerializationRecipe(Value value, func::FuncOp function) {
    return buildRematerializationRecipeImpl(value, function, [](Value) { return false; });
}

FailureOr<AdRematerializationRecipe>
buildAutodiffRematerializationRecipe(Value value, func::FuncOp function,
                                     llvm::function_ref<bool(Value)> isAvailableRoot) {
    return buildRematerializationRecipeImpl(value, function, isAvailableRoot);
}

StringRef stringifyAdResidualSourceKind(AdResidualSourceKind kind) {
    switch (kind) {
    case AdResidualSourceKind::Builtin:
        return "builtin";
    case AdResidualSourceKind::PrimalArgument:
        return "primal_argument";
    case AdResidualSourceKind::ExactVersionReload:
        return "exact_version_reload";
    case AdResidualSourceKind::PureRematerialization:
        return "pure_rematerialization";
    case AdResidualSourceKind::StaticCapture:
        return "static_capture";
    case AdResidualSourceKind::DynamicCapture:
        return "dynamic_capture";
    case AdResidualSourceKind::Unsupported:
        return "unsupported";
    }
    llvm_unreachable("unknown autodiff residual source kind");
}

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

    SmallVector<AdResidualInterval> minimumMemory = working;
    for (const AdRematerializationCandidate &candidate : candidates)
        for (unsigned residualIndex : candidate.residualIndices)
            minimumMemory[residualIndex].rematerialized = true;
    FailureOr<AdBufferAssignment> minimumAssignment = assignAdMemoryBuffers(minimumMemory);
    if (failed(minimumAssignment) || minimumAssignment->peakBytes > budgetBytes)
        return failure();

    for (const ScoredCandidate &scored : order) {
        if (result.buffers.peakBytes <= budgetBytes)
            break;
        const AdRematerializationCandidate &candidate = candidates[scored.index];
        bool changed = false;
        for (unsigned residualIndex : candidate.residualIndices)
            if (!working[residualIndex].rematerialized) {
                working[residualIndex].rematerialized = true;
                working[residualIndex].recomputationCost = candidate.recomputationCost;
                changed = true;
            }
        if (!changed)
            continue;
        FailureOr<AdBufferAssignment> assignment = assignAdMemoryBuffers(working);
        if (failed(assignment))
            return failure();
        FailureOr<uint64_t> cost = checkedAdd(result.recomputationCost, candidate.recomputationCost);
        if (failed(cost))
            return failure();
        result.recomputationCost = *cost;
        result.selectedCandidates[scored.index] = true;
        result.buffers = std::move(*assignment);
    }
    if (result.buffers.peakBytes > budgetBytes)
        return failure();

    // Remove expensive selections that are not required after interactions
    // between reusable physical buffers have been accounted for.
    SmallVector<ScoredCandidate> removalOrder;
    for (const ScoredCandidate &candidate : order)
        if (result.selectedCandidates[candidate.index])
            removalOrder.push_back(candidate);
    llvm::stable_sort(removalOrder, [](const ScoredCandidate &left, const ScoredCandidate &right) {
        if (left.cost != right.cost)
            return left.cost > right.cost;
        return left.index > right.index;
    });
    for (const ScoredCandidate &scored : removalOrder) {
        const AdRematerializationCandidate &candidate = candidates[scored.index];
        for (unsigned residualIndex : candidate.residualIndices) {
            working[residualIndex].rematerialized = residuals[residualIndex].rematerialized;
            working[residualIndex].recomputationCost = residuals[residualIndex].recomputationCost;
        }
        FailureOr<AdBufferAssignment> assignment = assignAdMemoryBuffers(working);
        if (failed(assignment))
            return failure();
        if (assignment->peakBytes <= budgetBytes) {
            result.selectedCandidates[scored.index] = false;
            result.recomputationCost -= candidate.recomputationCost;
            result.buffers = std::move(*assignment);
            continue;
        }
        for (unsigned residualIndex : candidate.residualIndices) {
            working[residualIndex].rematerialized = true;
            working[residualIndex].recomputationCost = candidate.recomputationCost;
        }
    }
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

    constexpr uint64_t defaultMemoryBudgetBytes = 64ull * 1024ull * 1024ull;
    uint64_t memoryBudgetBytes = defaultMemoryBudgetBytes;
    if (auto budget = function->getAttrOfType<IntegerAttr>("vernon.ad.memory_budget_bytes")) {
        const APInt &value = budget.getValue();
        if (value.isNegative() || value.isZero() || value.getActiveBits() > 64) {
            function.emitError("'vernon.ad.memory_budget_bytes' must be a positive 64-bit integer");
            return failure();
        }
        memoryBudgetBytes = value.getZExtValue();
    }
    StringRef selectedPolicy = "min_memory";
    if (auto policy = function->getAttrOfType<StringAttr>("vernon.ad.planning_policy")) {
        selectedPolicy = policy.getValue();
        if (selectedPolicy != "min_memory" && selectedPolicy != "balanced" && selectedPolicy != "min_runtime") {
            function.emitError("'vernon.ad.planning_policy' must be 'min_memory', 'balanced', or 'min_runtime'");
            return failure();
        }
    }

    VernonAutodiffTapePlan plan;
    plan.memoryPlan.memoryBudgetBytes = memoryBudgetBytes;
    plan.memoryPlan.selectedPolicy = selectedPolicy.str();
    plan.memoryPlan.wholeDispatchRetentionPermitted = selectedPolicy != "min_memory";
    SmallVector<AdResidualInterval> residuals;
    SmallVector<AdRematerializationRecipe> rematerializations;
    DenseMap<Operation *, uint64_t> operationOrder;
    for (auto [index, activity] : llvm::enumerate(analysis.getOperations()))
        operationOrder.try_emplace(activity.operation, static_cast<uint64_t>(index));
    auto isFunctionArgument = [&](Value value) {
        auto argument = dyn_cast<BlockArgument>(value);
        return argument && argument.getOwner() == &function.getBody().front();
    };
    const uint64_t scheduleLength = static_cast<uint64_t>(analysis.getOperations().size());
    FailureOr<uint64_t> reverseScheduleEnd = checkedMultiply(scheduleLength, 2);
    if (failed(reverseScheduleEnd)) {
        function.emitError("autodiff residual schedule length overflows");
        return failure();
    }
    DenseMap<Operation *, unsigned> regionPlanIndices;
    DenseMap<unsigned, unsigned> sourceToPlanIndex;
    DenseSet<Operation *> reconstructibleControl;
    DenseSet<Operation *> reconstructibleLoops;
    using AvailableRoot = std::function<bool(Value)>;
    auto isControlRoot = [&](Value value) {
        if (isFunctionArgument(value))
            return true;
        auto argument = dyn_cast<BlockArgument>(value);
        if (!argument)
            return false;
        auto loop = dyn_cast_or_null<scf::ForOp>(argument.getOwner()->getParentOp());
        if (!loop || argument != loop.getInductionVar())
            return false;
        auto step = loop.getStep().getDefiningOp<arith::ConstantIndexOp>();
        return step && step.value() > 0;
    };
    auto isExactReloadAvailable = [&](Value value, const AvailableRoot &isAvailableRoot) {
        auto load = value.getDefiningOp<LoadOp>();
        const AutodiffLoadInfo *loadInfo = load ? analysis.getLoadInfo(load) : nullptr;
        if (!load || !loadInfo || loadInfo->stability != AutodiffStorageStabilityRequirement::RetainedExactVersion ||
            !isa<BlockArgument>(load.getStorage()) || !isAvailableRoot(load.getStorage()))
            return false;
        return llvm::all_of(load.getIndices(), [&](Value index) {
            return succeeded(buildAutodiffRematerializationRecipe(index, function, isAvailableRoot));
        });
    };
    auto buildReconstructionRecipe = [&](Value value, const AvailableRoot &isAvailableRoot,
                                         SmallVectorImpl<Value> *exactReloadRoots = nullptr) {
        return buildAutodiffRematerializationRecipe(value, function, [&](Value root) {
            if (isAvailableRoot(root))
                return true;
            if (!isExactReloadAvailable(root, isAvailableRoot))
                return false;
            if (exactReloadRoots && !llvm::is_contained(*exactReloadRoots, root))
                exactReloadRoots->push_back(root);
            return true;
        });
    };
    auto canReconstructValue = [&](Value value, const AvailableRoot &isAvailableRoot) {
        return succeeded(buildReconstructionRecipe(value, isAvailableRoot));
    };
    auto canReconstructActiveOperation = [&](Operation *operation, const AvailableRoot &isAvailableRoot,
                                             bool rejectStorageEffects) {
        if (!analysis.isActive(operation))
            return true;
        if (rejectStorageEffects && analysis.getStorageEffect(operation))
            return false;
        if (const DifferentiationRule *rule = registry.lookup(operation)) {
            SmallVector<unsigned> activeOperands;
            for (auto [operandIndex, operand] : llvm::enumerate(operation->getOperands()))
                if (analysis.hasAnyActiveLeaf(operand))
                    activeOperands.push_back(static_cast<unsigned>(operandIndex));
            for (const AutodiffPrimalRequirement &requirement : rule->getVjpPrimalRequirements()) {
                if (!requirement.isRequiredFor(activeOperands))
                    continue;
                Value value = requirement.kind == AutodiffPrimalKind::Operand ? operation->getOperand(requirement.index)
                                                                              : operation->getResult(requirement.index);
                if (!canReconstructValue(value, isAvailableRoot))
                    return false;
            }
        }
        ValueRange indices;
        if (auto load = dyn_cast<LoadOp>(operation))
            indices = load.getIndices();
        else if (auto store = dyn_cast<StoreOp>(operation))
            indices = store.getIndices();
        else if (auto extract = dyn_cast<tensor::ExtractOp>(operation))
            indices = extract.getIndices();
        return llvm::all_of(indices, [&](Value index) {
            return succeeded(buildAutodiffRematerializationRecipe(index, function, isAvailableRoot));
        });
    };
    std::function<bool(scf::IfOp, const AvailableRoot &)> canReconstructIf;
    std::function<bool(scf::ForOp, const AvailableRoot &)> canReconstructFor;
    canReconstructIf = [&](scf::IfOp branch, const AvailableRoot &isAvailableRoot) {
        if (!canReconstructValue(branch.getCondition(), isAvailableRoot))
            return false;
        WalkResult result = branch.walk<WalkOrder::PreOrder>([&](Operation *operation) {
            if (operation == branch.getOperation())
                return WalkResult::advance();
            if (!analysis.isActive(operation))
                return operation->getNumRegions() == 0 ? WalkResult::advance() : WalkResult::skip();
            if (auto nestedIf = dyn_cast<scf::IfOp>(operation))
                return canReconstructIf(nestedIf, isAvailableRoot) ? WalkResult::skip() : WalkResult::interrupt();
            if (auto nestedFor = dyn_cast<scf::ForOp>(operation))
                return canReconstructFor(nestedFor, isAvailableRoot) ? WalkResult::skip() : WalkResult::interrupt();
            if (isa<scf::WhileOp>(operation))
                return analysis.isActive(operation) ? WalkResult::interrupt() : WalkResult::skip();
            return canReconstructActiveOperation(operation, isAvailableRoot, true) ? WalkResult::advance()
                                                                                   : WalkResult::interrupt();
        });
        return !result.wasInterrupted();
    };
    canReconstructFor = [&](scf::ForOp loop, const AvailableRoot &parentRoot) {
        auto step = loop.getStep().getDefiningOp<arith::ConstantIndexOp>();
        if (!step || step.value() <= 0)
            return false;
        AvailableRoot isAvailableRoot = [parentRoot, induction = loop.getInductionVar()](Value value) {
            return parentRoot(value) || value == induction;
        };
        if (failed(buildAutodiffRematerializationRecipe(loop.getLowerBound(), function, parentRoot)) ||
            failed(buildAutodiffRematerializationRecipe(loop.getUpperBound(), function, parentRoot)))
            return false;
        WalkResult result = loop.walk<WalkOrder::PreOrder>([&](Operation *operation) {
            if (operation == loop.getOperation())
                return WalkResult::advance();
            if (!analysis.isActive(operation))
                return operation->getNumRegions() == 0 ? WalkResult::advance() : WalkResult::skip();
            if (auto branch = dyn_cast<scf::IfOp>(operation)) {
                return canReconstructIf(branch, isAvailableRoot) ? WalkResult::skip() : WalkResult::interrupt();
            }
            if (auto nestedFor = dyn_cast<scf::ForOp>(operation))
                return canReconstructFor(nestedFor, isAvailableRoot) ? WalkResult::skip() : WalkResult::interrupt();
            if (isa<scf::WhileOp>(operation))
                return analysis.isActive(operation) ? WalkResult::interrupt() : WalkResult::skip();
            return canReconstructActiveOperation(operation, isAvailableRoot, false) ? WalkResult::advance()
                                                                                    : WalkResult::interrupt();
        });
        return !result.wasInterrupted();
    };
    for (const AutodiffRegion &source : analysis.getRegions()) {
        if (!analysis.isActive(source.operation))
            continue;
        if (auto loop = dyn_cast<scf::ForOp>(source.operation)) {
            auto step = loop.getStep().getDefiningOp<arith::ConstantIndexOp>();
            if (!step || step.value() <= 0 ||
                failed(buildAutodiffRematerializationRecipe(loop.getLowerBound(), function, isControlRoot)) ||
                failed(buildAutodiffRematerializationRecipe(loop.getUpperBound(), function, isControlRoot))) {
                loop.emitError(
                    "autodiff requires canonical positive-step scf.for bounds reconstructible from entry primals");
                return failure();
            }
        }
        AvailableRoot sourceRoot = isControlRoot;
        if (auto branch = dyn_cast<scf::IfOp>(source.operation); branch && canReconstructIf(branch, sourceRoot)) {
            reconstructibleControl.insert(source.operation);
            continue;
        }
        if (auto loop = dyn_cast<scf::ForOp>(source.operation); loop && canReconstructFor(loop, sourceRoot)) {
            reconstructibleLoops.insert(source.operation);
            continue;
        }
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
    auto isAvailablePlanningRoot = [&](Value value) {
        if (isFunctionArgument(value))
            return true;
        auto argument = dyn_cast<BlockArgument>(value);
        if (!argument)
            return false;
        auto loop = dyn_cast_or_null<scf::ForOp>(argument.getOwner()->getParentOp());
        return loop && argument == loop.getInductionVar() && reconstructibleLoops.contains(loop.getOperation());
    };

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
            region.parentRegionOrdinal = parentRegion.ordinal;
            FailureOr<unsigned> childOrdinal = checkedUnsigned(parentRegion.childRegionOrdinals.size());
            if (failed(childOrdinal)) {
                region.operation->emitError("autodiff child-region count exceeds the planner representation");
                return failure();
            }
            region.childOrdinal = *childOrdinal;
            parentRegion.childRegionOrdinals.push_back(region.ordinal);
        } else {
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
        else if (isa<scf::ForOp>(source.operation)) {
            // Canonical positive-step scf.for reconstructs its trip count and
            // induction value. A region, when present, stores only selected
            // per-iteration residuals and nested dynamic children.
        } else if (isa<scf::WhileOp>(source.operation)) {
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
            {AutodiffTapeFieldKind::ParentRecordIdentity, region.parentRegionOrdinal},
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
    uint64_t captureFixedBytes = plan.invocationHeader.size;
    for (auto [index, region] : llvm::enumerate(plan.regions)) {
        FailureOr<uint64_t> withHeader = checkedAdd(captureFixedBytes, region.header.size);
        FailureOr<uint64_t> withPrefix = succeeded(withHeader)
                                             ? checkedAdd(*withHeader, regionBuilders[index].prefix.size)
                                             : FailureOr<uint64_t>(failure());
        if (failed(withPrefix)) {
            function.emitError("autodiff minimum tape layout overflows");
            return failure();
        }
        captureFixedBytes = *withPrefix;
    }
    DenseSet<Value> plannedRequirements;
    DenseSet<Value> plannedResidualValues;
    DenseSet<Value> plannedSourceValues;
    uint64_t selectedCaptureBytes = 0;
    auto appendSourceSelections = [&](Value value, const ValueAbiLayout &layout, std::optional<unsigned> owner,
                                      const FailureOr<AdRematerializationRecipe> &recipe) {
        if (!plannedSourceValues.insert(value).second)
            return AdResidualSourceKind::Unsupported;
        AdResidualSourceKind selectedKind = AdResidualSourceKind::Unsupported;
        for (auto [leafIndex, unused] : llvm::enumerate(layout.leaves)) {
            (void)unused;
            AdResidualSourceSelection selection;
            selection.key.value = value;
            selection.key.abiLeafIndex = static_cast<unsigned>(leafIndex);
            const ValueAbiLeaf &leaf = layout.leaves[leafIndex];
            const uint64_t scalarSize = std::max<uint64_t>(leaf.scalarType.getIntOrFloatBitWidth() / 8, 1);
            FailureOr<uint64_t> leafBytes = checkedMultiply(scalarSize, leaf.scalarCount);
            auto addCandidate = [&](AdResidualSourceKind kind, bool current, uint64_t runtimeCost = 0,
                                    std::optional<unsigned> identity = std::nullopt,
                                    std::optional<unsigned> version = std::nullopt,
                                    ArrayRef<Operation *> operations = {}, bool legal = true) {
                AdResidualSource source;
                source.kind = kind;
                source.legal = legal;
                source.availableInCurrentContract = current;
                source.storageIdentity = identity;
                source.versionBefore = version;
                if (kind == AdResidualSourceKind::StaticCapture || kind == AdResidualSourceKind::DynamicCapture) {
                    source.captureStoreBytes = succeeded(leafBytes) ? *leafBytes : 0;
                    source.backwardLoadBytes = succeeded(leafBytes) ? *leafBytes : 0;
                } else if (kind == AdResidualSourceKind::ExactVersionReload) {
                    source.resourceReloadCost = runtimeCost;
                } else if (kind == AdResidualSourceKind::PureRematerialization) {
                    source.recomputationCost = runtimeCost;
                }
                llvm::append_range(source.recipe, operations);
                selection.candidates.push_back(std::move(source));
            };
            if (auto argument = dyn_cast<BlockArgument>(value)) {
                auto ownerFunction = dyn_cast_or_null<func::FuncOp>(argument.getOwner()->getParentOp());
                if (ownerFunction == function) {
                    if (function.getArgAttr(argument.getArgNumber(), kBuiltinAttrName))
                        addCandidate(AdResidualSourceKind::Builtin, true);
                    else
                        // TensorView primals require a retained logical resource
                        // version. The contract can describe that input, but a
                        // raw descriptor or unaccounted deep copy is not a
                        // legal planner source before graph versioning lands.
                        addCandidate(AdResidualSourceKind::PrimalArgument, !isa<TensorViewType>(argument.getType()));
                }
            }
            if (auto load = value.getDefiningOp<LoadOp>()) {
                if (const AutodiffLoadInfo *loadInfo = analysis.getLoadInfo(load)) {
                    const bool reconstructibleIndices = llvm::all_of(load.getIndices(), [&](Value index) {
                        return succeeded(
                            buildAutodiffRematerializationRecipe(index, function, isAvailablePlanningRoot));
                    });
                    const bool retainedExactVersion =
                        loadInfo->stability == AutodiffStorageStabilityRequirement::RetainedExactVersion &&
                        isa<BlockArgument>(load.getStorage());
                    if (reconstructibleIndices)
                        addCandidate(AdResidualSourceKind::ExactVersionReload, retainedExactVersion, 1,
                                     loadInfo->identity, loadInfo->versionBefore, {}, retainedExactVersion);
                    else
                        addCandidate(AdResidualSourceKind::Unsupported, false, 0, loadInfo->identity,
                                     loadInfo->versionBefore, {}, false);
                }
            }
            if (succeeded(recipe))
                addCandidate(AdResidualSourceKind::PureRematerialization, true, recipe->estimatedCost, std::nullopt,
                             std::nullopt, recipe->operations);
            addCandidate(owner ? AdResidualSourceKind::DynamicCapture : AdResidualSourceKind::StaticCapture, true);
            auto weightedRuntimeCost = [](const AdResidualSource &candidate) {
                auto saturatingAdd = [](uint64_t left, uint64_t right) {
                    return right > std::numeric_limits<uint64_t>::max() - left ? std::numeric_limits<uint64_t>::max()
                                                                               : left + right;
                };
                auto saturatingMultiply = [](uint64_t value, uint64_t weight) {
                    return value > std::numeric_limits<uint64_t>::max() / weight ? std::numeric_limits<uint64_t>::max()
                                                                                 : value * weight;
                };
                uint64_t result = saturatingAdd(candidate.captureStoreBytes, candidate.backwardLoadBytes);
                result = saturatingAdd(result, saturatingMultiply(candidate.resourceReloadCost, 4));
                return saturatingAdd(result, saturatingMultiply(candidate.recomputationCost, 8));
            };
            auto score = [&](const AdResidualSource &candidate) {
                const uint64_t retained = candidate.captureStoreBytes;
                const uint64_t runtime = weightedRuntimeCost(candidate);
                if (selectedPolicy == "min_runtime")
                    return std::pair(runtime, retained);
                if (selectedPolicy == "balanced") {
                    const uint64_t balanced = retained > std::numeric_limits<uint64_t>::max() - runtime
                                                  ? std::numeric_limits<uint64_t>::max()
                                                  : retained + runtime;
                    return std::pair(balanced, retained);
                }
                return std::pair(retained, runtime);
            };
            auto selected = std::prev(selection.candidates.end());
            auto fitsCaptureBudget = [&](const AdResidualSource &candidate) {
                const uint64_t padding =
                    candidate.captureStoreBytes ? std::max<uint64_t>(scalarSize, 1) - 1 : uint64_t{0};
                const uint64_t fixedBytes = selectedCaptureBytes || !plan.regions.empty() || candidate.captureStoreBytes
                                                ? captureFixedBytes
                                                : 0;
                if (fixedBytes > memoryBudgetBytes || selectedCaptureBytes > memoryBudgetBytes - fixedBytes ||
                    candidate.captureStoreBytes > memoryBudgetBytes - fixedBytes - selectedCaptureBytes)
                    return false;
                return padding <= memoryBudgetBytes - fixedBytes - selectedCaptureBytes - candidate.captureStoreBytes;
            };
            for (auto candidate = selection.candidates.begin(); candidate != selection.candidates.end(); ++candidate) {
                if (!candidate->legal || !candidate->availableInCurrentContract ||
                    !candidate->deterministicReductionLegal)
                    continue;
                if (!fitsCaptureBudget(*candidate))
                    continue;
                if (!selected->legal || !selected->availableInCurrentContract ||
                    !selected->deterministicReductionLegal || !fitsCaptureBudget(*selected) ||
                    score(*candidate) < score(*selected))
                    selected = candidate;
            }
            if (!fitsCaptureBudget(*selected)) {
                selection.selectedCandidate = static_cast<unsigned>(selection.candidates.size());
                plan.memoryPlan.sourceSelections.push_back(std::move(selection));
                return AdResidualSourceKind::Unsupported;
            }
            selectedCaptureBytes += selected->captureStoreBytes
                                        ? selected->captureStoreBytes + std::max<uint64_t>(scalarSize, 1) - 1
                                        : uint64_t{0};
            selection.selectedCandidate = static_cast<unsigned>(std::distance(selection.candidates.begin(), selected));
            if (selectedKind == AdResidualSourceKind::Unsupported)
                selectedKind = selected->kind;
            else if (selectedKind != selected->kind)
                selectedKind = owner ? AdResidualSourceKind::DynamicCapture : AdResidualSourceKind::StaticCapture;
            plan.memoryPlan.sourceSelections.push_back(std::move(selection));
        }
        return selectedKind;
    };
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
            if (analysis.hasAnyActiveLeaf(operand))
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
            SmallVector<Value> exactReloadRoots;
            FailureOr<AdRematerializationRecipe> recipe =
                buildReconstructionRecipe(value, isAvailablePlanningRoot, &exactReloadRoots);
            for (Value root : exactReloadRoots) {
                if (root == value || plannedSourceValues.contains(root))
                    continue;
                const ValueAbiLayout *rootLayout = analysis.getValueAbi(root);
                if (!rootLayout || appendSourceSelections(root, *rootLayout, owningRegion(root),
                                                          FailureOr<AdRematerializationRecipe>(failure())) !=
                                       AdResidualSourceKind::ExactVersionReload) {
                    operation->emitError("cannot select an exact-version reload dependency for rematerialization");
                    return failure();
                }
            }
            FailureOr<AdRematerializationRecipe> selectedRecipe = failure();
            if (!llvm::is_contained(exactReloadRoots, value))
                selectedRecipe = std::move(recipe);
            AdResidualSourceKind selectedSource = appendSourceSelections(value, *layout, owner, selectedRecipe);
            const bool externalPrimal = selectedSource == AdResidualSourceKind::Builtin ||
                                        selectedSource == AdResidualSourceKind::PrimalArgument;
            const bool rematerialized = selectedSource == AdResidualSourceKind::PureRematerialization ||
                                        selectedSource == AdResidualSourceKind::ExactVersionReload || externalPrimal;
            if (!rematerialized && failed(builder.add(value, *layout))) {
                operation->emitError("cannot lay out a rule-required primal value on the autodiff tape");
                return failure();
            }
            const uint64_t lifetimeEnd =
                *reverseScheduleEnd - std::min(scheduleLength, operationOrder.lookup(operation));
            const uint64_t selectedRecomputationCost =
                selectedSource == AdResidualSourceKind::PureRematerialization && succeeded(selectedRecipe)
                    ? selectedRecipe->estimatedCost
                    : uint64_t{0};
            if (failed(appendResidualIntervals(value, *layout, lifetimeEnd, rematerialized, selectedRecomputationCost,
                                               operation)))
                return failure();
            if (selectedSource == AdResidualSourceKind::PureRematerialization)
                rematerializations.push_back(std::move(*selectedRecipe));
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
            if (auto loop = dyn_cast_or_null<scf::ForOp>(argument.getOwner()->getParentOp());
                loop && argument == loop.getInductionVar())
                return success();
        }
        FailureOr<AdRematerializationRecipe> recipe =
            buildAutodiffRematerializationRecipe(index, function, isAvailablePlanningRoot);
        if (succeeded(recipe)) {
            const ValueAbiLayout *knownLayout = analysis.getValueAbi(index);
            FailureOr<ValueAbiLayout> canonical =
                knownLayout ? FailureOr<ValueAbiLayout>(*knownLayout)
                            : getValueStorageLayout(index.getType(), function->getParentOfType<ModuleOp>());
            if (succeeded(canonical)) {
                AdResidualSourceKind selectedSource =
                    appendSourceSelections(index, *canonical, owningRegion(index), recipe);
                if (failed(appendResidualIntervals(index, *canonical, lifetimeEnd, true, recipe->estimatedCost,
                                                   diagnostic)))
                    return failure();
                if (selectedSource == AdResidualSourceKind::PureRematerialization)
                    rematerializations.push_back(std::move(*recipe));
                return success();
            }
        }
        if (index.getType().isIntOrFloat()) {
            const ValueAbiLayout *layout = analysis.getValueAbi(index);
            if (layout) {
                appendSourceSelections(index, *layout, owningRegion(index), recipe);
                if (failed(builder.add(index, *layout)))
                    return failure();
                return appendResidualIntervals(index, *layout, lifetimeEnd, false, 0, diagnostic);
            }
            FailureOr<ValueAbiLayout> canonical =
                getValueStorageLayout(index.getType(), function->getParentOfType<ModuleOp>());
            if (failed(canonical) || failed(builder.add(index, *canonical)))
                return failure();
            appendSourceSelections(index, *canonical, owningRegion(index), recipe);
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
    auto appendControlSource = [&](Operation *operation, AdControlSourceKind controlKind, Value reconstructibleValue,
                                   bool captureRequired) {
        AdResidualSourceSelection selection;
        selection.key.value = reconstructibleValue;
        selection.key.controlOperation = operation;
        selection.key.controlKind = controlKind;
        auto add = [&](AdResidualSourceKind kind, bool current, uint64_t cost = 0,
                       ArrayRef<Operation *> operations = {}) {
            AdResidualSource candidate;
            candidate.kind = kind;
            candidate.legal = true;
            candidate.availableInCurrentContract = current;
            if (kind == AdResidualSourceKind::PureRematerialization)
                candidate.recomputationCost = cost;
            else if (kind == AdResidualSourceKind::DynamicCapture) {
                candidate.captureStoreBytes = 1;
                candidate.backwardLoadBytes = 1;
            }
            llvm::append_range(candidate.recipe, operations);
            selection.candidates.push_back(std::move(candidate));
        };
        if (reconstructibleValue) {
            if (auto argument = dyn_cast<BlockArgument>(reconstructibleValue)) {
                auto owner = dyn_cast_or_null<func::FuncOp>(argument.getOwner()->getParentOp());
                if (owner == function) {
                    if (function.getArgAttr(argument.getArgNumber(), kBuiltinAttrName))
                        add(AdResidualSourceKind::Builtin, true);
                    else
                        add(AdResidualSourceKind::PrimalArgument, true);
                }
            }
            FailureOr<AdRematerializationRecipe> recipe =
                buildAutodiffRematerializationRecipe(reconstructibleValue, function, isAvailablePlanningRoot);
            if (succeeded(recipe) && !recipe->operations.empty())
                add(AdResidualSourceKind::PureRematerialization, true, recipe->estimatedCost, recipe->operations);
        }
        if (captureRequired)
            add(AdResidualSourceKind::DynamicCapture, true);
        if (selection.candidates.empty())
            add(AdResidualSourceKind::Unsupported, false);
        selection.selectedCandidate = static_cast<unsigned>(selection.candidates.size() - 1);
        plan.memoryPlan.sourceSelections.push_back(std::move(selection));
    };
    for (AutodiffTapeRegion &region : plan.regions) {
        if (auto ifOp = dyn_cast<scf::IfOp>(region.operation))
            appendControlSource(region.operation, AdControlSourceKind::Predicate, ifOp.getCondition(), true);
        if (region.control.executedCount)
            appendControlSource(region.operation, AdControlSourceKind::ExecutedCount, {}, true);
        if (region.control.exitKind)
            appendControlSource(region.operation, AdControlSourceKind::ExitKind, {}, true);
    }
    for (Operation *operation : reconstructibleControl) {
        auto branch = cast<scf::IfOp>(operation);
        appendControlSource(operation, AdControlSourceKind::Predicate, branch.getCondition(), false);
    }
    for (const AutodiffRegion &source : analysis.getRegions()) {
        if (!analysis.isActive(source.operation))
            continue;
        auto loop = dyn_cast<scf::ForOp>(source.operation);
        if (!loop)
            continue;
        appendControlSource(source.operation, AdControlSourceKind::ExecutedCount, loop.getLowerBound(), false);
        appendControlSource(source.operation, AdControlSourceKind::ExecutedCount, loop.getUpperBound(), false);
        appendControlSource(source.operation, AdControlSourceKind::ExecutedCount, loop.getStep(), false);
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
    SmallVector<AdRematerializationCandidate> budgetCandidates;
    budgetCandidates.reserve(plan.memoryPlan.rematerializations.size());
    DenseSet<Value> budgetCandidateValues;
    for (const AdRematerializationRecipe &recipe : plan.memoryPlan.rematerializations) {
        if (!budgetCandidateValues.insert(recipe.value).second)
            continue;
        AdRematerializationCandidate candidate;
        candidate.recomputationCost = recipe.estimatedCost;
        for (auto [residualIndex, residual] : llvm::enumerate(plan.memoryPlan.residuals))
            if (residual.value == recipe.value)
                candidate.residualIndices.push_back(static_cast<unsigned>(residualIndex));
        if (!candidate.residualIndices.empty())
            budgetCandidates.push_back(std::move(candidate));
    }
    FailureOr<AdBudgetedBufferAssignment> assignment =
        assignAdMemoryBuffersWithinBudget(plan.memoryPlan.residuals, budgetCandidates, memoryBudgetBytes);
    if (failed(assignment)) {
        function.emitError() << "no legal autodiff residual plan fits the " << memoryBudgetBytes
                             << "-byte compiler memory budget";
        return failure();
    }
    plan.memoryPlan.bufferAssignment = std::move(assignment->buffers);
    const bool selectedCapture =
        llvm::any_of(plan.memoryPlan.sourceSelections, [](const AdResidualSourceSelection &selection) {
            if (selection.selectedCandidate >= selection.candidates.size())
                return true;
            AdResidualSourceKind kind = selection.candidates[selection.selectedCandidate].kind;
            return kind == AdResidualSourceKind::StaticCapture || kind == AdResidualSourceKind::DynamicCapture;
        });
    const uint64_t retainedTapeBytes = plan.regions.empty() && !selectedCapture ? 0 : *hint;
    if (retainedTapeBytes > memoryBudgetBytes) {
        function.emitError() << "no legal autodiff residual plan fits the " << memoryBudgetBytes
                             << "-byte compiler memory budget";
        return failure();
    }
    plan.memoryPlan.estimatedPersistentBytes = retainedTapeBytes;
    plan.memoryPlan.costComponents.retainedTapeBytes = retainedTapeBytes;
    for (const AdResidualSourceSelection &selection : plan.memoryPlan.sourceSelections) {
        if (selection.selectedCandidate >= selection.candidates.size())
            continue;
        const AdResidualSource &source = selection.candidates[selection.selectedCandidate];
        FailureOr<uint64_t> capture =
            checkedAdd(plan.memoryPlan.costComponents.captureStoreBytes, source.captureStoreBytes);
        FailureOr<uint64_t> load =
            checkedAdd(plan.memoryPlan.costComponents.backwardLoadBytes, source.backwardLoadBytes);
        FailureOr<uint64_t> reload =
            checkedAdd(plan.memoryPlan.costComponents.resourceReloadCost, source.resourceReloadCost);
        FailureOr<uint64_t> recompute =
            checkedAdd(plan.memoryPlan.costComponents.recomputationCost, source.recomputationCost);
        if (failed(capture) || failed(load) || failed(reload) || failed(recompute)) {
            function.emitError("autodiff selected source cost telemetry overflows");
            return failure();
        }
        plan.memoryPlan.costComponents.captureStoreBytes = *capture;
        plan.memoryPlan.costComponents.backwardLoadBytes = *load;
        plan.memoryPlan.costComponents.resourceReloadCost = *reload;
        plan.memoryPlan.costComponents.recomputationCost = *recompute;
    }
    return plan;
}

} // namespace mlir::vernon
