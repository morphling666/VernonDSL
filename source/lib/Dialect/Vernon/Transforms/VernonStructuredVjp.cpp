#include "mlir/Dialect/Vernon/Transforms/VernonStructuredVjp.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vernon/IR/Vernon.h"
#include "mlir/Dialect/Vernon/Transforms/VernonAggregateStorage.h"
#include "mlir/Dialect/Vernon/Transforms/VernonAutodiffAnalysis.h"
#include "mlir/Dialect/Vernon/Transforms/VernonAutodiffRules.h"
#include "mlir/Dialect/Vernon/Transforms/VernonAutodiffTapePlanning.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringSet.h"

#include <limits>

namespace mlir::vernon {
namespace {

constexpr StringLiteral kEntryAttr = "vernon.entry";
constexpr StringLiteral kStageAttr = "vernon.stage";

Value createTuple(OpBuilder &builder, Location location, TupleType type, ValueRange values) {
    OperationState state(location, TupleCreateOp::getOperationName());
    state.addOperands(values);
    state.addTypes(type);
    return builder.create(state)->getResult(0);
}

Value createZero(OpBuilder &builder, Location location, Type type) {
    if (auto floatType = dyn_cast<FloatType>(type))
        return arith::ConstantOp::create(builder, location, builder.getFloatAttr(floatType, 0.0));
    if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
        auto elementType = dyn_cast<FloatType>(tensorType.getElementType());
        if (!elementType || !tensorType.hasStaticShape())
            return {};
        return arith::ConstantOp::create(builder, location,
                                         DenseElementsAttr::get(tensorType, builder.getFloatAttr(elementType, 0.0)));
    }
    return {};
}

Operation *createOperation(OpBuilder &builder, Location location, StringRef name, ValueRange operands = {},
                           TypeRange results = {}, ArrayRef<NamedAttribute> attributes = {}) {
    OperationState state(location, name);
    state.addOperands(operands);
    state.addTypes(results);
    state.addAttributes(attributes);
    return builder.create(state);
}

Value createIndexConstant(OpBuilder &builder, Location location, int64_t value) {
    return arith::ConstantIndexOp::create(builder, location, value);
}

Value createI32Constant(OpBuilder &builder, Location location, int32_t value) {
    return arith::ConstantIntOp::create(builder, location, value, 32);
}

bool isValueDefinedIn(Value value, Region &region) {
    Region *owner = value.getParentRegion();
    return owner && (owner == &region || region.isAncestor(owner));
}

bool isValueDefinedDirectlyIn(Value value, Region &region) { return value.getParentRegion() == &region; }

void copyProfileFunctionAttrs(func::FuncOp source, func::FuncOp target) {
    for (NamedAttribute attribute : source->getAttrs()) {
        StringRef name = attribute.getName().strref();
        if (name == SymbolTable::getSymbolAttrName() || name == "function_type" ||
            name == source.getArgAttrsAttrName() || name == source.getResAttrsAttrName())
            continue;
        target->setAttr(attribute.getName(), attribute.getValue());
    }
    target->setAttr(kEntryAttr, UnitAttr::get(target.getContext()));
    target->setAttr(kStageAttr, StringAttr::get(target.getContext(), "compute"));
}

DictionaryAttr makeInterfaceAttrs(MLIRContext *context, StringRef interfaceName, StringRef sourceName,
                                  ArrayRef<StringRef> dtypes, int64_t location, Type valueType = {}) {
    NamedAttrList attributes;
    attributes.set("vernon.interface", StringAttr::get(context, interfaceName));
    attributes.set("vernon.source_name", StringAttr::get(context, sourceName));
    SmallVector<Attribute> dtypeAttrs;
    for (StringRef dtype : dtypes)
        dtypeAttrs.push_back(StringAttr::get(context, dtype));
    attributes.set("vernon.abi_leaf_dtypes", ArrayAttr::get(context, dtypeAttrs));
    if (isa_and_nonnull<RankedTensorType, TensorViewType>(valueType))
        attributes.set("vernon.element_abi_leaf_dtypes", ArrayAttr::get(context, dtypeAttrs));
    attributes.set("vernon.location", IntegerAttr::get(IntegerType::get(context, 64), location));
    return attributes.getDictionary(context);
}

StringRef scalarDtype(Type type) {
    if (type.isF16())
        return "f16";
    if (type.isF32())
        return "f32";
    if (type.isF64())
        return "f64";
    return {};
}

struct BackwardProfileTypes {
    SmallVector<Type> cotangents;
    SmallVector<Type> gradients;
    Type result;
    SmallVector<StringRef> cotangentDtypes;
    SmallVector<StringRef> gradientDtypes;
};

BackwardProfileTypes getBackwardProfileTypes(MLIRContext *context, const VernonAutodiffAnalysisResult &analysis) {
    BackwardProfileTypes types;
    for (const AutodiffLeaf &leaf : analysis.getActiveResultLeaves()) {
        types.cotangents.push_back(leaf.derivativeType);
        types.cotangentDtypes.push_back(scalarDtype(leaf.primalType));
        if (leaf.primalType.isF16())
            types.cotangentDtypes.back() = "f32";
    }
    for (const AutodiffLeaf &leaf : analysis.getWrtLeaves()) {
        types.gradients.push_back(leaf.derivativeType);
        types.gradientDtypes.push_back(leaf.primalType.isF64() ? "f64" : "f32");
    }
    types.result = types.gradients.size() == 1 ? types.gradients.front()
                                               : static_cast<Type>(TupleType::get(context, types.gradients));
    return types;
}

using AdjointKey = std::pair<Value, unsigned>;
using AdjointMap = DenseMap<AdjointKey, Value>;

void accumulateAdjoint(OpBuilder &builder, Location location, const VernonAutodiffAnalysisResult &analysis,
                       AdjointMap &adjoints, Value value, unsigned abiLeafIndex, Value contribution) {
    if (!contribution || !analysis.isActive(value, abiLeafIndex))
        return;
    AdjointKey key{value, abiLeafIndex};
    auto found = adjoints.find(key);
    if (found == adjoints.end())
        adjoints.try_emplace(key, contribution);
    else
        found->second = arith::AddFOp::create(builder, location, found->second, contribution);
}

LogicalResult reverseScalarOperation(Operation &operation, OpBuilder &builder,
                                     const VernonAutodiffAnalysisResult &analysis,
                                     const VernonAutodiffRuleRegistry &registry, const DenseMap<Value, Value> &primals,
                                     AdjointMap &adjoints) {
    if (!analysis.isActive(&operation))
        return success();
    if (operation.getNumResults() != 1) {
        if (llvm::any_of(operation.getResults(), [&](Value result) { return adjoints.contains({result, 0}); }))
            return operation.emitError("active structured VJP operation must have exactly one result");
        return success();
    }
    auto seed = adjoints.find({operation.getResult(0), 0});
    if (seed == adjoints.end())
        return success();
    const DifferentiationRule *rule = registry.lookup(&operation);
    if (!rule)
        return operation.emitError("active scalar operation has no VJP rule");
    SmallVector<Value> operands;
    for (Value operand : operation.getOperands())
        operands.push_back(primals.lookup(operand));
    SmallVector<Value> results = {primals.lookup(operation.getResult(0))};
    FailureOr<SmallVector<Value>> contributions = rule->buildVjp(
        &operation, AutodiffVjpBuildContext{builder, operation.getLoc(), operands, results, ValueRange(seed->second)});
    if (failed(contributions))
        return failure();
    for (auto [operand, contribution] : llvm::zip_equal(operation.getOperands(), *contributions))
        accumulateAdjoint(builder, operation.getLoc(), analysis, adjoints, operand, 0, contribution);
    return success();
}

LogicalResult validateStructuredPhase(func::FuncOp primal, const VernonAutodiffAnalysisResult &analysis) {
    if (!llvm::hasSingleElement(primal.getBody()))
        return primal.emitError("structured VJP requires one structured entry block");
    if (primal.getNumResults() != 1 || analysis.getActiveResultLeaves().empty() || analysis.getWrtLeaves().empty())
        return primal.emitError("structured VJP requires one differentiable result and at least one wrt leaf");
    WalkResult structured = primal.walk([&](Operation *operation) {
        if (operation->getNumRegions() != 0 && !isa<func::FuncOp, scf::IfOp, scf::WhileOp>(operation)) {
            operation->emitError("structured VJP supports only scf.if and scf.while regions");
            return WalkResult::interrupt();
        }
        return WalkResult::advance();
    });
    return structured.wasInterrupted() ? failure() : success();
}

struct DynamicRootLayout {
    SmallVector<uint64_t> invocationLeafOffsets;
    SmallVector<uint64_t> outputLeafOffsets;
    uint64_t stride{};
    uint64_t alignment{1};
};

FailureOr<DynamicRootLayout> buildDynamicRootLayout(const VernonAutodiffTapePlan &plan,
                                                    const ValueAbiLayout &outputLayout) {
    SmallVector<AutodiffTapeSlot> slots;
    for (const AutodiffTapeLeaf &leaf : plan.getInvocationRecord().leaves)
        slots.push_back({leaf.size, leaf.alignment});
    for (const ValueAbiLeaf &leaf : outputLayout.leaves) {
        const uint64_t scalarSize = std::max<uint64_t>(leaf.scalarType.getIntOrFloatBitWidth() / 8, 1);
        if (leaf.scalarCount == 0 || leaf.scalarCount > std::numeric_limits<uint64_t>::max() / scalarSize)
            return failure();
        slots.push_back({leaf.scalarCount * scalarSize, scalarSize});
    }
    FailureOr<AutodiffTapeLayout> layout = planAutodiffTapeLayout(slots);
    if (failed(layout) || layout->stride > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
        return failure();
    DynamicRootLayout result;
    const size_t invocationLeaves = plan.getInvocationRecord().leaves.size();
    result.invocationLeafOffsets.assign(layout->offsets.begin(), layout->offsets.begin() + invocationLeaves);
    result.outputLeafOffsets.assign(layout->offsets.begin() + invocationLeaves, layout->offsets.end());
    result.stride = layout->stride;
    result.alignment = layout->alignment;
    return result;
}

const AutodiffTapeField *findField(const AutodiffTapeHeaderSchema &schema, AutodiffTapeFieldKind kind) {
    auto found = llvm::find_if(schema.fields, [&](const AutodiffTapeField &field) { return field.kind == kind; });
    return found == schema.fields.end() ? nullptr : &*found;
}

Value beginRegion(OpBuilder &builder, Location location, Value tape, Value parentRegion = {}, Value parentRecord = {},
                  std::optional<unsigned> childOrdinal = std::nullopt) {
    SmallVector<Value> operands = {tape};
    SmallVector<NamedAttribute> attributes;
    if (parentRegion) {
        operands.push_back(parentRegion);
        operands.push_back(parentRecord);
        attributes.push_back(
            builder.getNamedAttr("child_ordinal", builder.getI64IntegerAttr(static_cast<int64_t>(*childOrdinal))));
    }
    return createOperation(builder, location, AdBeginRegionOp::getOperationName(), operands,
                           AdRegionHeaderType::get(builder.getContext()), attributes)
        ->getResult(0);
}

Value reserveRecord(OpBuilder &builder, Location location, Value region, uint64_t stride, uint64_t alignment) {
    SmallVector<NamedAttribute> attributes = {
        builder.getNamedAttr("record_size", builder.getI64IntegerAttr(static_cast<int64_t>(stride))),
        builder.getNamedAttr("record_alignment", builder.getI64IntegerAttr(static_cast<int64_t>(alignment))),
    };
    return createOperation(builder, location, AdReserveRecordOp::getOperationName(), region, builder.getIndexType(),
                           attributes)
        ->getResult(0);
}

void writeLeaf(OpBuilder &builder, Location location, Value region, Value record, Value value, uint64_t offset) {
    createOperation(builder, location, AdWriteLeafOp::getOperationName(), {region, record, value}, {},
                    builder.getNamedAttr("leaf_offset", builder.getI64IntegerAttr(static_cast<int64_t>(offset))));
}

Value readLeaf(OpBuilder &builder, Location location, Value region, Value recordIndex, Type type, uint64_t recordSize,
               uint64_t recordAlignment, uint64_t offset) {
    SmallVector<NamedAttribute> attributes = {
        builder.getNamedAttr("record_size", builder.getI64IntegerAttr(static_cast<int64_t>(recordSize))),
        builder.getNamedAttr("record_alignment", builder.getI64IntegerAttr(static_cast<int64_t>(recordAlignment))),
        builder.getNamedAttr("leaf_offset", builder.getI64IntegerAttr(static_cast<int64_t>(offset))),
    };
    return createOperation(builder, location, AdReadLeafOp::getOperationName(), {region, recordIndex}, type, attributes)
        ->getResult(0);
}

LogicalResult writeCanonicalTapeValue(OpBuilder &builder, Location location, ModuleOp module, Value region,
                                      Value record, Value value, Value sourceIdentity,
                                      ArrayRef<AutodiffTapeLeaf> tapeLeaves, ArrayRef<uint64_t> overrideOffsets = {}) {
    FailureOr<ValueAbiLayout> layout = getValueAbiLayout(value.getType(), module);
    FailureOr<SmallVector<Value>> scalars =
        decomposeAggregateValueToScalars(value.getType(), value, module, builder, location);
    if (failed(layout) || failed(scalars))
        return failure();
    uint64_t scalarCursor = 0;
    for (auto [leafIndex, abiLeaf] : llvm::enumerate(layout->leaves)) {
        const AutodiffTapeLeaf *tapeLeaf = nullptr;
        size_t tapeLeafIndex = 0;
        for (auto [candidateIndex, candidate] : llvm::enumerate(tapeLeaves)) {
            if (candidate.value == sourceIdentity && candidate.abiLeafIndex == leafIndex) {
                tapeLeaf = &candidate;
                tapeLeafIndex = candidateIndex;
                break;
            }
        }
        if (!tapeLeaf) {
            scalarCursor += abiLeaf.scalarCount;
            continue;
        }
        const uint64_t scalarSize = std::max<uint64_t>(abiLeaf.scalarType.getIntOrFloatBitWidth() / 8, 1);
        const uint64_t baseOffset = overrideOffsets.empty() ? tapeLeaf->offset : overrideOffsets[tapeLeafIndex];
        for (uint64_t lane = 0; lane < abiLeaf.scalarCount; ++lane) {
            if (scalarCursor + lane >= scalars->size())
                return failure();
            writeLeaf(builder, location, region, record, (*scalars)[scalarCursor + lane],
                      baseOffset + lane * scalarSize);
        }
        scalarCursor += abiLeaf.scalarCount;
    }
    return success(scalarCursor == scalars->size());
}

FailureOr<Value> readCanonicalTapeValue(OpBuilder &builder, Location location, ModuleOp module, Value region,
                                        Value recordIndex, Value source, ArrayRef<AutodiffTapeLeaf> tapeLeaves,
                                        uint64_t recordSize, uint64_t recordAlignment,
                                        ArrayRef<uint64_t> overrideOffsets = {}) {
    FailureOr<ValueAbiLayout> layout = getValueAbiLayout(source.getType(), module);
    if (failed(layout))
        return failure();
    SmallVector<Value> scalars;
    for (auto [leafIndex, abiLeaf] : llvm::enumerate(layout->leaves)) {
        const AutodiffTapeLeaf *tapeLeaf = nullptr;
        size_t tapeLeafIndex = 0;
        for (auto [candidateIndex, candidate] : llvm::enumerate(tapeLeaves)) {
            if (candidate.value == source && candidate.abiLeafIndex == leafIndex) {
                tapeLeaf = &candidate;
                tapeLeafIndex = candidateIndex;
                break;
            }
        }
        if (!tapeLeaf)
            return failure();
        const uint64_t scalarSize = std::max<uint64_t>(abiLeaf.scalarType.getIntOrFloatBitWidth() / 8, 1);
        const uint64_t baseOffset = overrideOffsets.empty() ? tapeLeaf->offset : overrideOffsets[tapeLeafIndex];
        for (uint64_t lane = 0; lane < abiLeaf.scalarCount; ++lane)
            scalars.push_back(readLeaf(builder, location, region, recordIndex, abiLeaf.scalarType, recordSize,
                                       recordAlignment, baseOffset + lane * scalarSize));
    }
    return buildAggregateValueFromScalars(source.getType(), scalars, module, builder, location);
}

class DynamicForwardEmitter {
public:
    DynamicForwardEmitter(func::FuncOp primal, const VernonAutodiffTapePlan &plan, Value tape, Value rootRegion,
                          Value rootRecord)
        : primal(primal), tape(tape), rootRegion(rootRegion), rootRecord(rootRecord) {
        for (const AutodiffTapeRegion &region : plan.getRegions()) {
            regions.try_emplace(region.operation, &region);
            regionsByOrdinal.try_emplace(region.ordinal, &region);
        }
    }

    IRMapping &getMapping() { return mapping; }

    LogicalResult emitTopLevel(OpBuilder &builder) {
        for (Operation &operation : primal.getBody().front().without_terminator())
            if (failed(emitOperation(operation, builder, rootRegion, rootRecord)))
                return failure();
        return success();
    }

private:
    LogicalResult emitOperation(Operation &operation, OpBuilder &builder, Value parentRegion, Value parentRecord) {
        if (planRegion(operation)) {
            if (auto ifOp = dyn_cast<scf::IfOp>(operation))
                return emitIf(ifOp, builder, parentRegion, parentRecord);
            if (auto whileOp = dyn_cast<scf::WhileOp>(operation))
                return emitWhile(whileOp, builder, parentRegion, parentRecord);
        }
        Operation *clone = builder.clone(operation, mapping);
        if (isa<StoreOp>(operation))
            clone->setAttr("vernon.ad.functionalized", UnitAttr::get(builder.getContext()));
        for (auto [source, target] : llvm::zip_equal(operation.getResults(), clone->getResults()))
            mapping.map(source, target);
        return success();
    }

    LogicalResult emitBlock(Block &source, Block &target, Value parentRegion, Value parentRecord) {
        OpBuilder builder(&target, target.end());
        if (!target.empty() && target.back().hasTrait<OpTrait::IsTerminator>())
            builder.setInsertionPoint(&target.back());
        for (Operation &operation : source.without_terminator())
            if (failed(emitOperation(operation, builder, parentRegion, parentRecord)))
                return failure();
        return success();
    }

    LogicalResult writePlannedLeaves(OpBuilder &builder, Location location, const AutodiffTapeRegion &region,
                                     Value handle, Value record, Region *sourceRegion = nullptr) {
        DenseSet<Value> written;
        for (const AutodiffTapeLeaf &leaf : region.record.leaves) {
            if (sourceRegion && !isValueDefinedDirectlyIn(leaf.value, *sourceRegion))
                continue;
            if (!written.insert(leaf.value).second)
                continue;
            Value mapped = mapping.lookupOrNull(leaf.value);
            if (!mapped) {
                Value sourceValue = leaf.value;
                return sourceValue.getParentRegion()->getParentOp()->emitError(
                    "dynamic tape value is unavailable in its owning record");
            }
            if (failed(writeCanonicalTapeValue(builder, location, primal->getParentOfType<ModuleOp>(), handle, record,
                                               mapped, leaf.value, region.record.leaves)))
                return Value(leaf.value)
                    .getParentRegion()
                    ->getParentOp()
                    ->emitError("cannot project a dynamic tape value through its canonical ABI");
        }
        return success();
    }

    LogicalResult emitIf(scf::IfOp source, OpBuilder &builder, Value parentRegion, Value parentRecord) {
        const AutodiffTapeRegion *region = regions.lookup(source.getOperation());
        if (!region)
            return source.emitError("active scf.if has no dynamic tape plan");
        Value handle = beginRegion(builder, source.getLoc(), tape, parentRegion, parentRecord, region->childOrdinal);
        Value record = reserveRecord(builder, source.getLoc(), handle, region->record.stride, region->record.alignment);
        const AutodiffTapeField *predicate = findField(region->record.prefix, AutodiffTapeFieldKind::Predicate);
        if (!predicate)
            return source.emitError("scf.if tape record has no predicate field");
        writeLeaf(builder, source.getLoc(), handle, record, mapping.lookup(source.getCondition()), predicate->offset);

        const bool needsSyntheticElse = source.getElseRegion().empty() && !region->childRegionOrdinals.empty();
        scf::IfOp target =
            scf::IfOp::create(builder, source.getLoc(), source.getResultTypes(), mapping.lookup(source.getCondition()),
                              !source.getElseRegion().empty() || needsSyntheticElse);
        for (auto [sourceRegion, targetRegion] : llvm::zip_equal(source->getRegions(), target->getRegions())) {
            if (sourceRegion.empty())
                continue;
            Block &targetBlock = targetRegion.front();
            if (failed(emitBlock(sourceRegion.front(), targetBlock, handle, record)))
                return failure();
            auto sourceYield = cast<scf::YieldOp>(sourceRegion.front().getTerminator());
            OpBuilder branchBuilder = OpBuilder::atBlockEnd(&targetBlock);
            if (!targetBlock.empty() && targetBlock.back().hasTrait<OpTrait::IsTerminator>())
                branchBuilder.setInsertionPoint(&targetBlock.back());
            if (failed(writePlannedLeaves(branchBuilder, source.getLoc(), *region, handle, record, &sourceRegion)))
                return failure();
            for (unsigned childOrdinal : region->childRegionOrdinals) {
                const AutodiffTapeRegion *child = regionsByOrdinal.lookup(childOrdinal);
                if (!child)
                    return source.emitError("active scf.if references an unknown child tape region");
                Region *owner = child->operation->getParentRegion();
                if (owner == &sourceRegion || sourceRegion.isAncestor(owner))
                    continue;
                Value empty = beginRegion(branchBuilder, source.getLoc(), tape, handle, record, child->childOrdinal);
                createOperation(branchBuilder, source.getLoc(), AdEndRegionOp::getOperationName(),
                                {empty, createIndexConstant(branchBuilder, source.getLoc(), 0),
                                 createI32Constant(branchBuilder, source.getLoc(), 0)});
            }
            SmallVector<Value> yielded =
                llvm::map_to_vector(sourceYield.getOperands(), [&](Value value) { return mapping.lookup(value); });
            if (!targetBlock.empty() && isa<scf::YieldOp>(targetBlock.back()))
                cast<scf::YieldOp>(targetBlock.back())->setOperands(yielded);
            else {
                OpBuilder yieldBuilder = OpBuilder::atBlockEnd(&targetBlock);
                scf::YieldOp::create(yieldBuilder, sourceYield.getLoc(), yielded);
            }
        }
        if (needsSyntheticElse) {
            Block &elseBlock = target.getElseRegion().front();
            OpBuilder elseBuilder = OpBuilder::atBlockBegin(&elseBlock);
            for (unsigned childOrdinal : region->childRegionOrdinals) {
                const AutodiffTapeRegion *child = regionsByOrdinal.lookup(childOrdinal);
                if (!child)
                    return source.emitError("active scf.if references an unknown child tape region");
                Value empty = beginRegion(elseBuilder, source.getLoc(), tape, handle, record, child->childOrdinal);
                createOperation(elseBuilder, source.getLoc(), AdEndRegionOp::getOperationName(),
                                {empty, createIndexConstant(elseBuilder, source.getLoc(), 0),
                                 createI32Constant(elseBuilder, source.getLoc(), 0)});
            }
            if (elseBlock.empty() || !isa<scf::YieldOp>(elseBlock.back()))
                scf::YieldOp::create(elseBuilder, source.getLoc());
        }
        for (auto [sourceResult, targetResult] : llvm::zip_equal(source.getResults(), target.getResults()))
            mapping.map(sourceResult, targetResult);
        builder.setInsertionPointAfter(target);
        createOperation(
            builder, source.getLoc(), AdEndRegionOp::getOperationName(),
            {handle, createIndexConstant(builder, source.getLoc(), 1), createI32Constant(builder, source.getLoc(), 0)});
        return success();
    }

    LogicalResult emitWhile(scf::WhileOp source, OpBuilder &builder, Value parentRegion, Value parentRecord) {
        const AutodiffTapeRegion *region = regions.lookup(source.getOperation());
        if (!region)
            return source.emitError("active scf.while has no dynamic tape plan");
        auto condition = dyn_cast<scf::ConditionOp>(source.getBefore().front().getTerminator());
        auto yield = dyn_cast<scf::YieldOp>(source.getAfter().front().getTerminator());
        if (!condition || !yield || condition.getArgs().size() != source.getBeforeArguments().size() ||
            yield.getResults().size() != source.getAfterArguments().size())
            return source.emitError("structured VJP requires canonical scf.while regions");
        for (auto [forwarded, argument] : llvm::zip_equal(condition.getArgs(), source.getBeforeArguments()))
            if (forwarded != argument)
                return source.emitError("structured VJP requires scf.while conditions to forward carried values");

        Value handle = beginRegion(builder, source.getLoc(), tape, parentRegion, parentRecord, region->childOrdinal);
        SmallVector<Value> inits;
        for (Value init : source.getInits())
            inits.push_back(mapping.lookup(init));
        inits.push_back(createIndexConstant(builder, source.getLoc(), 0));
        SmallVector<Type> resultTypes(source.getResultTypes());
        resultTypes.push_back(builder.getIndexType());
        OperationState state(source.getLoc(), scf::WhileOp::getOperationName());
        state.addOperands(inits);
        state.addTypes(resultTypes);
        state.addRegion();
        state.addRegion();
        auto target = cast<scf::WhileOp>(builder.create(state));

        Block *before = new Block();
        target.getBefore().push_back(before);
        for (BlockArgument argument : source.getBeforeArguments())
            before->addArgument(argument.getType(), source.getLoc());
        BlockArgument beforeCount = before->addArgument(builder.getIndexType(), source.getLoc());
        for (auto [sourceArgument, targetArgument] :
             llvm::zip_equal(source.getBeforeArguments(), before->getArguments().drop_back()))
            mapping.map(sourceArgument, targetArgument);
        OpBuilder beforeBuilder(before, before->end());
        for (Operation &operation : source.getBefore().front().without_terminator()) {
            if (isa<scf::IfOp, scf::WhileOp>(operation) && planRegion(operation))
                return operation.emitError("active nested control flow in an scf.while condition is unsupported");
            Operation *clone = beforeBuilder.clone(operation, mapping);
            for (auto [sourceResult, targetResult] : llvm::zip_equal(operation.getResults(), clone->getResults()))
                mapping.map(sourceResult, targetResult);
        }
        SmallVector<Value> forwarded;
        for (Value value : condition.getArgs())
            forwarded.push_back(mapping.lookup(value));
        forwarded.push_back(beforeCount);
        scf::ConditionOp::create(beforeBuilder, condition.getLoc(), mapping.lookup(condition.getCondition()),
                                 forwarded);

        Block *after = new Block();
        target.getAfter().push_back(after);
        for (BlockArgument argument : source.getAfterArguments())
            after->addArgument(argument.getType(), source.getLoc());
        BlockArgument afterCount = after->addArgument(builder.getIndexType(), source.getLoc());
        for (auto [sourceArgument, targetArgument] :
             llvm::zip_equal(source.getAfterArguments(), after->getArguments().drop_back()))
            mapping.map(sourceArgument, targetArgument);
        OpBuilder afterBuilder(after, after->end());
        Value record =
            reserveRecord(afterBuilder, source.getLoc(), handle, region->record.stride, region->record.alignment);
        if (failed(emitBlock(source.getAfter().front(), *after, handle, record)))
            return failure();
        afterBuilder.setInsertionPointToEnd(after);
        if (failed(writePlannedLeaves(afterBuilder, source.getLoc(), *region, handle, record)))
            return failure();
        Value nextCount = createOperation(afterBuilder, source.getLoc(), AdCheckedIncrementOp::getOperationName(),
                                          afterCount, afterBuilder.getIndexType())
                              ->getResult(0);
        SmallVector<Value> yielded;
        for (Value value : yield.getResults())
            yielded.push_back(mapping.lookup(value));
        yielded.push_back(nextCount);
        scf::YieldOp::create(afterBuilder, yield.getLoc(), yielded);

        for (auto [sourceResult, targetResult] : llvm::zip_equal(source.getResults(), target.getResults().drop_back()))
            mapping.map(sourceResult, targetResult);
        builder.setInsertionPointAfter(target);
        FailureOr<Value> exitKind = buildExitKind(source, target, builder);
        if (failed(exitKind))
            return failure();
        createOperation(builder, source.getLoc(), AdEndRegionOp::getOperationName(),
                        {handle, target.getResults().back(), *exitKind});
        return success();
    }

    const AutodiffTapeRegion *planRegion(Operation &operation) const { return regions.lookup(&operation); }

    FailureOr<Value> buildExitKind(scf::WhileOp source, scf::WhileOp target, OpBuilder &builder) {
        auto getStateIndex = [&](StringRef name, Type expectedType) -> FailureOr<std::optional<unsigned>> {
            Attribute attribute = source->getAttr(name);
            if (!attribute)
                return std::optional<unsigned>{};
            auto index = dyn_cast<IntegerAttr>(attribute);
            if (!index || index.getValue().isNegative() || index.getValue().getActiveBits() > 64)
                return source.emitError() << "'" << name << "' must be a valid non-negative loop-result index";
            uint64_t rawIndex = index.getValue().getZExtValue();
            if (rawIndex >= source.getNumResults())
                return source.emitError() << "'" << name << "' must be a valid non-negative loop-result index";
            unsigned value = static_cast<unsigned>(rawIndex);
            if (target.getResult(value).getType() != expectedType)
                return source.emitError() << "'" << name << "' refers to " << target.getResult(value).getType()
                                          << ", expected " << expectedType;
            return std::optional<unsigned>{value};
        };
        FailureOr<std::optional<unsigned>> breakIndex =
            getStateIndex("vernon.loop_control_index", builder.getI32Type());
        FailureOr<std::optional<unsigned>> returnIndex = getStateIndex("vernon.return_flag_index", builder.getI1Type());
        if (failed(breakIndex) || failed(returnIndex))
            return failure();
        if (*breakIndex && *returnIndex && **breakIndex == **returnIndex)
            return source.emitError("loop control and return flag must use distinct results");

        Value result = createI32Constant(builder, source.getLoc(), 0);
        if (*breakIndex) {
            Value one = createI32Constant(builder, source.getLoc(), 1);
            Value isBreak = arith::CmpIOp::create(builder, source.getLoc(), arith::CmpIPredicate::eq,
                                                  target.getResult(**breakIndex), one);
            result = arith::SelectOp::create(builder, source.getLoc(), isBreak, one, result);
        }
        if (*returnIndex) {
            Value three = createI32Constant(builder, source.getLoc(), 3);
            result = arith::SelectOp::create(builder, source.getLoc(), target.getResult(**returnIndex), three, result);
        }
        return result;
    }

    func::FuncOp primal;
    Value tape;
    Value rootRegion;
    Value rootRecord;
    IRMapping mapping;
    DenseMap<Operation *, const AutodiffTapeRegion *> regions;
    DenseMap<unsigned, const AutodiffTapeRegion *> regionsByOrdinal;
};

class DynamicReverseEmitter {
public:
    DynamicReverseEmitter(func::FuncOp primal, const VernonAutodiffAnalysisResult &analysis,
                          const VernonAutodiffRuleRegistry &registry, const VernonAutodiffTapePlan &plan,
                          const DynamicRootLayout &rootLayout, Value rootRegion)
        : primal(primal), analysis(analysis), registry(registry), plan(plan), rootLayout(rootLayout),
          rootRegion(rootRegion) {
        for (const AutodiffTapeRegion &region : plan.getRegions())
            regions.try_emplace(region.operation, &region);
    }

    LogicalResult initializePrimals(OpBuilder &builder) {
        Value zero = createIndexConstant(builder, primal.getLoc(), 0);
        DenseSet<Value> loaded;
        for (const AutodiffTapeLeaf &leaf : plan.getInvocationRecord().leaves) {
            if (!loaded.insert(leaf.value).second)
                continue;
            FailureOr<Value> value =
                readCanonicalTapeValue(builder, primal.getLoc(), primal->getParentOfType<ModuleOp>(), rootRegion, zero,
                                       leaf.value, plan.getInvocationRecord().leaves, rootLayout.stride,
                                       rootLayout.alignment, rootLayout.invocationLeafOffsets);
            if (failed(value))
                return Value(leaf.value)
                    .getParentRegion()
                    ->getParentOp()
                    ->emitError("cannot reconstruct an invocation tape value from canonical ABI leaves");
            primals.try_emplace(leaf.value, *value);
        }
        return success();
    }

    LogicalResult reverseTopLevel(OpBuilder &builder, AdjointMap &adjoints) {
        return reverseBlock(primal.getBody().front(), builder, adjoints, rootRegion,
                            createIndexConstant(builder, primal.getLoc(), 0));
    }

private:
    void accumulate(OpBuilder &builder, Location location, AdjointMap &adjoints, Value value, unsigned leafIndex,
                    Value contribution) {
        accumulateAdjoint(builder, location, analysis, adjoints, value, leafIndex, contribution);
    }

    FailureOr<Type> derivativeType(Value value, unsigned leafIndex) {
        const ValueAbiLayout *layout = analysis.getValueAbi(value);
        if (!layout || leafIndex >= layout->leaves.size())
            return failure();
        const ValueAbiLeaf &leaf = layout->leaves[leafIndex];
        FailureOr<Type> element = getAutodiffDerivativeType(leaf.scalarType);
        if (failed(element))
            return failure();
        if (leaf.shape.empty())
            return *element;
        return RankedTensorType::get(SmallVector<int64_t>(leaf.shape.begin(), leaf.shape.end()), *element);
    }

    Value zeroFor(OpBuilder &builder, Location location, Value value, unsigned leafIndex = 0) {
        FailureOr<Type> derivative = derivativeType(value, leafIndex);
        return succeeded(derivative) ? createZero(builder, location, *derivative) : Value{};
    }

    LogicalResult loadRegionPrimals(OpBuilder &builder, const AutodiffTapeRegion &region, Value handle,
                                    Value recordIndex, SmallVectorImpl<std::pair<Value, Value>> &saved,
                                    Region *sourceRegion = nullptr) {
        DenseSet<Value> loaded;
        for (const AutodiffTapeLeaf &leaf : region.record.leaves) {
            if (sourceRegion && !isValueDefinedDirectlyIn(leaf.value, *sourceRegion))
                continue;
            if (!loaded.insert(leaf.value).second)
                continue;
            saved.emplace_back(leaf.value, primals.lookup(leaf.value));
            FailureOr<Value> value = readCanonicalTapeValue(
                builder, region.operation->getLoc(), primal->getParentOfType<ModuleOp>(), handle, recordIndex,
                leaf.value, region.record.leaves, region.record.stride, region.record.alignment);
            if (failed(value))
                return Value(leaf.value)
                    .getParentRegion()
                    ->getParentOp()
                    ->emitError("cannot reconstruct a dynamic tape value from canonical ABI leaves");
            primals[leaf.value] = *value;
        }
        return success();
    }

    void restorePrimals(ArrayRef<std::pair<Value, Value>> saved) {
        for (auto [value, previous] : saved) {
            if (previous)
                primals[value] = previous;
            else
                primals.erase(value);
        }
    }

    SmallVector<AdjointKey> externalActiveLeaves(Region &region) {
        SmallVector<AdjointKey> values;
        DenseSet<AdjointKey> seen;
        region.walk([&](Operation *operation) {
            for (Value operand : operation->getOperands()) {
                if (isValueDefinedIn(operand, region))
                    continue;
                const ValueAbiLayout *layout = analysis.getValueAbi(operand);
                if (!layout)
                    continue;
                for (unsigned leafIndex = 0; leafIndex < layout->leaves.size(); ++leafIndex) {
                    AdjointKey key{operand, leafIndex};
                    if (analysis.isActive(operand, leafIndex) && seen.insert(key).second)
                        values.push_back(key);
                }
            }
        });
        return values;
    }

    FailureOr<Value> materializePrimal(Value value, OpBuilder &builder) {
        if (Value available = primals.lookup(value))
            return available;
        Operation *defining = value.getDefiningOp();
        if (!defining || defining->getNumRegions() != 0 || classifyAutodiffEffect(defining) != AutodiffEffectKind::Pure)
            return failure();
        IRMapping local;
        for (Value operand : defining->getOperands()) {
            FailureOr<Value> materialized = materializePrimal(operand, builder);
            if (failed(materialized))
                return failure();
            local.map(operand, *materialized);
        }
        Operation *clone = builder.clone(*defining, local);
        for (auto [source, target] : llvm::zip_equal(defining->getResults(), clone->getResults()))
            primals.try_emplace(source, target);
        return primals.lookup(value);
    }

    FailureOr<Value> replaceTensorElement(Value tensorValue, ValueRange indices, Value replacement, OpBuilder &builder,
                                          Location location) {
        auto type = dyn_cast<RankedTensorType>(tensorValue.getType());
        if (!type || !type.hasStaticShape() || indices.size() != static_cast<size_t>(type.getRank()) ||
            replacement.getType() != type.getElementType())
            return failure();
        SmallVector<Value> elements;
        SmallVector<SmallVector<int64_t>> coordinates(1);
        for (int64_t extent : type.getShape()) {
            SmallVector<SmallVector<int64_t>> expanded;
            for (const SmallVector<int64_t> &prefix : coordinates)
                for (int64_t index = 0; index < extent; ++index) {
                    SmallVector<int64_t> next(prefix);
                    next.push_back(index);
                    expanded.push_back(std::move(next));
                }
            coordinates = std::move(expanded);
        }
        for (const SmallVector<int64_t> &coordinate : coordinates) {
            SmallVector<Value> constants;
            for (int64_t index : coordinate)
                constants.push_back(createIndexConstant(builder, location, index));
            Value original = tensor::ExtractOp::create(builder, location, tensorValue, constants);
            Value matches = arith::ConstantIntOp::create(builder, location, 1, 1);
            for (auto [actual, expected] : llvm::zip_equal(indices, constants)) {
                Value equal = arith::CmpIOp::create(builder, location, arith::CmpIPredicate::eq, actual, expected);
                matches = arith::AndIOp::create(builder, location, matches, equal);
            }
            elements.push_back(arith::SelectOp::create(builder, location, matches, replacement, original));
        }
        return tensor::FromElementsOp::create(builder, location, type, elements).getResult();
    }

    LogicalResult reverseLoad(LoadOp load, OpBuilder &builder, AdjointMap &adjoints) {
        Value seed = adjoints.lookup({load.getResult(), 0});
        if (!seed)
            return success();
        SmallVector<Value> indices;
        for (Value index : load.getIndices()) {
            FailureOr<Value> materialized = materializePrimal(index, builder);
            if (failed(materialized))
                return load.emitError("cannot materialize a dynamic TensorView load index in reverse");
            indices.push_back(*materialized);
        }
        Value zero = zeroFor(builder, load.getLoc(), load.getStorage());
        FailureOr<Value> contribution = replaceTensorElement(zero, indices, seed, builder, load.getLoc());
        if (failed(contribution))
            return load.emitError("cannot construct a TensorView load scatter contribution");
        accumulate(builder, load.getLoc(), adjoints, load.getStorage(), 0, *contribution);
        return success();
    }

    LogicalResult reverseTensorExtract(tensor::ExtractOp extract, OpBuilder &builder, AdjointMap &adjoints) {
        Value seed = adjoints.lookup({extract.getResult(), 0});
        if (!seed)
            return success();
        SmallVector<Value> indices;
        for (Value index : extract.getIndices()) {
            FailureOr<Value> materialized = materializePrimal(index, builder);
            if (failed(materialized))
                return extract.emitError("cannot materialize a dynamic Tensor index in reverse");
            indices.push_back(*materialized);
        }
        Value zero = zeroFor(builder, extract.getLoc(), extract.getTensor());
        FailureOr<Value> contribution = replaceTensorElement(zero, indices, seed, builder, extract.getLoc());
        if (failed(contribution))
            return extract.emitError("cannot construct a Tensor index scatter contribution");
        accumulate(builder, extract.getLoc(), adjoints, extract.getTensor(), 0, *contribution);
        return success();
    }

    LogicalResult reverseStore(StoreOp store, OpBuilder &builder, AdjointMap &adjoints) {
        Value storageGradient = adjoints.lookup({store.getStorage(), 0});
        if (!storageGradient)
            storageGradient = zeroFor(builder, store.getLoc(), store.getStorage());
        SmallVector<Value> indices;
        for (Value index : store.getIndices()) {
            FailureOr<Value> materialized = materializePrimal(index, builder);
            if (failed(materialized))
                return store.emitError("cannot materialize a dynamic TensorView store index in reverse");
            indices.push_back(*materialized);
        }
        Value valueGradient = tensor::ExtractOp::create(builder, store.getLoc(), storageGradient, indices);
        accumulate(builder, store.getLoc(), adjoints, store.getValue(), 0, valueGradient);
        Value zero = createZero(builder, store.getLoc(), valueGradient.getType());
        FailureOr<Value> priorGradient = replaceTensorElement(storageGradient, indices, zero, builder, store.getLoc());
        if (failed(priorGradient))
            return store.emitError("cannot functionalize TensorView store gradient overwrite");
        adjoints[{store.getStorage(), 0}] = *priorGradient;
        return success();
    }

    LogicalResult reverseRule(Operation &operation, OpBuilder &builder, AdjointMap &adjoints) {
        return reverseScalarOperation(operation, builder, analysis, registry, primals, adjoints);
    }

    LogicalResult reverseProjection(Operation &operation, Value input, Value result,
                                    const ValueAbiPathComponent &prefix, OpBuilder &builder, AdjointMap &adjoints) {
        const ValueAbiLayout *inputLayout = analysis.getValueAbi(input);
        const ValueAbiLayout *resultLayout = analysis.getValueAbi(result);
        if (!inputLayout || !resultLayout)
            return operation.emitError("structured projection has no canonical Value ABI");
        for (auto [resultLeafIndex, resultLeaf] : llvm::enumerate(resultLayout->leaves)) {
            Value seed = adjoints.lookup({result, static_cast<unsigned>(resultLeafIndex)});
            if (!seed)
                continue;
            SmallVector<ValueAbiPathComponent> expected = {prefix};
            llvm::append_range(expected, resultLeaf.path);
            auto found = llvm::find_if(inputLayout->leaves, [&](const ValueAbiLeaf &leaf) {
                if (leaf.path.size() != expected.size())
                    return false;
                return llvm::all_of(llvm::zip_equal(leaf.path, expected), [](auto pair) {
                    return std::get<0>(pair).field == std::get<1>(pair).field &&
                           std::get<0>(pair).index == std::get<1>(pair).index;
                });
            });
            if (found == inputLayout->leaves.end())
                return operation.emitError("structured projection leaf is absent from the canonical input ABI");
            unsigned inputLeafIndex = static_cast<unsigned>(std::distance(inputLayout->leaves.begin(), found));
            accumulate(builder, operation.getLoc(), adjoints, input, inputLeafIndex, seed);
        }
        return success();
    }

    LogicalResult reverseCreate(Operation &operation, OpBuilder &builder, AdjointMap &adjoints) {
        Value result = operation.getResult(0);
        const ValueAbiLayout *resultLayout = analysis.getValueAbi(result);
        if (!resultLayout)
            return operation.emitError("aggregate creation has no canonical Value ABI");
        unsigned resultLeafIndex = 0;
        for (Value operand : operation.getOperands()) {
            const ValueAbiLayout *operandLayout = analysis.getValueAbi(operand);
            if (!operandLayout)
                return operation.emitError("aggregate creation operand has no canonical Value ABI");
            for (unsigned operandLeafIndex = 0; operandLeafIndex < operandLayout->leaves.size();
                 ++operandLeafIndex, ++resultLeafIndex) {
                Value seed = adjoints.lookup({result, resultLeafIndex});
                if (seed)
                    accumulate(builder, operation.getLoc(), adjoints, operand, operandLeafIndex, seed);
            }
        }
        return success(resultLeafIndex == resultLayout->leaves.size());
    }

    LogicalResult reverseStructural(Operation &operation, OpBuilder &builder, AdjointMap &adjoints) {
        if (auto get = dyn_cast<StructGetOp>(operation))
            return reverseProjection(operation, get.getInput(), get.getResult(),
                                     ValueAbiPathComponent::getField(get.getField()), builder, adjoints);
        if (auto get = dyn_cast<TupleGetOp>(operation))
            return reverseProjection(operation, get.getInput(), get.getResult(),
                                     ValueAbiPathComponent::getIndex(static_cast<uint64_t>(get.getIndex())), builder,
                                     adjoints);
        if (isa<StructCreateOp, TupleCreateOp>(operation))
            return reverseCreate(operation, builder, adjoints);
        return failure();
    }

    LogicalResult reverseBlock(Block &block, OpBuilder &builder, AdjointMap &adjoints, Value parentRegion,
                               Value parentRecordIndex) {
        for (Operation &operation : llvm::reverse(block.without_terminator())) {
            if (auto ifOp = dyn_cast<scf::IfOp>(operation)) {
                if (failed(reverseIf(ifOp, builder, adjoints, parentRegion, parentRecordIndex)))
                    return failure();
                continue;
            }
            if (auto whileOp = dyn_cast<scf::WhileOp>(operation)) {
                if (failed(reverseWhile(whileOp, builder, adjoints, parentRegion, parentRecordIndex)))
                    return failure();
                continue;
            }
            if (isa<StructGetOp, TupleGetOp, StructCreateOp, TupleCreateOp>(operation)) {
                if (failed(reverseStructural(operation, builder, adjoints)))
                    return failure();
                continue;
            }
            if (auto load = dyn_cast<LoadOp>(operation)) {
                if (failed(reverseLoad(load, builder, adjoints)))
                    return failure();
                continue;
            }
            if (auto extract = dyn_cast<tensor::ExtractOp>(operation)) {
                if (failed(reverseTensorExtract(extract, builder, adjoints)))
                    return failure();
                continue;
            }
            if (auto store = dyn_cast<StoreOp>(operation)) {
                if (failed(reverseStore(store, builder, adjoints)))
                    return failure();
                continue;
            }
            if (failed(reverseRule(operation, builder, adjoints)))
                return failure();
        }
        return success();
    }

    Value readNestedRegion(OpBuilder &builder, Location location, Value parentRegion, Value parentRecordIndex,
                           unsigned childOrdinal) {
        return createOperation(
                   builder, location, AdReadNestedRegionOp::getOperationName(), {parentRegion, parentRecordIndex},
                   AdRegionHeaderType::get(builder.getContext()),
                   builder.getNamedAttr("child_ordinal", builder.getI64IntegerAttr(static_cast<int64_t>(childOrdinal))))
            ->getResult(0);
    }

    LogicalResult reverseIf(scf::IfOp source, OpBuilder &builder, AdjointMap &adjoints, Value parentRegion,
                            Value parentRecordIndex) {
        bool seeded = llvm::any_of(source.getResults(), [&](Value result) {
            const ValueAbiLayout *layout = analysis.getValueAbi(result);
            if (!layout)
                return false;
            return llvm::any_of(llvm::seq<unsigned>(0, layout->leaves.size()),
                                [&](unsigned leaf) { return adjoints.contains({result, leaf}); });
        });
        if (!seeded)
            return success();
        const AutodiffTapeRegion *region = regions.lookup(source.getOperation());
        if (!region)
            return source.emitError("active scf.if has no reverse tape region");
        Value handle =
            readNestedRegion(builder, source.getLoc(), parentRegion, parentRecordIndex, region->childOrdinal);
        Value zero = createIndexConstant(builder, source.getLoc(), 0);
        const AutodiffTapeField *predicate = findField(region->record.prefix, AutodiffTapeFieldKind::Predicate);
        if (!predicate)
            return source.emitError("scf.if reverse tape record has no predicate");
        Value condition = readLeaf(builder, source.getLoc(), handle, zero, builder.getI1Type(), region->record.stride,
                                   region->record.alignment, predicate->offset);

        SmallVector<AdjointKey> external = externalActiveLeaves(source.getThenRegion());
        for (AdjointKey value : externalActiveLeaves(source.getElseRegion()))
            if (!llvm::is_contained(external, value))
                external.push_back(value);
        SmallVector<Type> resultTypes;
        for (auto [value, leafIndex] : external)
            resultTypes.push_back(*derivativeType(value, leafIndex));
        scf::IfOp reverse = scf::IfOp::create(builder, source.getLoc(), resultTypes, condition, true);
        for (auto [sourceRegion, reverseRegion] : llvm::zip_equal(source->getRegions(), reverse->getRegions())) {
            Block &reverseBlockRef = reverseRegion.front();
            scf::YieldOp reverseYield =
                reverseBlockRef.empty() ? scf::YieldOp{} : dyn_cast<scf::YieldOp>(reverseBlockRef.back());
            OpBuilder branchBuilder = reverseYield ? OpBuilder(reverseYield) : OpBuilder::atBlockEnd(&reverseBlockRef);
            SmallVector<std::pair<Value, Value>> savedPrimals;
            if (failed(loadRegionPrimals(branchBuilder, *region, handle, zero, savedPrimals, &sourceRegion)))
                return failure();
            AdjointMap local;
            auto sourceYield = cast<scf::YieldOp>(sourceRegion.front().getTerminator());
            for (auto [index, yielded] : llvm::enumerate(sourceYield.getOperands())) {
                const ValueAbiLayout *layout = analysis.getValueAbi(yielded);
                if (!layout)
                    continue;
                for (unsigned leafIndex = 0; leafIndex < layout->leaves.size(); ++leafIndex) {
                    Value seed = adjoints.lookup({source.getResult(index), leafIndex});
                    if (seed)
                        local[{yielded, leafIndex}] = seed;
                }
            }
            if (failed(reverseBlock(sourceRegion.front(), branchBuilder, local, handle, zero)))
                return failure();
            SmallVector<Value> yielded;
            for (auto [value, leafIndex] : external) {
                Value contribution = local.lookup({value, leafIndex});
                yielded.push_back(contribution ? contribution
                                               : zeroFor(branchBuilder, source.getLoc(), value, leafIndex));
            }
            if (reverseYield)
                reverseYield->setOperands(yielded);
            else
                scf::YieldOp::create(branchBuilder, source.getLoc(), yielded);
            restorePrimals(savedPrimals);
        }
        builder.setInsertionPointAfter(reverse);
        for (auto [key, contribution] : llvm::zip_equal(external, reverse.getResults()))
            accumulate(builder, source.getLoc(), adjoints, key.first, key.second, contribution);
        return success();
    }

    LogicalResult reverseWhile(scf::WhileOp source, OpBuilder &builder, AdjointMap &adjoints, Value parentRegion,
                               Value parentRecordIndex) {
        bool seeded = llvm::any_of(source.getResults(), [&](Value result) {
            const ValueAbiLayout *layout = analysis.getValueAbi(result);
            if (!layout)
                return false;
            return llvm::any_of(llvm::seq<unsigned>(0, layout->leaves.size()),
                                [&](unsigned leaf) { return adjoints.contains({result, leaf}); });
        });
        if (!seeded)
            return success();
        const AutodiffTapeRegion *region = regions.lookup(source.getOperation());
        if (!region)
            return source.emitError("active scf.while has no reverse tape region");
        auto condition = cast<scf::ConditionOp>(source.getBefore().front().getTerminator());
        auto yield = cast<scf::YieldOp>(source.getAfter().front().getTerminator());
        for (auto [forwarded, argument] : llvm::zip_equal(condition.getArgs(), source.getBeforeArguments()))
            if (forwarded != argument)
                return source.emitError("structured reverse requires scf.while conditions to forward carried values");

        Value handle =
            readNestedRegion(builder, source.getLoc(), parentRegion, parentRecordIndex, region->childOrdinal);
        Value count = createOperation(builder, source.getLoc(), AdReadExecutedCountOp::getOperationName(), handle,
                                      builder.getIndexType())
                          ->getResult(0);
        SmallVector<std::pair<unsigned, unsigned>> carriedIndices;
        for (auto [index, result] : llvm::enumerate(source.getResults())) {
            const ValueAbiLayout *layout = analysis.getValueAbi(result);
            if (!layout)
                continue;
            for (unsigned leafIndex = 0; leafIndex < layout->leaves.size(); ++leafIndex)
                if (analysis.isActive(result, leafIndex))
                    carriedIndices.emplace_back(index, leafIndex);
        }
        SmallVector<AdjointKey> external = externalActiveLeaves(source.getAfter());
        SmallVector<Value> initial;
        for (auto [index, leafIndex] : carriedIndices) {
            Value seed = adjoints.lookup({source.getResult(index), leafIndex});
            initial.push_back(seed ? seed : zeroFor(builder, source.getLoc(), source.getResult(index), leafIndex));
        }
        for (auto [value, leafIndex] : external)
            initial.push_back(zeroFor(builder, source.getLoc(), value, leafIndex));

        Value zero = createIndexConstant(builder, source.getLoc(), 0);
        Value one = createIndexConstant(builder, source.getLoc(), 1);
        scf::ForOp reverse = scf::ForOp::create(builder, source.getLoc(), zero, count, one, initial);
        Block *body = reverse.getBody();
        scf::YieldOp bodyYield = body->empty() ? scf::YieldOp{} : dyn_cast<scf::YieldOp>(body->back());
        OpBuilder bodyBuilder = bodyYield ? OpBuilder(bodyYield) : OpBuilder::atBlockEnd(body);
        Value last = arith::SubIOp::create(bodyBuilder, source.getLoc(), count, one);
        Value recordIndex = arith::SubIOp::create(bodyBuilder, source.getLoc(), last, reverse.getInductionVar());
        SmallVector<std::pair<Value, Value>> savedPrimals;
        if (failed(loadRegionPrimals(bodyBuilder, *region, handle, recordIndex, savedPrimals)))
            return failure();
        AdjointMap local;
        unsigned position = 0;
        for (auto [index, leafIndex] : carriedIndices)
            local[{yield.getOperand(index), leafIndex}] = reverse.getRegionIterArgs()[position++];
        for (AdjointKey value : external)
            local[value] = reverse.getRegionIterArgs()[position++];
        if (failed(reverseBlock(source.getAfter().front(), bodyBuilder, local, handle, recordIndex)))
            return failure();
        SmallVector<Value> next;
        for (auto [index, leafIndex] : carriedIndices) {
            Value argument = source.getAfterArguments()[index];
            Value contribution = local.lookup({argument, leafIndex});
            next.push_back(contribution ? contribution : zeroFor(bodyBuilder, source.getLoc(), argument, leafIndex));
        }
        for (auto [value, leafIndex] : external) {
            Value contribution = local.lookup({value, leafIndex});
            next.push_back(contribution ? contribution : zeroFor(bodyBuilder, source.getLoc(), value, leafIndex));
        }
        if (bodyYield)
            bodyYield->setOperands(next);
        else
            scf::YieldOp::create(bodyBuilder, source.getLoc(), next);
        restorePrimals(savedPrimals);

        builder.setInsertionPointAfter(reverse);
        position = 0;
        for (auto [index, leafIndex] : carriedIndices)
            accumulate(builder, source.getLoc(), adjoints, source.getInits()[index], leafIndex,
                       reverse.getResult(position++));
        for (auto [value, leafIndex] : external)
            accumulate(builder, source.getLoc(), adjoints, value, leafIndex, reverse.getResult(position++));
        return success();
    }

    func::FuncOp primal;
    const VernonAutodiffAnalysisResult &analysis;
    const VernonAutodiffRuleRegistry &registry;
    const VernonAutodiffTapePlan &plan;
    const DynamicRootLayout &rootLayout;
    Value rootRegion;
    DenseMap<Value, Value> primals;
    DenseMap<Operation *, const AutodiffTapeRegion *> regions;
};

FailureOr<func::FuncOp> createDynamicForward(func::FuncOp primal, StringRef symbol, const VernonAutodiffTapePlan &plan,
                                             const ValueAbiLayout &outputLayout, const DynamicRootLayout &rootLayout) {
    MLIRContext *context = primal.getContext();
    Type tapeType = AdTapeType::get(context);
    Type regionType = AdRegionHeaderType::get(context);
    TupleType resultType = TupleType::get(context, {primal.getResultTypes().front(), tapeType, regionType});
    OpBuilder moduleBuilder(primal);
    auto forward = func::FuncOp::create(moduleBuilder, primal.getLoc(), symbol,
                                        FunctionType::get(context, primal.getArgumentTypes(), {resultType}));
    bool committed = false;
    auto cleanup = llvm::make_scope_exit([&] {
        if (!committed)
            forward.erase();
    });
    copyProfileFunctionAttrs(primal, forward);
    forward.setAllArgAttrs(primal.getAllArgAttrs());
    SmallVector<StringRef> outputDtypes;
    for (const ValueAbiLeaf &leaf : outputLayout.leaves)
        outputDtypes.push_back(leaf.dtype);
    SmallVector<DictionaryAttr> forwardResultAttrs = {
        makeInterfaceAttrs(context, "output", "forward_result", outputDtypes, 0)};
    forward.setAllResultAttrs(forwardResultAttrs);
    Block *entry = forward.addEntryBlock();
    OpBuilder builder = OpBuilder::atBlockEnd(entry);

    OperationState captureState(primal.getLoc(), AdCaptureOp::getOperationName());
    captureState.addTypes({tapeType, regionType, builder.getI1Type(), builder.getIndexType(), builder.getI1Type()});
    captureState.addRegion();
    auto capture = cast<AdCaptureOp>(builder.create(captureState));
    Block *captureBlock = new Block();
    capture.getBody().push_back(captureBlock);
    OpBuilder captureBuilder(captureBlock, captureBlock->end());
    Value tape = createOperation(captureBuilder, primal.getLoc(), AdBeginInvocationOp::getOperationName(), {}, tapeType)
                     ->getResult(0);
    Value root = beginRegion(captureBuilder, primal.getLoc(), tape);
    Value rootRecord = reserveRecord(captureBuilder, primal.getLoc(), root, rootLayout.stride, rootLayout.alignment);
    DynamicForwardEmitter emitter(primal, plan, tape, root, rootRecord);
    for (auto [source, target] : llvm::zip_equal(primal.getArguments(), entry->getArguments()))
        emitter.getMapping().map(source, target);
    if (failed(emitter.emitTopLevel(captureBuilder)))
        return failure();
    DenseSet<Value> written;
    for (const AutodiffTapeLeaf &leaf : plan.getInvocationRecord().leaves) {
        if (!written.insert(leaf.value).second)
            continue;
        Value mapped = emitter.getMapping().lookupOrNull(leaf.value);
        if (!mapped)
            return primal.emitError("invocation tape value is unavailable after augmented forward");
        if (failed(writeCanonicalTapeValue(captureBuilder, primal.getLoc(), primal->getParentOfType<ModuleOp>(), root,
                                           rootRecord, mapped, leaf.value, plan.getInvocationRecord().leaves,
                                           rootLayout.invocationLeafOffsets)))
            return primal.emitError("cannot project an invocation tape value through its canonical ABI");
    }
    auto primalReturn = cast<func::ReturnOp>(primal.getBody().front().getTerminator());
    Value primalResult = emitter.getMapping().lookupOrNull(primalReturn.getOperand(0));
    if (!primalResult)
        return primal.emitError("augmented forward did not map the primal result");
    SmallVector<AutodiffTapeLeaf> outputLeaves;
    for (auto [index, leaf] : llvm::enumerate(outputLayout.leaves)) {
        const uint64_t scalarSize = std::max<uint64_t>(leaf.scalarType.getIntOrFloatBitWidth() / 8, 1);
        outputLeaves.push_back(AutodiffTapeLeaf{
            primalReturn.getOperand(0), static_cast<unsigned>(index), "", leaf.scalarType, leaf.dtype, leaf.scalarCount,
            rootLayout.outputLeafOffsets[index], leaf.scalarCount * scalarSize, scalarSize});
    }
    if (failed(writeCanonicalTapeValue(captureBuilder, primal.getLoc(), primal->getParentOfType<ModuleOp>(), root,
                                       rootRecord, primalResult, primalReturn.getOperand(0), outputLeaves)))
        return primal.emitError("cannot project the structured output through its canonical ABI");
    createOperation(captureBuilder, primal.getLoc(), AdEndRegionOp::getOperationName(),
                    {root, createIndexConstant(captureBuilder, primal.getLoc(), 1),
                     createI32Constant(captureBuilder, primal.getLoc(), 0)});
    createOperation(captureBuilder, primal.getLoc(), AdCaptureYieldOp::getOperationName(), {tape, root});

    FailureOr<Value> output =
        readCanonicalTapeValue(builder, primal.getLoc(), primal->getParentOfType<ModuleOp>(), capture.getRootRegion(),
                               createIndexConstant(builder, primal.getLoc(), 0), primalReturn.getOperand(0),
                               outputLeaves, rootLayout.stride, rootLayout.alignment);
    if (failed(output))
        return primal.emitError("cannot reconstruct the structured output from canonical ABI leaves");
    Value result =
        createTuple(builder, primal.getLoc(), resultType, {*output, capture.getTape(), capture.getRootRegion()});
    func::ReturnOp::create(builder, primal.getLoc(), result);
    committed = true;
    return forward;
}

FailureOr<func::FuncOp> createDynamicBackward(func::FuncOp primal, StringRef symbol,
                                              const VernonAutodiffAnalysisResult &analysis,
                                              const VernonAutodiffRuleRegistry &registry,
                                              const VernonAutodiffTapePlan &plan, const DynamicRootLayout &rootLayout) {
    MLIRContext *context = primal.getContext();
    Type tapeType = AdTapeType::get(context);
    Type regionType = AdRegionHeaderType::get(context);
    BackwardProfileTypes profileTypes = getBackwardProfileTypes(context, analysis);
    OpBuilder moduleBuilder(primal);
    SmallVector<Type> backwardArguments = {tapeType, regionType};
    llvm::append_range(backwardArguments, profileTypes.cotangents);
    auto backward = func::FuncOp::create(moduleBuilder, primal.getLoc(), symbol,
                                         FunctionType::get(context, backwardArguments, {profileTypes.result}));
    bool committed = false;
    auto cleanup = llvm::make_scope_exit([&] {
        if (!committed)
            backward.erase();
    });
    copyProfileFunctionAttrs(primal, backward);
    SmallVector<DictionaryAttr> backwardArgumentAttrs = {
        makeInterfaceAttrs(context, "input", "tape", {}, 0),
        makeInterfaceAttrs(context, "input", "tape_region", {}, 1),
    };
    for (auto [index, leaf] : llvm::enumerate(analysis.getActiveResultLeaves()))
        backwardArgumentAttrs.push_back(makeInterfaceAttrs(
            context, "input", leaf.path, {profileTypes.cotangentDtypes[index]}, index + 2, leaf.derivativeType));
    backward.setAllArgAttrs(backwardArgumentAttrs);
    SmallVector<DictionaryAttr> backwardResultAttrs = {
        makeInterfaceAttrs(context, "output", "gradients", profileTypes.gradientDtypes, 0)};
    backward.setAllResultAttrs(backwardResultAttrs);

    Block *entry = backward.addEntryBlock();
    OpBuilder builder = OpBuilder::atBlockEnd(entry);
    DynamicReverseEmitter emitter(primal, analysis, registry, plan, rootLayout, entry->getArgument(1));
    if (failed(emitter.initializePrimals(builder)))
        return failure();
    AdjointMap adjoints;
    Value returned = cast<func::ReturnOp>(primal.getBody().front().getTerminator()).getOperand(0);
    for (auto [index, leaf] : llvm::enumerate(analysis.getActiveResultLeaves()))
        adjoints.try_emplace(AdjointKey{returned, leaf.abiLeafIndex}, entry->getArgument(index + 2));
    if (failed(emitter.reverseTopLevel(builder, adjoints)))
        return failure();
    SmallVector<Value> gradients;
    for (const AutodiffLeaf &leaf : analysis.getWrtLeaves()) {
        Value gradient = adjoints.lookup({leaf.value, leaf.abiLeafIndex});
        gradients.push_back(gradient ? gradient : createZero(builder, primal.getLoc(), leaf.derivativeType));
    }
    Value result = gradients.size() == 1
                       ? gradients.front()
                       : createTuple(builder, primal.getLoc(), cast<TupleType>(profileTypes.result), gradients);
    func::ReturnOp::create(builder, primal.getLoc(), result);
    committed = true;
    return backward;
}

struct VernonStructuredVjpPass final : PassWrapper<VernonStructuredVjpPass, OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VernonStructuredVjpPass)

    VernonStructuredVjpPass() = default;
    VernonStructuredVjpPass(const VernonStructuredVjpPass &other) : PassWrapper(other), options(other.options) {}
    explicit VernonStructuredVjpPass(StructuredVjpOptions options) : options(std::move(options)) {}

    StringRef getArgument() const final { return "vernon-structured-vjp"; }
    StringRef getDescription() const final { return "Generate structured VJP profiles"; }

    void runOnOperation() override {
        StructuredVjpOptions selected = options;
        if (selected.wrtPaths.empty())
            selected.wrtPaths.assign(wrt.begin(), wrt.end());
        if (selected.forwardSymbol.empty())
            selected.forwardSymbol = forward;
        if (selected.backwardSymbol.empty())
            selected.backwardSymbol = backward;
        if (selected.wrtPaths.empty() || selected.forwardSymbol.empty() || selected.backwardSymbol.empty()) {
            getOperation().emitError("structured VJP pass requires wrt paths and forward/backward symbols");
            return signalPassFailure();
        }
        SmallVector<func::FuncOp> entries;
        getOperation().walk([&](func::FuncOp function) {
            if (function->hasAttr(kEntryAttr))
                entries.push_back(function);
        });
        if (entries.size() != 1) {
            getOperation().emitError("structured VJP pass requires exactly one entry function");
            return signalPassFailure();
        }
        if (failed(buildStructuredVjp(entries.front(), selected)))
            signalPassFailure();
    }

    StructuredVjpOptions options;
    ListOption<std::string> wrt{*this, "wrt", llvm::cl::desc("Canonical scalar wrt paths"), llvm::cl::ZeroOrMore};
    Option<std::string> forward{*this, "forward", llvm::cl::desc("Augmented forward symbol"), llvm::cl::init("")};
    Option<std::string> backward{*this, "backward", llvm::cl::desc("Reverse symbol"), llvm::cl::init("")};
};

} // namespace

FailureOr<StructuredVjpResult> buildStructuredVjp(func::FuncOp primal, const StructuredVjpOptions &options) {
    if (!primal || !primal->getParentOfType<ModuleOp>())
        return failure();
    if (options.wrtPaths.empty() || options.forwardSymbol.empty() || options.backwardSymbol.empty())
        return primal.emitError("structured VJP requires wrt paths and profile symbols");
    if (options.forwardSymbol == options.backwardSymbol)
        return primal.emitError("structured VJP requires distinct profile symbols");
    llvm::StringSet<> uniqueWrtPaths;
    for (const std::string &path : options.wrtPaths)
        if (path.empty() || !uniqueWrtPaths.insert(path).second)
            return primal.emitError("structured VJP requires unique non-empty wrt paths");
    ModuleOp module = primal->getParentOfType<ModuleOp>();
    if (module.lookupSymbol(options.forwardSymbol) || module.lookupSymbol(options.backwardSymbol))
        return primal.emitError("structured VJP profile symbol already exists");

    SmallVector<StringRef> wrtPaths;
    for (const std::string &path : options.wrtPaths)
        wrtPaths.push_back(path);
    VernonAutodiffRuleRegistry registry = createDefaultAutodiffRuleRegistry();
    FailureOr<VernonAutodiffAnalysisResult> analysis = analyzeAutodiffFunction(primal, wrtPaths, registry);
    if (failed(analysis) || failed(validateStructuredPhase(primal, *analysis)))
        return failure();
    FailureOr<VernonAutodiffTapePlan> plan = planAutodiffTape(primal, *analysis, registry);
    if (failed(plan))
        return failure();
    for (const AutodiffTapeRegion &region : plan->getRegions())
        if (region.record.stride > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
            return primal.emitError("structured VJP dynamic tape record is not representable");

    Value returned = cast<func::ReturnOp>(primal.getBody().front().getTerminator()).getOperand(0);
    const ValueAbiLayout *outputLayout = analysis->getValueAbi(returned);
    if (!outputLayout)
        return primal.emitError("structured VJP output has no canonical Value ABI");
    FailureOr<DynamicRootLayout> rootLayout = buildDynamicRootLayout(*plan, *outputLayout);
    if (failed(rootLayout))
        return primal.emitError("structured VJP root record layout overflow");
    FailureOr<func::FuncOp> forward =
        createDynamicForward(primal, options.forwardSymbol, *plan, *outputLayout, *rootLayout);
    if (failed(forward))
        return failure();
    FailureOr<func::FuncOp> backward =
        createDynamicBackward(primal, options.backwardSymbol, *analysis, registry, *plan, *rootLayout);
    if (failed(backward)) {
        forward->erase();
        return failure();
    }
    if (failed(verify(*forward)) || failed(verify(*backward))) {
        forward->erase();
        backward->erase();
        return primal.emitError("generated structured VJP profile failed verification");
    }
    primal->removeAttr(kEntryAttr);
    SmallVector<std::string> derivativeRules;
    for (const AutodiffOperationActivity &activity : analysis->getOperations()) {
        if (!activity.active)
            continue;
        if (const DifferentiationRule *rule = registry.lookup(activity.operation))
            derivativeRules.push_back(rule->getRegistryKey().str());
    }
    llvm::sort(derivativeRules);
    derivativeRules.erase(std::unique(derivativeRules.begin(), derivativeRules.end()), derivativeRules.end());
    if (plan->getStaticTapeBytesHint() > std::numeric_limits<uint64_t>::max() - rootLayout->stride) {
        forward->erase();
        backward->erase();
        return primal.emitError("structured VJP tape statistics hint overflow");
    }
    const uint64_t tapeBytes = plan->getStaticTapeBytesHint() + rootLayout->stride;
    return StructuredVjpResult{*forward, *backward, tapeBytes, std::move(derivativeRules)};
}

std::unique_ptr<Pass> createVernonStructuredVjpPass() { return std::make_unique<VernonStructuredVjpPass>(); }

std::unique_ptr<Pass> createVernonStructuredVjpPass(StructuredVjpOptions options) {
    return std::make_unique<VernonStructuredVjpPass>(std::move(options));
}

void registerVernonStructuredVjpPass() { PassRegistration<VernonStructuredVjpPass>(); }

} // namespace mlir::vernon
