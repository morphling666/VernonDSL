#include "mlir/Dialect/Vernon/Transforms/VernonStructuredVjp.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vernon/IR/Vernon.h"
#include "mlir/Dialect/Vernon/IR/VernonAttrs.h"
#include "mlir/Dialect/Vernon/Transforms/VernonAggregateStorage.h"
#include "mlir/Dialect/Vernon/Transforms/VernonAutodiffAnalysis.h"
#include "mlir/Dialect/Vernon/Transforms/VernonAutodiffRules.h"
#include "mlir/Dialect/Vernon/Transforms/VernonAutodiffTapePlanning.h"
#include "mlir/Dialect/Vernon/Transforms/VernonAutodiffUtils.h"
#include "mlir/Dialect/Vernon/Transforms/VernonLowerAccumulation.h"
#include "mlir/IR/Dominance.h"
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

Value buildPositiveStepTripCount(OpBuilder &builder, Location location, Value lower, Value upper, Value step) {
    Value zero = createIndexConstant(builder, location, 0);
    Value distance = arith::SubIOp::create(builder, location, upper, lower);
    Value positive = arith::CmpIOp::create(builder, location, arith::CmpIPredicate::sgt, distance, zero);
    Value count = arith::CeilDivSIOp::create(builder, location, distance, step);
    return arith::SelectOp::create(builder, location, positive, count, zero);
}

bool isValueDefinedIn(Value value, Region &region) {
    Region *owner = value.getParentRegion();
    return owner && (owner == &region || region.isAncestor(owner));
}

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

SmallVector<StringRef> languageLeafDtypes(DictionaryAttr attrs, Type valueType = {}) {
    SmallVector<StringRef> result;
    if (!attrs)
        return result;
    const bool container = isa_and_nonnull<RankedTensorType, TensorViewType>(valueType);
    if (container)
        if (auto dtypes = attrs.getAs<ArrayAttr>("vernon.element_abi_leaf_dtypes")) {
            for (Attribute attribute : dtypes)
                if (auto dtype = dyn_cast<StringAttr>(attribute))
                    result.push_back(dtype.getValue());
            if (!result.empty())
                return result;
        }
    if (auto dtypes = attrs.getAs<ArrayAttr>("vernon.abi_leaf_dtypes")) {
        for (Attribute attribute : dtypes)
            if (auto dtype = dyn_cast<StringAttr>(attribute))
                result.push_back(dtype.getValue());
        if (!result.empty())
            return result;
    }
    if (auto dtype = attrs.getAs<StringAttr>("vernon.dtype"); dtype && !dtype.getValue().empty())
        result.push_back(dtype.getValue());
    return result;
}

DictionaryAttr makeInterfaceAttrs(MLIRContext *context, StringRef interfaceName, StringRef sourceName,
                                  ArrayRef<StringRef> dtypes, int64_t location, Type valueType = {},
                                  StringRef autodiffRole = {}, StringRef autodiffSource = {}) {
    NamedAttrList attributes;
    attributes.set("vernon.interface", StringAttr::get(context, interfaceName));
    attributes.set("vernon.source_name", StringAttr::get(context, sourceName));
    SmallVector<Attribute> dtypeAttrs;
    for (StringRef dtype : dtypes)
        dtypeAttrs.push_back(StringAttr::get(context, dtype));
    attributes.set("vernon.abi_leaf_dtypes", ArrayAttr::get(context, dtypeAttrs));
    if (isa_and_nonnull<RankedTensorType, TensorViewType>(valueType))
        attributes.set("vernon.element_abi_leaf_dtypes", ArrayAttr::get(context, dtypeAttrs));
    if (dtypes.size() == 1)
        attributes.set("vernon.dtype", StringAttr::get(context, dtypes.front()));
    attributes.set("vernon.location", IntegerAttr::get(IntegerType::get(context, 64), location));
    if (!autodiffRole.empty())
        attributes.set("vernon.autodiff_role", StringAttr::get(context, autodiffRole));
    if (!autodiffSource.empty())
        attributes.set("vernon.autodiff_source", StringAttr::get(context, autodiffSource));
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
    SmallVector<Type> shapeSources;
    SmallVector<Type> cotangents;
    SmallVector<Type> storageGradients;
    SmallVector<Type> valueGradients;
    Type result;
    SmallVector<StringRef> cotangentDtypes;
    SmallVector<StringRef> gradientDtypes;
    SmallVector<StringRef> valueGradientDtypes;
};

bool hasExternalStorageShapeSource(const AutodiffStorageIdentity &identity) {
    auto view = dyn_cast<TensorViewType>(identity.binding.getType());
    return identity.internalAdjointOwnership != AutodiffInternalAdjointOwnership::None && view &&
           view.getAddressSpace() == "device" && isa<BlockArgument>(identity.binding);
}

bool requiresWorkgroupBackwardSynchronization(const VernonAutodiffAnalysisResult &analysis) {
    if (llvm::any_of(analysis.getStorageIdentities(), [](const AutodiffStorageIdentity &identity) {
            return identity.internalAdjointOwnership == AutodiffInternalAdjointOwnership::WorkgroupCoupled;
        }))
        return true;
    return llvm::any_of(analysis.getOperations(), [](const AutodiffOperationActivity &activity) {
        if (!activity.active)
            return false;
        if (activity.effect == AutodiffEffectKind::Barrier)
            return true;
        Value storage;
        if (auto load = dyn_cast<LoadOp>(activity.operation))
            storage = load.getStorage();
        else if (auto store = dyn_cast<StoreOp>(activity.operation))
            storage = store.getStorage();
        else if (auto atomic = dyn_cast<AtomicOp>(activity.operation))
            storage = atomic.getStorage();
        else if (auto reduce = dyn_cast<ReduceSumOp>(activity.operation))
            storage = reduce.getStorage();
        else if (auto scatter = dyn_cast<ScatterAddOp>(activity.operation))
            storage = scatter.getStorage();
        auto view = storage ? dyn_cast<TensorViewType>(storage.getType()) : TensorViewType{};
        return view && view.getAddressSpace() == "workgroup";
    });
}

BackwardProfileTypes getBackwardProfileTypes(MLIRContext *context, const VernonAutodiffAnalysisResult &analysis) {
    BackwardProfileTypes types;
    for (const AutodiffStorageIdentity &identity : analysis.getStorageIdentities()) {
        if (!hasExternalStorageShapeSource(identity))
            continue;
        auto source = cast<TensorViewType>(identity.binding.getType());
        types.shapeSources.push_back(
            TensorViewType::get(context, source.getElementType(), source.getShape(), "read", source.getAddressSpace()));
    }
    for (const AutodiffLeaf &leaf : analysis.getActiveResultLeaves()) {
        auto source = dyn_cast<TensorViewType>(leaf.value.getType());
        types.cotangents.push_back(
            source ? static_cast<Type>(TensorViewType::get(context, leaf.derivativeType, source.getShape(), "read",
                                                           source.getAddressSpace()))
                   : leaf.derivativeType);
        types.cotangentDtypes.push_back(scalarDtype(leaf.primalType));
        if (leaf.primalType.isF16())
            types.cotangentDtypes.back() = "f32";
    }
    for (const AutodiffLeaf &leaf : analysis.getWrtLeaves()) {
        StringRef dtype = leaf.primalType.isF64() ? "f64" : "f32";
        if (auto source = dyn_cast<TensorViewType>(leaf.value.getType())) {
            const auto identity =
                llvm::find_if(analysis.getStorageIdentities(), [&](const AutodiffStorageIdentity &candidate) {
                    return candidate.binding == leaf.value;
                });
            const bool invocationPrivate =
                identity != analysis.getStorageIdentities().end() &&
                identity->externalGradientOwnership == AutodiffExternalGradientOwnership::InvocationPrivate;
            Type elementType = leaf.derivativeType;
            SmallVector<int64_t> shape(source.getShape());
            if (auto aggregate = dyn_cast<RankedTensorType>(elementType)) {
                elementType = aggregate.getElementType();
                llvm::append_range(shape, aggregate.getShape());
            }
            types.storageGradients.push_back(TensorViewType::get(
                context, elementType, shape, invocationPrivate ? "read_write" : "write", source.getAddressSpace()));
        } else {
            types.valueGradients.push_back(leaf.derivativeType);
            types.valueGradientDtypes.push_back(dtype);
        }
        types.gradientDtypes.push_back(dtype);
    }
    if (!types.valueGradients.empty())
        types.result = types.valueGradients.size() == 1
                           ? types.valueGradients.front()
                           : static_cast<Type>(TupleType::get(context, types.valueGradients));
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
    SmallVector<unsigned> activeOperands;
    for (auto [operandIndex, operand] : llvm::enumerate(operation.getOperands())) {
        operands.push_back(primals.lookup(operand));
        if (analysis.hasAnyActiveLeaf(operand))
            activeOperands.push_back(static_cast<unsigned>(operandIndex));
    }
    SmallVector<Value> results = {primals.lookup(operation.getResult(0))};
    FailureOr<SmallVector<Value>> contributions =
        rule->buildVjp(&operation, AutodiffVjpBuildContext{builder, operation.getLoc(), operands, results,
                                                           ValueRange(seed->second), activeOperands});
    if (failed(contributions))
        return failure();
    for (auto [operand, contribution] : llvm::zip_equal(operation.getOperands(), *contributions))
        accumulateAdjoint(builder, operation.getLoc(), analysis, adjoints, operand, 0, contribution);
    return success();
}

LogicalResult validateStructuredPhase(func::FuncOp primal, const VernonAutodiffAnalysisResult &analysis) {
    if (!llvm::hasSingleElement(primal.getBody()))
        return primal.emitError("structured VJP requires one structured entry block");
    if (primal.getNumResults() != 0 || analysis.getActiveResultLeaves().empty() || analysis.getWrtLeaves().empty())
        return primal.emitError("structured VJP requires selected Storage outputs and at least one wrt leaf");
    WalkResult structured = primal.walk([&](Operation *operation) {
        if (operation->getNumRegions() != 0 && !isa<func::FuncOp, scf::IfOp, scf::ForOp, scf::WhileOp>(operation)) {
            operation->emitError("structured VJP supports only scf.if, scf.for, and scf.while regions");
            return WalkResult::interrupt();
        }
        return WalkResult::advance();
    });
    return structured.wasInterrupted() ? failure() : success();
}

struct ResidualRootLayout {
    SmallVector<uint64_t> invocationLeafOffsets;
    uint64_t stride{};
    uint64_t alignment{1};
};

FailureOr<ResidualRootLayout> buildDynamicRootLayout(const VernonAutodiffTapePlan &plan) {
    SmallVector<AutodiffTapeSlot> slots;
    for (const AutodiffTapeLeaf &leaf : plan.getInvocationRecord().leaves)
        slots.push_back({leaf.size, leaf.alignment});
    FailureOr<AutodiffTapeLayout> layout = planAutodiffTapeLayout(slots);
    if (failed(layout) || layout->stride > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
        return failure();
    ResidualRootLayout result;
    result.invocationLeafOffsets.assign(layout->offsets.begin(), layout->offsets.end());
    result.stride = layout->stride;
    result.alignment = layout->alignment;
    return result;
}

FailureOr<ResidualRootLayout> buildStaticRootLayout(const VernonAutodiffTapePlan &plan) {
    if (!plan.getRegions().empty())
        return failure();
    const AdMemoryPlan &memory = plan.getMemoryPlan();
    const AdBufferAssignment &assignment = memory.getBufferAssignment();
    ResidualRootLayout result;
    result.invocationLeafOffsets.reserve(plan.getInvocationRecord().leaves.size());
    for (const AutodiffTapeLeaf &leaf : plan.getInvocationRecord().leaves) {
        std::optional<uint64_t> offset;
        for (auto [index, residual] : llvm::enumerate(memory.getResiduals())) {
            if (residual.value != leaf.value || residual.abiLeafIndex != leaf.abiLeafIndex || residual.rematerialized)
                continue;
            if (index >= assignment.residualSlices.size())
                return failure();
            const AdBufferSlice &slice = assignment.residualSlices[index];
            if (slice.physicalBuffer >= assignment.physicalBuffers.size() || slice.byteSize != leaf.size)
                return failure();
            offset = slice.offset;
            result.alignment = std::max(result.alignment, residual.alignment);
            break;
        }
        if (!offset)
            return failure();
        result.invocationLeafOffsets.push_back(*offset);
    }
    result.stride = assignment.peakBytesByDomain[static_cast<size_t>(AdMemoryDomain::PersistentResidual)];
    return result.stride <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max())
               ? FailureOr<ResidualRootLayout>(std::move(result))
               : FailureOr<ResidualRootLayout>(failure());
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
    // Even an empty logical payload needs a distinct physical record so loop
    // iterations and nested-region parent links retain stable identity.
    stride = std::max<uint64_t>(stride, 1);
    alignment = std::max<uint64_t>(alignment, 1);
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
    FailureOr<ValueAbiLayout> layout = getValueStorageLayout(value.getType(), module);
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
    FailureOr<ValueAbiLayout> layout = getValueStorageLayout(source.getType(), module);
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

class ForwardEmitterCore {
public:
    ForwardEmitterCore(func::FuncOp primal, const VernonAutodiffAnalysisResult &analysis,
                       const VernonAutodiffTapePlan &plan, Value tape, Value rootRegion, Value rootRecord)
        : primal(primal), analysis(analysis), tape(tape), rootRegion(rootRegion), rootRecord(rootRecord) {
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
            if (auto forOp = dyn_cast<scf::ForOp>(operation))
                return emitFor(forOp, builder, parentRegion, parentRecord);
            if (auto whileOp = dyn_cast<scf::WhileOp>(operation))
                return emitWhile(whileOp, builder, parentRegion, parentRecord);
        }
        WalkResult validatedWrites = operation.walk([&](Operation *nested) {
            if (!isa<StoreOp, ReduceSumOp, ScatterAddOp, AtomicOp>(nested))
                return WalkResult::advance();
            if (!analysis.getStorageEffect(nested)) {
                nested->emitError("structured VJP cannot functionalize a write without analyzed Storage effects");
                return WalkResult::interrupt();
            }
            return WalkResult::advance();
        });
        if (validatedWrites.wasInterrupted())
            return failure();
        Operation *clone = builder.clone(operation, mapping);
        clone->walk([&](Operation *nested) {
            if (isa<StoreOp, ReduceSumOp, ScatterAddOp, AtomicOp>(nested))
                nested->setAttr("vernon.ad.functionalized", UnitAttr::get(builder.getContext()));
        });
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
            if (sourceRegion) {
                Value sourceValue = leaf.value;
                Region *owningChild = sourceValue.getParentRegion();
                while (owningChild && owningChild->getParentOp() != region.operation)
                    owningChild = owningChild->getParentOp()->getParentRegion();
                if (owningChild && owningChild != sourceRegion)
                    continue;
            }
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

    LogicalResult emitFor(scf::ForOp source, OpBuilder &builder, Value parentRegion, Value parentRecord) {
        const AutodiffTapeRegion *region = regions.lookup(source.getOperation());
        if (!region)
            return source.emitError("active scf.for has no residual region plan");
        Value handle = beginRegion(builder, source.getLoc(), tape, parentRegion, parentRecord, region->childOrdinal);
        SmallVector<Value> inits;
        for (Value init : source.getInitArgs())
            inits.push_back(mapping.lookup(init));
        inits.push_back(createIndexConstant(builder, source.getLoc(), 0));
        scf::ForOp target =
            scf::ForOp::create(builder, source.getLoc(), mapping.lookup(source.getLowerBound()),
                               mapping.lookup(source.getUpperBound()), mapping.lookup(source.getStep()), inits);
        mapping.map(source.getInductionVar(), target.getInductionVar());
        for (auto [sourceArgument, targetArgument] :
             llvm::zip_equal(source.getRegionIterArgs(), target.getRegionIterArgs().drop_back()))
            mapping.map(sourceArgument, targetArgument);
        Block *body = target.getBody();
        scf::YieldOp targetYield = body->empty() ? scf::YieldOp{} : dyn_cast<scf::YieldOp>(body->back());
        OpBuilder bodyBuilder = targetYield ? OpBuilder(targetYield) : OpBuilder::atBlockEnd(body);
        Value record =
            reserveRecord(bodyBuilder, source.getLoc(), handle, region->record.stride, region->record.alignment);
        if (failed(emitBlock(*source.getBody(), *body, handle, record)))
            return failure();
        if (targetYield)
            bodyBuilder.setInsertionPoint(targetYield);
        else
            bodyBuilder.setInsertionPointToEnd(body);
        if (failed(writePlannedLeaves(bodyBuilder, source.getLoc(), *region, handle, record)))
            return failure();
        auto sourceYield = cast<scf::YieldOp>(source.getBody()->getTerminator());
        SmallVector<Value> yielded;
        for (Value value : sourceYield.getResults())
            yielded.push_back(mapping.lookup(value));
        yielded.push_back(createOperation(bodyBuilder, source.getLoc(), AdCheckedIncrementOp::getOperationName(),
                                          target.getRegionIterArgs().back(), bodyBuilder.getIndexType())
                              ->getResult(0));
        if (targetYield)
            targetYield->setOperands(yielded);
        else
            scf::YieldOp::create(bodyBuilder, source.getLoc(), yielded);
        for (auto [sourceResult, targetResult] : llvm::zip_equal(source.getResults(), target.getResults().drop_back()))
            mapping.map(sourceResult, targetResult);
        builder.setInsertionPointAfter(target);
        createOperation(builder, source.getLoc(), AdEndRegionOp::getOperationName(),
                        {handle, target.getResults().back(), createI32Constant(builder, source.getLoc(), 0)});
        return success();
    }

    LogicalResult emitWhile(scf::WhileOp source, OpBuilder &builder, Value parentRegion, Value parentRecord) {
        const AutodiffTapeRegion *region = regions.lookup(source.getOperation());
        if (!region)
            return source.emitError("active scf.while has no dynamic tape plan");
        auto condition = dyn_cast<scf::ConditionOp>(source.getBefore().front().getTerminator());
        auto yield = dyn_cast<scf::YieldOp>(source.getAfter().front().getTerminator());
        if (!condition || !yield)
            return source.emitError("structured VJP requires canonical scf.while terminators");

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
    const VernonAutodiffAnalysisResult &analysis;
    Value tape;
    Value rootRegion;
    Value rootRecord;
    IRMapping mapping;
    DenseMap<Operation *, const AutodiffTapeRegion *> regions;
    DenseMap<unsigned, const AutodiffTapeRegion *> regionsByOrdinal;
};

class ReverseEmitterCore {
public:
    ReverseEmitterCore(func::FuncOp primal, const VernonAutodiffAnalysisResult &analysis,
                       const VernonAutodiffRuleRegistry &registry, const VernonAutodiffTapePlan &plan,
                       const ResidualRootLayout &rootLayout, Value rootRegion)
        : primal(primal), analysis(analysis), registry(registry), plan(plan), rootLayout(rootLayout),
          rootRegion(rootRegion) {
        for (const AutodiffTapeRegion &region : plan.getRegions())
            regions.try_emplace(region.operation, &region);
    }

    void initializeBuiltinPrimals(ArrayRef<BlockArgument> sources, ValueRange values) {
        for (auto [source, value] : llvm::zip_equal(sources, values))
            primals.try_emplace(source, value);
    }

    void initializePrimalArguments(ArrayRef<BlockArgument> sources, ValueRange values) {
        for (auto [source, value] : llvm::zip_equal(sources, values))
            primals.try_emplace(source, value);
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

    LogicalResult initializeStorageAdjoints(OpBuilder &builder, AdjointMap &adjoints, ValueRange shapeSources,
                                            ValueRange gradientDestinations) {
        const size_t expectedShapeSources =
            llvm::count_if(analysis.getStorageIdentities(), hasExternalStorageShapeSource);
        if (shapeSources.size() != expectedShapeSources)
            return primal.emitError("Storage adjoint shape-source count does not match Storage identities");
        const size_t expectedGradientDestinations =
            llvm::count_if(analysis.getWrtLeaves(),
                           [](const AutodiffLeaf &leaf) { return isa<TensorViewType>(leaf.value.getType()); });
        if (gradientDestinations.size() != expectedGradientDestinations)
            return primal.emitError("Storage gradient destination count does not match wrt Storage leaves");
        size_t gradientIndex = 0;
        for (const AutodiffLeaf &leaf : analysis.getWrtLeaves()) {
            if (!isa<TensorViewType>(leaf.value.getType()))
                continue;
            const AutodiffStorageIdentity *identity = analysis.getStorageIdentity(leaf.value);
            if (!identity)
                return primal.emitError("wrt Storage has no Storage identity");
            externalStorageAdjoints[{identity->id, leaf.abiLeafIndex}] = gradientDestinations[gradientIndex++];
        }
        size_t shapeSourceIndex = 0;
        for (const AutodiffStorageIdentity &identity : analysis.getStorageIdentities()) {
            const ValueAbiLayout *layout = analysis.getValueAbi(identity.binding);
            auto view = dyn_cast<TensorViewType>(identity.binding.getType());
            if (!layout || !view)
                return primal.emitError("Storage identity has no canonical TensorView ABI");
            for (auto [leafIndex, leaf] : llvm::enumerate(layout->leaves)) {
                if (!analysis.isActive(identity.binding, leafIndex))
                    continue;
                if (identity.internalAdjointOwnership == AutodiffInternalAdjointOwnership::None)
                    continue;
                FailureOr<Type> scalar = getAutodiffDerivativeType(leaf.scalarType);
                if (failed(scalar))
                    return primal.emitError("active Storage identity has no derivative scalar type");
                SmallVector<int64_t> shape(view.getShape());
                llvm::append_range(
                    shape, llvm::map_range(leaf.shape, [](uint64_t extent) { return static_cast<int64_t>(extent); }));
                auto bufferType = AdAdjointBufferType::get(primal.getContext(), *scalar, shape,
                                                           static_cast<unsigned>(view.getShape().size()));
                Value shapeSource = hasExternalStorageShapeSource(identity) ? shapeSources[shapeSourceIndex] : Value{};
                OperationState createState(primal.getLoc(), AdAdjointBufferCreateOp::getOperationName());
                if (shapeSource)
                    createState.addOperands(shapeSource);
                createState.addTypes(bufferType);
                StringRef ownership;
                switch (identity.internalAdjointOwnership) {
                case AutodiffInternalAdjointOwnership::LanePrivate:
                    ownership = "lane_private";
                    break;
                case AutodiffInternalAdjointOwnership::WorkgroupCoupled:
                    ownership = "workgroup_shared";
                    break;
                case AutodiffInternalAdjointOwnership::None:
                    return primal.emitError("active Storage identity has no internal adjoint ownership");
                }
                createState.addAttribute("ownership", builder.getStringAttr(ownership));
                AdAdjointBufferCreateOp create = cast<AdAdjointBufferCreateOp>(builder.create(createState));
                Value buffer = create.getBuffer();
                storageAdjoints[{identity.id, static_cast<unsigned>(leafIndex)}] = buffer;
            }
            shapeSourceIndex += hasExternalStorageShapeSource(identity);
        }
        for (const AutodiffLeaf &leaf : analysis.getActiveResultLeaves()) {
            const AutodiffStorageIdentity *identity = analysis.getStorageIdentity(leaf.value);
            if (!identity)
                continue;
            Value seed = adjoints.lookup({leaf.value, leaf.abiLeafIndex});
            Value buffer = storageAdjoints.lookup({identity->id, leaf.abiLeafIndex});
            if (!seed)
                return primal.emitError("Storage objective has no cotangent");
            if (buffer) {
                AdAdjointAccumulateDenseOp::create(builder, primal.getLoc(), buffer, seed);
            } else {
                externalStorageCotangents[{identity->id, leaf.abiLeafIndex}] = seed;
            }
            adjoints.erase({leaf.value, leaf.abiLeafIndex});
        }
        return success();
    }

    LogicalResult storeStorageAdjoint(OpBuilder &builder, const AutodiffLeaf &leaf, Value destination) {
        const AutodiffStorageIdentity *identity = analysis.getStorageIdentity(leaf.value);
        if (!identity)
            return failure();
        Value buffer = storageAdjoints.lookup({identity->id, leaf.abiLeafIndex});
        if (!buffer) {
            if (identity->internalAdjointOwnership == AutodiffInternalAdjointOwnership::None &&
                externalStorageAdjoints.contains({identity->id, leaf.abiLeafIndex}))
                return success();
            return failure();
        }
        AdAdjointStoreOp::create(builder, primal.getLoc(), buffer, destination);
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
            if (sourceRegion) {
                Value sourceValue = leaf.value;
                Region *owningChild = sourceValue.getParentRegion();
                while (owningChild && owningChild->getParentOp() != region.operation)
                    owningChild = owningChild->getParentOp()->getParentRegion();
                if (owningChild && owningChild != sourceRegion)
                    continue;
            }
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
                if (isa<TensorViewType>(operand.getType()))
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

    bool hasActiveStorageEffect(Region &region) {
        bool active = false;
        region.walk([&](Operation *operation) {
            active |= analysis.getStorageEffect(operation) && analysis.isActive(operation);
        });
        return active;
    }

    bool dominatesInsertion(Value value, OpBuilder &builder) {
        Block *insertionBlock = builder.getInsertionBlock();
        if (!value || !insertionBlock)
            return false;
        Operation *scope = insertionBlock->getParentOp();
        if (Operation *function = scope->getParentOfType<func::FuncOp>())
            scope = function;
        DominanceInfo dominance(scope);
        if (builder.getInsertionPoint() != insertionBlock->end())
            return dominance.dominates(value, &*builder.getInsertionPoint());
        if (value.getParentBlock() == insertionBlock)
            return true;
        return dominance.dominates(value.getParentBlock(), insertionBlock);
    }

    Value availablePrimal(Value source, OpBuilder &builder, const IRMapping *local = nullptr) {
        if (local)
            if (Value value = local->lookupOrNull(source))
                return value;
        Value value = primals.lookup(source);
        return dominatesInsertion(value, builder) ? value : Value{};
    }

    const AdResidualSource *selectedSource(Value value) const {
        const AdResidualSource *selected = nullptr;
        for (const AdResidualSourceSelection &selection : plan.getMemoryPlan().getSourceSelections()) {
            if (selection.key.value != value || selection.key.controlKind != AdControlSourceKind::None)
                continue;
            if (selection.selectedCandidate >= selection.candidates.size())
                return nullptr;
            const AdResidualSource &candidate = selection.candidates[selection.selectedCandidate];
            if (!selected) {
                selected = &candidate;
                continue;
            }
            if (selected->kind != candidate.kind || selected->storageIdentity != candidate.storageIdentity ||
                selected->versionBefore != candidate.versionBefore)
                return nullptr;
        }
        return selected;
    }

    FailureOr<Value> materializePrimal(Value value, OpBuilder &builder) {
        if (Value available = availablePrimal(value, builder))
            return available;
        const AdResidualSource *source = selectedSource(value);
        if (source && source->kind == AdResidualSourceKind::ExactVersionReload) {
            auto load = value.getDefiningOp<LoadOp>();
            const AutodiffLoadInfo *loadInfo = load ? analysis.getLoadInfo(load) : nullptr;
            if (!load || !loadInfo || !source->storageIdentity || !source->versionBefore ||
                loadInfo->identity != *source->storageIdentity || loadInfo->versionBefore != *source->versionBefore ||
                loadInfo->stability != AutodiffStorageStabilityRequirement::RetainedExactVersion)
                return failure();
            IRMapping mapping;
            FailureOr<Value> storage = materializePrimal(load.getStorage(), builder);
            if (failed(storage))
                return failure();
            mapping.map(load.getStorage(), *storage);
            for (Value index : load.getIndices()) {
                FailureOr<Value> materialized = materializePrimal(index, builder);
                if (failed(materialized))
                    return failure();
                mapping.map(index, *materialized);
            }
            Operation *clone = builder.clone(*load, mapping);
            Value reloaded = clone->getResult(0);
            primals[value] = reloaded;
            return reloaded;
        }
        FailureOr<AdRematerializationRecipe> rebuiltRecipe = failure();
        ArrayRef<Operation *> operations;
        if (source) {
            if (source->kind != AdResidualSourceKind::PureRematerialization)
                return failure();
            operations = source->recipe;
        } else {
            rebuiltRecipe = buildAutodiffRematerializationRecipe(
                value, primal, [&](Value root) { return static_cast<bool>(availablePrimal(root, builder)); });
            if (failed(rebuiltRecipe))
                return failure();
            operations = rebuiltRecipe->operations;
        }
        IRMapping local;
        for (Operation *operation : operations) {
            for (Value operand : operation->getOperands()) {
                Value materialized = availablePrimal(operand, builder, &local);
                if (!materialized) {
                    FailureOr<Value> dependency = materializePrimal(operand, builder);
                    if (failed(dependency))
                        return failure();
                    materialized = *dependency;
                }
                local.map(operand, materialized);
            }
            Operation *clone = builder.clone(*operation, local);
            for (auto [source, target] : llvm::zip_equal(operation->getResults(), clone->getResults()))
                local.map(source, target);
        }
        Value materialized = local.lookupOrNull(value);
        if (!materialized)
            return failure();
        primals[value] = materialized;
        return materialized;
    }

    FailureOr<Value> replaceTensorElement(Value tensorValue, ValueRange indices, Value replacement, OpBuilder &builder,
                                          Location location) {
        auto type = dyn_cast<RankedTensorType>(tensorValue.getType());
        if (!type || !type.hasStaticShape() || indices.size() != static_cast<size_t>(type.getRank()) ||
            replacement.getType() != type.getElementType())
            return failure();
        SmallVector<Value> elements;
        SmallVector<SmallVector<int64_t>> coordinates = enumerateStaticCoordinates(type.getShape());
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
        SmallVector<Value> indices;
        for (Value index : load.getIndices()) {
            FailureOr<Value> materialized = materializePrimal(index, builder);
            if (failed(materialized))
                return load.emitError("cannot materialize a dynamic TensorView load index in reverse");
            indices.push_back(*materialized);
        }
        const AutodiffStorageEffect *effect = analysis.getStorageEffect(load);
        const ValueAbiLayout *layout = analysis.getValueAbi(load.getResult());
        if (!effect || !layout)
            return load.emitError("active TensorView load has no canonical Storage effect ABI");
        for (unsigned leafIndex = 0; leafIndex < layout->leaves.size(); ++leafIndex) {
            Value seed = adjoints.lookup({load.getResult(), leafIndex});
            if (!seed)
                continue;
            Value buffer = storageAdjoints.lookup({effect->identity, leafIndex});
            Value external = externalStorageAdjoints.lookup({effect->identity, leafIndex});
            if (!buffer && !external)
                continue;
            if (!buffer) {
                SmallVector<NamedAttribute> attributes = {
                    builder.getNamedAttr("deterministic", builder.getBoolAttr(false))};
                const AutodiffExternalGradientOwnership ownership =
                    analysis.getStorageIdentities()[effect->identity].externalGradientOwnership;
                if (ownership == AutodiffExternalGradientOwnership::InvocationPrivate)
                    attributes.push_back(
                        builder.getNamedAttr(kAccumulationOwnershipAttrName,
                                             builder.getStringAttr(kInvocationPrivateAccumulationOwnership)));
                else if (ownership != AutodiffExternalGradientOwnership::AtomicShared)
                    return load.emitError("active external Storage gradient has no supported ownership proof");
                if (auto aggregate = dyn_cast<RankedTensorType>(seed.getType())) {
                    if (!aggregate.hasStaticShape())
                        return load.emitError("external aggregate Storage gradient has a dynamic element shape");
                    for (const SmallVector<int64_t> &coordinate : enumerateStaticCoordinates(aggregate.getShape())) {
                        SmallVector<Value> componentIndices(indices);
                        for (int64_t index : coordinate)
                            componentIndices.push_back(createIndexConstant(builder, load.getLoc(), index));
                        Value component = tensor::ExtractOp::create(
                            builder, load.getLoc(), seed, ValueRange(componentIndices).drop_front(indices.size()));
                        SmallVector<Value> operands = {component, external};
                        llvm::append_range(operands, componentIndices);
                        createOperation(builder, load.getLoc(), ScatterAddOp::getOperationName(), operands, {},
                                        attributes);
                    }
                } else {
                    SmallVector<Value> operands = {seed, external};
                    llvm::append_range(operands, indices);
                    createOperation(builder, load.getLoc(), ScatterAddOp::getOperationName(), operands, {}, attributes);
                }
                continue;
            }
            SmallVector<Value> operands = {buffer, seed};
            llvm::append_range(operands, indices);
            createOperation(builder, load.getLoc(), AdAdjointScatterAddOp::getOperationName(), operands);
        }
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
        SmallVector<Value> indices;
        for (Value index : store.getIndices()) {
            FailureOr<Value> materialized = materializePrimal(index, builder);
            if (failed(materialized))
                return store.emitError("cannot materialize a dynamic TensorView store index in reverse");
            indices.push_back(*materialized);
        }
        const AutodiffStorageEffect *effect = analysis.getStorageEffect(store);
        const ValueAbiLayout *layout = analysis.getValueAbi(store.getValue());
        if (!effect || !layout)
            return store.emitError("active TensorView store has no canonical Storage effect ABI");
        for (unsigned leafIndex = 0; leafIndex < layout->leaves.size(); ++leafIndex) {
            if (!analysis.isActive(store.getValue(), leafIndex))
                continue;
            Value buffer = storageAdjoints.lookup({effect->identity, leafIndex});
            FailureOr<Type> valueType = derivativeType(store.getValue(), leafIndex);
            if (failed(valueType))
                return store.emitError("cannot resolve TensorView store derivative leaf type");
            Value valueGradient;
            if (buffer) {
                SmallVector<Value> operands = {buffer};
                llvm::append_range(operands, indices);
                valueGradient = createOperation(builder, store.getLoc(), AdAdjointTakeAndClearOp::getOperationName(),
                                                operands, *valueType)
                                    ->getResult(0);
            } else if (Value cotangent = externalStorageCotangents.lookup({effect->identity, leafIndex})) {
                valueGradient = LoadOp::create(builder, store.getLoc(), *valueType, cotangent, indices);
            } else if (analysis.isActive(store.getStorage(), leafIndex)) {
                return store.emitError("active TensorView store leaf has no internal adjoint or output cotangent");
            } else {
                continue;
            }
            accumulate(builder, store.getLoc(), adjoints, store.getValue(), leafIndex, valueGradient);
        }
        return success();
    }

    LogicalResult reverseAdditiveRmw(Operation &operation, Value contribution, ValueRange sourceIndices, Value oldValue,
                                     OpBuilder &builder, AdjointMap &adjoints) {
        SmallVector<Value> indices;
        for (Value index : sourceIndices) {
            FailureOr<Value> materialized = materializePrimal(index, builder);
            if (failed(materialized))
                return operation.emitError("cannot materialize an additive Storage index in reverse");
            indices.push_back(*materialized);
        }
        const AutodiffStorageEffect *effect = analysis.getStorageEffect(&operation);
        const ValueAbiLayout *layout = analysis.getValueAbi(contribution);
        if (!effect || !layout)
            return operation.emitError("active additive Storage effect has no canonical effect ABI");
        for (unsigned leafIndex = 0; leafIndex < layout->leaves.size(); ++leafIndex) {
            if (!analysis.isActive(contribution, leafIndex) && !(oldValue && analysis.isActive(oldValue, leafIndex)))
                continue;
            FailureOr<Type> valueType = derivativeType(contribution, leafIndex);
            if (failed(valueType))
                return operation.emitError("cannot resolve additive contribution derivative leaf type");
            Value buffer = storageAdjoints.lookup({effect->identity, leafIndex});
            Value postCotangent;
            if (analysis.isActive(contribution, leafIndex) && buffer) {
                SmallVector<Value> operands = {buffer};
                llvm::append_range(operands, indices);
                postCotangent = createOperation(builder, operation.getLoc(), AdAdjointPeekOp::getOperationName(),
                                                operands, *valueType)
                                    ->getResult(0);
            } else if (Value cotangent = externalStorageCotangents.lookup({effect->identity, leafIndex});
                       analysis.isActive(contribution, leafIndex) && cotangent) {
                postCotangent = LoadOp::create(builder, operation.getLoc(), *valueType, cotangent, indices);
            } else if (analysis.isActive(contribution, leafIndex)) {
                return operation.emitError("active additive Storage contribution has no post-state cotangent");
            }
            if (postCotangent)
                accumulate(builder, operation.getLoc(), adjoints, contribution, leafIndex, postCotangent);

            if (!oldValue)
                continue;
            Value oldSeed = adjoints.lookup({oldValue, leafIndex});
            if (!oldSeed)
                continue;
            Value external = externalStorageAdjoints.lookup({effect->identity, leafIndex});
            if (!buffer && !external)
                return operation.emitError("active atomic old-value result has no pre-state Storage adjoint");
            if (buffer) {
                SmallVector<Value> operands = {buffer, oldSeed};
                llvm::append_range(operands, indices);
                createOperation(builder, operation.getLoc(), AdAdjointScatterAddOp::getOperationName(), operands);
            } else {
                SmallVector<Value> operands = {oldSeed, external};
                llvm::append_range(operands, indices);
                SmallVector<NamedAttribute> attributes = {
                    builder.getNamedAttr("deterministic", builder.getBoolAttr(false))};
                createOperation(builder, operation.getLoc(), ScatterAddOp::getOperationName(), operands, {},
                                attributes);
            }
        }
        return success();
    }

    LogicalResult reverseRule(Operation &operation, OpBuilder &builder, AdjointMap &adjoints) {
        const DifferentiationRule *rule = registry.lookup(&operation);
        if (!rule)
            return operation.emitError("active scalar operation has no VJP rule");
        SmallVector<unsigned> activeOperands;
        for (auto [operandIndex, operand] : llvm::enumerate(operation.getOperands()))
            if (analysis.hasAnyActiveLeaf(operand))
                activeOperands.push_back(static_cast<unsigned>(operandIndex));
        for (const AutodiffPrimalRequirement &requirement : rule->getVjpPrimalRequirements()) {
            if (!requirement.isRequiredFor(activeOperands))
                continue;
            Value source = requirement.kind == AutodiffPrimalKind::Operand ? operation.getOperand(requirement.index)
                                                                           : operation.getResult(requirement.index);
            if (availablePrimal(source, builder))
                continue;
            FailureOr<Value> materialized = materializePrimal(source, builder);
            if (failed(materialized))
                return operation.emitError("cannot rematerialize a rule-required primal value");
            primals.try_emplace(source, *materialized);
        }
        DenseMap<Value, Value> availablePrimals;
        for (Value operand : operation.getOperands())
            if (Value available = availablePrimal(operand, builder))
                availablePrimals.try_emplace(operand, available);
        for (Value result : operation.getResults())
            if (Value available = availablePrimal(result, builder))
                availablePrimals.try_emplace(result, available);
        return reverseScalarOperation(operation, builder, analysis, registry, availablePrimals, adjoints);
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
            if (auto barrier = dyn_cast<BarrierOp>(operation)) {
                BarrierOp::create(builder, barrier.getLoc(), barrier.getOrdering(), barrier.getScope());
                continue;
            }
            if (!analysis.isActive(&operation))
                continue;
            if (auto ifOp = dyn_cast<scf::IfOp>(operation)) {
                if (failed(reverseIf(ifOp, builder, adjoints, parentRegion, parentRecordIndex)))
                    return failure();
                continue;
            }
            if (auto forOp = dyn_cast<scf::ForOp>(operation)) {
                if (failed(reverseFor(forOp, builder, adjoints, parentRegion, parentRecordIndex)))
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
            if (auto reduce = dyn_cast<ReduceSumOp>(operation)) {
                if (failed(
                        reverseAdditiveRmw(operation, reduce.getValue(), reduce.getIndices(), {}, builder, adjoints)))
                    return failure();
                continue;
            }
            if (auto scatter = dyn_cast<ScatterAddOp>(operation)) {
                if (failed(
                        reverseAdditiveRmw(operation, scatter.getValue(), scatter.getIndices(), {}, builder, adjoints)))
                    return failure();
                continue;
            }
            if (auto atomic = dyn_cast<AtomicOp>(operation)) {
                if (failed(reverseAdditiveRmw(operation, atomic.getValue(), atomic.getIndices(), atomic.getResult(),
                                              builder, adjoints)))
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
        seeded |= hasActiveStorageEffect(source.getThenRegion()) || hasActiveStorageEffect(source.getElseRegion());
        if (!seeded)
            return success();
        const AutodiffTapeRegion *region = regions.lookup(source.getOperation());
        Value handle = parentRegion;
        Value recordIndex = parentRecordIndex;
        Value condition;
        if (region) {
            handle = readNestedRegion(builder, source.getLoc(), parentRegion, parentRecordIndex, region->childOrdinal);
            recordIndex = createIndexConstant(builder, source.getLoc(), 0);
            const AutodiffTapeField *predicate = findField(region->record.prefix, AutodiffTapeFieldKind::Predicate);
            if (!predicate)
                return source.emitError("scf.if reverse tape record has no predicate");
            condition = readLeaf(builder, source.getLoc(), handle, recordIndex, builder.getI1Type(),
                                 region->record.stride, region->record.alignment, predicate->offset);
        } else {
            FailureOr<Value> reconstructed = materializePrimal(source.getCondition(), builder);
            if (failed(reconstructed))
                return source.emitError("cannot reconstruct the selected scf.if predicate");
            condition = *reconstructed;
        }

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
            if (region &&
                failed(loadRegionPrimals(branchBuilder, *region, handle, recordIndex, savedPrimals, &sourceRegion)))
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
            if (failed(reverseBlock(sourceRegion.front(), branchBuilder, local, handle, recordIndex)))
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

    LogicalResult reverseFor(scf::ForOp source, OpBuilder &builder, AdjointMap &adjoints, Value parentRegion,
                             Value parentRecordIndex) {
        bool seeded = llvm::any_of(source.getResults(), [&](Value result) {
            const ValueAbiLayout *layout = analysis.getValueAbi(result);
            if (!layout)
                return false;
            return llvm::any_of(llvm::seq<unsigned>(0, layout->leaves.size()),
                                [&](unsigned leaf) { return adjoints.contains({result, leaf}); });
        });
        seeded |= hasActiveStorageEffect(source.getRegion());
        if (!seeded)
            return success();
        auto stepConstant = source.getStep().getDefiningOp<arith::ConstantIndexOp>();
        if (!stepConstant || stepConstant.value() <= 0)
            return source.emitError("structured VJP requires a canonical positive constant scf.for step");
        FailureOr<Value> lower = materializePrimal(source.getLowerBound(), builder);
        FailureOr<Value> upper = materializePrimal(source.getUpperBound(), builder);
        FailureOr<Value> step = materializePrimal(source.getStep(), builder);
        if (failed(lower) || failed(upper) || failed(step))
            return source.emitError("cannot reconstruct scf.for bounds in reverse");
        Value count = buildPositiveStepTripCount(builder, source.getLoc(), *lower, *upper, *step);

        const AutodiffTapeRegion *region = regions.lookup(source.getOperation());
        Value handle = parentRegion;
        if (region)
            handle = readNestedRegion(builder, source.getLoc(), parentRegion, parentRecordIndex, region->childOrdinal);
        auto yield = cast<scf::YieldOp>(source.getBody()->getTerminator());
        SmallVector<std::pair<unsigned, unsigned>> carriedIndices;
        for (auto [index, result] : llvm::enumerate(source.getResults())) {
            const ValueAbiLayout *layout = analysis.getValueAbi(result);
            if (!layout)
                continue;
            for (unsigned leafIndex = 0; leafIndex < layout->leaves.size(); ++leafIndex)
                if (analysis.isActive(result, leafIndex))
                    carriedIndices.emplace_back(index, leafIndex);
        }
        SmallVector<AdjointKey> external = externalActiveLeaves(source.getRegion());
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
        scf::YieldOp reverseYield = body->empty() ? scf::YieldOp{} : dyn_cast<scf::YieldOp>(body->back());
        OpBuilder bodyBuilder = reverseYield ? OpBuilder(reverseYield) : OpBuilder::atBlockEnd(body);
        Value last = arith::SubIOp::create(bodyBuilder, source.getLoc(), count, one);
        Value iteration = arith::SubIOp::create(bodyBuilder, source.getLoc(), last, reverse.getInductionVar());
        Value offset = arith::MulIOp::create(bodyBuilder, source.getLoc(), iteration, *step);
        Value primalInduction = arith::AddIOp::create(bodyBuilder, source.getLoc(), *lower, offset);
        Value previousInduction = primals.lookup(source.getInductionVar());
        primals[source.getInductionVar()] = primalInduction;

        SmallVector<std::pair<Value, Value>> savedPrimals;
        if (region && failed(loadRegionPrimals(bodyBuilder, *region, handle, iteration, savedPrimals)))
            return failure();
        AdjointMap local;
        unsigned position = 0;
        for (auto [index, leafIndex] : carriedIndices)
            local[{yield.getOperand(index), leafIndex}] = reverse.getRegionIterArgs()[position++];
        for (AdjointKey value : external)
            local[value] = reverse.getRegionIterArgs()[position++];
        if (failed(reverseBlock(*source.getBody(), bodyBuilder, local, handle, iteration)))
            return failure();
        SmallVector<Value> next;
        for (auto [index, leafIndex] : carriedIndices) {
            Value argument = source.getRegionIterArgs()[index];
            Value contribution = local.lookup({argument, leafIndex});
            next.push_back(contribution ? contribution : zeroFor(bodyBuilder, source.getLoc(), argument, leafIndex));
        }
        for (auto [value, leafIndex] : external) {
            Value contribution = local.lookup({value, leafIndex});
            next.push_back(contribution ? contribution : zeroFor(bodyBuilder, source.getLoc(), value, leafIndex));
        }
        if (reverseYield)
            reverseYield->setOperands(next);
        else
            scf::YieldOp::create(bodyBuilder, source.getLoc(), next);
        restorePrimals(savedPrimals);
        if (previousInduction)
            primals[source.getInductionVar()] = previousInduction;
        else
            primals.erase(source.getInductionVar());

        builder.setInsertionPointAfter(reverse);
        position = 0;
        for (auto [index, leafIndex] : carriedIndices)
            accumulate(builder, source.getLoc(), adjoints, source.getInitArgs()[index], leafIndex,
                       reverse.getResult(position++));
        for (auto [value, leafIndex] : external)
            accumulate(builder, source.getLoc(), adjoints, value, leafIndex, reverse.getResult(position++));
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
        seeded |= hasActiveStorageEffect(source.getBefore()) || hasActiveStorageEffect(source.getAfter());
        if (!seeded)
            return success();
        const AutodiffTapeRegion *region = regions.lookup(source.getOperation());
        if (!region)
            return source.emitError("active scf.while has no reverse tape region");
        auto yield = cast<scf::YieldOp>(source.getAfter().front().getTerminator());

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
    const ResidualRootLayout &rootLayout;
    Value rootRegion;
    DenseMap<Value, Value> primals;
    DenseMap<Operation *, const AutodiffTapeRegion *> regions;
    DenseMap<std::pair<unsigned, unsigned>, Value> storageAdjoints;
    DenseMap<std::pair<unsigned, unsigned>, Value> externalStorageAdjoints;
    DenseMap<std::pair<unsigned, unsigned>, Value> externalStorageCotangents;
};

SmallVector<BlockArgument> requiredPrimalArguments(const VernonAutodiffTapePlan &plan) {
    SmallVector<BlockArgument> result;
    DenseSet<Value> seen;
    auto append = [&](Value value) {
        auto argument = dyn_cast<BlockArgument>(value);
        if (!argument)
            return;
        auto function = dyn_cast_or_null<func::FuncOp>(argument.getOwner()->getParentOp());
        if (!function || argument.getOwner() != &function.getBody().front())
            return;
        if (function.getArgAttr(argument.getArgNumber(), kBuiltinAttrName) || !seen.insert(argument).second)
            return;
        result.push_back(argument);
    };
    for (const AdResidualSourceSelection &selection : plan.getMemoryPlan().getSourceSelections()) {
        if (selection.selectedCandidate >= selection.candidates.size())
            continue;
        const AdResidualSource &source = selection.candidates[selection.selectedCandidate];
        if (source.kind == AdResidualSourceKind::PrimalArgument)
            append(selection.key.value);
        if (source.kind == AdResidualSourceKind::ExactVersionReload)
            if (auto load = selection.key.value.getDefiningOp<LoadOp>())
                append(load.getStorage());
        if (source.kind == AdResidualSourceKind::PureRematerialization)
            for (Operation *operation : source.recipe)
                for (Value operand : operation->getOperands())
                    append(operand);
    }
    llvm::sort(result, [](BlockArgument left, BlockArgument right) {
        auto function = cast<func::FuncOp>(left.getOwner()->getParentOp());
        auto leftAttr = function.getArgAttrOfType<StringAttr>(left.getArgNumber(), "vernon.source_name");
        auto rightAttr = function.getArgAttrOfType<StringAttr>(right.getArgNumber(), "vernon.source_name");
        StringRef leftName = leftAttr ? leftAttr.getValue() : StringRef{};
        StringRef rightName = rightAttr ? rightAttr.getValue() : StringRef{};
        return std::tuple(leftName, left.getArgNumber()) < std::tuple(rightName, right.getArgNumber());
    });
    return result;
}

bool requiresLogicalTape(const VernonAutodiffTapePlan &plan) {
    if (!plan.getRegions().empty())
        return true;
    return llvm::any_of(plan.getMemoryPlan().getSourceSelections(), [](const AdResidualSourceSelection &selection) {
        if (selection.selectedCandidate >= selection.candidates.size())
            return true;
        AdResidualSourceKind kind = selection.candidates[selection.selectedCandidate].kind;
        return kind == AdResidualSourceKind::StaticCapture || kind == AdResidualSourceKind::DynamicCapture;
    });
}

FailureOr<func::FuncOp> createStructuredForward(func::FuncOp primal, StringRef symbol,
                                                const VernonAutodiffAnalysisResult &analysis,
                                                const VernonAutodiffTapePlan &plan,
                                                const ResidualRootLayout &rootLayout, bool usesTape) {
    MLIRContext *context = primal.getContext();
    if (!usesTape) {
        OpBuilder moduleBuilder(primal);
        auto forward = cast<func::FuncOp>(primal.clone());
        forward.setSymName(symbol);
        moduleBuilder.insert(forward);
        return forward;
    }
    Type tapeType = AdTapeType::get(context);
    Type regionType = AdRegionHeaderType::get(context);
    TupleType resultType = TupleType::get(context, {tapeType, regionType});
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
    for (unsigned index = 0; index < forward.getNumArguments(); ++index)
        if (!forward.getArgAttr(index, kBuiltinAttrName))
            forward.setArgAttr(index, "vernon.autodiff_role", StringAttr::get(context, "primal"));
    SmallVector<DictionaryAttr> forwardResultAttrs = {makeInterfaceAttrs(context, "output", "forward_state", {}, 0)};
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
    ForwardEmitterCore emitter(primal, analysis, plan, tape, root, rootRecord);
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
    createOperation(captureBuilder, primal.getLoc(), AdEndRegionOp::getOperationName(),
                    {root, createIndexConstant(captureBuilder, primal.getLoc(), 1),
                     createI32Constant(captureBuilder, primal.getLoc(), 0)});
    createOperation(captureBuilder, primal.getLoc(), AdCaptureYieldOp::getOperationName(), {tape, root});

    Value result = createTuple(builder, primal.getLoc(), resultType, {capture.getTape(), capture.getRootRegion()});
    func::ReturnOp::create(builder, primal.getLoc(), result);
    committed = true;
    return forward;
}

FailureOr<func::FuncOp> createStructuredBackward(func::FuncOp primal, StringRef symbol,
                                                 const VernonAutodiffAnalysisResult &analysis,
                                                 const VernonAutodiffRuleRegistry &registry,
                                                 const VernonAutodiffTapePlan &plan,
                                                 const ResidualRootLayout &rootLayout, bool usesTape) {
    MLIRContext *context = primal.getContext();
    Type tapeType = AdTapeType::get(context);
    Type regionType = AdRegionHeaderType::get(context);
    BackwardProfileTypes profileTypes = getBackwardProfileTypes(context, analysis);
    OpBuilder moduleBuilder(primal);
    SmallVector<Type> backwardArguments;
    if (usesTape) {
        backwardArguments.push_back(tapeType);
        backwardArguments.push_back(regionType);
    }
    SmallVector<BlockArgument> builtinArguments;
    for (BlockArgument argument : primal.getArguments())
        if (primal.getArgAttr(argument.getArgNumber(), kBuiltinAttrName)) {
            builtinArguments.push_back(argument);
            backwardArguments.push_back(argument.getType());
        }
    SmallVector<BlockArgument> primalArguments = requiredPrimalArguments(plan);
    for (BlockArgument argument : primalArguments)
        backwardArguments.push_back(argument.getType());
    llvm::append_range(backwardArguments, profileTypes.shapeSources);
    llvm::append_range(backwardArguments, profileTypes.cotangents);
    llvm::append_range(backwardArguments, profileTypes.storageGradients);
    SmallVector<Type> backwardResults;
    if (profileTypes.result)
        backwardResults.push_back(profileTypes.result);
    auto backward = func::FuncOp::create(moduleBuilder, primal.getLoc(), symbol,
                                         FunctionType::get(context, backwardArguments, backwardResults));
    bool committed = false;
    auto cleanup = llvm::make_scope_exit([&] {
        if (!committed)
            backward.erase();
    });
    copyProfileFunctionAttrs(primal, backward);
    backward->removeAttr("vernon.storage_effects");
    SmallVector<DictionaryAttr> backwardArgumentAttrs;
    if (usesTape) {
        backwardArgumentAttrs.push_back(makeInterfaceAttrs(context, "input", "tape", {}, 0));
        backwardArgumentAttrs.push_back(makeInterfaceAttrs(context, "input", "tape_region", {}, 1));
    }
    for (BlockArgument argument : builtinArguments)
        backwardArgumentAttrs.push_back(primal.getArgAttrDict(argument.getArgNumber()));
    const unsigned primalBase = (usesTape ? 2 : 0) + builtinArguments.size();
    for (auto [index, argument] : llvm::enumerate(primalArguments)) {
        auto sourceName = primal.getArgAttrOfType<StringAttr>(argument.getArgNumber(), "vernon.source_name");
        if (!sourceName || sourceName.getValue().empty())
            return primal.emitError("required backward primal argument has no source name");
        backwardArgumentAttrs.push_back(
            makeInterfaceAttrs(context, "input", ("primal." + sourceName.getValue()).str(),
                               languageLeafDtypes(primal.getArgAttrDict(argument.getArgNumber()), argument.getType()),
                               primalBase + index, argument.getType(), "retained_primal", sourceName.getValue()));
    }
    const unsigned shapeBase = primalBase + primalArguments.size();
    unsigned shapeSourceIndex = 0;
    for (const AutodiffStorageIdentity &identity : analysis.getStorageIdentities()) {
        if (!hasExternalStorageShapeSource(identity))
            continue;
        auto sourceName = primal.getArgAttrOfType<StringAttr>(cast<BlockArgument>(identity.binding).getArgNumber(),
                                                              "vernon.source_name");
        if (!sourceName || sourceName.getValue().empty())
            return primal.emitError("Storage identity has no source name for its backward shape source");
        backwardArgumentAttrs.push_back(makeInterfaceAttrs(
            context, "input", ("shape." + sourceName.getValue()).str(),
            languageLeafDtypes(primal.getArgAttrDict(cast<BlockArgument>(identity.binding).getArgNumber()),
                               identity.binding.getType()),
            shapeBase + shapeSourceIndex, identity.binding.getType(), "retained_primal", sourceName.getValue()));
        ++shapeSourceIndex;
    }
    const unsigned cotangentBase = shapeBase + profileTypes.shapeSources.size();
    for (auto [index, leaf] : llvm::enumerate(analysis.getActiveResultLeaves()))
        backwardArgumentAttrs.push_back(makeInterfaceAttrs(context, "input", leaf.path,
                                                           {profileTypes.cotangentDtypes[index]}, cotangentBase + index,
                                                           profileTypes.cotangents[index], "cotangent", leaf.rootPath));
    const unsigned gradientBase = cotangentBase + profileTypes.cotangents.size();
    unsigned storageGradientIndex = 0;
    for (auto [index, leaf] : llvm::enumerate(analysis.getWrtLeaves()))
        if (isa<TensorViewType>(leaf.value.getType())) {
            backwardArgumentAttrs.push_back(makeInterfaceAttrs(
                context, "input", leaf.path, {profileTypes.gradientDtypes[index]}, gradientBase + storageGradientIndex,
                profileTypes.storageGradients[storageGradientIndex], "gradient", leaf.rootPath));
            ++storageGradientIndex;
        }
    backward.setAllArgAttrs(backwardArgumentAttrs);
    for (const AutodiffStorageIdentity &identity : analysis.getStorageIdentities()) {
        if (!identity.externalGradientDestination)
            continue;
        StringRef ownership = identity.externalGradientOwnership == AutodiffExternalGradientOwnership::InvocationPrivate
                                  ? "invocation_private"
                              : identity.externalGradientOwnership == AutodiffExternalGradientOwnership::AtomicShared
                                  ? "atomic_shared"
                              : identity.externalGradientOwnership == AutodiffExternalGradientOwnership::WorkgroupShared
                                  ? "workgroup_shared"
                                  : "none";
        unsigned tensorGradientIndex = 0;
        for (const AutodiffLeaf &leaf : analysis.getWrtLeaves()) {
            if (!isa<TensorViewType>(leaf.value.getType()))
                continue;
            if (leaf.value == identity.binding)
                backward.setArgAttr(gradientBase + tensorGradientIndex, kAccumulationOwnershipAttrName,
                                    StringAttr::get(context, ownership));
            ++tensorGradientIndex;
        }
    }
    if (profileTypes.result) {
        NamedAttrList gradientResultAttrs(
            makeInterfaceAttrs(context, "output", "gradients", profileTypes.valueGradientDtypes, 0));
        SmallVector<Attribute> gradientPaths;
        for (const AutodiffLeaf &leaf : analysis.getWrtLeaves())
            if (!isa<TensorViewType>(leaf.value.getType()))
                gradientPaths.push_back(StringAttr::get(context, leaf.path));
        gradientResultAttrs.set("vernon.autodiff_gradient_paths", ArrayAttr::get(context, gradientPaths));
        SmallVector<DictionaryAttr> resultAttrs = {gradientResultAttrs.getDictionary(context)};
        backward.setAllResultAttrs(resultAttrs);
    }

    Block *entry = backward.addEntryBlock();
    OpBuilder builder = OpBuilder::atBlockEnd(entry);
    Value rootRegion = usesTape ? entry->getArgument(1) : Value{};
    ReverseEmitterCore emitter(primal, analysis, registry, plan, rootLayout, rootRegion);
    const unsigned builtinBase = usesTape ? 2 : 0;
    emitter.initializeBuiltinPrimals(builtinArguments,
                                     entry->getArguments().slice(builtinBase, builtinArguments.size()));
    emitter.initializePrimalArguments(primalArguments, entry->getArguments().slice(primalBase, primalArguments.size()));
    if (failed(emitter.initializePrimals(builder)))
        return failure();
    AdjointMap adjoints;
    for (auto [index, leaf] : llvm::enumerate(analysis.getActiveResultLeaves()))
        adjoints.try_emplace(AdjointKey{leaf.value, leaf.abiLeafIndex}, entry->getArgument(cotangentBase + index));
    if (failed(emitter.initializeStorageAdjoints(
            builder, adjoints, entry->getArguments().slice(shapeBase, profileTypes.shapeSources.size()),
            entry->getArguments().slice(gradientBase, profileTypes.storageGradients.size()))))
        return failure();
    const bool synchronizeWorkgroup = requiresWorkgroupBackwardSynchronization(analysis);
    if (synchronizeWorkgroup)
        BarrierOp::create(builder, primal.getLoc(), builder.getStringAttr("acquire_release"),
                          builder.getStringAttr("workgroup"));
    if (failed(emitter.reverseTopLevel(builder, adjoints)))
        return failure();
    if (synchronizeWorkgroup)
        BarrierOp::create(builder, primal.getLoc(), builder.getStringAttr("acquire_release"),
                          builder.getStringAttr("workgroup"));
    SmallVector<Value> valueGradients;
    storageGradientIndex = 0;
    for (const AutodiffLeaf &leaf : analysis.getWrtLeaves()) {
        Value gradient;
        if (isa<TensorViewType>(leaf.value.getType())) {
            if (failed(emitter.storeStorageAdjoint(builder, leaf,
                                                   entry->getArgument(gradientBase + storageGradientIndex++))))
                return primal.emitError("cannot store wrt Storage adjoint buffer");
        } else {
            gradient = adjoints.lookup({leaf.value, leaf.abiLeafIndex});
            valueGradients.push_back(gradient ? gradient : createZero(builder, primal.getLoc(), leaf.derivativeType));
        }
    }
    if (valueGradients.empty()) {
        func::ReturnOp::create(builder, primal.getLoc());
    } else {
        Value result = valueGradients.size() == 1 ? valueGradients.front()
                                                  : createTuple(builder, primal.getLoc(),
                                                                cast<TupleType>(profileTypes.result), valueGradients);
        func::ReturnOp::create(builder, primal.getLoc(), result);
    }
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
        if (selected.outputPaths.empty())
            selected.outputPaths.assign(outputs.begin(), outputs.end());
        if (selected.forwardSymbol.empty())
            selected.forwardSymbol = forward;
        if (selected.backwardSymbol.empty())
            selected.backwardSymbol = backward;
        if (selected.wrtPaths.empty() || selected.outputPaths.empty() || selected.forwardSymbol.empty() ||
            selected.backwardSymbol.empty()) {
            getOperation().emitError(
                "structured VJP pass requires wrt paths, Storage output paths, and forward/backward symbols");
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
    ListOption<std::string> outputs{*this, "outputs", llvm::cl::desc("Canonical Storage output paths"),
                                    llvm::cl::ZeroOrMore};
    Option<std::string> forward{*this, "forward", llvm::cl::desc("Augmented forward symbol"), llvm::cl::init("")};
    Option<std::string> backward{*this, "backward", llvm::cl::desc("Reverse symbol"), llvm::cl::init("")};
};

} // namespace

FailureOr<StructuredVjpResult> buildStructuredVjp(func::FuncOp primal, const StructuredVjpOptions &options) {
    if (!primal || !primal->getParentOfType<ModuleOp>())
        return failure();
    if (options.wrtPaths.empty() || options.outputPaths.empty() || options.forwardSymbol.empty() ||
        options.backwardSymbol.empty())
        return primal.emitError("structured VJP requires wrt paths, Storage output paths, and profile symbols");
    if (options.forwardSymbol == options.backwardSymbol)
        return primal.emitError("structured VJP requires distinct profile symbols");
    llvm::StringSet<> uniqueWrtPaths;
    for (const std::string &path : options.wrtPaths)
        if (path.empty() || !uniqueWrtPaths.insert(path).second)
            return primal.emitError("structured VJP requires unique non-empty wrt paths");
    llvm::StringSet<> uniqueOutputPaths;
    for (const std::string &path : options.outputPaths)
        if (path.empty() || !uniqueOutputPaths.insert(path).second)
            return primal.emitError("structured VJP requires unique non-empty Storage output paths");
    ModuleOp module = primal->getParentOfType<ModuleOp>();
    if (module.lookupSymbol(options.forwardSymbol) || module.lookupSymbol(options.backwardSymbol))
        return primal.emitError("structured VJP profile symbol already exists");

    func::FuncOp structuredPrimal = primal;

    SmallVector<StringRef> wrtPaths;
    for (const std::string &path : options.wrtPaths)
        wrtPaths.push_back(path);
    SmallVector<StringRef> outputPaths;
    for (const std::string &path : options.outputPaths)
        outputPaths.push_back(path);
    VernonAutodiffRuleRegistry registry = createDefaultAutodiffRuleRegistry();
    FailureOr<VernonAutodiffAnalysisResult> analysis =
        analyzeAutodiffFunction(structuredPrimal, wrtPaths, outputPaths, registry);
    if (failed(analysis) || failed(validateStructuredPhase(structuredPrimal, *analysis)))
        return failure();
    FailureOr<VernonAutodiffTapePlan> plan = planAutodiffTape(structuredPrimal, *analysis, registry);
    if (failed(plan))
        return failure();
    for (const AutodiffTapeRegion &region : plan->getRegions())
        if (region.record.stride > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
            return structuredPrimal.emitError("structured VJP dynamic tape record is not representable");

    FailureOr<ResidualRootLayout> rootLayout = buildDynamicRootLayout(*plan);
    const bool staticResiduals = plan->getRegions().empty();
    const bool usesTape = requiresLogicalTape(*plan);
    if (staticResiduals)
        rootLayout = buildStaticRootLayout(*plan);
    if (failed(rootLayout))
        return structuredPrimal.emitError("structured VJP root record layout overflow");
    FailureOr<func::FuncOp> forward =
        createStructuredForward(structuredPrimal, options.forwardSymbol, *analysis, *plan, *rootLayout, usesTape);
    if (failed(forward))
        return failure();
    FailureOr<func::FuncOp> backward = createStructuredBackward(structuredPrimal, options.backwardSymbol, *analysis,
                                                                registry, *plan, *rootLayout, usesTape);
    if (failed(backward)) {
        forward->erase();
        return failure();
    }
    if (failed(verify(*forward)) || failed(verify(*backward))) {
        forward->erase();
        backward->erase();
        return structuredPrimal.emitError("generated structured VJP profile failed verification");
    }
    const StringAttr residualStorage = StringAttr::get(primal.getContext(), !usesTape         ? "none"
                                                                            : staticResiduals ? "static"
                                                                                              : "dynamic");
    (*forward)->setAttr("vernon.ad.residual_storage", residualStorage);
    (*backward)->setAttr("vernon.ad.residual_storage", residualStorage);
    const uint64_t activeOperationCount = llvm::count_if(
        analysis->getOperations(), [](const AutodiffOperationActivity &activity) { return activity.active; });
    uint64_t recomputationCost = 0;
    for (const AdRematerializationRecipe &recipe : plan->getMemoryPlan().getRematerializations()) {
        if (recipe.estimatedCost > std::numeric_limits<uint64_t>::max() - recomputationCost) {
            forward->erase();
            backward->erase();
            return structuredPrimal.emitError("structured VJP recomputation telemetry overflows");
        }
        recomputationCost += recipe.estimatedCost;
    }
    (*backward)->setAttr("vernon.ad.active_operation_count",
                         IntegerAttr::get(IntegerType::get(primal.getContext(), 64), activeOperationCount));
    (*backward)->setAttr("vernon.ad.recomputation_cost",
                         IntegerAttr::get(IntegerType::get(primal.getContext(), 64), recomputationCost));
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
    if (!staticResiduals &&
        plan->getStaticTapeBytesHint() > std::numeric_limits<uint64_t>::max() - rootLayout->stride) {
        forward->erase();
        backward->erase();
        return structuredPrimal.emitError("structured VJP tape statistics hint overflow");
    }
    const uint64_t tapeBytes = !usesTape         ? 0
                               : staticResiduals ? rootLayout->stride
                                                 : plan->getStaticTapeBytesHint() + rootLayout->stride;
    SmallVector<std::string> requiredPrimalPaths;
    for (BlockArgument argument : requiredPrimalArguments(*plan)) {
        auto sourceName = primal.getArgAttrOfType<StringAttr>(argument.getArgNumber(), "vernon.source_name");
        if (!sourceName || sourceName.getValue().empty()) {
            forward->erase();
            backward->erase();
            return primal.emitError("required backward primal argument has no source name");
        }
        requiredPrimalPaths.push_back(("primal." + sourceName.getValue()).str());
    }
    SmallVector<std::pair<std::string, uint64_t>> sourceKindCounts;
    for (const AdResidualSourceSelection &selection : plan->getMemoryPlan().getSourceSelections()) {
        if (selection.selectedCandidate >= selection.candidates.size())
            continue;
        std::string kind = stringifyAdResidualSourceKind(selection.candidates[selection.selectedCandidate].kind).str();
        auto existing = llvm::find_if(sourceKindCounts, [&](const auto &entry) { return entry.first == kind; });
        if (existing == sourceKindCounts.end())
            sourceKindCounts.emplace_back(std::move(kind), 1);
        else
            ++existing->second;
    }
    llvm::sort(sourceKindCounts, [](const auto &left, const auto &right) { return left.first < right.first; });
    const AdPlanCostComponents &costs = plan->getMemoryPlan().getCostComponents();
    SmallVector<std::pair<std::string, uint64_t>> costComponents = {
        {"backward_load_bytes", costs.backwardLoadBytes},     {"capture_store_bytes", costs.captureStoreBytes},
        {"checkpoint_copy_bytes", costs.checkpointCopyBytes}, {"graph_replay_cost", costs.graphReplayCost},
        {"recomputation_cost", costs.recomputationCost},      {"resource_reload_cost", costs.resourceReloadCost},
        {"retained_tape_bytes", costs.retainedTapeBytes},
    };
    return StructuredVjpResult{*forward,
                               *backward,
                               tapeBytes,
                               std::move(derivativeRules),
                               std::move(requiredPrimalPaths),
                               std::move(sourceKindCounts),
                               std::move(costComponents),
                               plan->getMemoryPlan().getSelectedPolicy().str(),
                               usesTape && staticResiduals && plan->getMemoryPlan().permitsWholeDispatchRetention()};
}

std::unique_ptr<Pass> createVernonStructuredVjpPass() { return std::make_unique<VernonStructuredVjpPass>(); }

std::unique_ptr<Pass> createVernonStructuredVjpPass(StructuredVjpOptions options) {
    return std::make_unique<VernonStructuredVjpPass>(std::move(options));
}

void registerVernonStructuredVjpPass() { PassRegistration<VernonStructuredVjpPass>(); }

} // namespace mlir::vernon
