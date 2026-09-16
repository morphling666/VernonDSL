#ifndef MLIR_DIALECT_VERNON_TRANSFORMS_VERNONSTORAGEPROJECTION_H
#define MLIR_DIALECT_VERNON_TRANSFORMS_VERNONSTORAGEPROJECTION_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vernon/IR/Vernon.h"
#include "mlir/Dialect/Vernon/IR/VernonAttrs.h"
#include "mlir/Dialect/Vernon/IR/VernonMetadataAbi.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#include <limits>

namespace mlir::vernon {

inline LogicalResult appendTensorViewMetadataArgument(Operation *root) {
    SmallVector<func::FuncOp> functions;
    root->walk([&](func::FuncOp function) { functions.push_back(function); });
    for (func::FuncOp function : functions) {
        FailureOr<std::shared_ptr<const SemanticMetadataPlan>> semantic = getSemanticMetadataPlan(function);
        if (failed(semantic))
            return function.emitError("cannot build canonical TensorView metadata plan");
        std::optional<unsigned> carrierIndex;
        for (unsigned index = 0; index < function.getNumArguments(); ++index) {
            if (!function.getArgAttr(index, kTensorMetadataCarrierAttrName))
                continue;
            if (carrierIndex)
                return function.emitError("has multiple TensorView metadata carriers");
            carrierIndex = index;
        }
        if ((*semantic)->empty()) {
            if (carrierIndex)
                return function.emitError("has a metadata carrier but no device TensorView arguments");
            continue;
        }
        SmallVector<Type> fields((*semantic)->getFieldCount(), IndexType::get(function.getContext()));
        TupleType carrierType = TupleType::get(function.getContext(), fields);
        if (carrierIndex) {
            auto existing = dyn_cast<TupleType>(function.getArgumentTypes()[*carrierIndex]);
            auto count = function.getArgAttrOfType<IntegerAttr>(*carrierIndex, kTensorMetadataFieldCountAttrName);
            if (!existing || existing != carrierType || !count ||
                count.getInt() != static_cast<int64_t>((*semantic)->getFieldCount()))
                return function.emitError("has a malformed TensorView aggregate metadata carrier");
            continue;
        }
        OpBuilder builder(function);
        DictionaryAttr attrs = builder.getDictionaryAttr({
            builder.getNamedAttr(kTensorMetadataCarrierAttrName, builder.getUnitAttr()),
            builder.getNamedAttr(kTensorMetadataFieldCountAttrName,
                                 builder.getI64IntegerAttr((*semantic)->getFieldCount())),
        });
        SmallVector<DictionaryAttr> argumentAttrs;
        argumentAttrs.reserve(function.getNumArguments() + 1);
        for (unsigned index = 0; index < function.getNumArguments(); ++index) {
            DictionaryAttr existing = function.getArgAttrDict(index);
            argumentAttrs.push_back(existing ? existing : builder.getDictionaryAttr({}));
        }
        argumentAttrs.push_back(attrs);
        if (failed(
                function.insertArgument(function.getNumArguments(), carrierType, DictionaryAttr{}, function.getLoc())))
            return failure();
        function.setAllArgAttrs(argumentAttrs);
    }
    return success();
}

inline FailureOr<std::pair<Value, uint32_t>> tensorViewMetadataBase(Value storage) {
    auto view = dyn_cast<TensorViewType>(storage.getType());
    if (!view || view.getAddressSpace() != "device")
        return failure();
    auto argument = dyn_cast<BlockArgument>(storage);
    auto function =
        argument ? dyn_cast_or_null<FunctionOpInterface>(argument.getOwner()->getParentOp()) : FunctionOpInterface{};
    if (!function)
        return failure();
    FailureOr<std::shared_ptr<const SemanticMetadataPlan>> semantic = getSemanticMetadataPlan(function);
    if (failed(semantic))
        return failure();
    std::optional<unsigned> carrierIndex;
    for (unsigned index = 0; index < function.getNumArguments(); ++index)
        if (function.getArgAttr(index, kTensorMetadataCarrierAttrName)) {
            if (carrierIndex)
                return failure();
            carrierIndex = index;
        }
    if (!carrierIndex)
        return failure();
    auto owner = llvm::find_if((*semantic)->getViews(), [&](const SemanticMetadataView &candidate) {
        return candidate.argumentIndex == argument.getArgNumber();
    });
    if (owner == (*semantic)->getViews().end())
        return failure();
    return std::pair<Value, uint32_t>{function.getArgument(*carrierIndex), owner->firstFieldOrdinal};
}

inline FailureOr<Value> tensorViewMetadataField(Value storage, uint32_t ordinal, Location location,
                                                OpBuilder &builder) {
    FailureOr<std::pair<Value, uint32_t>> base = tensorViewMetadataBase(storage);
    if (failed(base))
        return failure();
    OperationState state(location, TupleGetOp::getOperationName());
    state.addOperands(base->first);
    state.addTypes(builder.getIndexType());
    state.addAttribute("index", builder.getI64IntegerAttr(base->second + ordinal));
    return builder.create(state)->getResult(0);
}

inline FailureOr<Value> tensorViewExtent(Value storage, unsigned axis, Location location, OpBuilder &builder) {
    auto view = dyn_cast<TensorViewType>(storage.getType());
    if (!view || axis >= view.getShape().size())
        return failure();
    if (view.getAddressSpace() == "device") {
        return tensorViewMetadataField(storage, 1 + axis, location, builder);
    }
    int64_t extent = view.getShape()[axis];
    if (extent <= 0)
        return failure();
    return arith::ConstantIndexOp::create(builder, location, extent).getResult();
}

inline LogicalResult materializeGetShape(GetShapeOp getShape, OpBuilder &builder) {
    Location location = getShape.getLoc();
    Type sourceType = getShape.getSource().getType();
    SmallVector<Value> extents;
    auto emitI32 = [&](int64_t extent) -> LogicalResult {
        if (extent <= 0 || extent > std::numeric_limits<int32_t>::max())
            return failure();
        extents.push_back(arith::ConstantIntOp::create(builder, location, static_cast<int32_t>(extent), 32));
        return success();
    };
    auto castIndexToI32 = [&](Value extent) {
        extents.push_back(arith::IndexCastOp::create(builder, location, builder.getI32Type(), extent));
    };
    if (auto view = dyn_cast<TensorViewType>(sourceType)) {
        for (unsigned axis = 0; axis < view.getShape().size(); ++axis) {
            FailureOr<Value> extent = tensorViewExtent(getShape.getSource(), axis, location, builder);
            if (failed(extent))
                return failure();
            castIndexToI32(*extent);
        }
    } else if (auto tensor = dyn_cast<RankedTensorType>(sourceType)) {
        for (int64_t axis = 0; axis < tensor.getRank(); ++axis) {
            if (tensor.isDynamicDim(axis)) {
                Value dim = tensor::DimOp::create(builder, location, getShape.getSource(), axis);
                castIndexToI32(dim);
            } else if (failed(emitI32(tensor.getDimSize(axis)))) {
                return failure();
            }
        }
    } else if (auto tensor = dyn_cast<TensorType>(sourceType)) {
        for (int64_t extent : tensor.getShape())
            if (failed(emitI32(extent)))
                return failure();
    } else {
        return failure();
    }
    Value packed = tensor::FromElementsOp::create(builder, location, getShape.getType(), extents);
    getShape.getResult().replaceAllUsesWith(packed);
    getShape.erase();
    return success();
}

/// Convert a logical ranked TensorView index to its canonical physical record
/// index. Device views consume the entry-scoped metadata carrier. Non-device
/// views are statically shaped and use canonical row-major projection.
inline FailureOr<Value> projectTensorViewIndex(Operation *operation, TensorViewType view, ValueRange indices,
                                               OpBuilder &builder) {
    if (!view || indices.size() != view.getShape().size())
        return failure();
    Value storage = isa<StoreOp>(operation) ? operation->getOperand(1) : operation->getOperand(0);
    if (view.getAddressSpace() == "device") {
        if (failed(tensorViewMetadataBase(storage)))
            return failure();
        FailureOr<Value> offsetValue = tensorViewMetadataField(storage, 0, operation->getLoc(), builder);
        if (failed(offsetValue))
            return failure();
        Value offset = *offsetValue;
        SmallVector<Value> strides;
        strides.reserve(view.getShape().size());
        const uint32_t strideBase = 1 + view.getShape().size();
        for (unsigned dimension = 0; dimension < view.getShape().size(); ++dimension) {
            FailureOr<Value> stride =
                tensorViewMetadataField(storage, strideBase + dimension, operation->getLoc(), builder);
            if (failed(stride))
                return failure();
            strides.push_back(*stride);
        }
        Value projected = offset;
        for (auto [index, stride] : llvm::zip_equal(indices, strides)) {
            Value term = arith::MulIOp::create(builder, operation->getLoc(), index, stride);
            projected = arith::AddIOp::create(builder, operation->getLoc(), projected, term);
        }
        return projected;
    }

    SmallVector<int64_t> strides(view.getShape().size());
    int64_t stride = 1;
    for (size_t dimension = view.getShape().size(); dimension-- > 0;) {
        if (view.getShape()[dimension] <= 0)
            return failure();
        strides[dimension] = stride;
        if (stride > std::numeric_limits<int64_t>::max() / view.getShape()[dimension])
            return failure();
        stride *= view.getShape()[dimension];
    }
    Location location = operation->getLoc();
    Value projected = arith::ConstantIndexOp::create(builder, location, 0);
    for (auto [index, stride] : llvm::zip_equal(indices, strides)) {
        Value term = index;
        if (stride != 1) {
            Value scale = arith::ConstantIndexOp::create(builder, location, stride);
            term = arith::MulIOp::create(builder, location, term, scale);
        }
        projected = arith::AddIOp::create(builder, location, projected, term);
    }
    return projected;
}

/// Replace logical storage operations with internal physical operations while
/// TensorView types and descriptor projection data are still available.
inline LogicalResult materializeTensorViewProjections(Operation *root) {
    if (failed(appendTensorViewMetadataArgument(root)))
        return failure();
    SmallVector<Operation *> operations;
    root->walk([&](Operation *operation) {
        if (isa<LoadOp, StoreOp, AtomicOp, GetShapeOp>(operation))
            operations.push_back(operation);
    });
    for (Operation *operation : operations) {
        if (auto getShape = dyn_cast<GetShapeOp>(operation)) {
            OpBuilder builder(operation);
            if (failed(materializeGetShape(getShape, builder)))
                return operation->emitError("cannot resolve Tensor or TensorView shape");
            continue;
        }
        Value storage = isa<StoreOp>(operation) ? operation->getOperand(1) : operation->getOperand(0);
        auto view = dyn_cast<TensorViewType>(storage.getType());
        if (!view)
            return operation->emitError("storage projection requires a TensorView operand");
        ValueRange indices = isa<LoadOp>(operation)    ? cast<LoadOp>(operation).getIndices()
                             : isa<StoreOp>(operation) ? cast<StoreOp>(operation).getIndices()
                                                       : cast<AtomicOp>(operation).getIndices();
        OpBuilder builder(operation);
        FailureOr<Value> projected = projectTensorViewIndex(operation, view, indices, builder);
        if (failed(projected))
            return operation->emitError("cannot resolve canonical TensorView projection");
        OperationState state(operation->getLoc(), isa<LoadOp>(operation)    ? PhysicalLoadOp::getOperationName()
                                                  : isa<StoreOp>(operation) ? PhysicalStoreOp::getOperationName()
                                                                            : PhysicalAtomicOp::getOperationName());
        if (auto store = dyn_cast<StoreOp>(operation))
            state.addOperands({store.getValue(), storage, *projected});
        else {
            state.addOperands({storage, *projected});
            if (auto atomic = dyn_cast<AtomicOp>(operation))
                state.addOperands(atomic.getValue());
        }
        state.addTypes(operation->getResultTypes());
        state.addAttributes(operation->getAttrs());
        Operation *physical = builder.create(state);
        operation->replaceAllUsesWith(physical);
        operation->erase();
    }
    return success();
}

} // namespace mlir::vernon

#endif
