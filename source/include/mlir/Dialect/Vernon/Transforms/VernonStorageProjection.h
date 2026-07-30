#ifndef MLIR_DIALECT_VERNON_TRANSFORMS_VERNONSTORAGEPROJECTION_H
#define MLIR_DIALECT_VERNON_TRANSFORMS_VERNONSTORAGEPROJECTION_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vernon/IR/Vernon.h"
#include "mlir/Dialect/Vernon/IR/VernonAttrs.h"
#include "mlir/Dialect/Vernon/IR/VernonValueAbi.h"
#include "mlir/IR/BuiltinOps.h"

#include <limits>

namespace mlir::vernon {

/// Convert a logical ranked TensorView index to its canonical physical record
/// index. Device views must carry their specialization layout. Non-device
/// views are statically shaped and use canonical row-major projection.
inline FailureOr<Value> projectTensorViewIndex(Operation *operation, TensorViewType view, ValueRange indices,
                                               OpBuilder &builder) {
    if (!view || indices.size() != view.getShape().size())
        return failure();
    SmallVector<int64_t> strides;
    int64_t offset = 0;
    DenseI64ArrayAttr operationStrides;
    IntegerAttr operationOffset;
    Value storage = isa<StoreOp>(operation) ? operation->getOperand(1) : operation->getOperand(0);
    if (auto argument = dyn_cast<BlockArgument>(storage)) {
        if (auto function = dyn_cast_or_null<func::FuncOp>(argument.getOwner()->getParentOp())) {
            operationStrides =
                function.getArgAttrOfType<DenseI64ArrayAttr>(argument.getArgNumber(), kTensorStridesAttrName);
            operationOffset = function.getArgAttrOfType<IntegerAttr>(argument.getArgNumber(), kTensorOffsetAttrName);
        }
    }
    if (operationStrides || operationOffset) {
        if (!operationStrides || !operationOffset || operationStrides.size() != indices.size() ||
            operationOffset.getInt() < 0)
            return failure();
        strides.append(operationStrides.asArrayRef().begin(), operationStrides.asArrayRef().end());
        offset = operationOffset.getInt();
    } else {
        if (view.getAddressSpace() == "device")
            return failure();
        int64_t stride = 1;
        strides.resize(view.getShape().size());
        for (size_t dimension = view.getShape().size(); dimension-- > 0;) {
            if (view.getShape()[dimension] <= 0)
                return failure();
            strides[dimension] = stride;
            if (stride > std::numeric_limits<int64_t>::max() / view.getShape()[dimension])
                return failure();
            stride *= view.getShape()[dimension];
        }
    }
    Location location = operation->getLoc();
    Value projected = arith::ConstantIndexOp::create(builder, location, offset);
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

/// Materialize logical-to-physical projection while TensorView types and
/// function argument layout metadata are still available. Backends then
/// consume only the first index and cannot diverge in projection behavior.
inline LogicalResult materializeTensorViewProjections(Operation *root) {
    SmallVector<Operation *> operations;
    root->walk([&](Operation *operation) {
        if (isa<LoadOp, StoreOp, AtomicOp>(operation))
            operations.push_back(operation);
    });
    for (Operation *operation : operations) {
        if (operation->hasAttr(kPhysicalIndexAttrName))
            continue;
        Value storage = isa<StoreOp>(operation) ? operation->getOperand(1) : operation->getOperand(0);
        auto view = dyn_cast<TensorViewType>(storage.getType());
        if (!view)
            return operation->emitError("storage projection requires a TensorView operand");
        if (view.getAddressSpace() != "device" && view.getElementType().isIntOrFloat())
            continue;
        ValueRange indices = isa<LoadOp>(operation)    ? cast<LoadOp>(operation).getIndices()
                             : isa<StoreOp>(operation) ? cast<StoreOp>(operation).getIndices()
                                                       : cast<AtomicOp>(operation).getIndices();
        OpBuilder builder(operation);
        FailureOr<Value> projected = projectTensorViewIndex(operation, view, indices, builder);
        if (failed(projected))
            return operation->emitError("cannot resolve canonical TensorView projection");
        SmallVector<Value> physicalIndices{*projected};
        while (physicalIndices.size() < indices.size())
            physicalIndices.push_back(arith::ConstantIndexOp::create(builder, operation->getLoc(), 0));
        if (auto load = dyn_cast<LoadOp>(operation))
            load.getIndicesMutable().assign(physicalIndices);
        else if (auto store = dyn_cast<StoreOp>(operation))
            store.getIndicesMutable().assign(physicalIndices);
        else
            cast<AtomicOp>(operation).getIndicesMutable().assign(physicalIndices);
        operation->setAttr(kPhysicalIndexAttrName, builder.getUnitAttr());
    }
    return success();
}

} // namespace mlir::vernon

#endif
