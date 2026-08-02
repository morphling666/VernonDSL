#ifndef MLIR_DIALECT_VERNON_TRANSFORMS_VERNONSTORAGEPROJECTION_H
#define MLIR_DIALECT_VERNON_TRANSFORMS_VERNONSTORAGEPROJECTION_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vernon/IR/Vernon.h"
#include "mlir/Dialect/Vernon/IR/VernonAttrs.h"

#include <limits>

namespace mlir::vernon {

inline LogicalResult appendTensorViewDescriptorArguments(Operation *root) {
    SmallVector<func::FuncOp> functions;
    root->walk([&](func::FuncOp function) { functions.push_back(function); });
    for (func::FuncOp function : functions) {
        bool alreadyExpanded = false;
        unsigned descriptorBase = function.getNumArguments();
        for (unsigned index = 0; index < function.getNumArguments(); ++index) {
            const bool hasDescriptorAttribute = function.getArgAttr(index, kTensorDescriptorOwnerAttrName) ||
                                                function.getArgAttr(index, kTensorDescriptorComponentAttrName) ||
                                                function.getArgAttr(index, kTensorDescriptorDimensionAttrName);
            alreadyExpanded |= hasDescriptorAttribute;
            if (hasDescriptorAttribute && descriptorBase == function.getNumArguments())
                descriptorBase = index;
        }
        if (alreadyExpanded) {
            struct ExpectedDescriptor {
                unsigned owner;
                StringRef component;
                std::optional<unsigned> dimension;
            };
            SmallVector<ExpectedDescriptor> expected;
            for (unsigned index = 0; index < descriptorBase; ++index) {
                auto view = dyn_cast<TensorViewType>(function.getArgumentTypes()[index]);
                if (!view || view.getAddressSpace() != "device")
                    continue;
                expected.push_back({index, "offset", std::nullopt});
                for (unsigned dimension = 0; dimension < view.getShape().size(); ++dimension)
                    expected.push_back({index, "extent", dimension});
                for (unsigned dimension = 0; dimension < view.getShape().size(); ++dimension)
                    expected.push_back({index, "stride", dimension});
            }
            bool canonical = descriptorBase + expected.size() == function.getNumArguments();
            for (auto [offset, descriptor] : llvm::enumerate(expected)) {
                const unsigned index = descriptorBase + offset;
                if (index >= function.getNumArguments()) {
                    canonical = false;
                    break;
                }
                auto owner = function.getArgAttrOfType<IntegerAttr>(index, kTensorDescriptorOwnerAttrName);
                auto component = function.getArgAttrOfType<StringAttr>(index, kTensorDescriptorComponentAttrName);
                auto dimension = function.getArgAttrOfType<IntegerAttr>(index, kTensorDescriptorDimensionAttrName);
                canonical &= function.getArgumentTypes()[index].isIndex() && owner &&
                             owner.getInt() == descriptor.owner && component &&
                             component.getValue() == descriptor.component &&
                             ((!descriptor.dimension && !dimension) ||
                              (descriptor.dimension && dimension &&
                               dimension.getInt() == static_cast<int64_t>(*descriptor.dimension)));
            }
            if (!canonical)
                return function.emitError("has a malformed internal TensorView descriptor argument sequence");
            continue;
        }
        const unsigned sourceArgumentCount = function.getNumArguments();
        OpBuilder builder(function);
        auto append = [&](unsigned owner, StringRef component,
                          std::optional<unsigned> dimension = std::nullopt) -> LogicalResult {
            SmallVector<NamedAttribute> attributes{
                builder.getNamedAttr(kTensorDescriptorOwnerAttrName, builder.getI64IntegerAttr(owner)),
                builder.getNamedAttr(kTensorDescriptorComponentAttrName, builder.getStringAttr(component))};
            if (dimension)
                attributes.push_back(
                    builder.getNamedAttr(kTensorDescriptorDimensionAttrName, builder.getI64IntegerAttr(*dimension)));
            return function.insertArgument(function.getNumArguments(), builder.getIndexType(),
                                           builder.getDictionaryAttr(attributes), function.getLoc());
        };
        for (unsigned index = 0; index < sourceArgumentCount; ++index) {
            auto view = dyn_cast<TensorViewType>(function.getArgumentTypes()[index]);
            if (!view || view.getAddressSpace() != "device")
                continue;
            if (failed(append(index, "offset")))
                return failure();
            for (unsigned dimension = 0; dimension < view.getShape().size(); ++dimension)
                if (failed(append(index, "extent", dimension)))
                    return failure();
            for (unsigned dimension = 0; dimension < view.getShape().size(); ++dimension)
                if (failed(append(index, "stride", dimension)))
                    return failure();
        }
    }
    return success();
}

/// Convert a logical ranked TensorView index to its canonical physical record
/// index. Device views consume per-dispatch descriptor operands. Non-device
/// views are statically shaped and use canonical row-major projection.
inline FailureOr<Value> projectTensorViewIndex(Operation *operation, TensorViewType view, ValueRange indices,
                                               OpBuilder &builder) {
    if (!view || indices.size() != view.getShape().size())
        return failure();
    Value storage = isa<StoreOp>(operation) ? operation->getOperand(1) : operation->getOperand(0);
    if (view.getAddressSpace() == "device") {
        auto argument = dyn_cast<BlockArgument>(storage);
        auto function = argument ? dyn_cast_or_null<func::FuncOp>(argument.getOwner()->getParentOp()) : func::FuncOp{};
        if (!function)
            return failure();
        unsigned descriptorBase = 0;
        while (descriptorBase < function.getNumArguments() &&
               !function.getArgAttr(descriptorBase, kTensorDescriptorComponentAttrName))
            ++descriptorBase;
        if (descriptorBase == function.getNumArguments() || argument.getArgNumber() >= descriptorBase)
            return failure();
        for (unsigned index = 0; index < argument.getArgNumber(); ++index)
            if (auto preceding = dyn_cast<TensorViewType>(function.getArgumentTypes()[index]);
                preceding && preceding.getAddressSpace() == "device")
                descriptorBase += 1 + 2 * preceding.getShape().size();
        if (descriptorBase + 1 + 2 * view.getShape().size() > function.getNumArguments())
            return failure();
        Value offset = function.getArgument(descriptorBase);
        SmallVector<Value> strides;
        strides.reserve(view.getShape().size());
        const unsigned strideBase = descriptorBase + 1 + view.getShape().size();
        for (unsigned dimension = 0; dimension < view.getShape().size(); ++dimension)
            strides.push_back(function.getArgument(strideBase + dimension));
        // Descriptor attributes identify generated arguments to later ABI passes;
        // projection follows their canonical positional sequence.
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
    if (failed(appendTensorViewDescriptorArguments(root)))
        return failure();
    SmallVector<Operation *> operations;
    root->walk([&](Operation *operation) {
        if (isa<LoadOp, StoreOp, AtomicOp>(operation))
            operations.push_back(operation);
    });
    for (Operation *operation : operations) {
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
