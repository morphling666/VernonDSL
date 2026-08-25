#include "mlir/Dialect/Vernon/Transforms/VernonAutodiffUtils.h"

#include "mlir/Dialect/Vernon/IR/Vernon.h"
#include "mlir/Dialect/Vernon/IR/VernonValueAbi.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <limits>

namespace mlir::vernon {

FailureOr<Type> getAutodiffDerivativeScalarType(Type scalarType) {
    if (scalarType.isF64())
        return scalarType;
    if (scalarType.isF16() || scalarType.isF32())
        return Float32Type::get(scalarType.getContext());
    return failure();
}

Type wrapAutodiffDerivativeTensorView(TensorViewType view, Type payload, StringRef access) {
    return TensorViewType::get(view.getContext(), payload, view.getShape(), access, view.getAddressSpace());
}

FailureOr<Type> getAutodiffDerivativeValueType(Type valueType, ModuleOp module) {
    if (isa<TensorViewType>(valueType))
        return failure();
    // Shaped Values keep their constructor; autodiff maps the ABI-stable element.
    if (auto tensor = dyn_cast<TensorType>(valueType)) {
        FailureOr<Type> element = getAutodiffDerivativeValueType(tensor.getElementType(), module);
        if (failed(element))
            return failure();
        return TensorType::get(valueType.getContext(), *element, tensor.getShape());
    }
    if (auto tensor = dyn_cast<RankedTensorType>(valueType)) {
        if (!tensor.hasStaticShape())
            return failure();
        FailureOr<Type> element = getAutodiffDerivativeValueType(tensor.getElementType(), module);
        if (failed(element))
            return failure();
        return RankedTensorType::get(tensor.getShape(), *element, tensor.getEncoding());
    }
    if (auto vector = dyn_cast<VectorType>(valueType)) {
        if (vector.isScalable())
            return failure();
        FailureOr<Type> element = getAutodiffDerivativeValueType(vector.getElementType(), module);
        if (failed(element))
            return failure();
        return VectorType::get(vector.getShape(), *element, vector.getScalableDims());
    }
    FailureOr<ValueAbiLayout> layout = getValueStorageLayout(valueType, module);
    if (failed(layout))
        return failure();
    SmallVector<Type> leaves;
    for (const ValueAbiLeaf &leaf : layout->leaves) {
        FailureOr<Type> scalar = getAutodiffDerivativeScalarType(leaf.scalarType);
        if (failed(scalar))
            continue;
        SmallVector<uint64_t> logicalShape(leaf.shape);
        if (logicalShape.empty()) {
            leaves.push_back(*scalar);
            continue;
        }
        SmallVector<int64_t> shape;
        shape.reserve(logicalShape.size());
        for (uint64_t extent : logicalShape) {
            if (extent > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
                return failure();
            shape.push_back(static_cast<int64_t>(extent));
        }
        leaves.push_back(RankedTensorType::get(shape, *scalar));
    }
    if (leaves.empty())
        return failure();
    if (leaves.size() == 1)
        return leaves.front();
    return TupleType::get(valueType.getContext(), leaves);
}

FailureOr<Type> getAutodiffDerivativeType(Type type, ModuleOp module) {
    if (auto view = dyn_cast<TensorViewType>(type)) {
        FailureOr<Type> payload = getAutodiffDerivativeValueType(view.getElementType(), module);
        if (failed(payload))
            return failure();
        return wrapAutodiffDerivativeTensorView(view, *payload, view.getAccess());
    }
    return getAutodiffDerivativeValueType(type, module);
}

FailureOr<ValueAbiLayout> getAutodiffDerivativeValueLayout(Type primalType, Type derivativeType, ModuleOp module,
                                                           ArrayRef<StringRef> logicalLeafDtypes) {
    SmallVector<uint64_t> outerShape;
    Type primalLayoutType = primalType;
    Type derivativeLayoutType = derivativeType;
    bool derivativeLeavesCarryOuterShape = false;
    if (auto tensor = dyn_cast<TensorType>(primalType)) {
        primalLayoutType = tensor.getElementType();
        derivativeLeavesCarryOuterShape = true;
        for (int64_t extent : tensor.getShape()) {
            if (extent <= 0)
                return failure();
            outerShape.push_back(static_cast<uint64_t>(extent));
        }
    } else if (auto tensor = dyn_cast<RankedTensorType>(primalType)) {
        auto derivativeTensor = dyn_cast<RankedTensorType>(derivativeType);
        if (!tensor.hasStaticShape() || !derivativeTensor || tensor.getShape() != derivativeTensor.getShape())
            return failure();
        primalLayoutType = tensor.getElementType();
        derivativeLayoutType = derivativeTensor.getElementType();
    }
    FailureOr<ValueAbiLayout> primal = getValueStorageLayout(primalLayoutType, module);
    FailureOr<ValueAbiLayout> derivative = getValueAbiLayout(derivativeLayoutType, module, logicalLeafDtypes);
    if (failed(primal) || failed(derivative) || primal->leaves.size() != derivative->leaves.size())
        return failure();
    for (auto [primalLeaf, derivativeLeaf] : llvm::zip_equal(primal->leaves, derivative->leaves)) {
        SmallVector<uint64_t> expectedShape;
        if (derivativeLeavesCarryOuterShape)
            expectedShape = outerShape;
        llvm::append_range(expectedShape, primalLeaf.shape);
        if (expectedShape != derivativeLeaf.shape)
            return failure();
    }
    std::string primalSpelling;
    llvm::raw_string_ostream stream(primalSpelling);
    primalType.print(stream);
    const std::string identity = "tangent<" + primalSpelling + ">";
    if (failed(rebaseValueAbiLayout(*derivative, *primal, identity)))
        return failure();
    return std::move(*derivative);
}

} // namespace mlir::vernon
