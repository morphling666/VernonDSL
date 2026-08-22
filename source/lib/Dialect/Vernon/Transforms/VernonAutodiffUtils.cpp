#include "mlir/Dialect/Vernon/Transforms/VernonAutodiffUtils.h"

#include "mlir/Dialect/Vernon/IR/VernonValueAbi.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/raw_ostream.h"

#include <limits>

namespace mlir::vernon {

FailureOr<Type> getAutodiffDerivativeType(Type scalarType) {
    if (scalarType.isF64())
        return scalarType;
    if (scalarType.isF16() || scalarType.isF32())
        return Float32Type::get(scalarType.getContext());
    return failure();
}

FailureOr<Type> getAutodiffDerivativeValueType(Type valueType, ModuleOp module) {
    if (auto tensor = dyn_cast<RankedTensorType>(valueType)) {
        FailureOr<Type> element = getAutodiffDerivativeType(tensor.getElementType());
        if (failed(element) || !tensor.hasStaticShape())
            return failure();
        return RankedTensorType::get(tensor.getShape(), *element);
    }
    FailureOr<ValueAbiLayout> layout = getValueStorageLayout(valueType, module);
    if (failed(layout))
        return failure();
    SmallVector<Type> leaves;
    for (const ValueAbiLeaf &leaf : layout->leaves) {
        FailureOr<Type> scalar = getAutodiffDerivativeType(leaf.scalarType);
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
