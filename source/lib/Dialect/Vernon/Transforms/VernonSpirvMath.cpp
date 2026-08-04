#include "VernonSpirvMath.h"

#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/APInt.h"

namespace mlir::vernon {

FailureOr<Value> lowerAtan2ToSpirv(Location location, Type resultType, Value y, Value x, OpBuilder &builder) {
    auto elementType = dyn_cast<FloatType>(getElementTypeOrSelf(resultType));
    if (!elementType)
        return failure();

    auto floatConstant = [&](double value) -> Value {
        Attribute attribute = builder.getFloatAttr(elementType, value);
        if (auto vectorType = dyn_cast<VectorType>(resultType))
            attribute = DenseElementsAttr::get(vectorType, cast<TypedAttr>(attribute));
        return spirv::ConstantOp::create(builder, location, resultType, cast<TypedAttr>(attribute));
    };
    Type integerElementType = builder.getIntegerType(elementType.getWidth());
    Type integerType = integerElementType;
    if (auto vectorType = dyn_cast<VectorType>(resultType))
        integerType = VectorType::get(vectorType.getShape(), integerElementType);
    auto integerConstant = [&](const APInt &value) -> Value {
        Attribute attribute = builder.getIntegerAttr(integerElementType, value);
        if (auto vectorType = dyn_cast<VectorType>(integerType))
            attribute = DenseElementsAttr::get(vectorType, cast<TypedAttr>(attribute));
        return spirv::ConstantOp::create(builder, location, integerType, cast<TypedAttr>(attribute));
    };
    auto signBit = [&](Value value) -> Value {
        Value bits = spirv::BitcastOp::create(builder, location, integerType, value);
        Value mask = integerConstant(APInt::getSignMask(elementType.getWidth()));
        Value zero = integerConstant(APInt::getZero(elementType.getWidth()));
        Value sign = spirv::BitwiseAndOp::create(builder, location, bits, mask);
        return spirv::INotEqualOp::create(builder, location, sign, zero);
    };

    Value zero = floatConstant(0.0);
    Value piOverFour = floatConstant(0.78539816339744830962);
    Value pi = floatConstant(3.14159265358979323846);
    Value yNegative = signBit(y);
    Value xNegative = signBit(x);
    Value absoluteY = spirv::GLFAbsOp::create(builder, location, resultType, y);
    Value absoluteX = spirv::GLFAbsOp::create(builder, location, resultType, x);
    Value ratio = spirv::FDivOp::create(builder, location, absoluteY, absoluteX);
    Value angle = spirv::GLAtanOp::create(builder, location, resultType, ratio);
    Value bothInfinite = spirv::LogicalAndOp::create(builder, location, spirv::IsInfOp::create(builder, location, y),
                                                     spirv::IsInfOp::create(builder, location, x));
    angle = spirv::SelectOp::create(builder, location, resultType, bothInfinite, piOverFour, angle);
    Value reflected = spirv::FSubOp::create(builder, location, pi, angle);
    angle = spirv::SelectOp::create(builder, location, resultType, xNegative, reflected, angle);
    Value negativeAngle = spirv::FSubOp::create(builder, location, zero, angle);
    angle = spirv::SelectOp::create(builder, location, resultType, yNegative, negativeAngle, angle);

    Value negativePi = spirv::FSubOp::create(builder, location, zero, pi);
    Value signedPi = spirv::SelectOp::create(builder, location, resultType, yNegative, negativePi, pi);
    Value zeroResult = spirv::SelectOp::create(builder, location, resultType, xNegative, signedPi, y);
    Value yIsZero = spirv::FOrdEqualOp::create(builder, location, y, zero);
    Value xIsNan = spirv::IsNanOp::create(builder, location, x);
    Value useZeroResult =
        spirv::LogicalAndOp::create(builder, location, yIsZero, spirv::LogicalNotOp::create(builder, location, xIsNan));
    return spirv::SelectOp::create(builder, location, resultType, useZeroResult, zeroResult, angle).getResult();
}

} // namespace mlir::vernon
