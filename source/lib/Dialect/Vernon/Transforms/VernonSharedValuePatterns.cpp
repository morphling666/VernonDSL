#include "mlir/Dialect/Vernon/Transforms/VernonSharedValuePatterns.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vernon/IR/Vernon.h"
#include "mlir/Dialect/Vernon/Transforms/VernonTensorShapeSemantics.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::vernon {

FailureOr<VectorType> convertStaticTensorToVector(Type type, std::optional<int64_t> elementLimit) {
    auto tensor = dyn_cast<RankedTensorType>(type);
    if (!tensor || !tensor.hasStaticShape() || tensor.getRank() == 0)
        return failure();
    int64_t elementCount = tensor.getNumElements();
    if (elementCount <= 0 || (elementLimit && elementCount > *elementLimit))
        return failure();
    return VectorType::get({elementCount}, tensor.getElementType());
}

namespace {

struct ConstantPattern final : OpConversionPattern<arith::ConstantOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(arith::ConstantOp op, OpAdaptor, ConversionPatternRewriter &rewriter) const override {
        auto tensorType = dyn_cast<RankedTensorType>(op.getType());
        if (!tensorType || !tensorType.hasStaticShape())
            return failure();
        auto elements = dyn_cast<DenseElementsAttr>(op.getValue());
        if (!elements)
            return failure();
        auto resultType = dyn_cast_if_present<VectorType>(getTypeConverter()->convertType(op.getType()));
        if (resultType) {
            rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, resultType, elements.reshape(resultType));
            return success();
        }

        SmallVector<Value> scalarElements;
        scalarElements.reserve(tensorType.getNumElements());
        for (Attribute element : elements.getValues<Attribute>())
            scalarElements.push_back(arith::ConstantOp::create(rewriter, op.getLoc(), cast<TypedAttr>(element)));
        rewriter.replaceOpWithNewOp<tensor::FromElementsOp>(op, tensorType, scalarElements);
        return success();
    }
};

struct FromElementsPattern final : OpConversionPattern<tensor::FromElementsOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(tensor::FromElementsOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        auto resultType = dyn_cast_if_present<VectorType>(getTypeConverter()->convertType(op.getType()));
        if (!resultType)
            return failure();
        rewriter.replaceOpWithNewOp<vector::FromElementsOp>(op, resultType, adaptor.getElements());
        return success();
    }
};

struct SplatPattern final : OpConversionPattern<tensor::SplatOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(tensor::SplatOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        auto resultType = dyn_cast_if_present<VectorType>(getTypeConverter()->convertType(op.getType()));
        if (!resultType)
            return failure();
        rewriter.replaceOpWithNewOp<vector::BroadcastOp>(op, resultType, adaptor.getInput());
        return success();
    }
};

struct ExtractPattern final : OpConversionPattern<tensor::ExtractOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(tensor::ExtractOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        auto sourceType = dyn_cast<RankedTensorType>(op.getTensor().getType());
        if (!sourceType || !sourceType.hasStaticShape() ||
            adaptor.getIndices().size() != static_cast<size_t>(sourceType.getRank()))
            return failure();

        Location location = op.getLoc();
        Value linear = adaptor.getIndices().front();
        for (auto [dimension, index] :
             llvm::zip_equal(sourceType.getShape().drop_front(), adaptor.getIndices().drop_front())) {
            Value extent = arith::ConstantIndexOp::create(rewriter, location, dimension);
            linear = arith::MulIOp::create(rewriter, location, linear, extent);
            linear = arith::AddIOp::create(rewriter, location, linear, index);
        }
        rewriter.replaceOpWithNewOp<vector::ExtractOp>(op, adaptor.getTensor(), OpFoldResult(linear));
        return success();
    }
};

template <typename Op> struct ElementwisePattern final : OpConversionPattern<Op> {
    using OpConversionPattern<Op>::OpConversionPattern;
    using OpAdaptor = typename Op::Adaptor;

    LogicalResult matchAndRewrite(Op op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
        Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
        if (!resultType || resultType == op.getResult().getType())
            return failure();
        rewriter.replaceOpWithNewOp<Op>(op, resultType, adaptor.getOperands()[0], adaptor.getOperands()[1]);
        return success();
    }
};

SmallVector<Value> flattenConstructOperands(Location location, ValueRange operands,
                                            ConversionPatternRewriter &rewriter) {
    SmallVector<Value> elements;
    for (Value operand : operands) {
        auto vectorType = dyn_cast<VectorType>(operand.getType());
        if (!vectorType) {
            elements.push_back(operand);
            continue;
        }
        for (int64_t index = 0; index < vectorType.getNumElements(); ++index)
            elements.push_back(vector::ExtractOp::create(rewriter, location, operand, index));
    }
    return elements;
}

Value createDot(Location location, Value lhs, Value rhs, ConversionPatternRewriter &rewriter) {
    Value product = arith::MulFOp::create(rewriter, location, lhs, rhs);
    return vector::ReductionOp::create(rewriter, location, vector::CombiningKind::ADD, product);
}

struct IntrinsicPattern final : OpConversionPattern<IntrinsicOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(IntrinsicOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        StringRef name = op.getName();
        Location location = op.getLoc();
        if (name == "construct") {
            Type converted = getTypeConverter()->convertType(op.getResult().getType());
            auto vectorType = dyn_cast_if_present<VectorType>(converted);
            if (!vectorType)
                return failure();
            SmallVector<Value> elements = flattenConstructOperands(location, adaptor.getOperands(), rewriter);
            if (elements.size() != static_cast<size_t>(vectorType.getNumElements()))
                return rewriter.notifyMatchFailure(op, "constructor element count mismatch");
            rewriter.replaceOpWithNewOp<vector::FromElementsOp>(op, vectorType, elements);
            return success();
        }
        if (name == "broadcast") {
            auto sourceType = dyn_cast<RankedTensorType>(op.getOperand(0).getType());
            auto resultTensorType = dyn_cast<RankedTensorType>(op.getResult().getType());
            auto resultType =
                dyn_cast_if_present<VectorType>(getTypeConverter()->convertType(op.getResult().getType()));
            if (!sourceType || !resultTensorType || !resultType || !isa<VectorType>(adaptor.getOperands()[0].getType()))
                return failure();
            SmallVector<Value> elements;
            for (int64_t resultIndex = 0; resultIndex < resultTensorType.getNumElements(); ++resultIndex) {
                FailureOr<int64_t> sourceIndex =
                    getStaticBroadcastLinearIndex(sourceType.getShape(), resultTensorType.getShape(), resultIndex);
                if (failed(sourceIndex))
                    return rewriter.notifyMatchFailure(op, "invalid broadcast shape");
                elements.push_back(
                    vector::ExtractOp::create(rewriter, location, adaptor.getOperands()[0], *sourceIndex));
            }
            rewriter.replaceOpWithNewOp<vector::FromElementsOp>(op, resultType, elements);
            return success();
        }
        if (name == "dot") {
            if (adaptor.getOperands().size() != 2 || !isa<VectorType>(adaptor.getOperands()[0].getType()))
                return failure();
            rewriter.replaceOp(op, createDot(location, adaptor.getOperands()[0], adaptor.getOperands()[1], rewriter));
            return success();
        }
        if (name == "normalize") {
            Value input = adaptor.getOperands().front();
            auto vectorType = dyn_cast<VectorType>(input.getType());
            if (!vectorType)
                return failure();
            Value squared = createDot(location, input, input, rewriter);
            Value length = math::SqrtOp::create(rewriter, location, squared);
            Value lengths = vector::BroadcastOp::create(rewriter, location, vectorType, length);
            rewriter.replaceOpWithNewOp<arith::DivFOp>(op, input, lengths);
            return success();
        }
        if (name == "cross") {
            auto vectorType = dyn_cast<VectorType>(adaptor.getOperands().front().getType());
            if (!vectorType || vectorType.getNumElements() != 3)
                return failure();
            Value lhs = adaptor.getOperands()[0];
            Value rhs = adaptor.getOperands()[1];
            constexpr int64_t lhsFirst[] = {1, 2, 0};
            constexpr int64_t rhsFirst[] = {2, 0, 1};
            constexpr int64_t lhsSecond[] = {2, 0, 1};
            constexpr int64_t rhsSecond[] = {1, 2, 0};
            SmallVector<Value> elements;
            for (int64_t index = 0; index < 3; ++index) {
                Value first = arith::MulFOp::create(
                    rewriter, location, vector::ExtractOp::create(rewriter, location, lhs, lhsFirst[index]),
                    vector::ExtractOp::create(rewriter, location, rhs, rhsFirst[index]));
                Value second = arith::MulFOp::create(
                    rewriter, location, vector::ExtractOp::create(rewriter, location, lhs, lhsSecond[index]),
                    vector::ExtractOp::create(rewriter, location, rhs, rhsSecond[index]));
                elements.push_back(arith::SubFOp::create(rewriter, location, first, second));
            }
            rewriter.replaceOpWithNewOp<vector::FromElementsOp>(op, vectorType, elements);
            return success();
        }
        if (name == "reflect") {
            Value incident = adaptor.getOperands()[0];
            Value normal = adaptor.getOperands()[1];
            auto vectorType = dyn_cast<VectorType>(incident.getType());
            if (!vectorType)
                return failure();
            Value projection = createDot(location, normal, incident, rewriter);
            Value two = arith::ConstantOp::create(rewriter, location, rewriter.getFloatAttr(projection.getType(), 2.0));
            Value factor = arith::MulFOp::create(rewriter, location, projection, two);
            Value factors = vector::BroadcastOp::create(rewriter, location, vectorType, factor);
            Value reflectedNormal = arith::MulFOp::create(rewriter, location, normal, factors);
            rewriter.replaceOpWithNewOp<arith::SubFOp>(op, incident, reflectedNormal);
            return success();
        }
        if (name == "matmul") {
            auto leftType = dyn_cast<RankedTensorType>(op.getOperand(0).getType());
            auto rightType = dyn_cast<RankedTensorType>(op.getOperand(1).getType());
            Type resultType = getTypeConverter()->convertType(op.getResult().getType());
            if (!leftType || !rightType || !resultType || !isa<VectorType>(adaptor.getOperands()[0].getType()) ||
                !isa<VectorType>(adaptor.getOperands()[1].getType()))
                return failure();
            FailureOr<StaticMatmulPlan> plan = getStaticMatmulPlan(leftType.getShape(), rightType.getShape());
            FailureOr<int64_t> resultCount =
                getStaticShapeElementCount(failed(plan) ? ArrayRef<int64_t>() : plan->resultShape);
            if (failed(plan) || failed(resultCount))
                return rewriter.notifyMatchFailure(op, "invalid matmul shape");
            Type elementType = leftType.getElementType();
            if (!isa<FloatType>(elementType))
                return rewriter.notifyMatchFailure(op, "matmul currently requires floating-point elements");
            SmallVector<Value> elements;
            for (int64_t resultIndex = 0; resultIndex < *resultCount; ++resultIndex) {
                Value sum = arith::ConstantOp::create(rewriter, location, rewriter.getFloatAttr(elementType, 0.0));
                for (int64_t reduction = 0; reduction < plan->reduction; ++reduction) {
                    FailureOr<int64_t> leftIndex = getStaticMatmulLeftLinearIndex(*plan, resultIndex, reduction);
                    FailureOr<int64_t> rightIndex = getStaticMatmulRightLinearIndex(*plan, resultIndex, reduction);
                    if (failed(leftIndex) || failed(rightIndex))
                        return rewriter.notifyMatchFailure(op, "invalid matmul index plan");
                    Value lhs = vector::ExtractOp::create(rewriter, location, adaptor.getOperands()[0], *leftIndex);
                    Value rhs = vector::ExtractOp::create(rewriter, location, adaptor.getOperands()[1], *rightIndex);
                    Value product = arith::MulFOp::create(rewriter, location, lhs, rhs);
                    sum = arith::AddFOp::create(rewriter, location, sum, product);
                }
                elements.push_back(sum);
            }
            if (auto vectorType = dyn_cast<VectorType>(resultType))
                rewriter.replaceOpWithNewOp<vector::FromElementsOp>(op, vectorType, elements);
            else if (elements.size() == 1 && elements.front().getType() == resultType)
                rewriter.replaceOp(op, elements.front());
            else
                return failure();
            return success();
        }
        if (name == "min" || name == "max" || name == "pow" || name == "clamp") {
            Value result;
            if (name == "min")
                result =
                    arith::MinimumFOp::create(rewriter, location, adaptor.getOperands()[0], adaptor.getOperands()[1]);
            else if (name == "max")
                result =
                    arith::MaximumFOp::create(rewriter, location, adaptor.getOperands()[0], adaptor.getOperands()[1]);
            else if (name == "pow")
                result = math::PowFOp::create(rewriter, location, adaptor.getOperands()[0], adaptor.getOperands()[1]);
            else {
                Value maximum =
                    arith::MaximumFOp::create(rewriter, location, adaptor.getOperands()[0], adaptor.getOperands()[1]);
                result = arith::MinimumFOp::create(rewriter, location, maximum, adaptor.getOperands()[2]);
            }
            rewriter.replaceOp(op, result);
            return success();
        }
        return failure();
    }
};

struct SwizzlePattern final : OpConversionPattern<SwizzleOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(SwizzleOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
        Value input = adaptor.getInput();
        if (!isa<VectorType>(input.getType()))
            return failure();
        SmallVector<Value> elements;
        for (char component : op.getMask()) {
            std::optional<unsigned> index = decodeSwizzleComponent(component);
            if (!index)
                return rewriter.notifyMatchFailure(op, "swizzle mask contains an invalid component alias");
            elements.push_back(vector::ExtractOp::create(rewriter, op.getLoc(), input, static_cast<int64_t>(*index)));
        }
        Type converted = getTypeConverter()->convertType(op.getResult().getType());
        if (auto vectorType = dyn_cast_if_present<VectorType>(converted))
            rewriter.replaceOpWithNewOp<vector::FromElementsOp>(op, vectorType, elements);
        else if (elements.size() == 1)
            rewriter.replaceOp(op, elements.front());
        else
            return failure();
        return success();
    }
};

} // namespace

void addVernonSharedValueTypeConversions(TypeConverter &converter, std::optional<int64_t> staticTensorElementLimit) {
    converter.addConversion([](Type type) { return type; });
    converter.addConversion([staticTensorElementLimit](RankedTensorType tensor) -> std::optional<Type> {
        FailureOr<VectorType> converted = convertStaticTensorToVector(tensor, staticTensorElementLimit);
        if (failed(converted))
            return std::nullopt;
        return *converted;
    });
}

void populateVernonSharedValuePatterns(TypeConverter &converter, RewritePatternSet &patterns) {
    MLIRContext *context = patterns.getContext();
    patterns
        .add<ConstantPattern, FromElementsPattern, SplatPattern, ExtractPattern, IntrinsicPattern, SwizzlePattern,
             ElementwisePattern<arith::AddFOp>, ElementwisePattern<arith::SubFOp>, ElementwisePattern<arith::MulFOp>,
             ElementwisePattern<arith::DivFOp>, ElementwisePattern<arith::AddIOp>, ElementwisePattern<arith::SubIOp>,
             ElementwisePattern<arith::MulIOp>>(converter, context);
}

void populateVernonSharedValueStructuralTypeConversions(TypeConverter &converter, RewritePatternSet &patterns,
                                                        ConversionTarget &target) {
    scf::populateSCFStructuralTypeConversionsAndLegality(converter, patterns, target);
}

} // namespace mlir::vernon
