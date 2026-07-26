#include "mlir/Dialect/Vernon/Transforms/VernonLowerGPUTensors.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vernon/IR/Vernon.h"
#include "mlir/Dialect/Vernon/Transforms/VernonSharedValuePatterns.h"
#include "mlir/Dialect/Vernon/Transforms/VernonTensorShapeSemantics.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/StringMap.h"

#include <functional>

namespace mlir::vernon {
namespace {

constexpr int64_t kRegisterTensorElementLimit = 16;
constexpr int64_t kSpirvVectorElementLimit = 4;

std::optional<unsigned> spirvCompositeElementCount(Type type) {
    if (auto vector = dyn_cast<VectorType>(type))
        return static_cast<unsigned>(vector.getNumElements());
    if (auto array = dyn_cast<spirv::ArrayType>(type))
        return array.getNumElements();
    return std::nullopt;
}

struct SpirvTensorConstantPattern final : OpConversionPattern<arith::ConstantOp> {
    SpirvTensorConstantPattern(TypeConverter &converter, MLIRContext *context)
        : OpConversionPattern(converter, context, PatternBenefit(2)) {}

    LogicalResult matchAndRewrite(arith::ConstantOp op, OpAdaptor, ConversionPatternRewriter &rewriter) const override {
        auto sourceType = dyn_cast<RankedTensorType>(op.getType());
        Type convertedType = getTypeConverter()->convertType(op.getType());
        std::optional<unsigned> elementCount = spirvCompositeElementCount(convertedType);
        auto elements = dyn_cast<DenseElementsAttr>(op.getValue());
        if (!sourceType || !elementCount || !elements ||
            static_cast<int64_t>(*elementCount) != sourceType.getNumElements())
            return failure();
        SmallVector<Value> values;
        for (Attribute element : elements.getValues<Attribute>())
            values.push_back(arith::ConstantOp::create(rewriter, op.getLoc(), cast<TypedAttr>(element)));
        rewriter.replaceOpWithNewOp<spirv::CompositeConstructOp>(op, convertedType, values);
        return success();
    }
};

struct SpirvTensorFromElementsPattern final : OpConversionPattern<tensor::FromElementsOp> {
    SpirvTensorFromElementsPattern(TypeConverter &converter, MLIRContext *context)
        : OpConversionPattern(converter, context, PatternBenefit(2)) {}

    LogicalResult matchAndRewrite(tensor::FromElementsOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        Type converted = getTypeConverter()->convertType(op.getType());
        std::optional<unsigned> elementCount = spirvCompositeElementCount(converted);
        if (!elementCount || adaptor.getElements().size() != *elementCount)
            return failure();
        rewriter.replaceOpWithNewOp<spirv::CompositeConstructOp>(op, converted, adaptor.getElements());
        return success();
    }
};

struct SpirvTensorSplatPattern final : OpConversionPattern<tensor::SplatOp> {
    SpirvTensorSplatPattern(TypeConverter &converter, MLIRContext *context)
        : OpConversionPattern(converter, context, PatternBenefit(2)) {}

    LogicalResult matchAndRewrite(tensor::SplatOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        Type convertedType = getTypeConverter()->convertType(op.getType());
        std::optional<unsigned> elementCount = spirvCompositeElementCount(convertedType);
        if (!elementCount)
            return failure();
        SmallVector<Value> values(*elementCount, adaptor.getInput());
        rewriter.replaceOpWithNewOp<spirv::CompositeConstructOp>(op, convertedType, values);
        return success();
    }
};

struct SpirvTensorExtractPattern final : OpConversionPattern<tensor::ExtractOp> {
    SpirvTensorExtractPattern(TypeConverter &converter, MLIRContext *context)
        : OpConversionPattern(converter, context, PatternBenefit(2)) {}

    LogicalResult matchAndRewrite(tensor::ExtractOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        auto sourceType = dyn_cast<RankedTensorType>(op.getTensor().getType());
        std::optional<unsigned> elementCount = spirvCompositeElementCount(adaptor.getTensor().getType());
        if (!sourceType || !elementCount || adaptor.getIndices().size() != static_cast<size_t>(sourceType.getRank()) ||
            static_cast<int64_t>(*elementCount) != sourceType.getNumElements())
            return failure();
        Location location = op.getLoc();
        Value linear;
        if (sourceType.getRank() == 0) {
            linear = arith::ConstantIndexOp::create(rewriter, location, 0);
        } else {
            linear = adaptor.getIndices().front();
            for (auto [extent, index] :
                 llvm::zip_equal(sourceType.getShape().drop_front(), adaptor.getIndices().drop_front())) {
                Value extentValue = arith::ConstantIndexOp::create(rewriter, location, extent);
                linear = arith::MulIOp::create(rewriter, location, linear, extentValue);
                linear = arith::AddIOp::create(rewriter, location, linear, index);
            }
        }
        Value selected =
            spirv::CompositeExtractOp::create(rewriter, location, adaptor.getTensor(), ArrayRef<int32_t>{0});
        for (int64_t index = 1; index < sourceType.getNumElements(); ++index) {
            Value candidate = spirv::CompositeExtractOp::create(rewriter, location, adaptor.getTensor(),
                                                                ArrayRef<int32_t>{static_cast<int32_t>(index)});
            Value expected = arith::ConstantIndexOp::create(rewriter, location, index);
            Value matches = arith::CmpIOp::create(rewriter, location, arith::CmpIPredicate::eq, linear, expected);
            selected = arith::SelectOp::create(rewriter, location, matches, candidate, selected);
        }
        rewriter.replaceOp(op, selected);
        return success();
    }
};

template <typename Op> struct SpirvTensorElementwisePattern final : OpConversionPattern<Op> {
    using OpConversionPattern<Op>::OpConversionPattern;
    using OpAdaptor = typename Op::Adaptor;

    LogicalResult matchAndRewrite(Op op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
        auto arrayType =
            dyn_cast_if_present<spirv::ArrayType>(this->getTypeConverter()->convertType(op.getResult().getType()));
        if (!arrayType || adaptor.getOperands().size() != 2)
            return failure();
        SmallVector<Value> values;
        for (unsigned index = 0; index < arrayType.getNumElements(); ++index) {
            Value lhs = spirv::CompositeExtractOp::create(rewriter, op.getLoc(), adaptor.getOperands()[0],
                                                          ArrayRef<int32_t>{static_cast<int32_t>(index)});
            Value rhs = spirv::CompositeExtractOp::create(rewriter, op.getLoc(), adaptor.getOperands()[1],
                                                          ArrayRef<int32_t>{static_cast<int32_t>(index)});
            values.push_back(Op::create(rewriter, op.getLoc(), lhs, rhs));
        }
        rewriter.replaceOpWithNewOp<spirv::CompositeConstructOp>(op, arrayType, values);
        return success();
    }
};

Value extractFlatTensorElement(Location location, Value value, int64_t index, bool useSpirv,
                               ConversionPatternRewriter &rewriter) {
    if (isa<VectorType>(value.getType()))
        return vector::ExtractOp::create(rewriter, location, value, index);
    if (useSpirv)
        return spirv::CompositeExtractOp::create(rewriter, location, value,
                                                 ArrayRef<int32_t>{static_cast<int32_t>(index)});
    return LLVM::ExtractValueOp::create(rewriter, location, value, ArrayRef<int64_t>{index});
}

FailureOr<Value> constructFlatTensor(Location location, Type resultType, ArrayRef<Value> elements, bool useSpirv,
                                     ConversionPatternRewriter &rewriter) {
    if (auto vectorType = dyn_cast<VectorType>(resultType))
        return vector::FromElementsOp::create(rewriter, location, vectorType, elements).getResult();
    if (auto arrayType = dyn_cast<spirv::ArrayType>(resultType); useSpirv && arrayType) {
        if (elements.size() != arrayType.getNumElements())
            return failure();
        return spirv::CompositeConstructOp::create(rewriter, location, arrayType, elements).getResult();
    }
    if (auto arrayType = dyn_cast<LLVM::LLVMArrayType>(resultType); !useSpirv && arrayType) {
        if (elements.size() != static_cast<size_t>(arrayType.getNumElements()))
            return failure();
        Value result = LLVM::UndefOp::create(rewriter, location, arrayType);
        for (auto [index, element] : llvm::enumerate(elements))
            result = LLVM::InsertValueOp::create(rewriter, location, result, element,
                                                 ArrayRef<int64_t>{static_cast<int64_t>(index)});
        return result;
    }
    if (elements.size() == 1 && elements.front().getType() == resultType)
        return elements.front();
    return failure();
}

struct GpuTensorShapeIntrinsicPattern final : OpConversionPattern<IntrinsicOp> {
    GpuTensorShapeIntrinsicPattern(TypeConverter &converter, MLIRContext *context, bool useSpirv)
        : OpConversionPattern(converter, context, PatternBenefit(3)), useSpirv(useSpirv) {}

    LogicalResult matchAndRewrite(IntrinsicOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        StringRef name = op.getName();
        if (name != "broadcast" && name != "matmul")
            return failure();
        Type resultType = getTypeConverter()->convertType(op.getResult().getType());
        if (!resultType)
            return failure();
        Location location = op.getLoc();
        SmallVector<Value> elements;
        if (name == "broadcast") {
            auto sourceType = dyn_cast<RankedTensorType>(op.getOperand(0).getType());
            auto resultTensorType = dyn_cast<RankedTensorType>(op.getResult().getType());
            if (!sourceType || !resultTensorType || adaptor.getOperands().size() != 1)
                return failure();
            for (int64_t resultIndex = 0; resultIndex < resultTensorType.getNumElements(); ++resultIndex) {
                FailureOr<int64_t> sourceIndex =
                    getStaticBroadcastLinearIndex(sourceType.getShape(), resultTensorType.getShape(), resultIndex);
                if (failed(sourceIndex))
                    return rewriter.notifyMatchFailure(op, "invalid broadcast shape");
                elements.push_back(
                    extractFlatTensorElement(location, adaptor.getOperands()[0], *sourceIndex, useSpirv, rewriter));
            }
        } else {
            auto leftType = dyn_cast<RankedTensorType>(op.getOperand(0).getType());
            auto rightType = dyn_cast<RankedTensorType>(op.getOperand(1).getType());
            if (!leftType || !rightType || adaptor.getOperands().size() != 2 ||
                !isa<FloatType>(leftType.getElementType()))
                return failure();
            FailureOr<StaticMatmulPlan> plan = getStaticMatmulPlan(leftType.getShape(), rightType.getShape());
            if (failed(plan))
                return rewriter.notifyMatchFailure(op, "invalid matmul shape");
            FailureOr<int64_t> resultCount = getStaticShapeElementCount(plan->resultShape);
            if (failed(resultCount))
                return failure();
            for (int64_t resultIndex = 0; resultIndex < *resultCount; ++resultIndex) {
                Value sum = arith::ConstantOp::create(rewriter, location,
                                                      rewriter.getFloatAttr(leftType.getElementType(), 0.0));
                for (int64_t reduction = 0; reduction < plan->reduction; ++reduction) {
                    FailureOr<int64_t> leftIndex = getStaticMatmulLeftLinearIndex(*plan, resultIndex, reduction);
                    FailureOr<int64_t> rightIndex = getStaticMatmulRightLinearIndex(*plan, resultIndex, reduction);
                    if (failed(leftIndex) || failed(rightIndex))
                        return failure();
                    Value lhs =
                        extractFlatTensorElement(location, adaptor.getOperands()[0], *leftIndex, useSpirv, rewriter);
                    Value rhs =
                        extractFlatTensorElement(location, adaptor.getOperands()[1], *rightIndex, useSpirv, rewriter);
                    Value product = arith::MulFOp::create(rewriter, location, lhs, rhs);
                    sum = arith::AddFOp::create(rewriter, location, sum, product);
                }
                elements.push_back(sum);
            }
        }
        FailureOr<Value> result = constructFlatTensor(location, resultType, elements, useSpirv, rewriter);
        if (failed(result))
            return failure();
        rewriter.replaceOp(op, *result);
        return success();
    }

    bool useSpirv;
};

struct GpuFlatTensorFromElementsPattern final : OpConversionPattern<tensor::FromElementsOp> {
    GpuFlatTensorFromElementsPattern(TypeConverter &converter, MLIRContext *context, bool useSpirv)
        : OpConversionPattern(converter, context, PatternBenefit(3)), useSpirv(useSpirv) {}

    LogicalResult matchAndRewrite(tensor::FromElementsOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        Type resultType = getTypeConverter()->convertType(op.getType());
        if (isa<VectorType>(resultType))
            return failure();
        SmallVector<Value> elements(adaptor.getElements().begin(), adaptor.getElements().end());
        FailureOr<Value> result = constructFlatTensor(op.getLoc(), resultType, elements, useSpirv, rewriter);
        if (failed(result))
            return failure();
        rewriter.replaceOp(op, *result);
        return success();
    }

    bool useSpirv;
};

struct GpuFlatTensorSplatPattern final : OpConversionPattern<tensor::SplatOp> {
    GpuFlatTensorSplatPattern(TypeConverter &converter, MLIRContext *context, bool useSpirv)
        : OpConversionPattern(converter, context, PatternBenefit(3)), useSpirv(useSpirv) {}

    LogicalResult matchAndRewrite(tensor::SplatOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        Type resultType = getTypeConverter()->convertType(op.getType());
        if (isa<VectorType>(resultType))
            return failure();
        SmallVector<Value> elements(op.getType().getNumElements(), adaptor.getInput());
        FailureOr<Value> result = constructFlatTensor(op.getLoc(), resultType, elements, useSpirv, rewriter);
        if (failed(result))
            return failure();
        rewriter.replaceOp(op, *result);
        return success();
    }

    bool useSpirv;
};

struct GpuFlatTensorExtractPattern final : OpConversionPattern<tensor::ExtractOp> {
    GpuFlatTensorExtractPattern(TypeConverter &converter, MLIRContext *context, bool useSpirv)
        : OpConversionPattern(converter, context, PatternBenefit(3)), useSpirv(useSpirv) {}

    LogicalResult matchAndRewrite(tensor::ExtractOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        auto sourceType = dyn_cast<RankedTensorType>(op.getTensor().getType());
        if (!sourceType || isa<VectorType>(adaptor.getTensor().getType()) ||
            adaptor.getIndices().size() != static_cast<size_t>(sourceType.getRank()))
            return failure();
        Location location = op.getLoc();
        Value linear = adaptor.getIndices().front();
        for (auto [extent, index] :
             llvm::zip_equal(sourceType.getShape().drop_front(), adaptor.getIndices().drop_front())) {
            linear = arith::MulIOp::create(rewriter, location, linear,
                                           arith::ConstantIndexOp::create(rewriter, location, extent));
            linear = arith::AddIOp::create(rewriter, location, linear, index);
        }
        Value selected = extractFlatTensorElement(location, adaptor.getTensor(), 0, useSpirv, rewriter);
        for (int64_t index = 1; index < sourceType.getNumElements(); ++index) {
            Value candidate = extractFlatTensorElement(location, adaptor.getTensor(), index, useSpirv, rewriter);
            Value expected = arith::ConstantIndexOp::create(rewriter, location, index);
            Value matches = arith::CmpIOp::create(rewriter, location, arith::CmpIPredicate::eq, linear, expected);
            selected = arith::SelectOp::create(rewriter, location, matches, candidate, selected);
        }
        rewriter.replaceOp(op, selected);
        return success();
    }

    bool useSpirv;
};

template <typename Op> struct GpuFlatTensorElementwisePattern final : OpConversionPattern<Op> {
    GpuFlatTensorElementwisePattern(TypeConverter &converter, MLIRContext *context, bool useSpirv)
        : OpConversionPattern<Op>(converter, context, PatternBenefit(3)), useSpirv(useSpirv) {}

    using OpAdaptor = typename Op::Adaptor;
    LogicalResult matchAndRewrite(Op op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
        auto sourceType = dyn_cast<RankedTensorType>(op.getResult().getType());
        Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
        if (!sourceType || isa<VectorType>(resultType) || adaptor.getOperands().size() != 2)
            return failure();
        SmallVector<Value> elements;
        for (int64_t index = 0; index < sourceType.getNumElements(); ++index) {
            Value lhs = extractFlatTensorElement(op.getLoc(), adaptor.getOperands()[0], index, useSpirv, rewriter);
            Value rhs = extractFlatTensorElement(op.getLoc(), adaptor.getOperands()[1], index, useSpirv, rewriter);
            elements.push_back(Op::create(rewriter, op.getLoc(), lhs, rhs));
        }
        FailureOr<Value> result = constructFlatTensor(op.getLoc(), resultType, elements, useSpirv, rewriter);
        if (failed(result))
            return failure();
        rewriter.replaceOp(op, *result);
        return success();
    }

    bool useSpirv;
};

struct GpuTupleCreatePattern final : OpConversionPattern<TupleCreateOp> {
    GpuTupleCreatePattern(TypeConverter &converter, MLIRContext *context, bool useSpirv)
        : OpConversionPattern(converter, context), useSpirv(useSpirv) {}

    LogicalResult matchAndRewrite(TupleCreateOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        Type converted = getTypeConverter()->convertType(op.getResult().getType());
        if (useSpirv) {
            auto structType = dyn_cast_if_present<spirv::StructType>(converted);
            if (!structType)
                return failure();
            rewriter.replaceOpWithNewOp<spirv::CompositeConstructOp>(op, structType, adaptor.getElements());
            return success();
        }
        auto structType = dyn_cast_if_present<LLVM::LLVMStructType>(converted);
        if (!structType)
            return failure();
        Value aggregate = LLVM::UndefOp::create(rewriter, op.getLoc(), structType);
        for (auto [index, element] : llvm::enumerate(adaptor.getElements()))
            aggregate = LLVM::InsertValueOp::create(rewriter, op.getLoc(), aggregate, element,
                                                    ArrayRef<int64_t>{static_cast<int64_t>(index)});
        rewriter.replaceOp(op, aggregate);
        return success();
    }

    bool useSpirv;
};

struct GpuTupleGetPattern final : OpConversionPattern<TupleGetOp> {
    GpuTupleGetPattern(TypeConverter &converter, MLIRContext *context, bool useSpirv)
        : OpConversionPattern(converter, context), useSpirv(useSpirv) {}

    LogicalResult matchAndRewrite(TupleGetOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        if (useSpirv) {
            rewriter.replaceOpWithNewOp<spirv::CompositeExtractOp>(
                op, adaptor.getInput(), ArrayRef<int32_t>{static_cast<int32_t>(op.getIndex())});
            return success();
        }
        rewriter.replaceOpWithNewOp<LLVM::ExtractValueOp>(op, adaptor.getInput(),
                                                          ArrayRef<int64_t>{static_cast<int64_t>(op.getIndex())});
        return success();
    }

    bool useSpirv;
};

struct GpuStructCreatePattern final : OpConversionPattern<StructCreateOp> {
    GpuStructCreatePattern(TypeConverter &converter, MLIRContext *context, bool useSpirv)
        : OpConversionPattern(converter, context), useSpirv(useSpirv) {}

    LogicalResult matchAndRewrite(StructCreateOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        SmallVector<Type> fieldTypes;
        llvm::transform(adaptor.getFields(), std::back_inserter(fieldTypes),
                        [](Value field) { return field.getType(); });
        if (useSpirv) {
            auto structType = spirv::StructType::get(fieldTypes);
            rewriter.replaceOpWithNewOp<spirv::CompositeConstructOp>(op, structType, adaptor.getFields());
            return success();
        }
        auto structType = LLVM::LLVMStructType::getLiteral(op.getContext(), fieldTypes);
        Value aggregate = LLVM::UndefOp::create(rewriter, op.getLoc(), structType);
        for (auto [index, field] : llvm::enumerate(adaptor.getFields()))
            aggregate = LLVM::InsertValueOp::create(rewriter, op.getLoc(), aggregate, field,
                                                    ArrayRef<int64_t>{static_cast<int64_t>(index)});
        rewriter.replaceOp(op, aggregate);
        return success();
    }

    bool useSpirv;
};

struct GpuStructGetPattern final : OpConversionPattern<StructGetOp> {
    GpuStructGetPattern(TypeConverter &converter, MLIRContext *context, bool useSpirv)
        : OpConversionPattern(converter, context), useSpirv(useSpirv) {}

    LogicalResult matchAndRewrite(StructGetOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        if (useSpirv) {
            rewriter.replaceOpWithNewOp<spirv::CompositeExtractOp>(
                op, adaptor.getInput(), ArrayRef<int32_t>{static_cast<int32_t>(op.getIndex())});
            return success();
        }
        rewriter.replaceOpWithNewOp<LLVM::ExtractValueOp>(op, adaptor.getInput(),
                                                          ArrayRef<int64_t>{static_cast<int64_t>(op.getIndex())});
        return success();
    }

    bool useSpirv;
};

struct GpuAggregateTensorConstructPattern final : OpConversionPattern<IntrinsicOp> {
    GpuAggregateTensorConstructPattern(TypeConverter &converter, MLIRContext *context, bool useSpirv)
        : OpConversionPattern(converter, context), useSpirv(useSpirv) {}

    LogicalResult matchAndRewrite(IntrinsicOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        if (op.getName() != "construct" || !isa<TensorType>(op.getResult().getType()))
            return failure();
        Type converted = getTypeConverter()->convertType(op.getResult().getType());
        if (useSpirv) {
            auto arrayType = dyn_cast_if_present<spirv::ArrayType>(converted);
            if (!arrayType || adaptor.getOperands().size() != arrayType.getNumElements())
                return failure();
            rewriter.replaceOpWithNewOp<spirv::CompositeConstructOp>(op, arrayType, adaptor.getOperands());
            return success();
        }
        auto arrayType = dyn_cast_if_present<LLVM::LLVMArrayType>(converted);
        if (!arrayType || adaptor.getOperands().size() != arrayType.getNumElements())
            return failure();
        Value aggregate = LLVM::UndefOp::create(rewriter, op.getLoc(), arrayType);
        for (auto [index, element] : llvm::enumerate(adaptor.getOperands()))
            aggregate = LLVM::InsertValueOp::create(rewriter, op.getLoc(), aggregate, element,
                                                    ArrayRef<int64_t>{static_cast<int64_t>(index)});
        rewriter.replaceOp(op, aggregate);
        return success();
    }

    bool useSpirv;
};

struct GpuAggregateTensorGetPattern final : OpConversionPattern<TensorGetOp> {
    GpuAggregateTensorGetPattern(TypeConverter &converter, MLIRContext *context, bool useSpirv)
        : OpConversionPattern(converter, context), useSpirv(useSpirv) {}

    LogicalResult matchAndRewrite(TensorGetOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        auto sourceType = dyn_cast<TensorType>(op.getInput().getType());
        if (!sourceType || adaptor.getIndices().size() != sourceType.getShape().size())
            return failure();
        Location location = op.getLoc();
        Value linear = adaptor.getIndices().front();
        for (auto [dimension, index] :
             llvm::zip_equal(sourceType.getShape().drop_front(), adaptor.getIndices().drop_front())) {
            Value extent = arith::ConstantIndexOp::create(rewriter, location, dimension);
            linear = arith::MulIOp::create(rewriter, location, linear, extent);
            linear = arith::AddIOp::create(rewriter, location, linear, index);
        }
        int64_t elementCount = 1;
        for (int64_t dimension : sourceType.getShape())
            elementCount *= dimension;
        auto extract = [&](int64_t index) -> Value {
            if (useSpirv)
                return spirv::CompositeExtractOp::create(rewriter, location, adaptor.getInput(),
                                                         ArrayRef<int32_t>{static_cast<int32_t>(index)});
            return LLVM::ExtractValueOp::create(rewriter, location, adaptor.getInput(), ArrayRef<int64_t>{index});
        };
        Value selected = extract(0);
        for (int64_t index = 1; index < elementCount; ++index) {
            Value candidate = extract(index);
            Value expected = arith::ConstantIndexOp::create(rewriter, location, index);
            Value matches = arith::CmpIOp::create(rewriter, location, arith::CmpIPredicate::eq, linear, expected);
            if (useSpirv) {
                std::function<Value(Type, Value, Value)> selectValue = [&](Type type, Value whenTrue,
                                                                           Value whenFalse) -> Value {
                    if (auto structType = dyn_cast<spirv::StructType>(type)) {
                        SmallVector<Value> fields;
                        for (unsigned field = 0; field < structType.getNumElements(); ++field) {
                            Value trueField = spirv::CompositeExtractOp::create(
                                rewriter, location, whenTrue, ArrayRef<int32_t>{static_cast<int32_t>(field)});
                            Value falseField = spirv::CompositeExtractOp::create(
                                rewriter, location, whenFalse, ArrayRef<int32_t>{static_cast<int32_t>(field)});
                            fields.push_back(selectValue(structType.getElementType(field), trueField, falseField));
                        }
                        return spirv::CompositeConstructOp::create(rewriter, location, structType, fields);
                    }
                    if (auto arrayType = dyn_cast<spirv::ArrayType>(type)) {
                        SmallVector<Value> elements;
                        for (unsigned element = 0; element < arrayType.getNumElements(); ++element) {
                            Value trueElement = spirv::CompositeExtractOp::create(
                                rewriter, location, whenTrue, ArrayRef<int32_t>{static_cast<int32_t>(element)});
                            Value falseElement = spirv::CompositeExtractOp::create(
                                rewriter, location, whenFalse, ArrayRef<int32_t>{static_cast<int32_t>(element)});
                            elements.push_back(selectValue(arrayType.getElementType(), trueElement, falseElement));
                        }
                        return spirv::CompositeConstructOp::create(rewriter, location, arrayType, elements);
                    }
                    return spirv::SelectOp::create(rewriter, location, type, matches, whenTrue, whenFalse);
                };
                selected = selectValue(candidate.getType(), candidate, selected);
            } else {
                selected = arith::SelectOp::create(rewriter, location, matches, candidate, selected);
            }
        }
        rewriter.replaceOp(op, selected);
        return success();
    }

    bool useSpirv;
};

struct VernonLowerGPUTensorsPass final : PassWrapper<VernonLowerGPUTensorsPass, OperationPass<gpu::GPUModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VernonLowerGPUTensorsPass)

    VernonLowerGPUTensorsPass() = default;
    explicit VernonLowerGPUTensorsPass(bool useSpirvTupleAbi) : useSpirvTupleAbi(useSpirvTupleAbi) {}

    StringRef getArgument() const final { return "vernon-lower-gpu-tensors"; }
    StringRef getDescription() const final { return "Lower Vernon value tensors to GPU register vectors"; }

    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<arith::ArithDialect, LLVM::LLVMDialect, math::MathDialect, scf::SCFDialect, spirv::SPIRVDialect,
                        vector::VectorDialect>();
    }

    void runOnOperation() override {
        MLIRContext *context = &getContext();
        WalkResult dynamicTensor = getOperation().walk([&](Operation *operation) {
            for (Type type : operation->getResultTypes()) {
                auto tensor = dyn_cast<RankedTensorType>(type);
                if (tensor && !tensor.hasStaticShape()) {
                    operation->emitError("dynamic local value Tensor cannot be allocated on this GPU "
                                         "backend; "
                                         "specialize its shape or pass it as an addressable Tensor");
                    return WalkResult::interrupt();
                }
            }
            for (Region &region : operation->getRegions()) {
                for (Block &block : region) {
                    for (BlockArgument argument : block.getArguments()) {
                        auto tensor = dyn_cast<RankedTensorType>(argument.getType());
                        if (tensor && !tensor.hasStaticShape()) {
                            operation->emitError("dynamic local value Tensor cannot be allocated on this GPU "
                                                 "backend; "
                                                 "specialize its shape or pass it as an addressable Tensor");
                            return WalkResult::interrupt();
                        }
                    }
                }
            }
            return WalkResult::advance();
        });
        if (dynamicTensor.wasInterrupted()) {
            signalPassFailure();
            return;
        }

        TypeConverter converter;
        llvm::StringMap<SmallVector<Type>> structFields;
        getOperation().walk([&](StructCreateOp create) {
            auto structure = dyn_cast<StructType>(create.getResult().getType());
            if (!structure)
                return;
            SmallVector<Type> fields;
            llvm::transform(create.getFields(), std::back_inserter(fields),
                            [](Value field) { return field.getType(); });
            structFields[structure.getName()] = std::move(fields);
        });
        if (useSpirvTupleAbi) {
            converter.addConversion([](Type type) { return type; });
            converter.addConversion([](RankedTensorType tensor) -> std::optional<Type> {
                if (!tensor.hasStaticShape() || tensor.getNumElements() <= 0 || !tensor.getElementType().isIntOrFloat())
                    return std::nullopt;
                // SPIR-V Shader vectors are limited to 2-4 components. Larger
                // register tensors must remain composites even when another
                // GPU backend could represent them as one register vector.
                if (tensor.getRank() != 0 && tensor.getNumElements() >= 2 &&
                    tensor.getNumElements() <= kSpirvVectorElementLimit)
                    return VectorType::get({tensor.getNumElements()}, tensor.getElementType());
                const unsigned stride = std::max<unsigned>(tensor.getElementType().getIntOrFloatBitWidth() / 8, 1);
                return spirv::ArrayType::get(tensor.getElementType(), static_cast<unsigned>(tensor.getNumElements()),
                                             stride);
            });
        } else {
            addVernonSharedValueTypeConversions(converter, kRegisterTensorElementLimit);
            converter.addConversion([](RankedTensorType tensor) -> std::optional<Type> {
                if (!tensor.hasStaticShape() || tensor.getNumElements() <= kRegisterTensorElementLimit ||
                    !tensor.getElementType().isIntOrFloat())
                    return std::nullopt;
                return LLVM::LLVMArrayType::get(tensor.getElementType(), tensor.getNumElements());
            });
        }
        converter.addConversion([&](TupleType tuple) -> std::optional<Type> {
            SmallVector<Type> elements;
            if (failed(converter.convertTypes(tuple.getTypes(), elements)))
                return std::nullopt;
            if (useSpirvTupleAbi)
                return spirv::StructType::get(elements);
            return LLVM::LLVMStructType::getLiteral(tuple.getContext(), elements);
        });
        converter.addConversion([&](TensorType tensor) -> std::optional<Type> {
            Type element = converter.convertType(tensor.getElementType());
            if (!element)
                return std::nullopt;
            int64_t count = 1;
            for (int64_t dimension : tensor.getShape())
                count *= dimension;
            if (useSpirvTupleAbi)
                return spirv::ArrayType::get(element, static_cast<unsigned>(count));
            return LLVM::LLVMArrayType::get(element, count);
        });
        converter.addConversion([&](StructType structure) -> std::optional<Type> {
            auto found = structFields.find(structure.getName());
            if (found == structFields.end())
                return std::nullopt;
            SmallVector<Type> fields;
            if (failed(converter.convertTypes(found->second, fields)))
                return std::nullopt;
            if (useSpirvTupleAbi)
                return spirv::StructType::get(fields);
            return LLVM::LLVMStructType::getLiteral(structure.getContext(), fields);
        });

        RewritePatternSet patterns(context);
        populateVernonSharedValuePatterns(converter, patterns);
        if (useSpirvTupleAbi)
            patterns.add<SpirvTensorConstantPattern, SpirvTensorExtractPattern, SpirvTensorFromElementsPattern,
                         SpirvTensorSplatPattern, SpirvTensorElementwisePattern<arith::AddFOp>,
                         SpirvTensorElementwisePattern<arith::SubFOp>, SpirvTensorElementwisePattern<arith::MulFOp>,
                         SpirvTensorElementwisePattern<arith::DivFOp>, SpirvTensorElementwisePattern<arith::AddIOp>,
                         SpirvTensorElementwisePattern<arith::SubIOp>, SpirvTensorElementwisePattern<arith::MulIOp>>(
                converter, context);
        patterns.add<GpuAggregateTensorConstructPattern, GpuAggregateTensorGetPattern, GpuStructCreatePattern,
                     GpuStructGetPattern, GpuTupleCreatePattern, GpuTupleGetPattern>(converter, context,
                                                                                     useSpirvTupleAbi);
        patterns.add<GpuTensorShapeIntrinsicPattern, GpuFlatTensorFromElementsPattern, GpuFlatTensorSplatPattern,
                     GpuFlatTensorExtractPattern>(converter, context, useSpirvTupleAbi);
        patterns.add<GpuFlatTensorElementwisePattern<arith::AddFOp>, GpuFlatTensorElementwisePattern<arith::SubFOp>,
                     GpuFlatTensorElementwisePattern<arith::MulFOp>, GpuFlatTensorElementwisePattern<arith::DivFOp>,
                     GpuFlatTensorElementwisePattern<arith::AddIOp>, GpuFlatTensorElementwisePattern<arith::SubIOp>,
                     GpuFlatTensorElementwisePattern<arith::MulIOp>>(converter, context, useSpirvTupleAbi);
        populateFunctionOpInterfaceTypeConversionPattern(gpu::GPUFuncOp::getOperationName(), patterns, converter);

        ConversionTarget target(*context);
        target.addLegalDialect<vector::VectorDialect>();
        if (useSpirvTupleAbi)
            target.addLegalDialect<spirv::SPIRVDialect>();
        else
            target.addLegalDialect<LLVM::LLVMDialect>();
        target.addIllegalDialect<VernonDialect>();
        target.addDynamicallyLegalDialect<tensor::TensorDialect>(
            [&](Operation *operation) { return converter.isLegal(operation); });
        target.addDynamicallyLegalDialect<arith::ArithDialect>([&](Operation *operation) {
            if (auto constant = dyn_cast<arith::ConstantOp>(operation))
                if (isa<RankedTensorType>(constant.getType()))
                    return false;
            return converter.isLegal(operation);
        });
        target.addDynamicallyLegalDialect<gpu::GPUDialect>(
            [&](Operation *operation) { return converter.isLegal(operation); });
        target.addDynamicallyLegalDialect<math::MathDialect>(
            [&](Operation *operation) { return converter.isLegal(operation); });
        target.addDynamicallyLegalOp<gpu::GPUFuncOp>(
            [&](gpu::GPUFuncOp op) { return converter.isSignatureLegal(op.getFunctionType()); });
        target.markUnknownOpDynamicallyLegal([&](Operation *operation) { return converter.isLegal(operation); });
        populateVernonSharedValueStructuralTypeConversions(converter, patterns, target);

        if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
            signalPassFailure();
    }

    bool useSpirvTupleAbi = false;
};

} // namespace

std::unique_ptr<Pass> createVernonLowerGPUTensorsPass(bool useSpirvTupleAbi) {
    return std::make_unique<VernonLowerGPUTensorsPass>(useSpirvTupleAbi);
}

void registerVernonLowerGPUTensorsPass() { PassRegistration<VernonLowerGPUTensorsPass>(); }

} // namespace mlir::vernon
