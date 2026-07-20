#include "mlir/Dialect/Vernon/Transforms/VernonLowerGPUTensors.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vernon/IR/Vernon.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::vernon {
namespace {

constexpr int64_t kRegisterTensorElementLimit = 16;

FailureOr<VectorType> convertTensorType(Type type) {
  auto tensor = dyn_cast<RankedTensorType>(type);
  if (!tensor || !tensor.hasStaticShape() || tensor.getRank() == 0)
    return failure();
  int64_t elementCount = tensor.getNumElements();
  if (elementCount <= 0 || elementCount > kRegisterTensorElementLimit)
    return failure();
  return VectorType::get({elementCount}, tensor.getElementType());
}

struct ConstantPattern final : OpConversionPattern<arith::ConstantOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::ConstantOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto tensorType = dyn_cast<RankedTensorType>(op.getType());
    if (!tensorType || !tensorType.hasStaticShape())
      return failure();
    auto elements = dyn_cast<DenseElementsAttr>(op.getValue());
    if (!elements)
      return failure();
    FailureOr<VectorType> resultType = convertTensorType(op.getType());
    if (succeeded(resultType)) {
      rewriter.replaceOpWithNewOp<arith::ConstantOp>(
          op, *resultType, elements.reshape(*resultType));
      return success();
    }

    SmallVector<Value> scalarElements;
    scalarElements.reserve(tensorType.getNumElements());
    for (Attribute element : elements.getValues<Attribute>())
      scalarElements.push_back(arith::ConstantOp::create(
          rewriter, op.getLoc(), cast<TypedAttr>(element)));
    rewriter.replaceOpWithNewOp<tensor::FromElementsOp>(op, tensorType,
                                                        scalarElements);
    return success();
  }
};

struct FromElementsPattern final : OpConversionPattern<tensor::FromElementsOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tensor::FromElementsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    FailureOr<VectorType> resultType = convertTensorType(op.getType());
    if (failed(resultType))
      return failure();
    rewriter.replaceOpWithNewOp<vector::FromElementsOp>(op, *resultType,
                                                        adaptor.getElements());
    return success();
  }
};

struct SplatPattern final : OpConversionPattern<tensor::SplatOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tensor::SplatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    FailureOr<VectorType> resultType = convertTensorType(op.getType());
    if (failed(resultType))
      return failure();
    rewriter.replaceOpWithNewOp<vector::BroadcastOp>(op, *resultType,
                                                     adaptor.getInput());
    return success();
  }
};

struct ExtractPattern final : OpConversionPattern<tensor::ExtractOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tensor::ExtractOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto sourceType = dyn_cast<RankedTensorType>(op.getTensor().getType());
    if (!sourceType || !sourceType.hasStaticShape() ||
        adaptor.getIndices().size() !=
            static_cast<size_t>(sourceType.getRank()))
      return failure();

    Location location = op.getLoc();
    Value linear = adaptor.getIndices().front();
    for (auto [dimension, index] :
         llvm::zip_equal(sourceType.getShape().drop_front(),
                         adaptor.getIndices().drop_front())) {
      Value extent =
          arith::ConstantIndexOp::create(rewriter, location, dimension);
      linear = arith::MulIOp::create(rewriter, location, linear, extent);
      linear = arith::AddIOp::create(rewriter, location, linear, index);
    }
    rewriter.replaceOpWithNewOp<vector::ExtractOp>(op, adaptor.getTensor(),
                                                   OpFoldResult(linear));
    return success();
  }
};

template <typename Op>
struct ElementwisePattern final : OpConversionPattern<Op> {
  using OpConversionPattern<Op>::OpConversionPattern;
  using OpAdaptor = typename Op::Adaptor;

  LogicalResult
  matchAndRewrite(Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType =
        this->getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType || resultType == op.getResult().getType())
      return failure();
    rewriter.replaceOpWithNewOp<Op>(op, resultType, adaptor.getOperands()[0],
                                    adaptor.getOperands()[1]);
    return success();
  }
};

SmallVector<Value>
flattenConstructOperands(Location location, ValueRange operands,
                         ConversionPatternRewriter &rewriter) {
  SmallVector<Value> elements;
  for (Value operand : operands) {
    auto vectorType = dyn_cast<VectorType>(operand.getType());
    if (!vectorType) {
      elements.push_back(operand);
      continue;
    }
    for (int64_t index = 0; index < vectorType.getNumElements(); ++index)
      elements.push_back(
          vector::ExtractOp::create(rewriter, location, operand, index));
  }
  return elements;
}

Value createDot(Location location, Value lhs, Value rhs,
                ConversionPatternRewriter &rewriter) {
  Value product = arith::MulFOp::create(rewriter, location, lhs, rhs);
  return vector::ReductionOp::create(rewriter, location,
                                     vector::CombiningKind::ADD, product);
}

struct IntrinsicPattern final : OpConversionPattern<IntrinsicOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(IntrinsicOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    StringRef name = op.getName();
    Location location = op.getLoc();
    if (name == "construct") {
      Type converted =
          getTypeConverter()->convertType(op.getResult().getType());
      auto vectorType = dyn_cast_if_present<VectorType>(converted);
      if (!vectorType)
        return failure();
      SmallVector<Value> elements =
          flattenConstructOperands(location, adaptor.getOperands(), rewriter);
      if (elements.size() != static_cast<size_t>(vectorType.getNumElements()))
        return rewriter.notifyMatchFailure(
            op, "constructor element count mismatch");
      rewriter.replaceOpWithNewOp<vector::FromElementsOp>(op, vectorType,
                                                          elements);
      return success();
    }
    if (name == "dot") {
      if (adaptor.getOperands().size() != 2 ||
          !isa<VectorType>(adaptor.getOperands()[0].getType()))
        return failure();
      rewriter.replaceOp(op, createDot(location, adaptor.getOperands()[0],
                                       adaptor.getOperands()[1], rewriter));
      return success();
    }
    if (name == "normalize") {
      Value input = adaptor.getOperands().front();
      auto vectorType = dyn_cast<VectorType>(input.getType());
      if (!vectorType)
        return failure();
      Value squared = createDot(location, input, input, rewriter);
      Value length = math::SqrtOp::create(rewriter, location, squared);
      Value lengths =
          vector::BroadcastOp::create(rewriter, location, vectorType, length);
      rewriter.replaceOpWithNewOp<arith::DivFOp>(op, input, lengths);
      return success();
    }
    if (name == "cross") {
      auto vectorType =
          dyn_cast<VectorType>(adaptor.getOperands().front().getType());
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
            rewriter, location,
            vector::ExtractOp::create(rewriter, location, lhs, lhsFirst[index]),
            vector::ExtractOp::create(rewriter, location, rhs,
                                      rhsFirst[index]));
        Value second = arith::MulFOp::create(
            rewriter, location,
            vector::ExtractOp::create(rewriter, location, lhs,
                                      lhsSecond[index]),
            vector::ExtractOp::create(rewriter, location, rhs,
                                      rhsSecond[index]));
        elements.push_back(
            arith::SubFOp::create(rewriter, location, first, second));
      }
      rewriter.replaceOpWithNewOp<vector::FromElementsOp>(op, vectorType,
                                                          elements);
      return success();
    }
    if (name == "reflect") {
      Value incident = adaptor.getOperands()[0];
      Value normal = adaptor.getOperands()[1];
      auto vectorType = dyn_cast<VectorType>(incident.getType());
      if (!vectorType)
        return failure();
      Value projection = createDot(location, normal, incident, rewriter);
      Value two = arith::ConstantOp::create(
          rewriter, location, rewriter.getFloatAttr(projection.getType(), 2.0));
      Value factor = arith::MulFOp::create(rewriter, location, projection, two);
      Value factors =
          vector::BroadcastOp::create(rewriter, location, vectorType, factor);
      Value reflectedNormal =
          arith::MulFOp::create(rewriter, location, normal, factors);
      rewriter.replaceOpWithNewOp<arith::SubFOp>(op, incident, reflectedNormal);
      return success();
    }
    if (name == "matmul") {
      auto leftType = dyn_cast<RankedTensorType>(op.getOperand(0).getType());
      auto rightType = dyn_cast<RankedTensorType>(op.getOperand(1).getType());
      Type converted =
          getTypeConverter()->convertType(op.getResult().getType());
      auto resultType = dyn_cast_if_present<VectorType>(converted);
      if (!leftType || !rightType || leftType.getRank() != 2 || !resultType)
        return failure();
      int64_t rows = leftType.getDimSize(0);
      int64_t innerCount = leftType.getDimSize(1);
      int64_t resultColumns =
          rightType.getRank() == 1 ? 1 : rightType.getDimSize(1);
      SmallVector<Value> elements;
      for (int64_t row = 0; row < rows; ++row) {
        for (int64_t column = 0; column < resultColumns; ++column) {
          Value sum = arith::ConstantOp::create(
              rewriter, location,
              rewriter.getFloatAttr(resultType.getElementType(), 0.0));
          for (int64_t inner = 0; inner < innerCount; ++inner) {
            Value lhs = vector::ExtractOp::create(rewriter, location,
                                                  adaptor.getOperands()[0],
                                                  row * innerCount + inner);
            int64_t rhsIndex = rightType.getRank() == 1
                                   ? inner
                                   : inner * resultColumns + column;
            Value rhs = vector::ExtractOp::create(
                rewriter, location, adaptor.getOperands()[1], rhsIndex);
            Value product = arith::MulFOp::create(rewriter, location, lhs, rhs);
            sum = arith::AddFOp::create(rewriter, location, sum, product);
          }
          elements.push_back(sum);
        }
      }
      rewriter.replaceOpWithNewOp<vector::FromElementsOp>(op, resultType,
                                                          elements);
      return success();
    }
    if (name == "min" || name == "max" || name == "pow" || name == "clamp") {
      Value result;
      if (name == "min")
        result = arith::MinimumFOp::create(rewriter, location,
                                           adaptor.getOperands()[0],
                                           adaptor.getOperands()[1]);
      else if (name == "max")
        result = arith::MaximumFOp::create(rewriter, location,
                                           adaptor.getOperands()[0],
                                           adaptor.getOperands()[1]);
      else if (name == "pow")
        result =
            math::PowFOp::create(rewriter, location, adaptor.getOperands()[0],
                                 adaptor.getOperands()[1]);
      else {
        Value maximum = arith::MaximumFOp::create(rewriter, location,
                                                  adaptor.getOperands()[0],
                                                  adaptor.getOperands()[1]);
        result = arith::MinimumFOp::create(rewriter, location, maximum,
                                           adaptor.getOperands()[2]);
      }
      rewriter.replaceOp(op, result);
      return success();
    }
    return failure();
  }
};

struct SwizzlePattern final : OpConversionPattern<SwizzleOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SwizzleOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getInput();
    if (!isa<VectorType>(input.getType()))
      return failure();
    SmallVector<Value> elements;
    for (char component : op.getMask()) {
      size_t index = StringRef("xyzw").find(component);
      if (index == StringRef::npos)
        return failure();
      elements.push_back(vector::ExtractOp::create(
          rewriter, op.getLoc(), input, static_cast<int64_t>(index)));
    }
    Type converted = getTypeConverter()->convertType(op.getResult().getType());
    if (auto vectorType = dyn_cast_if_present<VectorType>(converted))
      rewriter.replaceOpWithNewOp<vector::FromElementsOp>(op, vectorType,
                                                          elements);
    else if (elements.size() == 1)
      rewriter.replaceOp(op, elements.front());
    else
      return failure();
    return success();
  }
};

struct VernonLowerGPUTensorsPass final
    : PassWrapper<VernonLowerGPUTensorsPass, OperationPass<gpu::GPUModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VernonLowerGPUTensorsPass)

  StringRef getArgument() const final { return "vernon-lower-gpu-tensors"; }
  StringRef getDescription() const final {
    return "Lower Vernon value tensors to GPU register vectors";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, math::MathDialect, scf::SCFDialect,
                    vector::VectorDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    WalkResult dynamicTensor = getOperation().walk([&](Operation *operation) {
      for (Type type : operation->getResultTypes()) {
        auto tensor = dyn_cast<RankedTensorType>(type);
        if (tensor && !tensor.hasStaticShape()) {
          operation->emitError(
              "dynamic local value Tensor cannot be allocated on this GPU "
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
              operation->emitError(
                  "dynamic local value Tensor cannot be allocated on this GPU "
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
    converter.addConversion([](Type type) { return type; });
    converter.addConversion([](RankedTensorType tensor) -> std::optional<Type> {
      FailureOr<VectorType> converted = convertTensorType(tensor);
      if (failed(converted))
        return std::nullopt;
      return *converted;
    });

    RewritePatternSet patterns(context);
    patterns.add<
        ConstantPattern, FromElementsPattern, SplatPattern, ExtractPattern,
        IntrinsicPattern, SwizzlePattern, ElementwisePattern<arith::AddFOp>,
        ElementwisePattern<arith::SubFOp>, ElementwisePattern<arith::MulFOp>,
        ElementwisePattern<arith::DivFOp>, ElementwisePattern<arith::AddIOp>,
        ElementwisePattern<arith::SubIOp>, ElementwisePattern<arith::MulIOp>>(
        converter, context);
    populateFunctionOpInterfaceTypeConversionPattern(
        gpu::GPUFuncOp::getOperationName(), patterns, converter);

    ConversionTarget target(*context);
    target.addLegalDialect<vector::VectorDialect>();
    target.addIllegalDialect<VernonDialect>();
    target.addDynamicallyLegalDialect<tensor::TensorDialect>(
        [&](Operation *operation) { return converter.isLegal(operation); });
    target.addDynamicallyLegalDialect<arith::ArithDialect>(
        [&](Operation *operation) {
          if (auto constant = dyn_cast<arith::ConstantOp>(operation))
            if (isa<RankedTensorType>(constant.getType()))
              return false;
          return converter.isLegal(operation);
        });
    target.addDynamicallyLegalDialect<gpu::GPUDialect>(
        [&](Operation *operation) { return converter.isLegal(operation); });
    target.addDynamicallyLegalDialect<math::MathDialect>(
        [&](Operation *operation) { return converter.isLegal(operation); });
    target.addDynamicallyLegalOp<gpu::GPUFuncOp>([&](gpu::GPUFuncOp op) {
      return converter.isSignatureLegal(op.getFunctionType());
    });
    target.markUnknownOpDynamicallyLegal(
        [&](Operation *operation) { return converter.isLegal(operation); });
    scf::populateSCFStructuralTypeConversionsAndLegality(converter, patterns,
                                                         target);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createVernonLowerGPUTensorsPass() {
  return std::make_unique<VernonLowerGPUTensorsPass>();
}

void registerVernonLowerGPUTensorsPass() {
  PassRegistration<VernonLowerGPUTensorsPass>();
}

} // namespace mlir::vernon
