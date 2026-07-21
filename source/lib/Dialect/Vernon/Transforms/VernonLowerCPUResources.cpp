#include "mlir/Dialect/Vernon/Transforms/VernonLowerCPUResources.h"

#include "VernonCpuLoweringUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vernon/IR/Vernon.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::vernon {
namespace {

constexpr StringLiteral kTextureHelperName = "__vernon_cpu_texture_sample";

struct BufferLoadPattern final : OpConversionPattern<IntrinsicOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(IntrinsicOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (classifyCpuIntrinsic(op.getName()) != CpuIntrinsicKind::BufferLoad)
      return failure();
    if (adaptor.getOperands().size() != 2 || op.getNumResults() != 1)
      return rewriter.notifyMatchFailure(
          op, "buffer_load expects a buffer, index, and one result");
    rewriter.replaceOpWithNewOp<memref::LoadOp>(op, adaptor.getOperands()[0],
                                                adaptor.getOperands()[1]);
    return success();
  }
};

struct BufferStorePattern final : OpConversionPattern<IntrinsicOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(IntrinsicOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (classifyCpuIntrinsic(op.getName()) != CpuIntrinsicKind::BufferStore)
      return failure();
    if (adaptor.getOperands().size() != 3 || op.getNumResults() != 0)
      return rewriter.notifyMatchFailure(
          op, "buffer_store expects a buffer, index, value, and no result");
    rewriter.replaceOpWithNewOp<memref::StoreOp>(op, adaptor.getOperands()[2],
                                                 adaptor.getOperands()[0],
                                                 adaptor.getOperands()[1]);
    return success();
  }
};

struct TextureSamplePattern final : OpConversionPattern<IntrinsicOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(IntrinsicOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (classifyCpuIntrinsic(op.getName()) != CpuIntrinsicKind::TextureSample)
      return failure();
    if (op.getNumResults() != 1)
      return rewriter.notifyMatchFailure(
          op, "texture_sample expects exactly one result");

    Type resultType = getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(op,
                                         "texture_sample result is not legal");
    auto module = op->getParentOfType<ModuleOp>();
    auto function = op->getParentOfType<func::FuncOp>();
    if (!module || !function)
      return rewriter.notifyMatchFailure(
          op, "texture_sample must be nested in a function and module");

    SmallVector<Type> inputTypes;
    for (Value operand : adaptor.getOperands())
      inputTypes.push_back(operand.getType());
    Value callbacks = function.getArguments().back();
    inputTypes.push_back(callbacks.getType());
    FunctionType helperType =
        FunctionType::get(op.getContext(), inputTypes, {resultType});

    func::FuncOp helper = module.lookupSymbol<func::FuncOp>(kTextureHelperName);
    if (!helper) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      helper = func::FuncOp::create(rewriter, op.getLoc(), kTextureHelperName,
                                    helperType);
      helper.setPrivate();
    } else if (helper.getFunctionType() != helperType) {
      return op.emitError()
             << "CPU texture helper has incompatible type "
             << helper.getFunctionType() << "; expected " << helperType;
    }

    SmallVector<Value> helperOperands(adaptor.getOperands());
    helperOperands.push_back(callbacks);
    rewriter.replaceOpWithNewOp<func::CallOp>(
        op, helper.getSymName(), TypeRange{resultType}, helperOperands);
    return success();
  }
};

struct VernonLowerCPUResourcesPass final
    : PassWrapper<VernonLowerCPUResourcesPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VernonLowerCPUResourcesPass)

  StringRef getArgument() const final { return "vernon-lower-cpu-resources"; }
  StringRef getDescription() const final {
    return "Lower Vernon CPU resources to memrefs and integer handles";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, cf::ControlFlowDialect,
                    func::FuncDialect, index::IndexDialect, LLVM::LLVMDialect,
                    math::MathDialect, memref::MemRefDialect, scf::SCFDialect,
                    tensor::TensorDialect, vector::VectorDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    Type callbackType = LLVM::LLVMPointerType::get(context);
    for (func::FuncOp function : getOperation().getOps<func::FuncOp>()) {
      if (!function->hasAttr("vernon.entry"))
        continue;
      SmallVector<Type> inputs(function.getArgumentTypes());
      inputs.push_back(callbackType);
      function.setType(
          FunctionType::get(context, inputs, function.getResultTypes()));
      function.getBody().front().addArgument(callbackType, function.getLoc());
      function.setPrivate();
    }

    TypeConverter converter;
    converter.addConversion([](Type type) { return type; });
    converter.addConversion([](BufferType type) -> Type {
      return MemRefType::get({ShapedType::kDynamic}, type.getElementType());
    });
    converter.addConversion(
        [&](TextureType) -> Type { return IntegerType::get(context, 64); });
    converter.addConversion(
        [&](SamplerType) -> Type { return IntegerType::get(context, 64); });

    RewritePatternSet patterns(context);
    patterns.add<BufferLoadPattern, BufferStorePattern, TextureSamplePattern>(
        converter, context);
    populateFunctionOpInterfaceTypeConversionPattern(
        func::FuncOp::getOperationName(), patterns, converter);

    ConversionTarget target(*context);
    target.addLegalOp<ModuleOp>();
    target.addLegalDialect<arith::ArithDialect, cf::ControlFlowDialect,
                           index::IndexDialect, math::MathDialect,
                           memref::MemRefDialect, scf::SCFDialect,
                           tensor::TensorDialect, vector::VectorDialect>();
    target.addDynamicallyLegalDialect<func::FuncDialect>(
        [&](Operation *operation) { return converter.isLegal(operation); });
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp function) {
      return converter.isSignatureLegal(function.getFunctionType());
    });
    target.addIllegalDialect<VernonDialect>();

    if (failed(
            applyFullConversion(getOperation(), target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createVernonLowerCPUResourcesPass() {
  return std::make_unique<VernonLowerCPUResourcesPass>();
}

void registerVernonLowerCPUResourcesPass() {
  PassRegistration<VernonLowerCPUResourcesPass>();
}

} // namespace mlir::vernon
