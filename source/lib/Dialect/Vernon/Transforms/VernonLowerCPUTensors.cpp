#include "mlir/Dialect/Vernon/Transforms/VernonLowerCPUTensors.h"

#include "VernonCpuLoweringUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vernon/IR/Vernon.h"
#include "mlir/Dialect/Vernon/Transforms/VernonSharedValuePatterns.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::vernon {
namespace {

struct ReturnPattern final : OpConversionPattern<func::ReturnOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op, adaptor.getOperands());
    return success();
  }
};

struct ResourceIntrinsicTypePattern final : OpConversionPattern<IntrinsicOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(IntrinsicOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isCpuResourceIntrinsic(classifyCpuIntrinsic(op.getName())))
      return failure();
    SmallVector<Type> resultTypes;
    if (failed(getTypeConverter()->convertTypes(op->getResultTypes(),
                                                resultTypes)))
      return failure();
    OperationState state(op.getLoc(), IntrinsicOp::getOperationName());
    state.addOperands(adaptor.getOperands());
    state.addTypes(resultTypes);
    state.addAttributes(op->getAttrs());
    Operation *replacement = rewriter.create(state);
    rewriter.replaceOp(op, replacement->getResults());
    return success();
  }
};

struct VernonLowerCPUTensorsPass final
    : PassWrapper<VernonLowerCPUTensorsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VernonLowerCPUTensorsPass)

  StringRef getArgument() const final { return "vernon-lower-cpu-tensors"; }
  StringRef getDescription() const final {
    return "Lower static Vernon CPU value tensors to unrestricted vectors";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect, math::MathDialect,
                    scf::SCFDialect, tensor::TensorDialect,
                    vector::VectorDialect>();
  }

  void runOnOperation() override {
    // Struct declarations are source-side metadata. Helpers have already been
    // inlined, so declarations cannot affect CPU value conversion.
    for (StructDeclOp declaration :
         llvm::make_early_inc_range(getOperation().getOps<StructDeclOp>()))
      declaration.erase();

    bool invalid = false;
    getOperation().walk([&](IntrinsicOp intrinsic) {
      if (classifyCpuIntrinsic(intrinsic.getName()) !=
          CpuIntrinsicKind::Unknown)
        return;
      auto function = intrinsic->getParentOfType<func::FuncOp>();
      intrinsic.emitError()
          << "unknown Vernon intrinsic '" << intrinsic.getName()
          << "' in CPU entry '"
          << (function ? function.getSymName() : StringRef("<unknown>")) << "'";
      invalid = true;
    });
    if (invalid) {
      signalPassFailure();
      return;
    }

    MLIRContext *context = &getContext();
    TypeConverter converter;
    // CPU value tensors retain the legacy emitter's unrestricted static size.
    addVernonSharedValueTypeConversions(converter);

    RewritePatternSet patterns(context);
    populateVernonSharedValuePatterns(converter, patterns);
    patterns.add<ResourceIntrinsicTypePattern, ReturnPattern>(converter,
                                                              context);
    populateFunctionOpInterfaceTypeConversionPattern(
        func::FuncOp::getOperationName(), patterns, converter);

    ConversionTarget target(*context);
    target.addLegalOp<ModuleOp>();
    target.addDynamicallyLegalDialect<
        arith::ArithDialect, func::FuncDialect, math::MathDialect,
        scf::SCFDialect, tensor::TensorDialect, vector::VectorDialect>(
        [&](Operation *operation) { return converter.isLegal(operation); });
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp function) {
      return converter.isSignatureLegal(function.getFunctionType()) &&
             converter.isLegal(&function.getBody());
    });
    target.addDynamicallyLegalOp<IntrinsicOp>([&](IntrinsicOp intrinsic) {
      return isCpuResourceIntrinsic(
                 classifyCpuIntrinsic(intrinsic.getName())) &&
             converter.isLegal(intrinsic.getOperation());
    });
    target.addIllegalOp<SwizzleOp, StructDeclOp, StructCreateOp, StructGetOp>();
    populateVernonSharedValueStructuralTypeConversions(converter, patterns,
                                                       target);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createVernonLowerCPUTensorsPass() {
  return std::make_unique<VernonLowerCPUTensorsPass>();
}

void registerVernonLowerCPUTensorsPass() {
  PassRegistration<VernonLowerCPUTensorsPass>();
}

} // namespace mlir::vernon
