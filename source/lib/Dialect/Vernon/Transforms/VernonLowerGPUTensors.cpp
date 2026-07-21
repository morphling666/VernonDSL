#include "mlir/Dialect/Vernon/Transforms/VernonLowerGPUTensors.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vernon/IR/Vernon.h"
#include "mlir/Dialect/Vernon/Transforms/VernonSharedValuePatterns.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::vernon {
namespace {

constexpr int64_t kRegisterTensorElementLimit = 16;

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
    addVernonSharedValueTypeConversions(converter, kRegisterTensorElementLimit);

    RewritePatternSet patterns(context);
    populateVernonSharedValuePatterns(converter, patterns);
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
    populateVernonSharedValueStructuralTypeConversions(converter, patterns,
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
