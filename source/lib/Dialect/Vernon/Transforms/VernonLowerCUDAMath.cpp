#include "mlir/Dialect/Vernon/Transforms/VernonLowerCUDAMath.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::vernon {
namespace {

struct SqrtPattern final : OpRewritePattern<math::SqrtOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(math::SqrtOp op, PatternRewriter &rewriter) const override {
        auto intrinsic = LLVM::CallIntrinsicOp::create(rewriter, op.getLoc(), op.getType(),
                                                       rewriter.getStringAttr("llvm.sqrt"), op.getOperand());
        rewriter.replaceOp(op, intrinsic.getResults());
        return success();
    }
};

struct CosPattern final : OpRewritePattern<math::CosOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(math::CosOp op, PatternRewriter &rewriter) const override {
        auto fastmath = LLVM::FastmathFlagsAttr::get(op.getContext(), LLVM::FastmathFlags::afn);
        auto intrinsic = LLVM::CallIntrinsicOp::create(rewriter, op.getLoc(), TypeRange{op.getType()},
                                                       rewriter.getStringAttr("llvm.cos"), op.getOperand(), fastmath);
        rewriter.replaceOp(op, intrinsic.getResults());
        return success();
    }
};

// buildLowerToNVVMPassPipeline runs convert-arith-to-llvm before
// convert-gpu-to-nvvm, so arith.remf would reach the libdevice patterns as an
// llvm.frem that NVVM legality rejects; call libdevice's exact fmod directly.
struct RemFPattern final : OpRewritePattern<arith::RemFOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(arith::RemFOp op, PatternRewriter &rewriter) const override {
        Type type = op.getType();
        StringRef name = type.isF32() ? "__nv_fmodf" : type.isF64() ? "__nv_fmod" : StringRef();
        auto module = op->getParentOfType<gpu::GPUModuleOp>();
        if (name.empty() || !module)
            return failure();
        auto function = module.lookupSymbol<LLVM::LLVMFuncOp>(name);
        if (!function) {
            OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(module.getBody());
            function =
                LLVM::LLVMFuncOp::create(rewriter, op.getLoc(), name, LLVM::LLVMFunctionType::get(type, {type, type}));
        }
        rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, function, ValueRange{op.getLhs(), op.getRhs()});
        return success();
    }
};

struct VernonLowerCUDAMathPass final : PassWrapper<VernonLowerCUDAMathPass, OperationPass<gpu::GPUModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VernonLowerCUDAMathPass)

    StringRef getArgument() const final { return "vernon-lower-cuda-math"; }
    StringRef getDescription() const final { return "Lower CUDA math to LLVM intrinsics and libdevice calls"; }

    void getDependentDialects(DialectRegistry &registry) const override { registry.insert<LLVM::LLVMDialect>(); }

    void runOnOperation() override {
        RewritePatternSet patterns(&getContext());
        patterns.add<SqrtPattern, CosPattern, RemFPattern>(&getContext());
        if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
            signalPassFailure();
    }
};

} // namespace

std::unique_ptr<Pass> createVernonLowerCUDAMathPass() { return std::make_unique<VernonLowerCUDAMathPass>(); }

void registerVernonLowerCUDAMathPass() { PassRegistration<VernonLowerCUDAMathPass>(); }

} // namespace mlir::vernon
