#include "mlir/Dialect/Vernon/Transforms/VernonExpandFloorCeilDivision.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::vernon {
namespace {

// No SPIR-V, NVVM or LLVM conversion lowers arith.floordivsi / ceildivsi, so every backend pipeline runs this
// before dialect conversion. On SPIR-V it must follow vernon-lower-mul-overflow-checks, which matches floordivsi
// directly; once expanded, the quotient is no longer a single division the overflow matcher recognizes.
struct VernonExpandFloorCeilDivisionPass final : PassWrapper<VernonExpandFloorCeilDivisionPass, OperationPass<>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VernonExpandFloorCeilDivisionPass)

    StringRef getArgument() const final { return "vernon-expand-floor-ceil-division"; }
    StringRef getDescription() const final {
        return "Expand arith.floordivsi and arith.ceildivsi/ceildivui into truncating division";
    }

    void getDependentDialects(DialectRegistry &registry) const override { registry.insert<arith::ArithDialect>(); }

    void runOnOperation() override {
        RewritePatternSet patterns(&getContext());
        arith::populateCeilFloorDivExpandOpsPatterns(patterns);
        if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
            signalPassFailure();
    }
};

} // namespace

std::unique_ptr<Pass> createVernonExpandFloorCeilDivisionPass() {
    return std::make_unique<VernonExpandFloorCeilDivisionPass>();
}

void registerVernonExpandFloorCeilDivisionPass() { PassRegistration<VernonExpandFloorCeilDivisionPass>(); }

} // namespace mlir::vernon
