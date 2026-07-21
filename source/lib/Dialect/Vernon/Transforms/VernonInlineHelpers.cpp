#include "mlir/Dialect/Vernon/Transforms/VernonInlineHelpers.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::vernon {
namespace {

struct VernonInlineHelpersPass final
    : PassWrapper<VernonInlineHelpersPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VernonInlineHelpersPass)

  StringRef getArgument() const final { return "vernon-inline-helpers"; }
  StringRef getDescription() const final {
    return "Inline shared Vernon helper functions into their callers";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto eraseHelpers = [&] {
      for (func::FuncOp function :
           llvm::make_early_inc_range(module.getOps<func::FuncOp>()))
        if (!function->hasAttr("vernon.entry"))
          function.erase();
    };
    unsigned maxIterations = 1;
    for ([[maybe_unused]] func::FuncOp function : module.getOps<func::FuncOp>())
      ++maxIterations;

    for (unsigned iteration = 0; iteration < maxIterations; ++iteration) {
      SmallVector<func::CallOp> calls;
      module.walk([&](func::CallOp call) { calls.push_back(call); });
      if (calls.empty()) {
        eraseHelpers();
        return;
      }

      for (func::CallOp call : calls) {
        func::FuncOp callee =
            module.lookupSymbol<func::FuncOp>(call.getCallee());
        if (!callee) {
          call.emitError() << "cannot resolve helper '" << call.getCallee()
                           << "'";
          return signalPassFailure();
        }
        if (callee->hasAttr("vernon.entry")) {
          call.emitError("Vernon entry functions cannot be called");
          return signalPassFailure();
        }
        if (!llvm::hasSingleElement(callee.getBody())) {
          call.emitError("helper inlining requires a single function block");
          return signalPassFailure();
        }
        auto returnOp =
            dyn_cast<func::ReturnOp>(callee.front().getTerminator());
        if (!returnOp) {
          call.emitError("helper function has no func.return terminator");
          return signalPassFailure();
        }

        IRMapping mapping;
        mapping.map(callee.getArguments(), call.getOperands());
        OpBuilder builder(call);
        for (Operation &operation : callee.front().without_terminator())
          builder.clone(operation, mapping);
        SmallVector<Value> replacements;
        for (Value value : returnOp.getOperands())
          replacements.push_back(mapping.lookupOrDefault(value));
        call->replaceAllUsesWith(replacements);
        call.erase();
      }
    }

    SmallVector<func::CallOp> remainingCalls;
    module.walk([&](func::CallOp call) { remainingCalls.push_back(call); });
    if (!remainingCalls.empty()) {
      remainingCalls.front().emitError(
          "helper inlining did not converge; recursive calls are forbidden");
      signalPassFailure();
      return;
    }
    eraseHelpers();
  }
};

} // namespace

std::unique_ptr<Pass> createVernonInlineHelpersPass() {
  return std::make_unique<VernonInlineHelpersPass>();
}

void registerVernonInlineHelpersPass() {
  PassRegistration<VernonInlineHelpersPass>();
}

} // namespace mlir::vernon
