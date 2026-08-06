#pragma once

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"

#include <cstdint>
#include <memory>
#include <string>

namespace mlir::vernon {

struct StructuredVjpOptions {
    SmallVector<std::string> wrtPaths;
    std::string forwardSymbol;
    std::string backwardSymbol;
};

struct StructuredVjpResult {
    func::FuncOp forward;
    func::FuncOp backward;
    uint64_t tapeBytes{};
    SmallVector<std::string> derivativeRules;
};

/// Builds scalar VJP profiles directly from structured primal IR, including
/// runtime-bounded scf.if and scf.while regions.
FailureOr<StructuredVjpResult> buildStructuredScalarVjp(func::FuncOp primal, const StructuredVjpOptions &options);

std::unique_ptr<Pass> createVernonStructuredVjpPass();
std::unique_ptr<Pass> createVernonStructuredVjpPass(StructuredVjpOptions options);
void registerVernonStructuredVjpPass();

} // namespace mlir::vernon
