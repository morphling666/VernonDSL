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
    SmallVector<std::string> outputPaths;
};

struct StructuredVjpResult {
    func::FuncOp forward;
    func::FuncOp backward;
    uint64_t tapeBytes{};
    SmallVector<std::string> derivativeRules;
};

/// Builds VJP profiles directly from structured primal IR, including
/// aggregate values and runtime-bounded scf.if/scf.while regions.
FailureOr<StructuredVjpResult> buildStructuredVjp(func::FuncOp primal, const StructuredVjpOptions &options);

std::unique_ptr<Pass> createVernonStructuredVjpPass();
std::unique_ptr<Pass> createVernonStructuredVjpPass(StructuredVjpOptions options);
void registerVernonStructuredVjpPass();

} // namespace mlir::vernon
