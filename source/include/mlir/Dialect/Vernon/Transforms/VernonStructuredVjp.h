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

/// Builds straight-line scalar VJP profiles from the structured primal IR.
/// Structured control flow is intentionally rejected until its reverse
/// transform is implemented.
FailureOr<StructuredVjpResult> buildStructuredScalarVjp(func::FuncOp primal, const StructuredVjpOptions &options);

std::unique_ptr<Pass> createVernonStructuredVjpPass();
std::unique_ptr<Pass> createVernonStructuredVjpPass(StructuredVjpOptions options);
void registerVernonStructuredVjpPass();

} // namespace mlir::vernon
