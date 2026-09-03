#pragma once

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"

#include <memory>
#include <string>

namespace mlir::vernon::program {

struct ProgramVjpOptions {
    SmallVector<unsigned> wrtBoundaryIndices;
    std::string forwardSymbol{"forward"};
    std::string backwardSymbol{"backward"};
    SmallVector<unsigned> cotangentBoundaryIndices;
};

FailureOr<SmallVector<unsigned>> resolveProgramWrtBoundaryIndices(func::FuncOp primal, ArrayRef<StringRef> publicPaths);
FailureOr<SmallVector<unsigned>> resolveProgramCotangentBoundaryIndices(func::FuncOp primal,
                                                                        ArrayRef<StringRef> publicPaths);
LogicalResult buildProgramVjp(func::FuncOp primal, const ProgramVjpOptions &options);
std::unique_ptr<Pass> createVernonProgramVjpPass();
std::unique_ptr<Pass> createVernonProgramVjpPass(ProgramVjpOptions options);
void registerVernonProgramVjpPass();

} // namespace mlir::vernon::program
