#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/JSON.h"

#include <optional>

namespace vernon::compiler {

struct ProgramReflection {
    llvm::json::Object plan;
    llvm::json::Array kernelCompileRequests;
};

mlir::FailureOr<std::optional<ProgramReflection>> buildProgramReflection(mlir::ModuleOp module);

} // namespace vernon::compiler
