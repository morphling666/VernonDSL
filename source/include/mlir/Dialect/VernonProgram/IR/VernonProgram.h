//===- VernonProgram.h - Program scheduling dialect -------------*- C++ -*-===//

#ifndef MLIR_DIALECT_VERNONPROGRAM_IR_VERNONPROGRAM_H_
#define MLIR_DIALECT_VERNONPROGRAM_IR_VERNONPROGRAM_H_

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Vernon/IR/Vernon.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/StringRef.h"

#include "mlir/Dialect/VernonProgram/IR/VernonProgramDialect.h.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/VernonProgram/IR/VernonProgramOps.h.inc"

namespace mlir::vernon::program {

inline constexpr llvm::StringLiteral kCaptureForwardValueAttr = "vernon_program.capture_forward_value";

inline bool isStorageAllocIntrinsic(Operation *operation) {
    auto intrinsic = dyn_cast<mlir::vernon::IntrinsicOp>(operation);
    if (!intrinsic || intrinsic.getNumResults() != 1)
        return false;
    if (!isa<mlir::vernon::TensorViewType>(intrinsic.getResult().getType()))
        return false;
    StringRef name = intrinsic.getName();
    return name == "empty" || name == "empty_like" || name == "zeros" || name == "zeros_like" || name == "from_values";
}

} // namespace mlir::vernon::program

#endif // MLIR_DIALECT_VERNONPROGRAM_IR_VERNONPROGRAM_H_
