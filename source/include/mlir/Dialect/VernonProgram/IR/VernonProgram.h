//===- VernonProgram.h - Program scheduling dialect -------------*- C++ -*-===//

#ifndef MLIR_DIALECT_VERNONPROGRAM_IR_VERNONPROGRAM_H_
#define MLIR_DIALECT_VERNONPROGRAM_IR_VERNONPROGRAM_H_

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/StringRef.h"

#include "mlir/Dialect/VernonProgram/IR/VernonProgramDialect.h.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/VernonProgram/IR/VernonProgramOps.h.inc"

namespace mlir::vernon::program {

inline constexpr llvm::StringLiteral kCaptureForwardValueAttr = "vernon_program.capture_forward_value";

} // namespace mlir::vernon::program

#endif // MLIR_DIALECT_VERNONPROGRAM_IR_VERNONPROGRAM_H_
