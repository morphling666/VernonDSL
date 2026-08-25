//===- VernonProgram.h - Program scheduling dialect -------------*- C++ -*-===//

#ifndef MLIR_DIALECT_VERNONPROGRAM_IR_VERNONPROGRAM_H_
#define MLIR_DIALECT_VERNONPROGRAM_IR_VERNONPROGRAM_H_

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Vernon/IR/Vernon.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/StringRef.h"

#include "mlir/Dialect/VernonProgram/IR/VernonProgramDialect.h.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/VernonProgram/IR/VernonProgramOps.h.inc"

namespace mlir::vernon::program {

inline constexpr llvm::StringLiteral kCaptureForwardValueAttr = "vernon_program.capture_forward_value";
inline constexpr llvm::StringLiteral kOperandAccessesAttrName = "vernon_program.operand_accesses";

inline llvm::StringRef getProgramOperandAccess(Operation *operation, unsigned index) {
    auto accesses = operation->getAttrOfType<mlir::ArrayAttr>(kOperandAccessesAttrName);
    if (!accesses || index >= accesses.size())
        return {};
    if (auto name = mlir::dyn_cast<mlir::StringAttr>(accesses[index]))
        return name.getValue();
    return {};
}

/// Write-only DPS destinations are Storage objectives (kernel `outputs`), not
/// local VJP `wrt`. Nested kernel bind keeps the callee's parameter names.
inline bool isWriteOnlyProgramOperand(Operation *operation, unsigned index) {
    return getProgramOperandAccess(operation, index) == "write";
}

inline bool isProgramAllocIntrinsicName(llvm::StringRef name) {
    return name == "empty" || name == "empty_like" || name == "zeros" || name == "zeros_like" || name == "from_values";
}

inline bool isStorageAllocIntrinsic(Operation *operation) {
    auto intrinsic = dyn_cast<mlir::vernon::IntrinsicOp>(operation);
    if (!intrinsic || intrinsic.getNumResults() != 1)
        return false;
    if (!isa<mlir::vernon::TensorViewType>(intrinsic.getResult().getType()))
        return false;
    return isProgramAllocIntrinsicName(intrinsic.getName());
}

} // namespace mlir::vernon::program

#endif // MLIR_DIALECT_VERNONPROGRAM_IR_VERNONPROGRAM_H_
