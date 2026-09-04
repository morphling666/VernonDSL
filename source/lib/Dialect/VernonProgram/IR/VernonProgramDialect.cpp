//===- VernonProgramDialect.cpp - Program dialect implementation ----------===//

#include "mlir/Dialect/VernonProgram/IR/VernonProgram.h"

#include "mlir/IR/Builders.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::vernon::program;

#include "mlir/Dialect/VernonProgram/IR/VernonProgramDialect.cpp.inc"

void VernonProgramDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/VernonProgram/IR/VernonProgramOps.cpp.inc"
        >();
}

LogicalResult ComputeOp::verify() {
    if (getGrid().size() != 3 || llvm::any_of(getGrid(), [](int64_t value) { return value <= 0; }))
        return emitOpError("grid must contain three positive dimensions");
    if (auto controls = (*this)->getAttrOfType<DenseI64ArrayAttr>("vernon_program.grid_control_arguments");
        controls &&
        (controls.size() != 3 || llvm::any_of(controls.asArrayRef(), [](int64_t value) { return value < -1; })))
        return emitOpError("grid_control_arguments must contain three function argument indices or -1");
    if (getOperandNames().size() != getArguments().size())
        return emitOpError("operand_names must match operands");
    if (getResultNames().size() != getResults().size())
        return emitOpError("result_names must match results");
    return success();
}

LogicalResult GraphicsOp::verify() {
    if (getTopology().empty())
        return emitOpError("requires static primitive topology");
    if (getResults().empty() || getOperands().size() < getNumResults())
        return emitOpError("attachments must match results");
    if (getOperandNames().size() != getLogicalArguments().size())
        return emitOpError("operand_names must match non-attachment operands");
    if (getResultNames().size() != getResults().size())
        return emitOpError("result_names must match results");
    const int64_t colorCount = getColorCount();
    if (colorCount < 0 || static_cast<uint64_t>(colorCount) > getNumResults() ||
        getNumResults() - static_cast<uint64_t>(colorCount) > 1)
        return emitOpError("color_count must cover every color attachment and at most one depth attachment");
    for (auto [attachment, result] : llvm::zip_equal(getAttachments(), getResults()))
        if (attachment.getType() != result.getType())
            return emitOpError("each result must be the updated attachment of the same type");
    return success();
}

#define GET_OP_CLASSES
#include "mlir/Dialect/VernonProgram/IR/VernonProgramOps.cpp.inc"
