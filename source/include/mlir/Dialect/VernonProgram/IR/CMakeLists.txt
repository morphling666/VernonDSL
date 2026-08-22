set(LLVM_TARGET_DEFINITIONS VernonProgramOps.td)
mlir_tablegen(
    VernonProgramOps.h.inc
    -gen-op-decls
    EXTRA_INCLUDES
    ${MLIR_TABLEGEN_INCLUDE_DIRS})
mlir_tablegen(
    VernonProgramOps.cpp.inc
    -gen-op-defs
    EXTRA_INCLUDES
    ${MLIR_TABLEGEN_INCLUDE_DIRS})
add_mlir_dialect_tablegen_target(MLIRVernonProgramOpsIncGen)

set(LLVM_TARGET_DEFINITIONS VernonProgramDialect.td)
mlir_tablegen(
    VernonProgramDialect.h.inc
    -gen-dialect-decls
    -dialect=vernon_program
    EXTRA_INCLUDES
    ${MLIR_TABLEGEN_INCLUDE_DIRS})
mlir_tablegen(
    VernonProgramDialect.cpp.inc
    -gen-dialect-defs
    -dialect=vernon_program
    EXTRA_INCLUDES
    ${MLIR_TABLEGEN_INCLUDE_DIRS})
add_mlir_dialect_tablegen_target(MLIRVernonProgramDialectIncGen)
