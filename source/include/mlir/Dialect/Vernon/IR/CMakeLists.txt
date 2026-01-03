# Tablegen for Vernon dialect operations
set(LLVM_TARGET_DEFINITIONS VernonOps.td)
mlir_tablegen(VernonOps.h.inc -gen-op-decls
  EXTRA_INCLUDES ${MLIR_TABLEGEN_INCLUDE_DIRS})
mlir_tablegen(VernonOps.cpp.inc -gen-op-defs
  EXTRA_INCLUDES ${MLIR_TABLEGEN_INCLUDE_DIRS})
add_mlir_dialect_tablegen_target(MLIRVernonOpsIncGen)

# Tablegen for Vernon dialect types
set(LLVM_TARGET_DEFINITIONS VernonTypes.td)
mlir_tablegen(VernonTypes.h.inc -gen-typedef-decls -typedefs-dialect=vernon
  EXTRA_INCLUDES ${MLIR_TABLEGEN_INCLUDE_DIRS})
mlir_tablegen(VernonTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=vernon
  EXTRA_INCLUDES ${MLIR_TABLEGEN_INCLUDE_DIRS})
add_mlir_dialect_tablegen_target(MLIRVernonTypesIncGen)

# Tablegen for Vernon dialect
set(LLVM_TARGET_DEFINITIONS VernonDialect.td)
mlir_tablegen(VernonDialect.h.inc -gen-dialect-decls -dialect=vernon
  EXTRA_INCLUDES ${MLIR_TABLEGEN_INCLUDE_DIRS})
mlir_tablegen(VernonDialect.cpp.inc -gen-dialect-defs -dialect=vernon
  EXTRA_INCLUDES ${MLIR_TABLEGEN_INCLUDE_DIRS})
add_mlir_dialect_tablegen_target(MLIRVernonDialectIncGen)
