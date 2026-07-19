add_mlir_dialect_library(MLIRVernonDialect
  VernonAttrs.cpp
  VernonDialect.cpp
  VernonTypes.cpp
  VernonOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../../include/mlir/Dialect/Vernon

  DEPENDS
  MLIRVernonDialectIncGen
  MLIRVernonTypesIncGen
  MLIRVernonOpsIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRTensorDialect
  MLIRVectorDialect
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRIR
  MLIRSideEffectInterfaces
  MLIRInferTypeOpInterface
  MLIRSupport
)
