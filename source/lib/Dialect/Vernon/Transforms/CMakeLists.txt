add_mlir_library(MLIRVernonTransforms
  VernonToSpirv.cpp

  DEPENDS
  MLIRVernonDialect
  MLIRSPIRVDialect

  LINK_LIBS PUBLIC
  MLIRVernonDialect
  MLIRSPIRVDialect
  MLIRIR
  MLIRPass
  MLIRSupport
)
