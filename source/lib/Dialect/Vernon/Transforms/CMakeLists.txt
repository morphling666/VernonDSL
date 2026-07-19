add_mlir_library(MLIRVernonTransforms
  VernonValidation.cpp
  VernonToGPU.cpp
  VernonToSpirv.cpp

  DEPENDS
  MLIRVernonDialect
  MLIRGPUDialect
  MLIRSPIRVDialect
  MLIRSPIRVUtils

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRGPUDialect
  MLIRMemRefDialect
  MLIRVernonDialect
  MLIRSPIRVDialect
  MLIRIR
  MLIRPass
  MLIRSupport
)
