add_mlir_library(MLIRVernonTransforms
  VernonLowerCUDAMath.cpp
  VernonLowerGPUTensors.cpp
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
  MLIRLLVMDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRSCFTransforms
  MLIRVernonDialect
  MLIRSPIRVDialect
  MLIRTensorDialect
  MLIRVectorDialect
  MLIRIR
  MLIRPass
  MLIRSupport
)
