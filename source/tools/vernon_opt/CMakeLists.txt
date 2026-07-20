add_executable(vernon-opt
  vernon_opt.cpp
)

target_link_libraries(vernon-opt PRIVATE
  MLIRVernonDialect
  MLIRVernonTransforms

  # Dialects
  MLIRFuncDialect
  MLIRFuncInlinerExtension
  MLIRArithDialect
  MLIRTensorDialect
  MLIRVectorDialect
  MLIRLinalgDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRBufferizationDialect
  MLIRSPIRVDialect

  # Passes
  MLIRTransforms
  MLIRBufferizationTransforms
  MLIRLinalgTransforms
  MLIRMemRefTransforms
  MLIRSPIRVTransforms

  # Core
  MLIRIR
  MLIRPass
  MLIRParser
  MLIRSupport
  MLIROptLib
)