add_executable(vernon-opt
  vernon_opt.cpp
)

target_link_libraries(vernon-opt PRIVATE
  MLIRVernonDialect
  MLIRVernonTransforms

  MLIRFuncDialect
  MLIRSPIRVDialect
  MLIRArithDialect
  MLIRTensorDialect
  MLIRVectorDialect
  MLIRLinalgDialect
  
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIROptLib
  MLIRParser
)