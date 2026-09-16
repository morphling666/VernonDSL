add_executable(vernon-opt vernon_opt.cpp)

target_link_libraries(
    vernon-opt
    PRIVATE MLIRVernonDialect
            MLIRVernonProgramDialect
            MLIRVernonProgramTransforms
            MLIRVernonTransforms
            # Dialects
            MLIRFuncDialect
            MLIRFuncInlinerExtension
            MLIRArithDialect
            MLIRControlFlowDialect
            MLIRIndexDialect
            MLIRLLVMDialect
            MLIRTensorDialect
            MLIRVectorDialect
            MLIRLinalgDialect
            MLIRMathDialect
            MLIRMemRefDialect
            MLIRSCFDialect
            MLIRBufferizationDialect
            MLIRSPIRVDialect
            # Passes
            MLIRArithToLLVM
            MLIRControlFlowToLLVM
            MLIRConvertToLLVMPass
            MLIRFuncToLLVM
            MLIRIndexToLLVM
            MLIRMathToLLVM
            MLIRMemRefToLLVM
            MLIRSCFToControlFlow
            MLIRUBToLLVM
            MLIRVectorToLLVM
            MLIRTransforms
            MLIRBufferizationTransforms
            MLIRLinalgTransforms
            MLIRMemRefTransforms
            MLIRSPIRVTransforms
            # Core
            MLIRIR
            MLIRPass
            MLIRParser
            MLIRRegisterAllExtensions
            MLIRSupport
            MLIROptLib)
