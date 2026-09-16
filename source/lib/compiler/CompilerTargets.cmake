add_subdirectory("${VERNON_SOURCE_DIR}/include/mlir/Dialect/Vernon"
                 "${CMAKE_CURRENT_BINARY_DIR}/include/mlir/Dialect/Vernon")
add_subdirectory("${VERNON_SOURCE_DIR}/lib/Dialect/Vernon" "${CMAKE_CURRENT_BINARY_DIR}/lib/Dialect/Vernon")
add_subdirectory("${VERNON_SOURCE_DIR}/include/mlir/Dialect/VernonProgram"
                 "${CMAKE_CURRENT_BINARY_DIR}/include/mlir/Dialect/VernonProgram")
add_subdirectory("${VERNON_SOURCE_DIR}/lib/Dialect/VernonProgram"
                 "${CMAKE_CURRENT_BINARY_DIR}/lib/Dialect/VernonProgram")
include("${VERNON_REPOSITORY_ROOT}/cmake/IncludeDxc.cmake")

set(VERNON_COMPILER_ENGINE_SOURCES
    compiler_artifacts.cpp
    compiler_cpu.cpp
    compiler_cuda.cpp
    compiler_dispatch.cpp
    compiler_dxc.cpp
    compiler_frontend.cpp
    compiler_internal.cpp
    compiler_json.cpp
    compiler_program_abi.cpp
    compiler_program_aggregation.cpp
    compiler_program_assembly.cpp
    compiler_program_boundary.cpp
    compiler_program_builtin.cpp
    compiler_program_capture.cpp
    compiler_program_compute.cpp
    compiler_program_derivative.cpp
    compiler_program_finalization.cpp
    compiler_program_graph.cpp
    compiler_program_graphics.cpp
    compiler_program_implementation.cpp
    compiler_program_lowering.cpp
    compiler_program_publication.cpp
    compiler_program_reflection.cpp
    compiler_program_semantic_type.cpp
    compiler_program_serializer.cpp
    compiler_program_stage.cpp
    compiler_program_storage.cpp
    compiler_program_tape.cpp
    compiler_program_target_aggregation.cpp
    compiler_reflection.cpp
    compiler_spirv.cpp
    compiler_spirv_cross.cpp
    VernonCpuAbiWrapper.cpp
    VernonCpuHalfConversion.cpp)

llvm_map_components_to_libnames(
    VERNON_LLVM_JIT_LIBS
    AllTargetsAsmParsers
    AllTargetsCodeGens
    AllTargetsDescs
    AllTargetsInfos
    AsmParser
    Core
    OrcJIT
    Passes
    Target
    TargetParser
    native)

add_library(VernonCompilerEngine OBJECT ${VERNON_COMPILER_ENGINE_SOURCES})
set_target_properties(VernonCompilerEngine PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_definitions(VernonCompilerEngine PRIVATE VERNON_DSL_COMPILER_STATIC)
target_include_directories(
    VernonCompilerEngine
    PUBLIC $<BUILD_INTERFACE:${VERNON_SOURCE_DIR}/include>
    PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/include")
target_link_libraries(
    VernonCompilerEngine
    PUBLIC VernonCompilerBuildHeaders
           MLIRArithToLLVM
           MLIRBufferizationTransforms
           MLIRControlFlowToLLVM
           MLIRConvertToLLVMPass
           MLIRVernonDialect
           MLIRVernonProgramDialect
           MLIRVernonProgramTransforms
           MLIRVernonTransforms
           MLIRFuncDialect
           MLIRFuncToLLVM
           MLIRGPUToSPIRV
           MLIRGPUPipelines
           MLIRIR
           MLIRIndexToLLVM
           MLIRLinalgTransforms
           MLIRMathToLLVM
           MLIRMathToSPIRV
           MLIRMemRefToLLVM
           MLIRParser
           MLIRPass
           MLIRRegisterAllDialects
           MLIRRegisterAllExtensions
           MLIRSCFTransforms
           MLIRSCFToControlFlow
           MLIRSPIRVDialect
           MLIRSPIRVSerialization
           MLIRSPIRVTransforms
           MLIRSupport
           MLIRToLLVMIRTranslationRegistration
           MLIRUBToLLVM
           MLIRVectorToLLVM
           ${VERNON_LLVM_JIT_LIBS}
           spirv-cross-glsl
           spirv-cross-hlsl
           spirv-cross-msl)

add_library(VernonDSLCompiler SHARED VernonCompiler.cpp)
target_compile_definitions(VernonDSLCompiler PRIVATE VERNON_DSL_COMPILER_BUILD)
target_link_libraries(VernonDSLCompiler PRIVATE VernonCompilerEngine)
if(MSVC)
    target_compile_options(VernonCompilerEngine PRIVATE /EHsc)
    target_compile_options(VernonDSLCompiler PRIVATE /EHsc)
elseif(NOT APPLE)
    # Keep LLVM/MLIR symbols pulled from static archives private to the compiler library. Exporting them lets Mesa's
    # Vulkan driver bind against Vernon's LLVM copy and causes duplicate command-line option registration at load.
    target_link_options(VernonDSLCompiler PRIVATE "LINKER:--exclude-libs,ALL")
endif()
if(NOT MSVC)
    target_compile_options(VernonCompilerEngine PRIVATE -fexceptions)
    target_compile_options(VernonDSLCompiler PRIVATE -fexceptions)
endif()
target_include_directories(VernonDSLCompiler PUBLIC $<BUILD_INTERFACE:${VERNON_SOURCE_DIR}/include>
                                                    $<INSTALL_INTERFACE:include>)

if(SKBUILD)
    set(_vernon_compiler_runtime_destination vernon_dsl)
    set(_vernon_compiler_library_destination vernon_dsl)
else()
    set(_vernon_compiler_runtime_destination bin)
    set(_vernon_compiler_library_destination lib)
endif()
vernon_configure_dxc(VernonDSLCompiler "${_vernon_compiler_runtime_destination}")
if(VERNON_DXC_EXECUTABLE)
    file(TO_CMAKE_PATH "${VERNON_DXC_EXECUTABLE}" _vernon_compiler_dxc_path)
    target_compile_definitions(VernonCompilerEngine PRIVATE VERNON_DXC_EXECUTABLE="${_vernon_compiler_dxc_path}")
endif()
install(
    TARGETS VernonDSLCompiler
    RUNTIME DESTINATION ${_vernon_compiler_runtime_destination} COMPONENT VernonWheel
    LIBRARY DESTINATION ${_vernon_compiler_library_destination} COMPONENT VernonWheel
    ARCHIVE DESTINATION lib COMPONENT VernonDevelopment)
install(
    FILES "${VERNON_SOURCE_DIR}/include/VernonCommon.h" "${VERNON_SOURCE_DIR}/include/VernonCompiler.h"
          "${VERNON_SOURCE_DIR}/include/VernonCpuWorkgroupABI.h"
    DESTINATION include
    COMPONENT VernonDevelopment)
install(
    FILES "${VERNON_SOURCE_DIR}/include/vernon-c/Common.h" "${VERNON_SOURCE_DIR}/include/vernon-c/Compiler.h"
    DESTINATION include/vernon-c
    COMPONENT VernonDevelopment)

add_library(vernon-compile-packaging STATIC "${VERNON_SOURCE_DIR}/tools/vernon_compile_packaging.cpp")
target_include_directories(vernon-compile-packaging PUBLIC "${VERNON_SOURCE_DIR}/tools")
target_link_libraries(vernon-compile-packaging PRIVATE LLVMSupport LLVMTargetParser)
add_executable(vernon-compile "${VERNON_SOURCE_DIR}/tools/vernon_compile.cpp")
target_link_libraries(vernon-compile PRIVATE vernon-compile-packaging VernonDSLCompiler)
if(MSVC)
    target_compile_options(vernon-compile-packaging PRIVATE /EHsc)
    target_compile_options(vernon-compile PRIVATE /EHsc)
endif()
install(TARGETS vernon-compile RUNTIME DESTINATION ${_vernon_compiler_runtime_destination} COMPONENT VernonWheel)

add_subdirectory("${VERNON_SOURCE_DIR}/tools/vernon_opt" "${CMAKE_CURRENT_BINARY_DIR}/tools/vernon_opt")
