if(VERNON_ENABLE_RUNTIME)
  include(VernonRuntimeTarget.cmake)
  vernon_add_runtime()

  if(SKBUILD)
    install(DIRECTORY "${CMAKE_SOURCE_DIR}/runtime_src/"
      DESTINATION vernon_dsl/runtime_src
      COMPONENT VernonWheel)
    install(FILES
      "${CMAKE_CURRENT_SOURCE_DIR}/VernonRuntimeTarget.cmake"
      DESTINATION vernon_dsl/runtime_src/cmake
      COMPONENT VernonWheel)
    install(FILES
      "${CMAKE_CURRENT_SOURCE_DIR}/include/VernonCommon.h"
      "${CMAKE_CURRENT_SOURCE_DIR}/include/VernonRuntime.h"
      DESTINATION vernon_dsl/runtime_src/include
      COMPONENT VernonWheel)
    install(FILES
      "${CMAKE_CURRENT_SOURCE_DIR}/include/vernon-c/Common.h"
      "${CMAKE_CURRENT_SOURCE_DIR}/include/vernon-c/Runtime.h"
      DESTINATION vernon_dsl/runtime_src/include/vernon-c
      COMPONENT VernonWheel)
    install(FILES
      "${CMAKE_CURRENT_SOURCE_DIR}/lib/VernonRuntime.cpp"
      DESTINATION vernon_dsl/runtime_src/lib
      COMPONENT VernonWheel)
    install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/lib/runtime/"
      DESTINATION vernon_dsl/runtime_src/lib/runtime
      COMPONENT VernonWheel
      FILES_MATCHING PATTERN "*.cpp" PATTERN "*.h")
  endif()
endif()

if(NOT VERNON_ENABLE_COMPILER)
  if(BUILD_TESTING)
    add_subdirectory(tests)
  endif()
  return()
endif()

add_subdirectory(include/mlir/Dialect/Vernon)
add_subdirectory(lib/Dialect/Vernon)

if(VERNON_ENABLE_PYTHON_BINDINGS OR BUILD_TESTING)
  set(vernon_python_components Interpreter)
  if(VERNON_ENABLE_PYTHON_BINDINGS)
    list(APPEND vernon_python_components Development.Module)
  endif()
  find_package(Python 3.11.9...<3.12
    COMPONENTS ${vernon_python_components} QUIET)
endif()

add_library(VernonDSLCompiler SHARED
  lib/VernonCpuAbiWrapper.cpp
  lib/VernonCompiler.cpp
)
llvm_map_components_to_libnames(VERNON_LLVM_JIT_LIBS
  AllTargetsAsmParsers
  AllTargetsCodeGens
  AllTargetsDescs
  AllTargetsInfos
  AsmParser
  Core
  OrcJIT
  Target
  native
)
target_compile_definitions(VernonDSLCompiler PRIVATE
  VERNON_DSL_COMPILER_BUILD
)
if(VERNON_ENABLE_SPIRV_CROSS)
  target_compile_definitions(VernonDSLCompiler PRIVATE
    VERNON_HAS_SPIRV_CROSS=1
  )
  if(MSVC)
    target_compile_options(VernonDSLCompiler PRIVATE /EHsc)
  else()
    target_compile_options(VernonDSLCompiler PRIVATE -fexceptions)
  endif()
endif()
target_include_directories(VernonDSLCompiler
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(VernonDSLCompiler PRIVATE
  MLIRArithToLLVM
  MLIRBufferizationTransforms
  MLIRControlFlowToLLVM
  MLIRConvertToLLVMPass
  MLIRVernonDialect
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
  MLIRSCFToControlFlow
  MLIRSPIRVDialect
  MLIRSPIRVSerialization
  MLIRSPIRVTransforms
  MLIRSupport
  MLIRToLLVMIRTranslationRegistration
  MLIRUBToLLVM
  MLIRVectorToLLVM
  ${VERNON_LLVM_JIT_LIBS}
)
if(WIN32)
  target_link_libraries(VernonDSLCompiler PRIVATE lldCommon lldCOFF)
elseif(APPLE)
  target_link_libraries(VernonDSLCompiler PRIVATE lldCommon lldMachO)
else()
  target_link_libraries(VernonDSLCompiler PRIVATE lldCommon lldELF)
endif()
if(VERNON_ENABLE_SPIRV_CROSS)
  target_link_libraries(VernonDSLCompiler PRIVATE
    spirv-cross-glsl
    spirv-cross-hlsl
    spirv-cross-msl
  )
endif()

if(SKBUILD)
  set(_vernon_compiler_runtime_destination vernon_dsl)
  set(_vernon_compiler_library_destination vernon_dsl)
else()
  set(_vernon_compiler_runtime_destination bin)
  set(_vernon_compiler_library_destination lib)
endif()
install(TARGETS VernonDSLCompiler
  RUNTIME DESTINATION ${_vernon_compiler_runtime_destination}
    COMPONENT VernonWheel
  LIBRARY DESTINATION ${_vernon_compiler_library_destination}
    COMPONENT VernonWheel
  ARCHIVE DESTINATION lib
    COMPONENT VernonDevelopment
)
install(FILES
  include/VernonCommon.h
  include/VernonCompiler.h
  DESTINATION include)
install(FILES
  include/vernon-c/Common.h
  include/vernon-c/Compiler.h
  DESTINATION include/vernon-c)

if(VERNON_ENABLE_RUNTIME)
  if(VERNON_ENABLE_PYTHON_BINDINGS)
    if(Python_FOUND)
      execute_process(
        COMMAND ${Python_EXECUTABLE} -m nanobind --cmake_dir
        OUTPUT_VARIABLE VERNON_NANOBIND_CMAKE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE VERNON_NANOBIND_RESULT
      )
      if(VERNON_NANOBIND_RESULT EQUAL 0)
        list(APPEND CMAKE_PREFIX_PATH "${VERNON_NANOBIND_CMAKE_DIR}")
        find_package(nanobind CONFIG QUIET)
      endif()
    endif()
    if(nanobind_FOUND)
      nanobind_add_module(vernon-dsl-native python/native_module.cpp)
      target_link_libraries(vernon-dsl-native PRIVATE
        VernonDSLCompiler Vernon::Runtime)
      if(MSVC)
        target_compile_options(vernon-dsl-native PRIVATE /EHsc)
      endif()
      set_target_properties(vernon-dsl-native PROPERTIES OUTPUT_NAME "_native")
      if(APPLE)
        set_target_properties(vernon-dsl-native PROPERTIES
          INSTALL_RPATH "@loader_path")
      elseif(UNIX)
        set_target_properties(vernon-dsl-native PROPERTIES
          INSTALL_RPATH "$ORIGIN")
      endif()
      install(TARGETS vernon-dsl-native
        RUNTIME DESTINATION vernon_dsl COMPONENT VernonWheel
        LIBRARY DESTINATION vernon_dsl COMPONENT VernonWheel
      )
    else()
      message(STATUS "nanobind not found; skipping vernon_dsl._native")
    endif()
  endif()
endif()

add_library(vernon-compile-packaging STATIC
  tools/vernon_compile_packaging.cpp
)
target_include_directories(vernon-compile-packaging PUBLIC tools)
target_link_libraries(vernon-compile-packaging PUBLIC
  VernonDSLCompiler
  PRIVATE
  LLVMSupport
  LLVMTargetParser
)
add_executable(vernon-compile
  tools/vernon_compile.cpp
)
target_link_libraries(vernon-compile PRIVATE vernon-compile-packaging)
if(MSVC)
  target_compile_options(vernon-compile-packaging PRIVATE /EHsc)
  target_compile_options(vernon-compile PRIVATE /EHsc)
endif()
if(SKBUILD)
  install(TARGETS vernon-compile
    RUNTIME DESTINATION vernon_dsl COMPONENT VernonWheel)
else()
  install(TARGETS vernon-compile RUNTIME DESTINATION bin)
endif()

add_subdirectory(tools/vernon_opt)

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
