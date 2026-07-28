cmake_minimum_required(VERSION 3.20)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(
        VernonDSLCompiler
        VERSION 0.1.0
        LANGUAGES C CXX)
    set(CMAKE_CXX_STANDARD
        17
        CACHE STRING "C++ standard to conform to")
    include(CTest)
endif()

get_filename_component(VERNON_REPOSITORY_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
set(VERNON_SOURCE_DIR "${VERNON_REPOSITORY_ROOT}/source")

include("${VERNON_REPOSITORY_ROOT}/cmake/IncludeSpirvCross.cmake")
include_spirv_cross()

set(MLIR_DIR
    "${VERNON_REPOSITORY_ROOT}/llvm-project/install/lib/cmake/mlir"
    CACHE PATH "Directory containing MLIRConfig.cmake")
set(LLD_DIR
    "${VERNON_REPOSITORY_ROOT}/llvm-project/install/lib/cmake/lld"
    CACHE PATH "Directory containing LLDConfig.cmake")
find_package(MLIR REQUIRED CONFIG)
find_package(LLD REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Using LLDConfig.cmake in: ${LLD_DIR}")

list(
    APPEND
    CMAKE_MODULE_PATH
    "${MLIR_CMAKE_DIR}"
    "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

set(LLVM_RUNTIME_OUTPUT_INTDIR "${CMAKE_CURRENT_BINARY_DIR}/bin")
set(LLVM_LIBRARY_OUTPUT_INTDIR "${CMAKE_CURRENT_BINARY_DIR}/lib")
set(MLIR_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")

set(MLIR_SOURCE_DIR "${VERNON_REPOSITORY_ROOT}/llvm-project/mlir")
set(MLIR_TABLEGEN_INCLUDE_DIRS "${VERNON_SOURCE_DIR}/include" "${MLIR_SOURCE_DIR}/include" ${LLVM_INCLUDE_DIRS})

add_library(VernonCompilerBuildHeaders INTERFACE)
target_include_directories(
    VernonCompilerBuildHeaders
    INTERFACE "${VERNON_SOURCE_DIR}/include"
              "${CMAKE_CURRENT_BINARY_DIR}/include"
              ${LLVM_INCLUDE_DIRS}
              ${MLIR_INCLUDE_DIRS}
              ${LLD_INCLUDE_DIRS})

include("${CMAKE_CURRENT_SOURCE_DIR}/CompilerTargets.cmake")
foreach(
    _vernon_compiler_target IN
    ITEMS MLIRVernonDialect
          MLIRVernonTransforms
          VernonDSLCompiler
          vernon-compile-packaging
          vernon-compile
          vernon-opt)
    target_link_libraries(${_vernon_compiler_target} PRIVATE VernonCompilerBuildHeaders)
    target_include_directories(${_vernon_compiler_target} BEFORE PRIVATE "${VERNON_SOURCE_DIR}/include"
                                                                         "${CMAKE_CURRENT_BINARY_DIR}/include")
endforeach()

# The combined build registers tests after Runtime and Python targets exist. Register here only when the compiler is
# configured as a standalone project.
if(BUILD_TESTING AND CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    include("${VERNON_REPOSITORY_ROOT}/cmake/VernonTests.cmake")
    vernon_add_tests()
endif()
