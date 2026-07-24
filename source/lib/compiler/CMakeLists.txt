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

include("${VERNON_REPOSITORY_ROOT}/cmake/IncludeGTest.cmake")
include("${VERNON_REPOSITORY_ROOT}/cmake/IncludeSpirvCross.cmake")
include_spirv_cross()

set(MLIR_DIR
    "${VERNON_REPOSITORY_ROOT}/llvm-project/install/lib/cmake/mlir"
    CACHE PATH "Directory containing MLIRConfig.cmake")
set(LLD_DIR
    "${VERNON_REPOSITORY_ROOT}/llvm-project/install/lib/cmake/lld"
    CACHE PATH "Directory containing LLDConfig.cmake")
if(LLD_DIR MATCHES "-NOTFOUND$")
    set(LLD_DIR
        "${VERNON_REPOSITORY_ROOT}/llvm-project/install/lib/cmake/lld"
        CACHE PATH "Directory containing LLDConfig.cmake" FORCE)
endif()
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

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${LLD_INCLUDE_DIRS})
# Project dialect headers and their freshly generated TableGen files must win over any older Vernon headers present in
# the MLIR installation prefix.
include_directories(BEFORE "${VERNON_SOURCE_DIR}/include")
include_directories(BEFORE "${CMAKE_CURRENT_BINARY_DIR}/include")

set(MLIR_SOURCE_DIR "${VERNON_REPOSITORY_ROOT}/llvm-project/mlir")
set(MLIR_TABLEGEN_INCLUDE_DIRS "${MLIR_SOURCE_DIR}/include" ${LLVM_INCLUDE_DIRS})

include("${CMAKE_CURRENT_SOURCE_DIR}/CompilerTargets.cmake")

if(BUILD_TESTING AND NOT VERNON_COMPOSITION_BUILD)
    add_subdirectory("${VERNON_SOURCE_DIR}/tests" "${CMAKE_CURRENT_BINARY_DIR}/tests")
endif()
