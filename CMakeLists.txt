cmake_minimum_required(VERSION 3.20.0)

project(VernonDSL VERSION 0.1.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")
set(MLIR_DIR ${CMAKE_CURRENT_SOURCE_DIR}/llvm-project/install/lib/cmake/mlir)
# Find MLIR
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

# Add MLIR's CMake modules to the path
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

# Include MLIR CMake functions
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/source/include)
include_directories(${CMAKE_BINARY_DIR}/source/include)

# For tablegen, we need access to MLIR source .td files (they're not installed)
# Point to the source directory for tablegen includes
set(MLIR_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/llvm-project/mlir)
set(MLIR_TABLEGEN_INCLUDE_DIRS 
  ${MLIR_SOURCE_DIR}/include
  ${LLVM_INCLUDE_DIRS}
)

# Add Vernon dialect subdirectories
add_subdirectory(source)
