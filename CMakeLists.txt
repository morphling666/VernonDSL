cmake_minimum_required(VERSION 3.20.0)

project(VernonDSL VERSION 0.1.0 LANGUAGES CXX C)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")
include(CTest)

option(VERNON_ENABLE_SPIRV_CROSS
  "Build OpenGL, OpenGL ES, Metal, and HLSL source backends" ON)
option(VERNON_ENABLE_COMPILER "Build the Vernon compiler" ON)
option(VERNON_ENABLE_RUNTIME "Build the Vernon kernel runtime" ON)
option(VERNON_ENABLE_CUDA_RUNTIME
  "Enable dynamically loaded CUDA Driver runtime backend" ON)
option(VERNON_ENABLE_VULKAN_RUNTIME
  "Enable dynamically loaded Vulkan runtime backend" ON)
option(VERNON_ENABLE_PYTHON_BINDINGS "Build the nanobind Python module" ON)
if(VERNON_ENABLE_RUNTIME)
  find_package(nlohmann_json CONFIG QUIET)
  if(NOT nlohmann_json_FOUND)
    include(FetchContent)
    FetchContent_Declare(nlohmann_json
      GIT_REPOSITORY https://github.com/nlohmann/json.git
      GIT_TAG v3.12.0
      GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(nlohmann_json)
  endif()
endif()
if(VERNON_ENABLE_VULKAN_RUNTIME AND VERNON_ENABLE_RUNTIME)
  include(FetchContent)
  FetchContent_Declare(vulkan_headers
    GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Headers.git
    GIT_TAG v1.4.357
    GIT_SHALLOW TRUE
  )
  FetchContent_MakeAvailable(vulkan_headers)
endif()
if(NOT VERNON_ENABLE_COMPILER)
  add_subdirectory(source)
  return()
endif()
if(VERNON_ENABLE_SPIRV_CROSS)
  include(FetchContent)
  set(SPIRV_CROSS_CLI OFF CACHE BOOL "" FORCE)
  set(SPIRV_CROSS_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
  set(SPIRV_CROSS_ENABLE_C_API OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(spirv_cross
    GIT_REPOSITORY https://github.com/KhronosGroup/SPIRV-Cross.git
    GIT_TAG vulkan-sdk-1.4.350.1
    GIT_SHALLOW TRUE
  )
  FetchContent_GetProperties(spirv_cross)
  if(NOT spirv_cross_POPULATED)
    if(POLICY CMP0169)
      cmake_policy(PUSH)
      cmake_policy(SET CMP0169 OLD)
    endif()
    FetchContent_Populate(spirv_cross)
    if(POLICY CMP0169)
      cmake_policy(POP)
    endif()
    # SPIRV-Cross is statically linked into the compiler. Excluding its
    # directory keeps its unrelated install rules out of the Vernon SDK.
    add_subdirectory(
      ${spirv_cross_SOURCE_DIR}
      ${spirv_cross_BINARY_DIR}
      EXCLUDE_FROM_ALL
    )
  endif()
endif()
set(MLIR_DIR "${CMAKE_CURRENT_SOURCE_DIR}/llvm-project/install/lib/cmake/mlir"
  CACHE PATH "Directory containing MLIRConfig.cmake")
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
