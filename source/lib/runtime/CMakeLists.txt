cmake_minimum_required(VERSION 3.20)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/VernonVersions.cmake")
    include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/VernonVersions.cmake")
else()
    include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/VernonVersions.cmake")
endif()
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(
        VernonRuntime
        VERSION ${VERNON_PROJECT_VERSION}
        LANGUAGES C CXX)
    set(CMAKE_CXX_STANDARD
        17
        CACHE STRING "C++ standard to conform to")
    include(CTest)
endif()

file(READ "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" _vernon_runtime_version)
string(STRIP "${_vernon_runtime_version}" _vernon_runtime_version)
if(NOT
   _vernon_runtime_version
   STREQUAL
   VERNON_RELEASE_VERSION)
    message(FATAL_ERROR "Runtime VERSION (${_vernon_runtime_version}) must match "
                        "Vernon release version (${VERNON_RELEASE_VERSION})")
endif()

set(VERNON_RUNTIME_PROFILE
    "desktop"
    CACHE STRING "Runtime platform profile: desktop, mobile, or web")
set_property(
    CACHE VERNON_RUNTIME_PROFILE
    PROPERTY STRINGS
             desktop
             mobile
             web)
if(NOT
   VERNON_RUNTIME_PROFILE
   MATCHES
   "^(desktop|mobile|web)$")
    message(FATAL_ERROR "VERNON_RUNTIME_PROFILE must be desktop, mobile, or web")
endif()

set(_vernon_default_library_type SHARED)
if(VERNON_RUNTIME_PROFILE STREQUAL "mobile" OR VERNON_RUNTIME_PROFILE STREQUAL "web")
    set(_vernon_default_library_type STATIC)
endif()
set(VERNON_RUNTIME_LIBRARY_TYPE
    "${_vernon_default_library_type}"
    CACHE STRING "Runtime library type: SHARED or STATIC")
set_property(CACHE VERNON_RUNTIME_LIBRARY_TYPE PROPERTY STRINGS SHARED STATIC)
if(NOT
   VERNON_RUNTIME_LIBRARY_TYPE
   MATCHES
   "^(SHARED|STATIC)$")
    message(FATAL_ERROR "VERNON_RUNTIME_LIBRARY_TYPE must be SHARED or STATIC")
endif()

set(_vernon_default_desktop_backend OFF)
if(VERNON_RUNTIME_PROFILE STREQUAL "desktop")
    set(_vernon_default_desktop_backend ON)
endif()

string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _vernon_target_processor)
set(_vernon_cuda_runtime_supported OFF)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND _vernon_target_processor MATCHES "^(aarch64|amd64|arm64|x86_64)$")
    set(_vernon_cuda_runtime_supported ON)
elseif(WIN32 AND _vernon_target_processor MATCHES "^(amd64|x64|x86_64)$")
    set(_vernon_cuda_runtime_supported ON)
endif()

if(_vernon_cuda_runtime_supported)
    option(VERNON_ENABLE_CUDA_RUNTIME "Enable the CUDA Driver backend" ${_vernon_default_desktop_backend})
else()
    set(VERNON_ENABLE_CUDA_RUNTIME
        OFF
        CACHE BOOL "Enable the CUDA Driver backend" FORCE)
endif()
option(VERNON_ENABLE_VULKAN_RUNTIME "Enable the Vulkan backend" ${_vernon_default_desktop_backend})
if(WIN32)
    option(VERNON_ENABLE_DIRECTX12_RUNTIME "Enable the DirectX 12 backend" ${_vernon_default_desktop_backend})
else()
    set(VERNON_ENABLE_DIRECTX12_RUNTIME
        OFF
        CACHE BOOL "Enable the DirectX 12 backend" FORCE)
endif()
if(VERNON_RUNTIME_PROFILE STREQUAL "web")
    set(VERNON_ENABLE_CUDA_RUNTIME
        OFF
        CACHE BOOL "" FORCE)
    set(VERNON_ENABLE_VULKAN_RUNTIME
        OFF
        CACHE BOOL "" FORCE)
    set(VERNON_ENABLE_DIRECTX12_RUNTIME
        OFF
        CACHE BOOL "" FORCE)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../../include/VernonRuntime.h")
    get_filename_component(_vernon_repository_root "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
    set(_vernon_runtime_target_file "${CMAKE_CURRENT_SOURCE_DIR}/VernonRuntimeTarget.cmake")
else()
    set(_vernon_repository_root "${CMAKE_CURRENT_SOURCE_DIR}")
    set(_vernon_runtime_target_file "${CMAKE_CURRENT_SOURCE_DIR}/cmake/VernonRuntimeTarget.cmake")
endif()
set(VERNON_REPOSITORY_ROOT "${_vernon_repository_root}")

if(NOT TARGET nlohmann_json::nlohmann_json)
    include("${_vernon_repository_root}/cmake/IncludeNlohmannJson.cmake")
    include_nlohmann_json()
endif()
if(VERNON_ENABLE_VULKAN_RUNTIME AND NOT TARGET Vulkan::Headers)
    include("${_vernon_repository_root}/cmake/IncludeVulkanHeaders.cmake")
    include_vulkan_headers()
endif()

include("${_vernon_runtime_target_file}")
vernon_add_runtime()

# The combined build registers tests after compiler and Python targets exist. Register here only when Runtime is
# configured as a standalone project.
if(BUILD_TESTING AND CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    include("${VERNON_REPOSITORY_ROOT}/cmake/VernonTests.cmake")
    vernon_add_tests()
endif()

include(CMakePackageConfigHelpers)
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/VernonRuntimeConfig.cmake.in" "${CMAKE_CURRENT_BINARY_DIR}/VernonRuntimeConfig.cmake"
    INSTALL_DESTINATION lib/cmake/VernonRuntime)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/VernonRuntimeConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)
install(
    FILES "${CMAKE_CURRENT_BINARY_DIR}/VernonRuntimeConfig.cmake"
          "${CMAKE_CURRENT_BINARY_DIR}/VernonRuntimeConfigVersion.cmake"
    DESTINATION lib/cmake/VernonRuntime
    COMPONENT VernonDevelopment)

if(SKBUILD)
    install(
        FILES "${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt" "${CMAKE_CURRENT_SOURCE_DIR}/VERSION"
              "${CMAKE_CURRENT_SOURCE_DIR}/VernonRuntimeConfig.cmake.in"
        DESTINATION vernon_dsl/runtime_src
        COMPONENT VernonWheel)
    install(
        FILES "${_vernon_runtime_target_file}" "${_vernon_repository_root}/cmake/VernonVersions.cmake"
        DESTINATION vernon_dsl/runtime_src/cmake
        COMPONENT VernonWheel)
    install(
        FILES "${_VERNON_RUNTIME_INCLUDE_DIR}/VernonVersions.h"
              "${_VERNON_RUNTIME_INCLUDE_DIR}/VernonCommon.h"
              "${_VERNON_RUNTIME_INCLUDE_DIR}/VernonOpenGLContext.h"
              "${_VERNON_RUNTIME_INCLUDE_DIR}/VernonRHI.h"
              "${_VERNON_RUNTIME_INCLUDE_DIR}/VernonRHI.hpp"
              "${_VERNON_RUNTIME_INCLUDE_DIR}/VernonExecutionGraph.h"
              "${_VERNON_RUNTIME_INCLUDE_DIR}/VernonRuntime.h"
              "${_VERNON_RUNTIME_INCLUDE_DIR}/VernonRuntime.hpp"
              "${_VERNON_RUNTIME_INCLUDE_DIR}/VernonRuntimeCore.h"
              "${_VERNON_RUNTIME_INCLUDE_DIR}/VernonRuntimeProvider.h"
              "${_VERNON_RUNTIME_INCLUDE_DIR}/VernonRuntimeRHIAdapter.h"
        DESTINATION vernon_dsl/runtime_src/include
        COMPONENT VernonWheel)
    install(
        FILES "${_VERNON_RUNTIME_INCLUDE_DIR}/vernon-c/Common.h" "${_VERNON_RUNTIME_INCLUDE_DIR}/vernon-c/Runtime.h"
        DESTINATION vernon_dsl/runtime_src/include/vernon-c
        COMPONENT VernonWheel)
    install(
        DIRECTORY "${_VERNON_RUNTIME_IMPL_DIR}/"
        DESTINATION vernon_dsl/runtime_src/lib/runtime
        COMPONENT VernonWheel
        FILES_MATCHING
        PATTERN "*.cpp"
        PATTERN "*.h")
    install(
        DIRECTORY "${_VERNON_RUNTIME_IMPL_DIR}/../execution_graph/"
        DESTINATION vernon_dsl/runtime_src/lib/execution_graph
        COMPONENT VernonWheel
        FILES_MATCHING
        PATTERN "*.cpp"
        PATTERN "*.h")
    install(
        DIRECTORY "${_VERNON_RUNTIME_IMPL_DIR}/../rhi/"
        DESTINATION vernon_dsl/runtime_src/lib/rhi
        COMPONENT VernonWheel
        FILES_MATCHING
        PATTERN "*.cpp"
        PATTERN "*.h")
    install(
        DIRECTORY "${_VERNON_RUNTIME_IMPL_DIR}/../platform/"
        DESTINATION vernon_dsl/runtime_src/lib/platform
        COMPONENT VernonWheel
        FILES_MATCHING
        PATTERN "*.cpp"
        PATTERN "*.h")
    install(
        DIRECTORY "${_vernon_repository_root}/cmake/"
        DESTINATION vernon_dsl/runtime_src/cmake
        COMPONENT VernonWheel
        FILES_MATCHING
        PATTERN "IncludeNlohmannJson.cmake"
        PATTERN "IncludeVulkanHeaders.cmake")
endif()
