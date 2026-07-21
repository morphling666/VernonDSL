cmake_minimum_required(VERSION 3.20)

project(VernonRuntime LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")

set(VERNON_RUNTIME_PROFILE "desktop" CACHE STRING
    "Runtime platform profile: desktop, mobile, or web")
set_property(CACHE VERNON_RUNTIME_PROFILE PROPERTY STRINGS desktop mobile web)

if(NOT VERNON_RUNTIME_PROFILE MATCHES "^(desktop|mobile|web)$")
  message(FATAL_ERROR
    "VERNON_RUNTIME_PROFILE must be desktop, mobile, or web")
endif()

set(_vernon_default_library_type SHARED)
if(VERNON_RUNTIME_PROFILE STREQUAL "mobile" OR
   VERNON_RUNTIME_PROFILE STREQUAL "web")
  set(_vernon_default_library_type STATIC)
endif()
set(VERNON_RUNTIME_LIBRARY_TYPE "${_vernon_default_library_type}" CACHE STRING
    "Runtime library type: SHARED or STATIC")
set_property(CACHE VERNON_RUNTIME_LIBRARY_TYPE PROPERTY STRINGS SHARED STATIC)
if(NOT VERNON_RUNTIME_LIBRARY_TYPE MATCHES "^(SHARED|STATIC)$")
  message(FATAL_ERROR "VERNON_RUNTIME_LIBRARY_TYPE must be SHARED or STATIC")
endif()

if(VERNON_RUNTIME_PROFILE STREQUAL "desktop")
  set(_vernon_default_desktop_backend ON)
else()
  set(_vernon_default_desktop_backend OFF)
endif()
option(VERNON_ENABLE_CUDA_RUNTIME "Enable the CUDA Driver backend"
       ${_vernon_default_desktop_backend})
option(VERNON_ENABLE_VULKAN_RUNTIME "Enable the Vulkan backend"
       ${_vernon_default_desktop_backend})

if(VERNON_RUNTIME_PROFILE STREQUAL "web")
  set(VERNON_ENABLE_CUDA_RUNTIME OFF CACHE BOOL "" FORCE)
  set(VERNON_ENABLE_VULKAN_RUNTIME OFF CACHE BOOL "" FORCE)
endif()

if(NOT TARGET nlohmann_json::nlohmann_json)
  find_package(nlohmann_json CONFIG REQUIRED)
endif()
if(VERNON_ENABLE_VULKAN_RUNTIME AND NOT TARGET Vulkan::Headers)
  find_package(VulkanHeaders CONFIG REQUIRED)
endif()
if(VERNON_ENABLE_VULKAN_RUNTIME AND NOT TARGET Vulkan::Headers)
  message(FATAL_ERROR "VulkanHeaders did not provide Vulkan::Headers")
endif()

include(cmake/VernonRuntimeTarget.cmake)
vernon_add_runtime()
