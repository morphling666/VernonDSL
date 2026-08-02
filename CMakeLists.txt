cmake_minimum_required(VERSION 3.20.0)

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/VernonVersions.cmake")
if(CMAKE_SYSTEM_NAME STREQUAL "iOS" OR CMAKE_OSX_SYSROOT MATCHES "(^|/)(iphoneos|iphonesimulator)(\\.sdk)?$")
    message(FATAL_ERROR "The top-level VernonDSL project does not support iOS cross-builds. "
                        "Build the packaged standalone Runtime source project instead.")
endif()
if(APPLE)
    option(VERNON_ENABLE_METAL_RUNTIME "Enable the Metal backend" ON)
endif()
if(APPLE AND VERNON_ENABLE_METAL_RUNTIME)
    project(
        VernonDSL
        VERSION ${VERNON_PROJECT_VERSION}
        LANGUAGES CXX
                  C
                  OBJC
                  OBJCXX)
else()
    project(
        VernonDSL
        VERSION ${VERNON_PROJECT_VERSION}
        LANGUAGES CXX C)
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE
        Release
        CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD
    17
    CACHE STRING "C++ standard to conform to")
include(CTest)

option(VERNON_BUILD_RUNTIME "Build the Vernon Runtime subproject" ON)
option(VERNON_BUILD_COMPILER "Build the Vernon Compiler subproject" ON)
option(VERNON_ENABLE_PYTHON_BINDINGS "Build the nanobind Python module" ON)
option(VERNON_INSTALL_GIT_HOOK "Install the staged-file formatting hook" ON)
option(VERNON_ENABLE_SANITIZERS "Build Vernon targets with AddressSanitizer" OFF)

# Every native artifact in a wheel is installed into vernon_dsl. Make that layout relocatable by default instead of
# configuring each target separately.
if(SKBUILD)
    set(CMAKE_BUILD_RPATH_USE_ORIGIN ON)
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH OFF)
    if(APPLE)
        set(CMAKE_INSTALL_RPATH "@loader_path")
    elseif(UNIX)
        set(CMAKE_INSTALL_RPATH "$ORIGIN")
    endif()
endif()

if(VERNON_ENABLE_SANITIZERS)
    if(MSVC)
        add_compile_options(/fsanitize=address /Zi)
        add_link_options(/INCREMENTAL:NO /DEBUG)
    else()
        add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
        add_link_options(-fsanitize=address)
    endif()
endif()

set(VERNON_REPOSITORY_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")

include(cmake/VernonPythonEnvironment.cmake)
if(VERNON_BUILD_RUNTIME
   AND VERNON_BUILD_COMPILER
   AND VERNON_ENABLE_PYTHON_BINDINGS)
    vernon_prepare_python_environment("${VERNON_REPOSITORY_ROOT}")
endif()

if(VERNON_INSTALL_GIT_HOOK AND EXISTS "${CMAKE_SOURCE_DIR}/.git")
    include(cmake/GitHook.cmake)
endif()

add_subdirectory(source)
