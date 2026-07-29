cmake_minimum_required(VERSION 3.20.0)

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/VernonVersions.cmake")
project(
    VernonDSL
    VERSION ${VERNON_RELEASE_VERSION}
    LANGUAGES CXX C)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD
    17
    CACHE STRING "C++ standard to conform to")
include(CTest)

option(VERNON_BUILD_RUNTIME "Build the Vernon Runtime subproject" ON)
option(VERNON_BUILD_COMPILER "Build the Vernon Compiler subproject" ON)
option(VERNON_INSTALL_GIT_HOOK "Install the staged-file formatting hook" OFF)
option(VERNON_ENABLE_SANITIZERS "Build Vernon targets with AddressSanitizer" OFF)

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

if(VERNON_INSTALL_GIT_HOOK AND EXISTS "${CMAKE_SOURCE_DIR}/.git")
    include(cmake/GitHook.cmake)
endif()

add_subdirectory(source)
