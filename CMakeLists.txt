cmake_minimum_required(VERSION 3.20.0)

project(
    VernonDSL
    VERSION 0.1.0
    LANGUAGES CXX C)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD
    17
    CACHE STRING "C++ standard to conform to")
include(CTest)

option(VERNON_BUILD_RUNTIME "Build the Vernon Runtime subproject" ON)
option(VERNON_BUILD_COMPILER "Build the Vernon Compiler subproject" ON)
option(VERNON_INSTALL_GIT_HOOK "Install the staged-file formatting hook" ON)
set(VERNON_COMPOSITION_BUILD ON)
set(VERNON_REPOSITORY_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")

if(VERNON_INSTALL_GIT_HOOK AND EXISTS "${CMAKE_SOURCE_DIR}/.git")
    include(cmake/GitHook.cmake)
endif()

add_subdirectory(source)
