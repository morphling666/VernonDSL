cmake_minimum_required(VERSION 3.20)
project(VernonRuntimePackageSmoke LANGUAGES C)

find_package(VernonRuntime CONFIG REQUIRED)
get_target_property(vernon_runtime_type Vernon::Runtime TYPE)
if(vernon_runtime_type STREQUAL "STATIC_LIBRARY")
    enable_language(CXX)
endif()

add_executable(vernon-runtime-package-smoke main.c)
target_link_libraries(vernon-runtime-package-smoke PRIVATE Vernon::Runtime)
