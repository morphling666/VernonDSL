cmake_minimum_required(VERSION 3.20)
project(VernonRuntimePackageSmoke LANGUAGES C)

find_package(VernonRuntime CONFIG REQUIRED)

add_executable(vernon-runtime-package-smoke main.c)
target_link_libraries(vernon-runtime-package-smoke PRIVATE Vernon::Runtime)
