cmake_minimum_required(VERSION 3.24)

find_package(
    Python 3.11...<3.15
    COMPONENTS Interpreter
    REQUIRED)
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m vernon_dsl.runtime_source --cmake-dir
    RESULT_VARIABLE _vernon_runtime_source_result
    OUTPUT_VARIABLE VERNON_RUNTIME_SOURCE_DIR
    ERROR_VARIABLE _vernon_runtime_source_error
    OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT
   _vernon_runtime_source_result
   EQUAL
   0
   OR NOT EXISTS "${VERNON_RUNTIME_SOURCE_DIR}/CMakeLists.txt")
    message(
        FATAL_ERROR
            "Python '${Python_EXECUTABLE}' did not provide VernonRuntime sources from an installed vernon-lang wheel: ${_vernon_runtime_source_error}"
    )
endif()
file(READ "${VERNON_RUNTIME_SOURCE_DIR}/VERSION" _vernon_runtime_version)
string(STRIP "${_vernon_runtime_version}" _vernon_runtime_version)
project(
    VernonExternalEngine
    VERSION "${_vernon_runtime_version}"
    LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(EMSCRIPTEN)
    set(_vernon_runtime_profile web)
    add_compile_options("-fexceptions")
    add_link_options("-sDISABLE_EXCEPTION_CATCHING=0")
else()
    set(_vernon_runtime_profile desktop)
    include(FetchContent)
    set(GLFW_BUILD_DOCS
        OFF
        CACHE BOOL "" FORCE)
    set(GLFW_BUILD_EXAMPLES
        OFF
        CACHE BOOL "" FORCE)
    set(GLFW_BUILD_TESTS
        OFF
        CACHE BOOL "" FORCE)
    set(GLFW_INSTALL
        OFF
        CACHE BOOL "" FORCE)
    FetchContent_Declare(
        glfw
        GIT_REPOSITORY "https://github.com/glfw/glfw.git"
        GIT_TAG "3.4"
        GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(glfw)
endif()

set(VERNON_RUNTIME_PROFILE
    "${_vernon_runtime_profile}"
    CACHE STRING "" FORCE)
set(VERNON_RUNTIME_LIBRARY_TYPE
    STATIC
    CACHE STRING "" FORCE)
set(VERNON_ENABLE_METAL_RUNTIME
    OFF
    CACHE BOOL "" FORCE)
set(VERNON_ENABLE_CUDA_RUNTIME
    OFF
    CACHE BOOL "" FORCE)
set(VERNON_ENABLE_VULKAN_RUNTIME
    OFF
    CACHE BOOL "" FORCE)
set(VERNON_ENABLE_DIRECTX12_RUNTIME
    OFF
    CACHE BOOL "" FORCE)
set(BUILD_TESTING
    OFF
    CACHE BOOL "" FORCE)
message(STATUS "Using VernonRuntime sources from installed wheel: ${VERNON_RUNTIME_SOURCE_DIR}")
add_subdirectory("${VERNON_RUNTIME_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/vernon-runtime")

# The DSL sources import the repository's examples packages; vernon_dsl itself comes from the installed wheel.
get_filename_component(_vernon_repository_root "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
vernon_configure_program_cooker(
    PYTHON
    "${Python_EXECUTABLE}"
    PYTHON_PATH
    "${_vernon_repository_root}")

set(_vernon_cooked_root "${CMAKE_CURRENT_BINARY_DIR}/cooked")
if(EMSCRIPTEN)
    set(_vernon_cpu_cook_options --cpu-triple wasm32-unknown-emscripten)
    set(_vernon_graphics_target opengles)
    set(_vernon_graphics_cook_options --opengl-version 300)
else()
    set(_vernon_cpu_cook_options)
    set(_vernon_graphics_target opengl)
    set(_vernon_graphics_cook_options --opengl-version 330)
endif()
vernon_add_program_asset(
    vernon-external-engine-fractal
    SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/fractal_pipeline.py:asset"
    TARGET
    cpu
    OUTPUT_DIRECTORY
    "${_vernon_cooked_root}/fractal"
    COOK_OPTIONS
    ${_vernon_cpu_cook_options}
    DEPENDS
    "${_vernon_repository_root}/examples/fractal.py"
    MANIFEST_VARIABLE
    _vernon_cpu_manifest)
vernon_add_program_asset(
    vernon-external-engine-mandelbulb
    SOURCE
    "${_vernon_repository_root}/examples/shader_lib/mandelbulb.py:mandelbulb_asset"
    TARGET
    ${_vernon_graphics_target}
    OUTPUT_DIRECTORY
    "${_vernon_cooked_root}/mandelbulb"
    COOK_OPTIONS
    ${_vernon_graphics_cook_options}
    DEPENDS
    "${_vernon_repository_root}/examples/shader_lib/fullscreen.py"
    MANIFEST_VARIABLE
    _vernon_graphics_manifest)

if(EMSCRIPTEN)
    set(_vernon_cpu_runtime_dir "/vernon-cpu-bundle")
    set(_vernon_graphics_runtime_dir "/vernon-graphics-bundle")
else()
    set(_vernon_cpu_runtime_dir "${_vernon_cooked_root}/fractal")
    set(_vernon_graphics_runtime_dir "${_vernon_cooked_root}/mandelbulb")
endif()
set(_vernon_embedded_bundles "${CMAKE_CURRENT_BINARY_DIR}/generated/embedded_bundles.h")
add_custom_command(
    OUTPUT "${_vernon_embedded_bundles}"
    COMMAND
        "${CMAKE_COMMAND}" "-DTEMPLATE=${CMAKE_CURRENT_SOURCE_DIR}/embedded_bundles.h.in"
        "-DOUTPUT=${_vernon_embedded_bundles}" "-DCPU_MANIFEST=${_vernon_cpu_manifest}"
        "-DCPU_RUNTIME_DIR=${_vernon_cpu_runtime_dir}" "-DGRAPHICS_MANIFEST=${_vernon_graphics_manifest}"
        "-DGRAPHICS_RUNTIME_DIR=${_vernon_graphics_runtime_dir}" -P "${CMAKE_CURRENT_SOURCE_DIR}/embed_bundles.cmake"
    DEPENDS "${_vernon_cpu_manifest}"
            "${_vernon_graphics_manifest}"
            "${CMAKE_CURRENT_SOURCE_DIR}/embedded_bundles.h.in"
            "${CMAKE_CURRENT_SOURCE_DIR}/embed_bundles.cmake"
    COMMENT "Embedding cooked Program manifests"
    VERBATIM)

add_executable(
    vernon-external-engine
    "${_vernon_embedded_bundles}"
    cpu_fractal.cpp
    graphics_host.cpp
    main.cpp
    mandelbulb.cpp
    split_screen.cpp)
target_include_directories(vernon-external-engine PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
target_link_libraries(
    vernon-external-engine
    PRIVATE VernonRuntime
            Vernon::RHI
            vernon-external-engine-fractal
            vernon-external-engine-mandelbulb)
if(NOT EMSCRIPTEN)
    target_link_libraries(vernon-external-engine PRIVATE glfw)
endif()

if(EMSCRIPTEN)
    target_link_options(
        vernon-external-engine
        PRIVATE
        "-sALLOW_MEMORY_GROWTH=1"
        "-sASSERTIONS=1"
        "-sEXIT_RUNTIME=1"
        "-sENVIRONMENT=web,node"
        "-sFULL_ES3=1"
        "-sUSE_GLFW=3"
        "--embed-file=${_vernon_cooked_root}/fractal@${_vernon_cpu_runtime_dir}"
        "--embed-file=${_vernon_cooked_root}/mandelbulb@${_vernon_graphics_runtime_dir}")
    add_custom_command(
        TARGET vernon-external-engine
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_LIST_DIR}/browser.html"
                "$<TARGET_FILE_DIR:vernon-external-engine>/index.html"
        VERBATIM)
endif()

enable_testing()
add_test(NAME vernon-external-engine-headless COMMAND vernon-external-engine --headless)
set_tests_properties(vernon-external-engine-headless PROPERTIES PASS_REGULAR_EXPRESSION
                                                                "Vernon CPU fractal headless checksum: [1-9]")
