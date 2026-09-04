cmake_minimum_required(VERSION 3.20)

file(READ "${CMAKE_CURRENT_LIST_DIR}/../../source/lib/runtime/VERSION" _vernon_runtime_version)
string(STRIP "${_vernon_runtime_version}" _vernon_runtime_version)
project(
    VernonExternalEngine
    VERSION "${_vernon_runtime_version}"
    LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(VERNON_EXTERNAL_ENGINE_CPU_COOKED_DIR
    ""
    CACHE PATH "Cooked native or wasm CPU fractal bundle")
set(VERNON_EXTERNAL_ENGINE_GRAPHICS_COOKED_DIR
    ""
    CACHE PATH "Cooked OpenGL or OpenGL ES Mandelbulb bundle")
if(NOT VERNON_EXTERNAL_ENGINE_CPU_COOKED_DIR OR NOT VERNON_EXTERNAL_ENGINE_GRAPHICS_COOKED_DIR)
    message(FATAL_ERROR "Set VERNON_EXTERNAL_ENGINE_CPU_COOKED_DIR and VERNON_EXTERNAL_ENGINE_GRAPHICS_COOKED_DIR")
endif()
get_filename_component(VERNON_EXTERNAL_ENGINE_CPU_COOKED_DIR "${VERNON_EXTERNAL_ENGINE_CPU_COOKED_DIR}" ABSOLUTE)
get_filename_component(VERNON_EXTERNAL_ENGINE_GRAPHICS_COOKED_DIR "${VERNON_EXTERNAL_ENGINE_GRAPHICS_COOKED_DIR}"
                       ABSOLUTE)

function(vernon_find_single_manifest directory output)
    file(
        GLOB
        manifests
        CONFIGURE_DEPENDS
        "${directory}/*.program.json")
    list(LENGTH manifests manifest_count)
    if(NOT
       manifest_count
       EQUAL
       1)
        message(FATAL_ERROR "${directory} must contain exactly one pipeline manifest")
    endif()
    list(
        GET
        manifests
        0
        manifest)
    set(${output}
        "${manifest}"
        PARENT_SCOPE)
endfunction()

vernon_find_single_manifest("${VERNON_EXTERNAL_ENGINE_CPU_COOKED_DIR}" _vernon_cpu_manifest)
vernon_find_single_manifest("${VERNON_EXTERNAL_ENGINE_GRAPHICS_COOKED_DIR}" _vernon_graphics_manifest)
file(READ "${_vernon_cpu_manifest}" VERNON_EXTERNAL_ENGINE_CPU_MANIFEST)
file(READ "${_vernon_graphics_manifest}" _vernon_graphics_external_manifest)

string(
    JSON
    _vernon_cpu_target_kind
    GET
    "${VERNON_EXTERNAL_ENGINE_CPU_MANIFEST}"
    target
    kind)
if(NOT
   _vernon_cpu_target_kind
   STREQUAL
   "cpu")
    message(FATAL_ERROR "The CPU bundle must target cpu")
endif()
string(
    JSON
    _vernon_cpu_target_triple
    GET
    "${VERNON_EXTERNAL_ENGINE_CPU_MANIFEST}"
    runtime_requirements
    target_triple)
string(
    JSON
    _vernon_graphics_target_kind
    GET
    "${_vernon_graphics_external_manifest}"
    target
    kind)

file(
    GLOB
    _vernon_registration_sources
    CONFIGURE_DEPENDS
    "${VERNON_EXTERNAL_ENGINE_CPU_COOKED_DIR}/vernon_cpu_registration_*.c")
file(
    GLOB
    _vernon_registration_headers
    CONFIGURE_DEPENDS
    "${VERNON_EXTERNAL_ENGINE_CPU_COOKED_DIR}/vernon_cpu_registration_*.h")
file(
    GLOB
    _vernon_cpu_objects
    CONFIGURE_DEPENDS
    "${VERNON_EXTERNAL_ENGINE_CPU_COOKED_DIR}/artifacts/*.o"
    "${VERNON_EXTERNAL_ENGINE_CPU_COOKED_DIR}/artifacts/*.obj")
list(LENGTH _vernon_registration_sources _vernon_registration_source_count)
list(LENGTH _vernon_registration_headers _vernon_registration_header_count)
if(NOT
   _vernon_registration_source_count
   EQUAL
   1
   OR NOT
      _vernon_registration_header_count
      EQUAL
      1
   OR NOT _vernon_cpu_objects)
    message(FATAL_ERROR "The CPU bundle requires registration source/header and relocatable objects")
endif()
list(
    GET
    _vernon_registration_sources
    0
    _vernon_registration_source)
list(
    GET
    _vernon_registration_headers
    0
    _vernon_registration_header)
file(READ "${_vernon_registration_header}" _vernon_registration_declaration)
string(
    REGEX MATCH
          "VernonStatus[ \t\r\n]+(vernonRegisterCpuArtifacts_[0-9a-f]+)\\(void\\)"
          _vernon_registration_match
          "${_vernon_registration_declaration}")
if(NOT CMAKE_MATCH_1)
    message(FATAL_ERROR "Could not identify the generated CPU registration function")
endif()
set(_vernon_registration_function "${CMAKE_MATCH_1}")
set_source_files_properties(${_vernon_cpu_objects} PROPERTIES EXTERNAL_OBJECT TRUE)

if(EMSCRIPTEN)
    if(NOT
       _vernon_cpu_target_triple
       STREQUAL
       "wasm32-unknown-emscripten"
       OR NOT
          _vernon_graphics_target_kind
          STREQUAL
          "opengles")
        message(FATAL_ERROR "Emscripten requires wasm CPU and OpenGL ES bundles")
    endif()
    set(_vernon_runtime_profile web)
else()
    if(_vernon_cpu_target_triple MATCHES "^wasm"
       OR NOT
          _vernon_graphics_target_kind
          STREQUAL
          "opengl")
        message(FATAL_ERROR "Desktop requires native CPU and OpenGL bundles")
    endif()
    set(_vernon_runtime_profile desktop)
    include("${CMAKE_CURRENT_LIST_DIR}/../../cmake/IncludeGlfw.cmake")
    include_glfw()
endif()

find_package(
    Python3
    COMPONENTS Interpreter
    REQUIRED)
set(_vernon_embedded_graphics_manifest "${CMAKE_CURRENT_BINARY_DIR}/generated/graphics.program.json")
execute_process(
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_LIST_DIR}/embed_text_artifacts.py" "${_vernon_graphics_manifest}"
            "${_vernon_embedded_graphics_manifest}"
    RESULT_VARIABLE _vernon_embed_result
    ERROR_VARIABLE _vernon_embed_error)
if(NOT
   _vernon_embed_result
   EQUAL
   0)
    message(FATAL_ERROR "Failed to embed graphics artifacts: ${_vernon_embed_error}")
endif()
file(READ "${_vernon_embedded_graphics_manifest}" VERNON_EXTERNAL_ENGINE_GRAPHICS_MANIFEST)
configure_file(embedded_bundles.h.in "${CMAKE_CURRENT_BINARY_DIR}/generated/embedded_bundles.h" @ONLY)

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
add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/../../source/lib/runtime" "${CMAKE_CURRENT_BINARY_DIR}/vernon-runtime")

add_executable(
    vernon-external-engine
    "${_vernon_registration_source}"
    ${_vernon_cpu_objects}
    cpu_fractal.cpp
    graphics_host.cpp
    main.cpp
    mandelbulb.cpp
    split_screen.cpp)
target_include_directories(vernon-external-engine PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated"
                                                          "${VERNON_EXTERNAL_ENGINE_CPU_COOKED_DIR}")
target_compile_definitions(vernon-external-engine
                           PRIVATE "VERNON_EXTERNAL_ENGINE_REGISTRATION_FUNCTION=${_vernon_registration_function}")
target_link_libraries(vernon-external-engine PRIVATE VernonRuntime VernonExecutionGraph)
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
        "-sUSE_GLFW=3")
    add_custom_command(
        TARGET vernon-external-engine
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_LIST_DIR}/browser.html"
                "$<TARGET_FILE_DIR:vernon-external-engine>/index.html"
        VERBATIM)
endif()
