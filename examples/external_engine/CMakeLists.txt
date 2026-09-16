cmake_minimum_required(VERSION 3.20)

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
        message(FATAL_ERROR "${directory} must contain exactly one Program manifest")
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
    target
    options
    triple)
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
    add_compile_options("-fexceptions")
    add_link_options("-sDISABLE_EXCEPTION_CATCHING=0")
else()
    if(_vernon_cpu_target_triple MATCHES "^wasm"
       OR NOT
          _vernon_graphics_target_kind
          STREQUAL
          "opengl")
        message(FATAL_ERROR "Desktop requires native CPU and OpenGL bundles")
    endif()
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

set(VERNON_EXTERNAL_ENGINE_GRAPHICS_MANIFEST "${_vernon_graphics_external_manifest}")
if(EMSCRIPTEN)
    set(VERNON_EXTERNAL_ENGINE_CPU_RUNTIME_DIR "/vernon-cpu-bundle")
    set(VERNON_EXTERNAL_ENGINE_GRAPHICS_RUNTIME_DIR "/vernon-graphics-bundle")
else()
    set(VERNON_EXTERNAL_ENGINE_CPU_RUNTIME_DIR "${VERNON_EXTERNAL_ENGINE_CPU_COOKED_DIR}")
    set(VERNON_EXTERNAL_ENGINE_GRAPHICS_RUNTIME_DIR "${VERNON_EXTERNAL_ENGINE_GRAPHICS_COOKED_DIR}")
endif()

function(
    vernon_chunk_raw_string
    input_variable
    delimiter
    output_variable)
    set(content "${${input_variable}}")
    set(literal)
    set(first_chunk TRUE)
    string(LENGTH "${content}" remaining)
    while(remaining GREATER 0)
        if(remaining GREATER 12000)
            set(chunk_size 12000)
        else()
            set(chunk_size "${remaining}")
        endif()
        string(
            SUBSTRING "${content}"
                      0
                      "${chunk_size}"
                      chunk)
        if(NOT first_chunk)
            string(APPEND literal ")${delimiter}\"\n    R\"${delimiter}(")
        endif()
        string(APPEND literal "${chunk}")
        set(first_chunk FALSE)
        string(
            SUBSTRING "${content}"
                      "${chunk_size}"
                      -1
                      content)
        string(LENGTH "${content}" remaining)
    endwhile()
    set(${output_variable}
        "${literal}"
        PARENT_SCOPE)
endfunction()

vernon_chunk_raw_string(VERNON_EXTERNAL_ENGINE_CPU_MANIFEST vernon_cpu VERNON_EXTERNAL_ENGINE_CPU_MANIFEST_LITERAL)
vernon_chunk_raw_string(VERNON_EXTERNAL_ENGINE_GRAPHICS_MANIFEST vernon_graphics
                        VERNON_EXTERNAL_ENGINE_GRAPHICS_MANIFEST_LITERAL)
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
message(STATUS "Using VernonRuntime sources from installed wheel: ${VERNON_RUNTIME_SOURCE_DIR}")
add_subdirectory("${VERNON_RUNTIME_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/vernon-runtime")

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
target_link_libraries(vernon-external-engine PRIVATE VernonRuntime Vernon::RHI)
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
        "--embed-file=${VERNON_EXTERNAL_ENGINE_CPU_COOKED_DIR}@${VERNON_EXTERNAL_ENGINE_CPU_RUNTIME_DIR}"
        "--embed-file=${VERNON_EXTERNAL_ENGINE_GRAPHICS_COOKED_DIR}@${VERNON_EXTERNAL_ENGINE_GRAPHICS_RUNTIME_DIR}")
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
