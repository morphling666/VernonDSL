if(NOT Python_EXECUTABLE)
    message(FATAL_ERROR "VERNON_BUILD_BENCHMARKS requires the configured Python environment")
endif()
if(NOT TARGET VernonRuntime OR NOT TARGET vernon-dsl-native)
    message(FATAL_ERROR "VERNON_BUILD_BENCHMARKS requires Runtime, Compiler, and Python bindings")
endif()

include("${VERNON_REPOSITORY_ROOT}/cmake/VernonProgramFixture.cmake")
include("${VERNON_REPOSITORY_ROOT}/cmake/VernonTargetHelpers.cmake")
if(NOT TARGET nlohmann_json::nlohmann_json)
    include("${VERNON_REPOSITORY_ROOT}/cmake/IncludeNlohmannJson.cmake")
    include_nlohmann_json()
endif()

file(
    GLOB_RECURSE
    VERNON_PROGRAM_FIXTURE_PYTHON_SOURCES
    CONFIGURE_DEPENDS
    "${VERNON_REPOSITORY_ROOT}/python/vernon_dsl/*.py")
list(APPEND VERNON_PROGRAM_FIXTURE_PYTHON_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/workloads.py")

# Keep this matrix aligned with source/tests/CMakeLists.txt. Runtime capability probing still decides whether a compiled
# backend is usable on this machine.
set(_vernon_benchmark_targets "cpu|cpu")
if(VERNON_ENABLE_CUDA_RUNTIME)
    list(APPEND _vernon_benchmark_targets "cuda|cuda")
endif()
if(VERNON_ENABLE_VULKAN_RUNTIME)
    list(APPEND _vernon_benchmark_targets "vulkan|vulkan")
endif()
if(VERNON_ENABLE_DIRECTX12_RUNTIME)
    list(APPEND _vernon_benchmark_targets "directx12|directx")
endif()
if(VERNON_ENABLE_METAL_RUNTIME)
    list(APPEND _vernon_benchmark_targets "metal|metal")
endif()
list(
    APPEND
    _vernon_benchmark_targets
    "opengl|opengl"
    "opengl_es|opengles")

set(_vernon_benchmark_assets
    "binding|binding_asset"
    "dynamic_binding|dynamic_binding_asset"
    "square|square_asset"
    "chain_1|chain_1_asset"
    "chain_8|chain_8_asset"
    "chain_64|chain_64_asset"
    "control|control_asset"
    "empty_kernel|empty_kernel_asset"
    "elementwise|elementwise_asset"
    "reduction|reduction_asset"
    "draw_1|draw_1_asset"
    "draw_64|draw_64_asset")
set(_vernon_benchmark_dependencies)
set(_vernon_benchmark_cpu_libraries)
set(_vernon_benchmark_manifest_rows)
set(_vernon_benchmark_registration_declarations)

foreach(_vernon_target_row IN LISTS _vernon_benchmark_targets)
    string(
        REPLACE "|"
                ";"
                _vernon_target_fields
                "${_vernon_target_row}")
    list(
        GET
        _vernon_target_fields
        0
        _vernon_backend)
    list(
        GET
        _vernon_target_fields
        1
        _vernon_architecture)
    foreach(_vernon_asset_row IN LISTS _vernon_benchmark_assets)
        string(
            REPLACE "|"
                    ";"
                    _vernon_asset_fields
                    "${_vernon_asset_row}")
        list(
            GET
            _vernon_asset_fields
            0
            _vernon_asset_name)
        list(
            GET
            _vernon_asset_fields
            1
            _vernon_asset_symbol)
        if(_vernon_backend STREQUAL "cpu" AND (_vernon_asset_name STREQUAL "control" OR _vernon_asset_name MATCHES
                                                                                        "^draw_"))
            continue()
        endif()
        set(_vernon_asset_reference "${CMAKE_CURRENT_SOURCE_DIR}/workloads.py:${_vernon_asset_symbol}")
        set(_vernon_output_stem "${_vernon_asset_name}_${_vernon_backend}")
        if(_vernon_backend STREQUAL "cpu")
            string(MAKE_C_IDENTIFIER "benchmark_${_vernon_asset_name}" _vernon_registration_suffix)
            set(_vernon_registration "vernonRegisterFixture_${_vernon_registration_suffix}")
            vernon_add_cpu_program_fixture_library(
                "benchmark-${_vernon_asset_name}-cpu"
                "${CMAKE_CURRENT_BINARY_DIR}/${_vernon_output_stem}"
                "${_vernon_output_stem}.program.json"
                "${_vernon_asset_reference}"
                "${_vernon_registration}"
                "${VERNON_REPOSITORY_ROOT}/source/tests/scripts/prepare_cpu_program_fixture.py"
                _vernon_manifest
                _vernon_fixture_target)
            list(APPEND _vernon_benchmark_cpu_libraries "${_vernon_fixture_target}")
            string(APPEND _vernon_benchmark_registration_declarations
                   "extern \"C\" VernonStatus ${_vernon_registration}(void);\n")
            set(_vernon_registration_expression "&${_vernon_registration}")
        else()
            vernon_add_program_asset_bundle(
                "benchmark-${_vernon_asset_name}-${_vernon_backend}"
                "${_vernon_architecture}"
                "${_vernon_asset_reference}"
                "${CMAKE_CURRENT_BINARY_DIR}"
                "${_vernon_output_stem}"
                _vernon_manifest)
            list(APPEND _vernon_benchmark_dependencies
                 "vernon-benchmark-${_vernon_asset_name}-${_vernon_backend}-bundle")
            set(_vernon_registration_expression "nullptr")
        endif()
        string(
            APPEND
            _vernon_benchmark_manifest_rows
            "    {\"${_vernon_asset_name}\", \"${_vernon_backend}\", R\"vernon(${_vernon_manifest})vernon\", ${_vernon_registration_expression}},\n"
        )
    endforeach()
endforeach()

file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(
    GENERATE
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/generated/benchmark_manifest_rows.inc"
    CONTENT "${_vernon_benchmark_manifest_rows}")
file(
    GENERATE
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/generated/benchmark_registration_declarations.inc"
    CONTENT "${_vernon_benchmark_registration_declarations}")

add_executable(VernonBenchmarkDriver driver.cpp)
set_target_properties(VernonBenchmarkDriver PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/benchmarks")
target_include_directories(
    VernonBenchmarkDriver
    PRIVATE "${VERNON_REPOSITORY_ROOT}/source/include"
            "${VERNON_REPOSITORY_ROOT}/source/lib"
            "${VERNON_REPOSITORY_ROOT}/source/tests/support"
            "${CMAKE_CURRENT_BINARY_DIR}/generated")
target_link_libraries(
    VernonBenchmarkDriver
    PRIVATE Vernon::Runtime
            Vernon::RHI
            nlohmann_json::nlohmann_json
            ${_vernon_benchmark_cpu_libraries})
if(TARGET VernonGlfwContextOwner)
    target_link_libraries(VernonBenchmarkDriver PRIVATE Vernon::GlfwContextOwner)
    target_compile_definitions(VernonBenchmarkDriver PRIVATE VERNON_GLFW_CONTEXT_OWNER_AVAILABLE=1)
endif()
add_dependencies(VernonBenchmarkDriver ${_vernon_benchmark_dependencies})
vernon_stage_target_files(VernonBenchmarkDriver VernonRuntime VernonRHI)

add_custom_target(
    VernonBenchmarkFixtures
    DEPENDS VernonBenchmarkDriver
    COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/run.py" --catalog
            "${CMAKE_CURRENT_SOURCE_DIR}/fixtures.json" --validate
    WORKING_DIRECTORY "${VERNON_REPOSITORY_ROOT}"
    COMMENT "Build and validate Vernon benchmark fixtures"
    VERBATIM)
