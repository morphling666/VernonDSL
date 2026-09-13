include("${VERNON_REPOSITORY_ROOT}/cmake/IncludeGTest.cmake")
include_gtest()
include(GoogleTest)
include("${VERNON_REPOSITORY_ROOT}/cmake/VernonTargetHelpers.cmake")

if(NOT TARGET nlohmann_json::nlohmann_json)
    include("${VERNON_REPOSITORY_ROOT}/cmake/IncludeNlohmannJson.cmake")
    include_nlohmann_json()
endif()

set(VERNON_TEST_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
set(VERNON_TEST_BINARY_ROOT "${CMAKE_CURRENT_BINARY_DIR}")

if(TARGET vernon-opt)
    add_subdirectory(mlir)
endif()

if(Python_EXECUTABLE)
    add_test(NAME VernonGeneratedVersionsCheck COMMAND "${Python_EXECUTABLE}"
                                                       "${VERNON_REPOSITORY_ROOT}/tools/generate_versions.py" --check)
    add_test(NAME VernonRuntimeSourcePolicyCheck
             COMMAND "${Python_EXECUTABLE}" "${VERNON_REPOSITORY_ROOT}/tools/check_runtime_source_policy.py"
                     --repository "${VERNON_REPOSITORY_ROOT}")
endif()

file(
    GLOB_RECURSE
    VERNON_TEST_BUNDLE_PYTHON_SOURCES
    CONFIGURE_DEPENDS
    "${VERNON_REPOSITORY_ROOT}/python/vernon_dsl/*.py")
set_property(
    DIRECTORY
    APPEND
    PROPERTY CMAKE_CONFIGURE_DEPENDS "${VERNON_REPOSITORY_ROOT}/python/tests/language_contract_cases.py"
             "${VERNON_REPOSITORY_ROOT}/python/tests/generate_language_contract_acceptance_cmake.py")

function(vernon_discover_gtests target)
    if(MSVC)
        target_compile_options(${target} PRIVATE /EHsc)
    endif()
    gtest_discover_tests(${target} DISCOVERY_MODE POST_BUILD WORKING_DIRECTORY "${VERNON_TEST_BINARY_ROOT}")
endfunction()

function(vernon_enable_glfw_context_owner_for_test target)
    if(TARGET VernonGlfwContextOwner)
        target_link_libraries(${target} PRIVATE Vernon::GlfwContextOwner)
        target_compile_definitions(${target} PRIVATE VERNON_GLFW_CONTEXT_OWNER_AVAILABLE=1)
    endif()
endfunction()

function(
    vernon_add_program_asset_test_bundle
    name
    architecture
    asset_reference
    output_stem
    output_variable)
    set(_vernon_bundle_directory "${VERNON_TEST_BINARY_ROOT}/${output_stem}")
    set(_vernon_bundle_manifest "${_vernon_bundle_directory}/${output_stem}.program.json")
    string(
        REGEX
        REPLACE ":[^:]+$"
                ""
                _vernon_asset_source
                "${asset_reference}")
    add_custom_command(
        OUTPUT "${_vernon_bundle_manifest}"
        COMMAND ${CMAKE_COMMAND} -E rm -rf "${_vernon_bundle_directory}"
        COMMAND
            ${CMAKE_COMMAND} -E env "PYTHONPATH=${VERNON_REPOSITORY_ROOT}/python" ${Python_EXECUTABLE} -m
            vernon_dsl.program_asset_cli "${asset_reference}" --target "${architecture}" --output
            "${_vernon_bundle_directory}"
        DEPENDS vernon-dsl-native
                "$<TARGET_FILE:vernon-dsl-native>"
                "${_vernon_asset_source}"
                ${VERNON_TEST_BUNDLE_PYTHON_SOURCES}
        VERBATIM)
    add_custom_target("vernon-${name}-test-bundle" DEPENDS "${_vernon_bundle_manifest}")
    set(${output_variable}
        "${_vernon_bundle_manifest}"
        PARENT_SCOPE)
endfunction()

function(
    vernon_add_cpu_program_fixture
    name
    directory_stem
    manifest_name
    asset_reference
    wrapper_function
    manifest_variable
    target_variable)
    cmake_parse_arguments(
        FIXTURE
        ""
        "COOK_SCRIPT"
        ""
        ${ARGN})
    if(FIXTURE_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "unknown CPU autodiff fixture arguments: ${FIXTURE_UNPARSED_ARGUMENTS}")
    endif()
    set(_vernon_directory "${VERNON_TEST_BINARY_ROOT}/${directory_stem}")
    set(_vernon_manifest "${_vernon_directory}/${manifest_name}")
    set(_vernon_registration "${_vernon_directory}/program_registration.c")
    set(_vernon_wrapper "${_vernon_directory}/program_registration_wrapper.c")
    set(_vernon_artifact_archive "${_vernon_directory}/program_artifacts.a")
    string(
        REGEX
        REPLACE ":[^:]+$"
                ""
                _vernon_asset_source
                "${asset_reference}")
    if(FIXTURE_COOK_SCRIPT)
        set(_vernon_cook_command ${Python_EXECUTABLE} "${FIXTURE_COOK_SCRIPT}" "${asset_reference}"
                                 "${_vernon_directory}")
    else()
        set(_vernon_cook_command ${Python_EXECUTABLE} -m vernon_dsl.program_asset_cli "${asset_reference}" --target cpu
                                 --output "${_vernon_directory}")
    endif()
    add_custom_command(
        OUTPUT "${_vernon_manifest}"
               "${_vernon_registration}"
               "${_vernon_wrapper}"
               "${_vernon_artifact_archive}"
        COMMAND ${CMAKE_COMMAND} -E rm -rf "${_vernon_directory}"
        COMMAND ${CMAKE_COMMAND} -E env "PYTHONPATH=${VERNON_REPOSITORY_ROOT}/python" ${_vernon_cook_command}
        COMMAND ${Python_EXECUTABLE} "${VERNON_TEST_ROOT}/scripts/prepare_cpu_program_fixture.py" "${_vernon_manifest}"
                "${CMAKE_C_OUTPUT_EXTENSION}" --wrapper-function "${wrapper_function}" --archiver "${CMAKE_AR}"
        DEPENDS vernon-dsl-native
                "$<TARGET_FILE:vernon-dsl-native>"
                "${_vernon_asset_source}"
                ${FIXTURE_COOK_SCRIPT}
                ${VERNON_TEST_BUNDLE_PYTHON_SOURCES}
                "${VERNON_TEST_ROOT}/scripts/prepare_cpu_program_fixture.py"
        VERBATIM)
    set_source_files_properties("${_vernon_registration}" "${_vernon_wrapper}" PROPERTIES GENERATED TRUE)
    set(_vernon_target "vernon-${name}-fixture")
    add_library(${_vernon_target} STATIC "${_vernon_registration}" "${_vernon_wrapper}")
    add_custom_target(${_vernon_target}-artifacts DEPENDS "${_vernon_artifact_archive}")
    add_dependencies(${_vernon_target} ${_vernon_target}-artifacts)
    target_link_libraries(${_vernon_target} PUBLIC "${_vernon_artifact_archive}")
    set_target_properties(${_vernon_target} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_include_directories(${_vernon_target} PRIVATE "${VERNON_REPOSITORY_ROOT}/source/include")
    set(${manifest_variable}
        "${_vernon_manifest}"
        PARENT_SCOPE)
    set(${target_variable}
        "${_vernon_target}"
        PARENT_SCOPE)
endfunction()

function(
    vernon_add_python_program_fixture
    target
    module
    registration_function
    fixture_target)
    python_add_library(
        ${target}
        MODULE
        WITH_SOABI
        "${VERNON_TEST_ROOT}/fixtures/autodiff_python_fixture.c")
    set_target_properties(${target} PROPERTIES OUTPUT_NAME "${module}")
    target_compile_definitions(${target} PRIVATE VERNON_AUTODIFF_FIXTURE_MODULE=${module}
                                                 VERNON_AUTODIFF_FIXTURE_REGISTRATION=${registration_function})
    target_include_directories(${target} PRIVATE "${VERNON_REPOSITORY_ROOT}/source/include")
    target_link_libraries(${target} PRIVATE Vernon::Runtime ${fixture_target})
    vernon_stage_target_files(${target} VernonRuntime)
endfunction()

if(TARGET VernonRuntime)
    add_library(VernonRuntimePublicCHeaderCheck OBJECT runtime/runtime_public_header_compile.c)
    target_link_libraries(VernonRuntimePublicCHeaderCheck PRIVATE Vernon::Runtime)
    add_library(VernonRuntimeAutodiffAbiCHeaderCheck OBJECT runtime/runtime_autodiff_abi_compile.c)
    target_include_directories(VernonRuntimeAutodiffAbiCHeaderCheck PRIVATE "${VERNON_REPOSITORY_ROOT}/source/lib")
    add_library(VernonRhiPublicCHeaderCheck OBJECT runtime/runtime_rhi_public_header_compile.c)
    target_link_libraries(VernonRhiPublicCHeaderCheck PRIVATE Vernon::RHI Vernon::RuntimeCore)

    get_target_property(_vernon_runtime_core_links VernonRuntimeCore LINK_LIBRARIES)
    if(_vernon_runtime_core_links MATCHES "(^|;)Vernon(RHI|::RHI)(;|$)")
        message(FATAL_ERROR "VernonRuntimeCore must not link VernonRHI")
    endif()

    add_executable(VernonRuntimeCoreProviderTests runtime/runtime_foreign_provider_test.cpp)
    target_link_libraries(VernonRuntimeCoreProviderTests PRIVATE GTest::gtest_main Vernon::RuntimeCore)
    add_dependencies(VernonRuntimeCoreProviderTests VernonRhiPublicCHeaderCheck)
    vernon_discover_gtests(VernonRuntimeCoreProviderTests)

    add_executable(VernonRuntimeSupportTests runtime/fallible_value_test.cpp runtime/lifecycle_benchmark_test.cpp
                                             runtime/runtime_binding_benchmark_test.cpp)
    target_link_libraries(VernonRuntimeSupportTests PRIVATE GTest::gtest_main Vernon::Runtime)
    vernon_discover_gtests(VernonRuntimeSupportTests)
    add_executable(VernonLifecycleControlTests runtime/lifecycle_control_test.cpp)
    target_link_libraries(VernonLifecycleControlTests PRIVATE GTest::gtest_main Vernon::Runtime)
    vernon_discover_gtests(VernonLifecycleControlTests)
    add_executable(VernonRhiLifecycleCoreTests rhi/rhi_lifecycle_core_test.cpp)
    target_include_directories(VernonRhiLifecycleCoreTests PRIVATE "${VERNON_REPOSITORY_ROOT}/source/include"
                                                                   "${VERNON_REPOSITORY_ROOT}/source/lib")
    target_link_libraries(VernonRhiLifecycleCoreTests PRIVATE GTest::gtest_main)
    if(MSVC)
        target_compile_options(VernonRhiLifecycleCoreTests PRIVATE /EHs-c-)
    else()
        target_compile_options(VernonRhiLifecycleCoreTests PRIVATE -fno-exceptions)
    endif()
    vernon_discover_gtests(VernonRhiLifecycleCoreTests)
    add_executable(VernonRhiDeferredActionTests rhi/rhi_deferred_action_test.cpp)
    target_include_directories(VernonRhiDeferredActionTests PRIVATE "${VERNON_REPOSITORY_ROOT}/source/include"
                                                                    "${VERNON_REPOSITORY_ROOT}/source/lib")
    target_link_libraries(VernonRhiDeferredActionTests PRIVATE GTest::gtest_main Vernon::RHI)
    vernon_discover_gtests(VernonRhiDeferredActionTests)
    vernon_stage_target_files(VernonRhiDeferredActionTests VernonRHI)
    add_executable(VernonRhiStrictBoundaryTests rhi/rhi_strict_boundary_test.cpp)
    target_include_directories(VernonRhiStrictBoundaryTests PRIVATE "${VERNON_REPOSITORY_ROOT}/source/include"
                                                                    "${VERNON_REPOSITORY_ROOT}/source/lib")
    target_link_libraries(VernonRhiStrictBoundaryTests PRIVATE GTest::gtest_main Vernon::RHI)
    vernon_discover_gtests(VernonRhiStrictBoundaryTests)
    vernon_stage_target_files(VernonRhiStrictBoundaryTests VernonRHI)
    add_executable(VernonRhiCommandLifecycleTests rhi/rhi_command_lifecycle_test.cpp)
    set_target_properties(VernonRhiCommandLifecycleTests PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                                    "${CMAKE_BINARY_DIR}/source")
    target_include_directories(VernonRhiCommandLifecycleTests PRIVATE "${VERNON_REPOSITORY_ROOT}/source/lib"
                                                                      "${VERNON_TEST_ROOT}/support")
    target_link_libraries(VernonRhiCommandLifecycleTests PRIVATE GTest::gtest_main Vernon::Runtime Vernon::RHI)
    vernon_enable_glfw_context_owner_for_test(VernonRhiCommandLifecycleTests)
    vernon_discover_gtests(VernonRhiCommandLifecycleTests)
    vernon_stage_target_files(VernonRhiCommandLifecycleTests VernonRuntime VernonRHI)
    add_library(VernonRuntimeSupportNoExceptionsCheck OBJECT runtime/runtime_support_no_exceptions_compile.cpp)
    target_link_libraries(VernonRuntimeSupportNoExceptionsCheck PRIVATE Vernon::Runtime)
    if(MSVC)
        target_compile_options(VernonRuntimeSupportNoExceptionsCheck PRIVATE /EHs-c-)
    else()
        target_compile_options(VernonRuntimeSupportNoExceptionsCheck PRIVATE -fno-exceptions)
    endif()
    add_dependencies(VernonRuntimeSupportTests VernonRuntimeSupportNoExceptionsCheck)
    add_dependencies(VernonLifecycleControlTests VernonRuntimeSupportNoExceptionsCheck)

    add_executable(VernonRhiCommandBenchmarkTests rhi/rhi_command_benchmark_test.cpp)
    set_target_properties(VernonRhiCommandBenchmarkTests PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                                    "${CMAKE_BINARY_DIR}/source")
    target_include_directories(VernonRhiCommandBenchmarkTests PRIVATE "${VERNON_REPOSITORY_ROOT}/source/lib"
                                                                      "${VERNON_TEST_ROOT}/support")
    target_link_libraries(VernonRhiCommandBenchmarkTests PRIVATE GTest::gtest_main Vernon::Runtime Vernon::RHI)
    vernon_enable_glfw_context_owner_for_test(VernonRhiCommandBenchmarkTests)
    vernon_stage_target_files(VernonRhiCommandBenchmarkTests VernonRuntime VernonRHI)

    if(Python_EXECUTABLE)
        if(APPLE)
            set(_vernon_lifecycle_baseline_platform "macos")
        elseif(WIN32)
            set(_vernon_lifecycle_baseline_platform "windows")
        else()
            string(TOLOWER "${CMAKE_SYSTEM_NAME}" _vernon_lifecycle_baseline_platform)
        endif()
        string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _vernon_lifecycle_baseline_architecture)
        if(CMAKE_CONFIGURATION_TYPES)
            set(_vernon_lifecycle_baseline_configuration "$<LOWER_CASE:$<CONFIG>>")
        else()
            string(TOLOWER "${CMAKE_BUILD_TYPE}" _vernon_lifecycle_baseline_configuration)
        endif()
        set(_vernon_lifecycle_baseline
            "${VERNON_REPOSITORY_ROOT}/reports/baselines/2026-09-11-${_vernon_lifecycle_baseline_platform}-${_vernon_lifecycle_baseline_architecture}-lifecycle-phase2-${_vernon_lifecycle_baseline_configuration}.json"
        )
        add_test(
            NAME VernonLifecyclePhase2PerformanceGate
            COMMAND "${Python_EXECUTABLE}" "${VERNON_REPOSITORY_ROOT}/tools/check_lifecycle_benchmark.py" --benchmark
                    "$<TARGET_FILE:VernonRuntimeSupportTests>" --baseline "${_vernon_lifecycle_baseline}")
        set_tests_properties(
            VernonLifecyclePhase2PerformanceGate
            PROPERTIES RUN_SERIAL
                       TRUE
                       SKIP_RETURN_CODE
                       77)

        set(_vernon_runtime_binding_baseline
            "${VERNON_REPOSITORY_ROOT}/reports/baselines/2026-09-12-${_vernon_lifecycle_baseline_platform}-${_vernon_lifecycle_baseline_architecture}-runtime-binding-phase4-${_vernon_lifecycle_baseline_configuration}.json"
        )
        add_test(
            NAME VernonRuntimeBindingPhase4PerformanceGate
            COMMAND
                "${Python_EXECUTABLE}" "${VERNON_REPOSITORY_ROOT}/tools/check_runtime_binding_benchmark.py" --benchmark
                "$<TARGET_FILE:VernonRuntimeSupportTests>" --baseline "${_vernon_runtime_binding_baseline}")
        set_tests_properties(
            VernonRuntimeBindingPhase4PerformanceGate
            PROPERTIES RUN_SERIAL
                       TRUE
                       SKIP_RETURN_CODE
                       77
                       LABELS
                       benchmark)

        set(_vernon_runtime_session_baseline
            "${VERNON_REPOSITORY_ROOT}/reports/baselines/2026-09-13-${_vernon_lifecycle_baseline_platform}-${_vernon_lifecycle_baseline_architecture}-runtime-session-phase9-${_vernon_lifecycle_baseline_configuration}.json"
        )
        add_test(NAME VernonRuntimeSessionPhase9PerformanceGate
                 COMMAND "${Python_EXECUTABLE}" "${VERNON_REPOSITORY_ROOT}/tools/check_runtime_session_benchmark.py"
                         --repository "${VERNON_REPOSITORY_ROOT}" --baseline "${_vernon_runtime_session_baseline}")
        set_tests_properties(
            VernonRuntimeSessionPhase9PerformanceGate
            PROPERTIES RUN_SERIAL
                       TRUE
                       SKIP_RETURN_CODE
                       77
                       LABELS
                       benchmark)

        set(_vernon_rhi_command_baseline
            "${VERNON_REPOSITORY_ROOT}/reports/baselines/2026-09-12-${_vernon_lifecycle_baseline_platform}-${_vernon_lifecycle_baseline_architecture}-rhi-command-phase3-${_vernon_lifecycle_baseline_configuration}.json"
        )
        add_test(
            NAME VernonRhiCommandPhase3PerformanceGate
            COMMAND
                "${Python_EXECUTABLE}" "${VERNON_REPOSITORY_ROOT}/tools/check_rhi_command_benchmark.py" --benchmark
                "$<TARGET_FILE:VernonRhiCommandBenchmarkTests>" --baseline "${_vernon_rhi_command_baseline}"
                --repository "${VERNON_REPOSITORY_ROOT}" --output
                "${VERNON_TEST_BINARY_ROOT}/rhi_command_phase3_benchmark.json")
        set_tests_properties(
            VernonRhiCommandPhase3PerformanceGate
            PROPERTIES RUN_SERIAL
                       TRUE
                       TIMEOUT
                       300
                       LABELS
                       benchmark)
    endif()

    if(VERNON_ENABLE_CUDA_RUNTIME)
        add_executable(VernonRuntimeRHIAdapterTests rhi/cuda_adapter_test.cpp)
        target_include_directories(VernonRuntimeRHIAdapterTests PRIVATE "${VERNON_REPOSITORY_ROOT}/source/lib")
        target_link_libraries(VernonRuntimeRHIAdapterTests PRIVATE GTest::gtest_main Vernon::RuntimeCore
                                                                   Vernon::RuntimeRHIAdapter)
        vernon_stage_target_files(VernonRuntimeRHIAdapterTests VernonRHI)
        vernon_discover_gtests(VernonRuntimeRHIAdapterTests)
    endif()

    add_library(vernon-cpu-aot-fixture SHARED fixtures/cpu_aot_fixture.c)
    target_include_directories(vernon-cpu-aot-fixture PRIVATE "${VERNON_REPOSITORY_ROOT}/source/include")
    add_test(
        NAME vernon-cpu-aot-bundle-platform-mapping
        COMMAND
            ${CMAKE_COMMAND} "-DGENERATOR=${VERNON_TEST_ROOT}/scripts/write_cpu_aot_bundle.cmake"
            "-DTEST_OUTPUT_DIRECTORY=${VERNON_TEST_BINARY_ROOT}/cpu_aot_bundle_platform_mapping" -P
            "${VERNON_TEST_ROOT}/scripts/test_cpu_aot_bundle_platform_mapping.cmake")

    add_executable(
        VernonRuntimeTests
        runtime/compute_launch_planner_test.cpp
        runtime/directx_command_state_test.cpp
        runtime/dirty_range_set_test.cpp
        runtime/execution_command_model_test.cpp
        runtime/execution_graph_test.cpp
        runtime/graphics_invocation_planner_test.cpp
        runtime/program_instance_test.cpp
        runtime/program_manifest_test.cpp
        runtime/resolved_execution_plan_test.cpp
        runtime/rhi_adapter_backend_storage_test.cpp
        runtime/rhi_resource_lifetime_test.cpp
        runtime/runtime_architecture_type_test.cpp
        runtime/runtime_autodiff_test.cpp
        runtime/runtime_c_api_test.cpp
        runtime/runtime_common_api_test.cpp
        runtime/runtime_cpu_pipeline_test.cpp
        runtime/runtime_external_gl_test.cpp
        runtime/shape_layout_test.cpp
        runtime/stage_artifact_contract_test.cpp
        runtime/tensor_bridge_test.cpp
        runtime/vertex_attribute_capabilities_test.cpp)
    set_target_properties(VernonRuntimeTests PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/source")
    target_include_directories(
        VernonRuntimeTests PRIVATE "${VERNON_REPOSITORY_ROOT}/source/lib" "${VERNON_TEST_ROOT}/support"
                                   "${VERNON_TEST_BINARY_ROOT}/generated")
    target_link_libraries(
        VernonRuntimeTests
        PRIVATE GTest::gtest_main
                Vernon::Runtime
                VernonCommandGraph
                Vernon::RuntimeRHIAdapter
                nlohmann_json::nlohmann_json)
    vernon_enable_glfw_context_owner_for_test(VernonRuntimeTests)
    if(VERNON_RUNTIME_LIBRARY_TYPE STREQUAL "SHARED")
        target_link_libraries(VernonRuntimeTests PRIVATE VernonRuntimeInternals Vernon::RuntimeCore)
    endif()
    add_dependencies(VernonRuntimeTests VernonRuntimeAutodiffAbiCHeaderCheck VernonRuntimePublicCHeaderCheck)
    vernon_stage_target_files(VernonRuntimeTests VernonRuntime VernonRHI)
    add_executable(VernonRuntimeCpuWorkgroupTests runtime/cpu_workgroup_dispatch_test.cpp)
    set_target_properties(VernonRuntimeCpuWorkgroupTests PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                                    "${CMAKE_BINARY_DIR}/source")
    target_include_directories(VernonRuntimeCpuWorkgroupTests PRIVATE "${VERNON_REPOSITORY_ROOT}/source/lib")
    target_link_libraries(VernonRuntimeCpuWorkgroupTests PRIVATE GTest::gtest_main Vernon::Runtime)
    vernon_stage_target_files(VernonRuntimeCpuWorkgroupTests VernonRuntime)
    vernon_discover_gtests(VernonRuntimeCpuWorkgroupTests)
    if(TARGET VernonDSLCompiler)
        add_executable(VernonCompilerRuntimeIntegrationTests runtime/runtime_synchronization_test.cpp)
        set_target_properties(VernonCompilerRuntimeIntegrationTests PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                                               "${CMAKE_BINARY_DIR}/source")
        target_include_directories(VernonCompilerRuntimeIntegrationTests PRIVATE "${VERNON_REPOSITORY_ROOT}/source/lib"
                                                                                 "${VERNON_TEST_ROOT}/support")
        target_link_libraries(
            VernonCompilerRuntimeIntegrationTests
            PRIVATE GTest::gtest_main
                    Vernon::Runtime
                    VernonCommandGraph
                    Vernon::RuntimeRHIAdapter
                    VernonDSLCompiler)
        vernon_enable_glfw_context_owner_for_test(VernonCompilerRuntimeIntegrationTests)
        vernon_stage_target_files(
            VernonCompilerRuntimeIntegrationTests
            VernonRuntime
            VernonRHI
            VernonDSLCompiler)
        vernon_discover_gtests(VernonCompilerRuntimeIntegrationTests)
    endif()
    if(TARGET vernon-dsl-native AND Python_FOUND)
        set(gpu_autodiff_fixture_targets)
        set(module_program_fixture_targets)
        set(module_graphics_fixture_targets)
        set(synchronization_fixture_targets)
        set(language_contract_fixture_targets)
        set(module_program_cpu_fixture_targets)
        set(synchronization_cpu_fixture_targets)
        set(language_contract_cpu_fixture_targets)
        set(VERNON_TEST_FIXTURE_REGISTRATION_DECLARATIONS)
        set(enabled_fixture_targets)
        list(APPEND enabled_fixture_targets "cpu|VERNON_RUNTIME_CPU|compute,autodiff,synchronization")
        if(VERNON_ENABLE_CUDA_RUNTIME)
            list(APPEND enabled_fixture_targets "cuda|VERNON_RUNTIME_CUDA|compute,autodiff,gpu,synchronization,texture")
        endif()
        if(VERNON_ENABLE_VULKAN_RUNTIME)
            list(APPEND enabled_fixture_targets
                 "vulkan|VERNON_RUNTIME_VULKAN|compute,autodiff,gpu,graphics,synchronization,texture")
        endif()
        if(VERNON_ENABLE_DIRECTX12_RUNTIME)
            list(APPEND enabled_fixture_targets
                 "directx|VERNON_RUNTIME_DIRECTX12|compute,autodiff,gpu,graphics,synchronization,texture")
        endif()
        if(VERNON_ENABLE_METAL_RUNTIME)
            list(APPEND enabled_fixture_targets
                 "metal|VERNON_RUNTIME_METAL|compute,autodiff,gpu,graphics,synchronization,texture")
        endif()
        list(
            APPEND
            enabled_fixture_targets
            "opengl|VERNON_RUNTIME_OPENGL|compute,autodiff,gpu,graphics,synchronization,texture"
            "opengles|VERNON_RUNTIME_OPENGL_ES|compute,autodiff,gpu,graphics,synchronization,texture")

        set(language_contract_fixture_specs_file
            "${VERNON_TEST_BINARY_ROOT}/generated/language_contract_fixture_specs.cmake")
        execute_process(
            COMMAND
                "${CMAKE_COMMAND}" -E env
                "PYTHONPATH=${VERNON_REPOSITORY_ROOT}/python:${VERNON_REPOSITORY_ROOT}/python/tests"
                "${Python_EXECUTABLE}"
                "${VERNON_REPOSITORY_ROOT}/python/tests/generate_language_contract_acceptance_cmake.py" --repository
                "${VERNON_REPOSITORY_ROOT}" --output "${language_contract_fixture_specs_file}" --cpp-output
                "${VERNON_TEST_BINARY_ROOT}/generated/language_contract_acceptance_descriptors.inc"
            RESULT_VARIABLE language_contract_fixture_specs_status
            ERROR_VARIABLE language_contract_fixture_specs_error)
        if(NOT
           language_contract_fixture_specs_status
           EQUAL
           0)
            message(FATAL_ERROR "failed to generate language contract fixture matrix: "
                                "${language_contract_fixture_specs_error}")
        endif()
        include("${language_contract_fixture_specs_file}")
        set(cross_backend_fixture_specs ${VERNON_LANGUAGE_CONTRACT_FIXTURE_SPECS})
        set(VERNON_TEST_FIXTURE_MANIFEST_ROWS)
        set(VERNON_TEST_FIXTURE_PAIRS)
        foreach(enabled_fixture_target IN LISTS enabled_fixture_targets)
            string(
                REPLACE "|"
                        ";"
                        enabled_fixture_target_fields
                        "${enabled_fixture_target}")
            list(LENGTH enabled_fixture_target_fields enabled_fixture_target_field_count)
            if(NOT
               enabled_fixture_target_field_count
               EQUAL
               3)
                message(FATAL_ERROR "invalid fixture target row: ${enabled_fixture_target}")
            endif()
            list(
                GET
                enabled_fixture_target_fields
                0
                fixture_architecture)
            list(
                GET
                enabled_fixture_target_fields
                1
                fixture_runtime)
            list(
                GET
                enabled_fixture_target_fields
                2
                fixture_capabilities)
            string(
                REPLACE ","
                        ";"
                        fixture_capability_list
                        "${fixture_capabilities}")
            foreach(cross_backend_fixture_spec IN LISTS cross_backend_fixture_specs)
                string(
                    REPLACE "|"
                            ";"
                            fixture_fields
                            "${cross_backend_fixture_spec}")
                list(LENGTH fixture_fields fixture_field_count)
                if(NOT
                   fixture_field_count
                   EQUAL
                   4)
                    message(FATAL_ERROR "invalid cross-backend fixture row: ${cross_backend_fixture_spec}")
                endif()
                list(
                    GET
                    fixture_fields
                    0
                    fixture_group)
                list(
                    GET
                    fixture_fields
                    1
                    fixture_id)
                list(
                    GET
                    fixture_fields
                    2
                    fixture_asset)
                list(
                    GET
                    fixture_fields
                    3
                    fixture_requirements)
                string(
                    REPLACE ","
                            ";"
                            fixture_requirement_list
                            "${fixture_requirements}")
                set(fixture_supported TRUE)
                foreach(fixture_requirement IN LISTS fixture_requirement_list)
                    if(NOT
                       fixture_requirement
                       IN_LIST
                       fixture_capability_list)
                        set(fixture_supported FALSE)
                    endif()
                endforeach()
                if(NOT fixture_supported)
                    continue()
                endif()
                set(fixture_pair "${fixture_id}@${fixture_architecture}")
                if(fixture_pair IN_LIST VERNON_TEST_FIXTURE_PAIRS)
                    message(FATAL_ERROR "duplicate cross-backend fixture pair: ${fixture_pair}")
                endif()
                list(APPEND VERNON_TEST_FIXTURE_PAIRS "${fixture_pair}")
                string(
                    REPLACE "_"
                            "-"
                            fixture_name
                            "${fixture_id}")
                set(fixture_manifest_variable "${fixture_id}_${fixture_architecture}_manifest")
                if(fixture_runtime STREQUAL "VERNON_RUNTIME_CPU")
                    string(MAKE_C_IDENTIFIER "${fixture_id}_${fixture_architecture}" fixture_registration_suffix)
                    set(fixture_registration "vernonRegisterFixture_${fixture_registration_suffix}")
                    set(fixture_target_variable "${fixture_id}_${fixture_architecture}_fixture_target")
                    vernon_add_cpu_program_fixture(
                        "${fixture_name}-${fixture_architecture}"
                        "${fixture_id}_${fixture_architecture}_fixture"
                        "${fixture_id}_${fixture_architecture}_fixture.program.json"
                        "${fixture_asset}"
                        "${fixture_registration}"
                        "${fixture_manifest_variable}"
                        "${fixture_target_variable}")
                    list(APPEND ${fixture_group}_cpu_fixture_targets "${${fixture_target_variable}}")
                    string(TOUPPER "${fixture_group}" fixture_group_macro)
                    string(MAKE_C_IDENTIFIER "${fixture_group_macro}" fixture_group_macro)
                    string(
                        APPEND
                        VERNON_TEST_FIXTURE_REGISTRATION_DECLARATIONS
                        "#if defined(VERNON_${fixture_group_macro}_CPU_FIXTURES_AVAILABLE)\nextern \"C\" VernonStatus ${fixture_registration}(void);\n#endif\n"
                    )
                    set(fixture_registration_expression "&${fixture_registration}")
                else()
                    vernon_add_program_asset_test_bundle(
                        "${fixture_name}-${fixture_architecture}"
                        "${fixture_architecture}"
                        "${fixture_asset}"
                        "${fixture_id}_${fixture_architecture}_fixture"
                        "${fixture_manifest_variable}")
                    list(APPEND ${fixture_group}_fixture_targets
                         "vernon-${fixture_name}-${fixture_architecture}-test-bundle")
                    set(fixture_registration_expression "nullptr")
                endif()
                string(FIND "${${fixture_manifest_variable}}" ")vernon\"" fixture_raw_string_terminator)
                if(NOT
                   fixture_raw_string_terminator
                   EQUAL
                   -1)
                    message(
                        FATAL_ERROR "fixture manifest path contains reserved raw-string terminator: ${fixture_pair}")
                endif()
                if(fixture_runtime STREQUAL "VERNON_RUNTIME_CPU")
                    string(TOUPPER "${fixture_group}" fixture_group_macro)
                    string(MAKE_C_IDENTIFIER "${fixture_group_macro}" fixture_group_macro)
                    string(
                        APPEND
                        VERNON_TEST_FIXTURE_MANIFEST_ROWS
                        "#if defined(VERNON_${fixture_group_macro}_CPU_FIXTURES_AVAILABLE)\n        {\"${fixture_id}\", \"${fixture_architecture}\", ${fixture_runtime}, R\"vernon(${${fixture_manifest_variable}})vernon\", ${fixture_registration_expression}},\n#endif\n"
                    )
                else()
                    string(
                        APPEND
                        VERNON_TEST_FIXTURE_MANIFEST_ROWS
                        "        {\"${fixture_id}\", \"${fixture_architecture}\", ${fixture_runtime}, R\"vernon(${${fixture_manifest_variable}})vernon\", ${fixture_registration_expression}},\n"
                    )
                endif()
            endforeach()
        endforeach()
        target_compile_definitions(
            VernonCompilerRuntimeIntegrationTests PRIVATE VERNON_SYNCHRONIZATION_FIXTURES_AVAILABLE=1
                                                          VERNON_SYNCHRONIZATION_CPU_FIXTURES_AVAILABLE=1)
        target_include_directories(VernonCompilerRuntimeIntegrationTests PRIVATE "${VERNON_TEST_BINARY_ROOT}/generated")
        target_link_libraries(VernonCompilerRuntimeIntegrationTests PRIVATE ${synchronization_cpu_fixture_targets})
        add_dependencies(VernonCompilerRuntimeIntegrationTests ${synchronization_fixture_targets})

        vernon_add_cpu_program_fixture(
            cooked-autodiff
            autodiff_cooked_fixture
            autodiff_cooked_fixture.program.json
            "${VERNON_TEST_ROOT}/fixtures/autodiff_cooked_aggregate_asset.py:asset"
            vernonRegisterAutodiffFixture
            cooked_autodiff_manifest
            cooked_autodiff_fixture_target)
        file(MAKE_DIRECTORY "${VERNON_TEST_BINARY_ROOT}/generated")
        file(
            GENERATE
            OUTPUT "${VERNON_TEST_BINARY_ROOT}/generated/program_fixture_registration_declarations.inc"
            CONTENT "${VERNON_TEST_FIXTURE_REGISTRATION_DECLARATIONS}")
        file(
            GENERATE
            OUTPUT "${VERNON_TEST_BINARY_ROOT}/generated/program_fixture_manifest_rows.inc"
            CONTENT "${VERNON_TEST_FIXTURE_MANIFEST_ROWS}")
        vernon_add_cpu_program_fixture(
            typed-specialization
            typed_specialization_fixture
            typed_specialization_fixture.program.json
            "${VERNON_TEST_ROOT}/fixtures/typed_specialization_program_asset.py:asset"
            vernonRegisterTypedSpecializationFixture
            typed_specialization_manifest
            typed_specialization_fixture_target)
        vernon_add_cpu_program_fixture(
            native-autodiff-f16
            autodiff_native_f16_fixture
            autodiff_native_f16_fixture.program.json
            "${VERNON_TEST_ROOT}/fixtures/autodiff_native_f16_asset.py:asset"
            vernonRegisterNativeAutodiffF16Fixture
            native_autodiff_f16_manifest
            native_autodiff_f16_fixture_target)
        vernon_add_cpu_program_fixture(
            native-autodiff-numeric-cpu
            autodiff_native_numeric_cpu_fixture
            autodiff_native_numeric_cpu_fixture.program.json
            "${VERNON_TEST_ROOT}/fixtures/autodiff_native_numeric_asset.py:asset"
            vernonRegisterNativeAutodiffNumericCpuFixture
            native_autodiff_numeric_cpu_manifest
            native_autodiff_numeric_cpu_fixture_target)
        vernon_add_cpu_program_fixture(
            structured-scalar-autodiff
            autodiff_structured_scalar_fixture
            autodiff_structured_scalar_fixture.program.json
            "${VERNON_TEST_ROOT}/fixtures/autodiff_structured_scalar_asset.py:asset"
            vernonRegisterStructuredScalarAutodiffFixture
            structured_scalar_autodiff_manifest
            structured_scalar_autodiff_fixture_target)
        vernon_add_cpu_program_fixture(
            structured-scalar-balanced-autodiff
            autodiff_structured_scalar_balanced_fixture
            autodiff_structured_scalar_balanced_fixture.program.json
            "${VERNON_TEST_ROOT}/fixtures/autodiff_structured_scalar_asset.py:balanced_asset"
            vernonRegisterStructuredScalarBalancedAutodiffFixture
            structured_scalar_balanced_autodiff_manifest
            structured_scalar_balanced_autodiff_fixture_target)
        vernon_add_cpu_program_fixture(
            structured-dynamic-autodiff
            autodiff_structured_dynamic_fixture
            autodiff_structured_dynamic_fixture.program.json
            "${VERNON_TEST_ROOT}/fixtures/autodiff_structured_scalar_asset.py:dynamic_asset"
            vernonRegisterStructuredDynamicAutodiffFixture
            structured_dynamic_autodiff_manifest
            structured_dynamic_autodiff_fixture_target)
        vernon_add_cpu_program_fixture(
            structured-f64-autodiff
            autodiff_structured_f64_fixture
            autodiff_structured_f64_fixture.program.json
            "${VERNON_TEST_ROOT}/fixtures/autodiff_structured_scalar_asset.py:double_asset"
            vernonRegisterStructuredF64AutodiffFixture
            structured_f64_autodiff_manifest
            structured_f64_autodiff_fixture_target)
        vernon_add_cpu_program_fixture(
            structured-aggregate-autodiff
            autodiff_structured_aggregate_fixture
            autodiff_structured_aggregate_fixture.program.json
            "${VERNON_TEST_ROOT}/fixtures/autodiff_structured_aggregate_asset.py:asset"
            vernonRegisterStructuredAggregateAutodiffFixture
            structured_aggregate_autodiff_manifest
            structured_aggregate_autodiff_fixture_target)
        vernon_add_cpu_program_fixture(
            structured-storage-autodiff
            autodiff_structured_storage_fixture
            autodiff_structured_storage_fixture.program.json
            "${VERNON_TEST_ROOT}/fixtures/autodiff_structured_storage_asset.py:asset"
            vernonRegisterStructuredStorageAutodiffFixture
            structured_storage_autodiff_manifest
            structured_storage_autodiff_fixture_target)
        vernon_add_cpu_program_fixture(
            structured-storage-aggregate-autodiff
            autodiff_structured_storage_aggregate_fixture
            autodiff_structured_storage_aggregate_fixture.program.json
            "${VERNON_TEST_ROOT}/fixtures/autodiff_structured_storage_asset.py:aggregate_asset"
            vernonRegisterStructuredStorageAggregateAutodiffFixture
            structured_storage_aggregate_autodiff_manifest
            structured_storage_aggregate_autodiff_fixture_target)
        vernon_add_cpu_program_fixture(
            structured-storage-workgroup-autodiff
            autodiff_structured_storage_workgroup_fixture
            autodiff_structured_storage_workgroup_fixture.program.json
            "${VERNON_TEST_ROOT}/fixtures/autodiff_workgroup_storage_asset.py:asset"
            vernonRegisterStructuredStorageWorkgroupAutodiffFixture
            structured_storage_workgroup_autodiff_manifest
            structured_storage_workgroup_autodiff_fixture_target)
        vernon_add_cpu_program_fixture(
            dynamic-v2-signed-stride
            autodiff_dynamic_v2_signed_stride_fixture
            autodiff_dynamic_v2_signed_stride_fixture.program.json
            "${VERNON_TEST_ROOT}/fixtures/autodiff_dynamic_v2_parity_asset.py:signed_stride_asset"
            vernonRegisterDynamicV2SignedStrideFixture
            dynamic_v2_signed_stride_manifest
            dynamic_v2_signed_stride_fixture_target)
        vernon_add_cpu_program_fixture(
            dynamic-v2-field-path
            autodiff_dynamic_v2_field_path_fixture
            autodiff_dynamic_v2_field_path_fixture.program.json
            "${VERNON_TEST_ROOT}/fixtures/autodiff_dynamic_v2_parity_asset.py:field_path_asset"
            vernonRegisterDynamicV2FieldPathFixture
            dynamic_v2_field_path_manifest
            dynamic_v2_field_path_fixture_target)
        vernon_add_cpu_program_fixture(
            dynamic-v2-strided-scatter
            autodiff_dynamic_v2_strided_scatter_fixture
            autodiff_dynamic_v2_strided_scatter_fixture.program.json
            "${VERNON_TEST_ROOT}/fixtures/autodiff_dynamic_v2_parity_asset.py:strided_scatter_asset"
            vernonRegisterDynamicV2StridedScatterFixture
            dynamic_v2_strided_scatter_manifest
            dynamic_v2_strided_scatter_fixture_target)
        vernon_add_cpu_program_fixture(
            dynamic-v2-gather
            autodiff_dynamic_v2_gather_fixture
            autodiff_dynamic_v2_gather_fixture.program.json
            "${VERNON_TEST_ROOT}/fixtures/autodiff_dynamic_v2_parity_asset.py:gather_asset"
            vernonRegisterDynamicV2GatherFixture
            dynamic_v2_gather_manifest
            dynamic_v2_gather_fixture_target)
        vernon_add_cpu_program_fixture(
            dynamic-v2-multi-output
            autodiff_dynamic_v2_multi_output_fixture
            autodiff_dynamic_v2_multi_output_fixture.program.json
            "${VERNON_TEST_ROOT}/fixtures/autodiff_dynamic_v2_parity_asset.py:multi_output_asset"
            vernonRegisterDynamicV2MultiOutputFixture
            dynamic_v2_multi_output_manifest
            dynamic_v2_multi_output_fixture_target)
        vernon_add_cpu_program_fixture(
            dynamic-v2-partially-dynamic
            autodiff_dynamic_v2_partially_dynamic_fixture
            autodiff_dynamic_v2_partially_dynamic_fixture.program.json
            "${VERNON_TEST_ROOT}/fixtures/autodiff_dynamic_v2_parity_asset.py:partially_dynamic_asset"
            vernonRegisterDynamicV2PartiallyDynamicFixture
            dynamic_v2_partially_dynamic_manifest
            dynamic_v2_partially_dynamic_fixture_target)
        vernon_add_cpu_program_fixture(
            dynamic-v2-mixed-alias
            autodiff_dynamic_v2_mixed_alias_fixture
            autodiff_dynamic_v2_mixed_alias_fixture.program.json
            "${VERNON_TEST_ROOT}/fixtures/autodiff_dynamic_v2_parity_asset.py:mixed_alias_asset"
            vernonRegisterDynamicV2MixedAliasFixture
            dynamic_v2_mixed_alias_manifest
            dynamic_v2_mixed_alias_fixture_target)
        add_executable(VernonModuleProgramRuntimeTests runtime/runtime_module_program_c_api_test.cpp)
        target_link_libraries(VernonRuntimeTests PRIVATE ${module_program_cpu_fixture_targets})
        target_compile_definitions(
            VernonRuntimeTests PRIVATE VERNON_CPU_CANONICAL_PROGRAM_MANIFEST="${module_program_cpu_manifest}"
                                       VERNON_MODULE_PROGRAM_CPU_FIXTURES_AVAILABLE=1)
        target_link_libraries(
            VernonModuleProgramRuntimeTests
            PRIVATE GTest::gtest_main
                    Vernon::Runtime
                    nlohmann_json::nlohmann_json
                    ${module_program_cpu_fixture_targets}
                    ${typed_specialization_fixture_target})
        target_include_directories(
            VernonModuleProgramRuntimeTests
            PRIVATE "${VERNON_REPOSITORY_ROOT}/source/lib" "${VERNON_TEST_ROOT}/support"
                    "${VERNON_TEST_BINARY_ROOT}/generated")
        target_compile_definitions(
            VernonModuleProgramRuntimeTests
            PRIVATE VERNON_MODULE_PROGRAM_MANIFEST="${module_program_cpu_manifest}"
                    VERNON_MODULE_PROGRAM_CPU_FIXTURES_AVAILABLE=1
                    VERNON_TYPED_SPECIALIZATION_MANIFEST="${typed_specialization_manifest}")
        set_target_properties(VernonModuleProgramRuntimeTests PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                                         "${CMAKE_BINARY_DIR}/source")
        vernon_stage_target_files(VernonModuleProgramRuntimeTests VernonRuntime)
        vernon_discover_gtests(VernonModuleProgramRuntimeTests)

        if(gpu_autodiff_fixture_targets)
            add_executable(VernonGpuAutodiffTests runtime/runtime_gpu_autodiff_test.cpp)
            target_include_directories(
                VernonGpuAutodiffTests PRIVATE "${VERNON_REPOSITORY_ROOT}/source/lib" "${VERNON_TEST_ROOT}/support"
                                               "${VERNON_TEST_BINARY_ROOT}/generated")
            target_link_libraries(VernonGpuAutodiffTests PRIVATE GTest::gtest_main Vernon::Runtime Vernon::RHI)
            vernon_enable_glfw_context_owner_for_test(VernonGpuAutodiffTests)
            add_dependencies(VernonGpuAutodiffTests ${gpu_autodiff_fixture_targets})
            set_target_properties(VernonGpuAutodiffTests PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                                    "${CMAKE_BINARY_DIR}/source")
            vernon_stage_target_files(VernonGpuAutodiffTests VernonRuntime VernonRHI)
            vernon_discover_gtests(VernonGpuAutodiffTests)
        endif()
        if(module_program_fixture_targets)
            add_executable(VernonModuleProgramGpuRuntimeTests runtime/runtime_module_program_gpu_c_api_test.cpp)
            target_link_libraries(
                VernonModuleProgramGpuRuntimeTests
                PRIVATE GTest::gtest_main
                        Vernon::Runtime
                        Vernon::RHI
                        nlohmann_json::nlohmann_json)
            vernon_enable_glfw_context_owner_for_test(VernonModuleProgramGpuRuntimeTests)
            target_include_directories(
                VernonModuleProgramGpuRuntimeTests
                PRIVATE "${VERNON_REPOSITORY_ROOT}/source/lib" "${VERNON_TEST_ROOT}/support"
                        "${VERNON_TEST_BINARY_ROOT}/generated")
            add_dependencies(VernonModuleProgramGpuRuntimeTests ${module_program_fixture_targets})
            set_target_properties(VernonModuleProgramGpuRuntimeTests PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                                                "${CMAKE_BINARY_DIR}/source")
            vernon_stage_target_files(VernonModuleProgramGpuRuntimeTests VernonRuntime VernonRHI)
            vernon_discover_gtests(VernonModuleProgramGpuRuntimeTests)
        endif()
        if(module_graphics_fixture_targets)
            add_executable(VernonModuleGraphicsProgramRuntimeTests
                           runtime/runtime_module_graphics_program_c_api_test.cpp)
            target_link_libraries(
                VernonModuleGraphicsProgramRuntimeTests
                PRIVATE GTest::gtest_main
                        Vernon::Runtime
                        Vernon::RHI
                        nlohmann_json::nlohmann_json)
            vernon_enable_glfw_context_owner_for_test(VernonModuleGraphicsProgramRuntimeTests)
            target_include_directories(
                VernonModuleGraphicsProgramRuntimeTests
                PRIVATE "${VERNON_REPOSITORY_ROOT}/source/lib" "${VERNON_TEST_ROOT}/support"
                        "${VERNON_TEST_BINARY_ROOT}/generated")
            add_dependencies(VernonModuleGraphicsProgramRuntimeTests ${module_graphics_fixture_targets})
            set_target_properties(VernonModuleGraphicsProgramRuntimeTests PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                                                     "${CMAKE_BINARY_DIR}/source")
            vernon_stage_target_files(VernonModuleGraphicsProgramRuntimeTests VernonRuntime VernonRHI)
            vernon_discover_gtests(VernonModuleGraphicsProgramRuntimeTests)
        endif()
        if(language_contract_fixture_targets OR language_contract_cpu_fixture_targets)
            add_executable(VernonLanguageContractAcceptanceTests runtime/runtime_language_contract_acceptance_test.cpp)
            target_link_libraries(
                VernonLanguageContractAcceptanceTests
                PRIVATE GTest::gtest_main
                        Vernon::Runtime
                        Vernon::RHI
                        nlohmann_json::nlohmann_json
                        ${language_contract_cpu_fixture_targets})
            vernon_enable_glfw_context_owner_for_test(VernonLanguageContractAcceptanceTests)
            target_include_directories(
                VernonLanguageContractAcceptanceTests
                PRIVATE "${VERNON_REPOSITORY_ROOT}/source/lib" "${VERNON_TEST_ROOT}/support"
                        "${VERNON_TEST_BINARY_ROOT}/generated")
            target_compile_definitions(VernonLanguageContractAcceptanceTests
                                       PRIVATE VERNON_LANGUAGE_CONTRACT_CPU_FIXTURES_AVAILABLE=1)
            add_dependencies(VernonLanguageContractAcceptanceTests ${language_contract_fixture_targets})
            set_target_properties(VernonLanguageContractAcceptanceTests PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                                                   "${CMAKE_BINARY_DIR}/source")
            vernon_stage_target_files(VernonLanguageContractAcceptanceTests VernonRuntime VernonRHI)
            vernon_discover_gtests(VernonLanguageContractAcceptanceTests)
        endif()

        add_executable(VernonStructuredScalarAutodiffTests runtime/runtime_structured_scalar_autodiff_test.cpp)
        target_link_libraries(
            VernonStructuredScalarAutodiffTests
            PRIVATE GTest::gtest_main
                    Vernon::Runtime
                    nlohmann_json::nlohmann_json
                    ${structured_scalar_autodiff_fixture_target}
                    ${structured_scalar_balanced_autodiff_fixture_target}
                    ${structured_dynamic_autodiff_fixture_target}
                    ${structured_f64_autodiff_fixture_target})
        target_include_directories(VernonStructuredScalarAutodiffTests PRIVATE "${VERNON_REPOSITORY_ROOT}/source/lib"
                                                                               "${VERNON_TEST_ROOT}/support")
        target_compile_definitions(
            VernonStructuredScalarAutodiffTests
            PRIVATE
                VERNON_STRUCTURED_SCALAR_AUTODIFF_MANIFEST="${structured_scalar_autodiff_manifest}"
                VERNON_STRUCTURED_SCALAR_BALANCED_AUTODIFF_MANIFEST="${structured_scalar_balanced_autodiff_manifest}"
                VERNON_STRUCTURED_DYNAMIC_AUTODIFF_MANIFEST="${structured_dynamic_autodiff_manifest}"
                VERNON_STRUCTURED_F64_AUTODIFF_MANIFEST="${structured_f64_autodiff_manifest}")
        if(VERNON_HOST_TAPE_INSTRUMENTATION)
            target_compile_definitions(VernonStructuredScalarAutodiffTests PRIVATE VERNON_HOST_TAPE_INSTRUMENTATION=1)
        endif()
        set_target_properties(VernonStructuredScalarAutodiffTests PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                                             "${CMAKE_BINARY_DIR}/source")
        vernon_stage_target_files(VernonStructuredScalarAutodiffTests VernonRuntime)
        vernon_discover_gtests(VernonStructuredScalarAutodiffTests)

        add_executable(VernonStructuredAggregateAutodiffTests runtime/runtime_structured_aggregate_autodiff_test.cpp)
        target_link_libraries(VernonStructuredAggregateAutodiffTests
                              PRIVATE GTest::gtest_main Vernon::Runtime ${structured_aggregate_autodiff_fixture_target})
        target_include_directories(VernonStructuredAggregateAutodiffTests PRIVATE "${VERNON_TEST_ROOT}/support")
        target_compile_definitions(
            VernonStructuredAggregateAutodiffTests
            PRIVATE VERNON_STRUCTURED_AGGREGATE_AUTODIFF_MANIFEST="${structured_aggregate_autodiff_manifest}")
        set_target_properties(VernonStructuredAggregateAutodiffTests PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                                                "${CMAKE_BINARY_DIR}/source")
        vernon_stage_target_files(VernonStructuredAggregateAutodiffTests VernonRuntime)
        vernon_discover_gtests(VernonStructuredAggregateAutodiffTests)

        add_executable(VernonStructuredStorageAutodiffTests runtime/runtime_structured_storage_autodiff_test.cpp)
        target_link_libraries(VernonStructuredStorageAutodiffTests
                              PRIVATE GTest::gtest_main Vernon::Runtime ${structured_storage_autodiff_fixture_target})
        target_compile_definitions(
            VernonStructuredStorageAutodiffTests
            PRIVATE VERNON_STRUCTURED_STORAGE_AUTODIFF_MANIFEST="${structured_storage_autodiff_manifest}")
        if(VERNON_HOST_TAPE_INSTRUMENTATION)
            target_compile_definitions(VernonStructuredStorageAutodiffTests PRIVATE VERNON_HOST_TAPE_INSTRUMENTATION=1)
        endif()
        target_include_directories(VernonStructuredStorageAutodiffTests PRIVATE "${VERNON_REPOSITORY_ROOT}/source/lib"
                                                                                "${VERNON_TEST_ROOT}/support")
        set_target_properties(VernonStructuredStorageAutodiffTests PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                                              "${CMAKE_BINARY_DIR}/source")
        vernon_stage_target_files(VernonStructuredStorageAutodiffTests VernonRuntime)
        vernon_discover_gtests(VernonStructuredStorageAutodiffTests)

        add_executable(VernonRuntimeWorkgroupAutodiffTests runtime/runtime_workgroup_autodiff_test.cpp)
        target_link_libraries(
            VernonRuntimeWorkgroupAutodiffTests PRIVATE GTest::gtest_main Vernon::Runtime
                                                        ${structured_storage_workgroup_autodiff_fixture_target})
        target_compile_definitions(
            VernonRuntimeWorkgroupAutodiffTests
            PRIVATE
                VERNON_STRUCTURED_STORAGE_WORKGROUP_AUTODIFF_MANIFEST="${structured_storage_workgroup_autodiff_manifest}"
        )
        target_include_directories(VernonRuntimeWorkgroupAutodiffTests PRIVATE "${VERNON_REPOSITORY_ROOT}/source/lib"
                                                                               "${VERNON_TEST_ROOT}/support")
        set_target_properties(VernonRuntimeWorkgroupAutodiffTests PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                                             "${CMAKE_BINARY_DIR}/source")
        vernon_stage_target_files(VernonRuntimeWorkgroupAutodiffTests VernonRuntime)
        vernon_discover_gtests(VernonRuntimeWorkgroupAutodiffTests)

        vernon_add_python_program_fixture(
            VernonCookedAutodiffPythonFixture
            vernon_cooked_autodiff_fixture
            vernonRegisterAutodiffFixture
            ${cooked_autodiff_fixture_target})
        vernon_add_python_program_fixture(
            VernonNativeAutodiffF16PythonFixture
            vernon_native_autodiff_f16_fixture
            vernonRegisterNativeAutodiffF16Fixture
            ${native_autodiff_f16_fixture_target})
        vernon_add_python_program_fixture(
            VernonTypedSpecializationPythonFixture
            vernon_typed_specialization_fixture
            vernonRegisterTypedSpecializationFixture
            ${typed_specialization_fixture_target})
        vernon_add_python_program_fixture(
            VernonNativeAutodiffNumericCpuPythonFixture
            vernon_native_autodiff_numeric_cpu_fixture
            vernonRegisterNativeAutodiffNumericCpuFixture
            ${native_autodiff_numeric_cpu_fixture_target})
        vernon_add_python_program_fixture(
            VernonStructuredStorageAggregateAutodiffPythonFixture
            vernon_structured_storage_aggregate_autodiff_fixture
            vernonRegisterStructuredStorageAggregateAutodiffFixture
            ${structured_storage_aggregate_autodiff_fixture_target})
        vernon_add_python_program_fixture(
            VernonDynamicV2SignedStridePythonFixture
            vernon_dynamic_v2_signed_stride_fixture
            vernonRegisterDynamicV2SignedStrideFixture
            ${dynamic_v2_signed_stride_fixture_target})
        vernon_add_python_program_fixture(
            VernonDynamicV2FieldPathPythonFixture
            vernon_dynamic_v2_field_path_fixture
            vernonRegisterDynamicV2FieldPathFixture
            ${dynamic_v2_field_path_fixture_target})
        vernon_add_python_program_fixture(
            VernonDynamicV2StridedScatterPythonFixture
            vernon_dynamic_v2_strided_scatter_fixture
            vernonRegisterDynamicV2StridedScatterFixture
            ${dynamic_v2_strided_scatter_fixture_target})
        vernon_add_python_program_fixture(
            VernonDynamicV2GatherPythonFixture
            vernon_dynamic_v2_gather_fixture
            vernonRegisterDynamicV2GatherFixture
            ${dynamic_v2_gather_fixture_target})
        vernon_add_python_program_fixture(
            VernonDynamicV2MultiOutputPythonFixture
            vernon_dynamic_v2_multi_output_fixture
            vernonRegisterDynamicV2MultiOutputFixture
            ${dynamic_v2_multi_output_fixture_target})
        vernon_add_python_program_fixture(
            VernonDynamicV2PartiallyDynamicPythonFixture
            vernon_dynamic_v2_partially_dynamic_fixture
            vernonRegisterDynamicV2PartiallyDynamicFixture
            ${dynamic_v2_partially_dynamic_fixture_target})
        vernon_add_python_program_fixture(
            VernonDynamicV2MixedAliasPythonFixture
            vernon_dynamic_v2_mixed_alias_fixture
            vernonRegisterDynamicV2MixedAliasFixture
            ${dynamic_v2_mixed_alias_fixture_target})
        add_test(
            NAME VernonPythonCookedAutodiffTest
            COMMAND
                "${Python_EXECUTABLE}" "${VERNON_TEST_ROOT}/python/autodiff_cooked_runtime_test.py"
                "${cooked_autodiff_manifest}" "$<TARGET_FILE_DIR:VernonCookedAutodiffPythonFixture>"
                "${VERNON_REPOSITORY_ROOT}/python" "${VERNON_TEST_ROOT}/fixtures")
        add_test(
            NAME VernonPythonNativeAutodiffF16Test
            COMMAND
                "${Python_EXECUTABLE}" "${VERNON_TEST_ROOT}/python/autodiff_native_f16_runtime_test.py"
                "${native_autodiff_f16_manifest}" "$<TARGET_FILE_DIR:VernonNativeAutodiffF16PythonFixture>"
                "${VERNON_REPOSITORY_ROOT}/python")
        add_test(
            NAME VernonPythonTypedSpecializationTest
            COMMAND
                "${Python_EXECUTABLE}" "${VERNON_TEST_ROOT}/python/typed_specialization_runtime_test.py"
                "${typed_specialization_manifest}" "$<TARGET_FILE_DIR:VernonTypedSpecializationPythonFixture>"
                "${VERNON_REPOSITORY_ROOT}/python")
        add_test(
            NAME VernonPythonNativeAutodiffNumericCpuTest
            COMMAND
                "${Python_EXECUTABLE}" "${VERNON_TEST_ROOT}/python/autodiff_native_numeric_runtime_test.py"
                "${native_autodiff_numeric_cpu_manifest}"
                "$<TARGET_FILE_DIR:VernonNativeAutodiffNumericCpuPythonFixture>" "${VERNON_REPOSITORY_ROOT}/python")
        add_test(
            NAME VernonPythonStructuredStorageAggregateAutodiffTest
            COMMAND
                "${Python_EXECUTABLE}"
                "${VERNON_TEST_ROOT}/python/autodiff_structured_storage_aggregate_runtime_test.py"
                "${structured_storage_aggregate_autodiff_manifest}"
                "$<TARGET_FILE_DIR:VernonStructuredStorageAggregateAutodiffPythonFixture>"
                "${VERNON_REPOSITORY_ROOT}/python" "${VERNON_TEST_ROOT}/fixtures")
        add_test(
            NAME VernonPythonStructuredScalarDirectAutodiffTest
            COMMAND
                "${Python_EXECUTABLE}" "${VERNON_TEST_ROOT}/python/autodiff_structured_scalar_direct_runtime_test.py"
                "${VERNON_REPOSITORY_ROOT}/python" "${VERNON_TEST_ROOT}/fixtures")
        add_test(
            NAME VernonPythonDynamicV2ParityTest
            COMMAND
                "${Python_EXECUTABLE}" "${VERNON_TEST_ROOT}/python/autodiff_dynamic_v2_parity_runtime_test.py"
                "${dynamic_v2_signed_stride_manifest}" "${dynamic_v2_field_path_manifest}"
                "${dynamic_v2_strided_scatter_manifest}" "${dynamic_v2_gather_manifest}"
                "${dynamic_v2_multi_output_manifest}" "${dynamic_v2_partially_dynamic_manifest}"
                "${dynamic_v2_mixed_alias_manifest}" "$<TARGET_FILE_DIR:VernonDynamicV2SignedStridePythonFixture>"
                "${VERNON_REPOSITORY_ROOT}/python" "${VERNON_TEST_ROOT}/fixtures")
    endif()

    if(VERNON_ENABLE_VULKAN_RUNTIME AND TARGET VernonDSLCompiler)
        target_sources(VernonRuntimeTests PRIVATE runtime/runtime_vulkan_test.cpp)
        target_link_libraries(VernonRuntimeTests PRIVATE VernonDSLCompiler)
    endif()

    if(VERNON_ENABLE_VULKAN_RUNTIME)
        target_sources(VernonRuntimeTests PRIVATE rhi/vulkan_native_interop_test.cpp)
        target_link_libraries(VernonRuntimeTests PRIVATE Vernon::RHI Vulkan::Headers)
    endif()

    if(VERNON_ENABLE_METAL_RUNTIME)
        target_sources(VernonRuntimeTests PRIVATE runtime/runtime_metal_test.cpp)
        set_source_files_properties(runtime/runtime_metal_test.cpp PROPERTIES LANGUAGE OBJCXX)
        if(TARGET vernon-dsl-native AND Python_FOUND)
            vernon_add_program_asset_test_bundle(
                metal-compute-pipeline
                metal
                "${VERNON_REPOSITORY_ROOT}/python/tests/program_asset_fixture.py:scale_asset"
                metal_compute_program_bundle
                metal_compute_program_test_manifest)
            vernon_add_program_asset_test_bundle(
                metal-graphics-pipeline
                metal
                "${VERNON_REPOSITORY_ROOT}/python/tests/program_asset_fixture.py:sampled_asset"
                metal_graphics_program_bundle
                metal_graphics_program_test_manifest)
            vernon_add_program_asset_test_bundle(
                metal-resolution-pipeline
                metal
                "${VERNON_REPOSITORY_ROOT}/python/tests/program_asset_fixture.py:resolution_asset"
                metal_resolution_program_bundle
                metal_resolution_program_test_manifest)
            target_compile_definitions(
                VernonRuntimeTests
                PRIVATE VERNON_METAL_COMPUTE_PROGRAM_BUNDLE="${metal_compute_program_test_manifest}"
                        VERNON_METAL_GRAPHICS_PROGRAM_BUNDLE="${metal_graphics_program_test_manifest}"
                        VERNON_METAL_RESOLUTION_PROGRAM_BUNDLE="${metal_resolution_program_test_manifest}")
            add_dependencies(
                VernonRuntimeTests
                vernon-metal-compute-pipeline-test-bundle
                vernon-metal-graphics-pipeline-test-bundle
                vernon-metal-resolution-pipeline-test-bundle)
            add_test(
                NAME vernon-metal-mandelbulb-showcase
                COMMAND
                    ${Python_EXECUTABLE} "${VERNON_REPOSITORY_ROOT}/examples/mandelbulb_showcase.py" --architecture
                    metal --headless --preset smoke --output "${VERNON_TEST_BINARY_ROOT}/metal_mandelbulb_showcase.png")
            add_test(NAME vernon-metal-fractal-example
                     COMMAND ${Python_EXECUTABLE} "${VERNON_REPOSITORY_ROOT}/examples/fractal.py" --architecture metal
                             --headless --frames 1)
            add_test(
                NAME vernon-metal-pbr-example
                COMMAND ${Python_EXECUTABLE} "${VERNON_REPOSITORY_ROOT}/examples/pbr.py" --architecture metal
                        --headless --frames 1 --size 128 --output "${VERNON_TEST_BINARY_ROOT}/metal_pbr_example.png")
            add_test(
                NAME vernon-metal-complete-pipeline-example
                COMMAND
                    ${Python_EXECUTABLE} "${VERNON_REPOSITORY_ROOT}/examples/complete_pipeline.py" --architecture metal
                    --headless --frames 1 --size 64 --instances 2 --output
                    "${VERNON_TEST_BINARY_ROOT}/metal_complete_pipeline_example.png")
            add_test(
                NAME vernon-metal-dynamic-ocean-example
                COMMAND
                    ${Python_EXECUTABLE} "${VERNON_REPOSITORY_ROOT}/examples/dynamic_ocean.py" --arch metal --headless
                    --frames 1 --size 64 --grid 16 --output "${VERNON_TEST_BINARY_ROOT}/metal_dynamic_ocean.png")
            add_test(
                NAME vernon-metal-dynamic-terrain-erosion-example
                COMMAND
                    ${Python_EXECUTABLE} "${VERNON_REPOSITORY_ROOT}/examples/dynamic_terrain_erosion.py" --arch metal
                    --headless --frames 1 --size 64 --grid 16 --output
                    "${VERNON_TEST_BINARY_ROOT}/metal_dynamic_terrain_erosion.png")
            set_tests_properties(
                vernon-metal-mandelbulb-showcase
                vernon-metal-fractal-example
                vernon-metal-pbr-example
                vernon-metal-complete-pipeline-example
                vernon-metal-dynamic-ocean-example
                vernon-metal-dynamic-terrain-erosion-example
                PROPERTIES ENVIRONMENT
                           "PYTHONPATH=${VERNON_REPOSITORY_ROOT}/python"
                           SKIP_REGULAR_EXPRESSION
                           "Metal argument-buffer encoding is unavailable on this device")
        endif()
    endif()

    if(VERNON_ENABLE_DIRECTX12_RUNTIME)
        target_sources(VernonRuntimeTests PRIVATE runtime/runtime_directx12_test.cpp
                                                  rhi/directx12_native_interop_test.cpp)
        target_link_libraries(VernonRuntimeTests PRIVATE Vernon::RHI d3d12 dxgi)
        if(TARGET vernon-dsl-native AND Python_FOUND)
            vernon_add_program_asset_test_bundle(
                directx-pipeline
                directx
                "${VERNON_REPOSITORY_ROOT}/python/tests/program_asset_fixture.py:sampled_depth_asset"
                directx_program_bundle
                directx_program_test_manifest)
            vernon_add_program_asset_test_bundle(
                directx-compute-pipeline
                directx
                "${VERNON_REPOSITORY_ROOT}/python/tests/program_asset_fixture.py:scale_asset"
                directx_compute_program_bundle
                directx_compute_program_test_manifest)
            vernon_add_program_asset_test_bundle(
                directx-resolution-pipeline
                directx
                "${VERNON_REPOSITORY_ROOT}/python/tests/program_asset_fixture.py:resolution_asset"
                directx_resolution_program_bundle
                directx_resolution_program_test_manifest)
            target_sources(VernonRuntimeTests PRIVATE runtime/runtime_directx12_pipeline_test.cpp)
            target_compile_definitions(
                VernonRuntimeTests
                PRIVATE VERNON_DIRECTX_PROGRAM_BUNDLE="${directx_program_test_manifest}"
                        VERNON_DIRECTX_COMPUTE_PROGRAM_BUNDLE="${directx_compute_program_test_manifest}"
                        VERNON_DIRECTX_RESOLUTION_PROGRAM_BUNDLE="${directx_resolution_program_test_manifest}")
            target_link_libraries(VernonRuntimeTests PRIVATE VernonDSLCompiler)
            add_dependencies(
                VernonRuntimeTests
                vernon-directx-pipeline-test-bundle
                vernon-directx-compute-pipeline-test-bundle
                vernon-directx-resolution-pipeline-test-bundle)
        endif()
    endif()

    if(VERNON_ENABLE_VULKAN_RUNTIME
       AND TARGET vernon-dsl-native
       AND Python_FOUND)
        vernon_add_program_asset_test_bundle(
            vulkan-pipeline
            vulkan
            "${VERNON_REPOSITORY_ROOT}/python/tests/program_asset_fixture.py:sampled_depth_asset"
            vulkan_program_bundle
            vulkan_program_test_manifest)
        vernon_add_program_asset_test_bundle(
            vulkan-compute-pipeline
            vulkan
            "${VERNON_REPOSITORY_ROOT}/python/tests/program_asset_fixture.py:scale_asset"
            vulkan_compute_program_bundle
            vulkan_compute_program_test_manifest)
        vernon_add_program_asset_test_bundle(
            vulkan-cube-map
            vulkan
            "${VERNON_REPOSITORY_ROOT}/python/tests/cube_map_shader.py:cube_map_asset"
            vulkan_cube_map_bundle
            vulkan_cube_map_test_manifest)
        vernon_add_program_asset_test_bundle(
            vulkan-multiset-texture
            vulkan
            "${VERNON_TEST_ROOT}/fixtures/language_contract_texture_asset.py:multiset_asset"
            vulkan_multiset_texture_bundle
            vulkan_multiset_texture_test_manifest)
        target_sources(VernonRuntimeTests PRIVATE runtime/runtime_vulkan_pipeline_test.cpp
                                                  runtime/runtime_vulkan_cube_map_test.cpp)
        target_compile_definitions(
            VernonRuntimeTests
            PRIVATE VERNON_VULKAN_PROGRAM_BUNDLE="${vulkan_program_test_manifest}"
                    VERNON_VULKAN_CUBE_MAP_BUNDLE="${vulkan_cube_map_test_manifest}"
                    VERNON_VULKAN_COMPUTE_PROGRAM_BUNDLE="${vulkan_compute_program_test_manifest}"
                    VERNON_VULKAN_MULTISET_TEXTURE_BUNDLE="${vulkan_multiset_texture_test_manifest}")
        add_dependencies(
            VernonRuntimeTests
            vernon-vulkan-pipeline-test-bundle
            vernon-vulkan-cube-map-test-bundle
            vernon-vulkan-compute-pipeline-test-bundle
            vernon-vulkan-multiset-texture-test-bundle)
    endif()

    if(TARGET vernon-dsl-native AND Python_FOUND)
        vernon_add_program_asset_test_bundle(
            opengl-runtime-acceptance
            opengl
            "${VERNON_REPOSITORY_ROOT}/python/tests/program_asset_fixture.py:opengl_runtime_acceptance_asset"
            opengl_runtime_acceptance_bundle
            opengl_runtime_acceptance_manifest)
        vernon_add_program_asset_test_bundle(
            opengl-matrix-uniform
            opengl
            "${VERNON_REPOSITORY_ROOT}/python/tests/program_asset_fixture.py:matrix_asset"
            opengl_matrix_uniform_bundle
            opengl_matrix_uniform_manifest)
        vernon_add_program_asset_test_bundle(
            opengles-matrix-uniform
            opengles
            "${VERNON_REPOSITORY_ROOT}/python/tests/program_asset_fixture.py:matrix_asset"
            opengles_matrix_uniform_bundle
            opengles_matrix_uniform_manifest)
        vernon_add_program_asset_test_bundle(
            opengl-signed-uniform
            opengl
            "${VERNON_REPOSITORY_ROOT}/python/tests/program_asset_fixture.py:signed_uniform_asset"
            opengl_signed_uniform_bundle
            opengl_signed_uniform_manifest)
        vernon_add_program_asset_test_bundle(
            opengl-unsigned-uniform
            opengl
            "${VERNON_REPOSITORY_ROOT}/python/tests/program_asset_fixture.py:unsigned_uniform_asset"
            opengl_unsigned_uniform_bundle
            opengl_unsigned_uniform_manifest)
        vernon_add_program_asset_test_bundle(
            opengl-explicit-sampler
            opengl
            "${VERNON_REPOSITORY_ROOT}/python/tests/program_asset_fixture.py:explicit_sampler_asset"
            opengl_explicit_sampler_bundle
            opengl_explicit_sampler_manifest)
        vernon_add_program_asset_test_bundle(
            opengl-indexed-vertex
            opengl
            "${VERNON_REPOSITORY_ROOT}/python/tests/program_asset_fixture.py:indexed_asset"
            opengl_indexed_vertex_bundle
            opengl_indexed_vertex_manifest)
        vernon_add_program_asset_test_bundle(
            opengl-i32-vertex
            opengl
            "${VERNON_REPOSITORY_ROOT}/python/tests/program_asset_fixture.py:i32_vertex_asset"
            opengl_i32_vertex_bundle
            opengl_i32_vertex_manifest)
        vernon_add_program_asset_test_bundle(
            opengl-u32-vertex
            opengl
            "${VERNON_REPOSITORY_ROOT}/python/tests/program_asset_fixture.py:u32_vertex_asset"
            opengl_u32_vertex_bundle
            opengl_u32_vertex_manifest)
        vernon_add_program_asset_test_bundle(
            opengl-f32-vertex
            opengl
            "${VERNON_REPOSITORY_ROOT}/python/tests/program_asset_fixture.py:f32_vertex_asset"
            opengl_f32_vertex_bundle
            opengl_f32_vertex_manifest)
        vernon_add_program_asset_test_bundle(
            opengl-storage-image
            opengl
            "${VERNON_REPOSITORY_ROOT}/python/tests/program_asset_fixture.py:storage_image_asset"
            opengl_storage_image_bundle
            opengl_storage_image_manifest)
        target_compile_definitions(
            VernonRuntimeTests
            PRIVATE VERNON_OPENGL_RUNTIME_ACCEPTANCE_BUNDLE="${opengl_runtime_acceptance_manifest}"
                    VERNON_OPENGL_MATRIX_UNIFORM_BUNDLE="${opengl_matrix_uniform_manifest}"
                    VERNON_OPENGLES_MATRIX_UNIFORM_BUNDLE="${opengles_matrix_uniform_manifest}"
                    VERNON_OPENGL_SIGNED_UNIFORM_BUNDLE="${opengl_signed_uniform_manifest}"
                    VERNON_OPENGL_UNSIGNED_UNIFORM_BUNDLE="${opengl_unsigned_uniform_manifest}"
                    VERNON_OPENGL_EXPLICIT_SAMPLER_BUNDLE="${opengl_explicit_sampler_manifest}"
                    VERNON_OPENGL_INDEXED_VERTEX_BUNDLE="${opengl_indexed_vertex_manifest}"
                    VERNON_OPENGL_I32_VERTEX_BUNDLE="${opengl_i32_vertex_manifest}"
                    VERNON_OPENGL_U32_VERTEX_BUNDLE="${opengl_u32_vertex_manifest}"
                    VERNON_OPENGL_F32_VERTEX_BUNDLE="${opengl_f32_vertex_manifest}"
                    VERNON_OPENGL_STORAGE_IMAGE_BUNDLE="${opengl_storage_image_manifest}")
        add_dependencies(
            VernonRuntimeTests
            vernon-opengl-runtime-acceptance-test-bundle
            vernon-opengl-matrix-uniform-test-bundle
            vernon-opengles-matrix-uniform-test-bundle
            vernon-opengl-signed-uniform-test-bundle
            vernon-opengl-unsigned-uniform-test-bundle
            vernon-opengl-explicit-sampler-test-bundle
            vernon-opengl-indexed-vertex-test-bundle
            vernon-opengl-i32-vertex-test-bundle
            vernon-opengl-u32-vertex-test-bundle
            vernon-opengl-f32-vertex-test-bundle
            vernon-opengl-storage-image-test-bundle)
    endif()

    vernon_discover_gtests(VernonRuntimeTests)
    add_test(
        NAME vernon-runtime-package-smoke
        COMMAND
            ${CMAKE_COMMAND} "-DVERNON_BUILD_DIR=${CMAKE_BINARY_DIR}"
            "-DVERNON_SMOKE_SOURCE_DIR=${VERNON_TEST_ROOT}/runtime/package_smoke"
            "-DVERNON_SMOKE_BUILD_DIR=${VERNON_TEST_BINARY_ROOT}/runtime_package_smoke_build"
            "-DVERNON_INSTALL_PREFIX=${VERNON_TEST_BINARY_ROOT}/runtime_package_smoke_install"
            "-DVERNON_TEST_CONFIG=$<CONFIG>" "-DVERNON_TEST_GENERATOR=${CMAKE_GENERATOR}"
            "-DVERNON_TEST_PLATFORM=${CMAKE_GENERATOR_PLATFORM}" -P
            "${VERNON_TEST_ROOT}/scripts/run_runtime_package_smoke.cmake")
    set_tests_properties(vernon-runtime-package-smoke PROPERTIES RUN_SERIAL TRUE)
endif()

if(TARGET VernonCompilerTestSupport AND TARGET MLIRVernonTransforms)
    add_executable(
        VernonAutodiffAnalysisTests
        compiler/vernon_autodiff_analysis_test.cpp
        compiler/vernon_autodiff_rules_test.cpp
        compiler/vernon_autodiff_tape_planning_test.cpp
        compiler/vernon_structured_vjp_test.cpp)
    set_target_properties(VernonAutodiffAnalysisTests PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/source")
    target_link_libraries(
        VernonAutodiffAnalysisTests
        PRIVATE GTest::gtest_main
                VernonCompilerTestSupport
                MLIRVernonTransforms
                MLIRVernonProgramTransforms)
    vernon_discover_gtests(VernonAutodiffAnalysisTests)
endif()

if(TARGET VernonDSLCompiler)
    add_executable(
        VernonCompilerTests
        compiler/compiled_program_ownership_test.cpp
        compiler/compiler_c_api_test.cpp
        compiler/compiler_cpp_api_test.cpp
        compiler/compiler_graphics_output_test.cpp
        compiler/compiler_program_finalization_test.cpp
        compiler/compiler_synchronization_test.cpp
        compiler/spirv_image_query_size_lod_materialization_test.cpp
        compiler/vernon_compile_packaging_test.cpp
        compiler/vernon_value_abi_test.cpp)
    set_target_properties(VernonCompilerTests PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/source")
    target_include_directories(
        VernonCompilerTests
        PRIVATE "${VERNON_REPOSITORY_ROOT}/source/lib/compiler"
                "${VERNON_REPOSITORY_ROOT}/source/tools"
                "${VERNON_TEST_ROOT}/support"
                $<TARGET_PROPERTY:VernonDSLCompiler,INCLUDE_DIRECTORIES>
                ${LLVM_INCLUDE_DIRS}
                ${MLIR_INCLUDE_DIRS})
    target_link_libraries(
        VernonCompilerTests
        PRIVATE GTest::gtest_main
                VernonDSLCompiler
                vernon-compile-packaging
                nlohmann_json::nlohmann_json)
    vernon_stage_target_files(VernonCompilerTests VernonDSLCompiler)
    add_dependencies(VernonCompilerTests vernon-opt vernon-compile)
    vernon_discover_gtests(VernonCompilerTests)

    if(Python_Interpreter_FOUND)
        foreach(test_module IN ITEMS test_compiler.py test_shared_definitions.py test_blinn_phong_numeric.py)
            string(
                REPLACE ".py"
                        ""
                        test_name
                        "${test_module}")
            add_test(NAME "vernon-python-${test_name}"
                     COMMAND ${Python_EXECUTABLE} -m unittest discover -s "${VERNON_REPOSITORY_ROOT}/python/tests" -p
                             "${test_module}")
            set_tests_properties("vernon-python-${test_name}" PROPERTIES ENVIRONMENT
                                                                         "PYTHONPATH=${VERNON_REPOSITORY_ROOT}/python")
        endforeach()
    endif()

    if(TARGET vernon-dsl-native)
        add_test(NAME vernon-python-test_storage_vjp_contract
                 COMMAND ${Python_EXECUTABLE} -m unittest discover -s "${VERNON_REPOSITORY_ROOT}/python/tests" -p
                         "test_storage_vjp_contract.py")
        add_test(NAME vernon-python-smoke-fluid-graph
                 COMMAND ${Python_EXECUTABLE} -m unittest discover -s "${VERNON_REPOSITORY_ROOT}/python/tests" -p
                         "test_smoke_fluid_graph.py")
        add_test(NAME vernon-python-module-program
                 COMMAND ${Python_EXECUTABLE} -m unittest discover -s "${VERNON_REPOSITORY_ROOT}/python/tests" -p
                         "test_program_execution.py")
        add_test(NAME vernon-python-autodiff-phase2-benchmark
                 COMMAND ${Python_EXECUTABLE} "${VERNON_REPOSITORY_ROOT}/scripts/benchmark_autodiff_smoke_phase2.py"
                         --grids 32 64 --output "${VERNON_TEST_BINARY_ROOT}/autodiff_phase2_benchmark_ci.json")
        add_test(
            NAME vernon-python-atomic-reduction-benchmark
            COMMAND
                ${Python_EXECUTABLE} "${VERNON_REPOSITORY_ROOT}/scripts/benchmark_atomic_reduction.py" --architectures
                cpu --patterns one_bin workgroup_publication --invocations 128 --iterations 2 --output
                "${VERNON_TEST_BINARY_ROOT}/atomic_reduction_benchmark_ci.json" --markdown
                "${VERNON_TEST_BINARY_ROOT}/atomic_reduction_benchmark_ci.md")
        set_tests_properties(
            vernon-python-test_storage_vjp_contract
            vernon-python-smoke-fluid-graph
            vernon-python-module-program
            vernon-python-autodiff-phase2-benchmark
            vernon-python-atomic-reduction-benchmark
            PROPERTIES ENVIRONMENT
                       "PYTHONPATH=${VERNON_REPOSITORY_ROOT}/python"
                       WORKING_DIRECTORY
                       "${VERNON_REPOSITORY_ROOT}")
        set_tests_properties(
            vernon-python-autodiff-phase2-benchmark vernon-python-atomic-reduction-benchmark
            PROPERTIES TIMEOUT
                       300
                       LABELS
                       benchmark)
        add_test(NAME vernon-python-native-compiled-program
                 COMMAND ${Python_EXECUTABLE} "${VERNON_REPOSITORY_ROOT}/python/tests/test_native_compiled_program.py")
        add_test(NAME vernon-python-kernel-memory-phase-compile
                 COMMAND ${Python_EXECUTABLE} "${VERNON_TEST_ROOT}/python/kernel_memory_phase_compile_test.py")
        add_test(NAME vernon-python-compile-surface-parity
                 COMMAND ${Python_EXECUTABLE} "${VERNON_TEST_ROOT}/compiler/compile_surface_parity_test.py"
                         $<TARGET_FILE:VernonDSLCompiler> $<TARGET_FILE:vernon-compile>)
        set_tests_properties(
            vernon-python-native-compiled-program vernon-python-kernel-memory-phase-compile
            vernon-python-compile-surface-parity PROPERTIES ENVIRONMENT "PYTHONPATH=${VERNON_REPOSITORY_ROOT}/python")
    endif()

    if(WIN32)
        add_test(
            NAME vernon-python-native-integration
            COMMAND
                powershell -ExecutionPolicy Bypass -File "${VERNON_TEST_ROOT}/scripts/run_frontend_integration.ps1"
                -ProjectRoot "${VERNON_REPOSITORY_ROOT}" -BinaryDir "${VERNON_TEST_BINARY_ROOT}" -VernonOpt
                $<TARGET_FILE:vernon-opt> -VernonCompiler $<TARGET_FILE:vernon-compile>)
    endif()
endif()
