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

function(vernon_discover_gtests target)
    gtest_discover_tests(${target} DISCOVERY_MODE POST_BUILD WORKING_DIRECTORY "${VERNON_TEST_BINARY_ROOT}")
endfunction()

if(TARGET VernonRuntime)
    add_library(VernonRuntimePublicCHeaderCheck OBJECT runtime/runtime_public_header_compile.c)
    target_link_libraries(VernonRuntimePublicCHeaderCheck PRIVATE Vernon::Runtime)
    add_library(VernonRhiPublicCHeaderCheck OBJECT runtime/runtime_rhi_public_header_compile.c)
    target_link_libraries(VernonRhiPublicCHeaderCheck PRIVATE Vernon::RHI Vernon::RuntimeCore)

    get_target_property(_vernon_runtime_core_links VernonRuntimeCore LINK_LIBRARIES)
    if(_vernon_runtime_core_links MATCHES "(^|;)Vernon(RHI|::RHI)(;|$)")
        message(FATAL_ERROR "VernonRuntimeCore must not link VernonRHI")
    endif()

    add_executable(VernonRuntimeCoreProviderTests runtime/runtime_foreign_provider_test.cpp)
    target_link_libraries(VernonRuntimeCoreProviderTests PRIVATE GTest::gtest_main Vernon::RuntimeCore)
    add_dependencies(VernonRuntimeCoreProviderTests VernonRhiPublicCHeaderCheck)
    if(MSVC)
        target_compile_options(VernonRuntimeCoreProviderTests PRIVATE /EHsc)
    endif()
    vernon_discover_gtests(VernonRuntimeCoreProviderTests)

    if(VERNON_ENABLE_CUDA_RUNTIME)
        add_executable(VernonRuntimeRHIAdapterTests rhi/cuda_adapter_test.cpp)
        target_include_directories(VernonRuntimeRHIAdapterTests PRIVATE "${VERNON_REPOSITORY_ROOT}/source/lib")
        target_link_libraries(VernonRuntimeRHIAdapterTests PRIVATE GTest::gtest_main Vernon::RuntimeCore
                                                                   Vernon::RuntimeRHIAdapter)
        vernon_stage_target_files(VernonRuntimeRHIAdapterTests VernonRHI)
        if(MSVC)
            target_compile_options(VernonRuntimeRHIAdapterTests PRIVATE /EHsc)
        endif()
        vernon_discover_gtests(VernonRuntimeRHIAdapterTests)
    endif()

    if(WIN32)
        set(runtime_test_os windows)
    elseif(APPLE)
        set(runtime_test_os macos)
    else()
        set(runtime_test_os linux)
    endif()
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(AMD64|amd64|x86_64)$")
        set(runtime_test_arch x86_64)
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(ARM64|arm64|aarch64)$")
        set(runtime_test_arch aarch64)
    else()
        set(runtime_test_arch unknown)
    endif()

    add_library(vernon-cpu-aot-fixture SHARED fixtures/cpu_aot_fixture.c)
    target_include_directories(vernon-cpu-aot-fixture PRIVATE "${VERNON_REPOSITORY_ROOT}/source/include")
    set(cpu_bundle "${VERNON_TEST_BINARY_ROOT}/cpu_aot_bundle")
    add_custom_command(
        OUTPUT "${cpu_bundle}/compute.json" "${cpu_bundle}/cpu_fill.pipeline.json"
        COMMAND
            ${CMAKE_COMMAND} "-DARTIFACT=$<TARGET_FILE:vernon-cpu-aot-fixture>" "-DOUTPUT=${cpu_bundle}"
            "-DOPERATING_SYSTEM=${runtime_test_os}" "-DARCHITECTURE=${runtime_test_arch}" -P
            "${VERNON_TEST_ROOT}/scripts/write_cpu_aot_bundle.cmake"
        DEPENDS vernon-cpu-aot-fixture "${VERNON_TEST_ROOT}/scripts/write_cpu_aot_bundle.cmake"
        VERBATIM)
    add_custom_target(vernon-cpu-aot-test-bundle DEPENDS "${cpu_bundle}/compute.json"
                                                         "${cpu_bundle}/cpu_fill.pipeline.json")

    add_executable(
        VernonRuntimeTests
        runtime/compute_launch_planner_test.cpp
        runtime/directx_command_state_test.cpp
        runtime/execution_graph_test.cpp
        runtime/graphics_invocation_planner_test.cpp
        runtime/pipeline_manifest_requirements_test.cpp
        runtime/rhi_resource_lifetime_test.cpp
        runtime/runtime_c_api_test.cpp
        runtime/runtime_common_api_test.cpp
        runtime/runtime_cpu_pipeline_test.cpp
        runtime/runtime_external_gl_test.cpp
        runtime/structured_value_binding_test.cpp
        runtime/tensor_bridge_test.cpp
        runtime/vertex_attribute_capabilities_test.cpp)
    set_target_properties(VernonRuntimeTests PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/source")
    target_include_directories(VernonRuntimeTests PRIVATE "${VERNON_REPOSITORY_ROOT}/source/lib"
                                                          "${VERNON_TEST_ROOT}/support")
    target_link_libraries(
        VernonRuntimeTests
        PRIVATE GTest::gtest_main
                Vernon::Runtime
                Vernon::ExecutionGraph
                Vernon::RuntimeRHIAdapter
                VernonRuntimeInternals
                nlohmann_json::nlohmann_json)
    target_compile_definitions(
        VernonRuntimeTests PRIVATE VERNON_CPU_BUNDLE_PATH="${cpu_bundle}" VERNON_RUNTIME_TEST_OS="${runtime_test_os}"
                                   VERNON_RUNTIME_TEST_ARCH="${runtime_test_arch}")
    add_dependencies(VernonRuntimeTests vernon-cpu-aot-test-bundle)
    add_dependencies(VernonRuntimeTests VernonRuntimePublicCHeaderCheck)
    vernon_stage_target_files(VernonRuntimeTests VernonRuntime VernonRHI)
    if(TARGET VernonDSLCompiler)
        target_sources(VernonRuntimeTests PRIVATE runtime/runtime_synchronization_test.cpp)
        target_link_libraries(VernonRuntimeTests PRIVATE VernonDSLCompiler)
        vernon_stage_target_files(VernonRuntimeTests VernonDSLCompiler)
    endif()
    if(MSVC)
        target_compile_options(VernonRuntimeTests PRIVATE /EHsc)
    endif()

    if(TARGET vernon-compile)
        set(compiler_cpu_bundle "${VERNON_TEST_BINARY_ROOT}/compiler_cpu_aot_bundle")
        add_custom_command(
            OUTPUT "${compiler_cpu_bundle}/compute.json"
            COMMAND $<TARGET_FILE:vernon-compile> --target cpu "${VERNON_TEST_ROOT}/integration/cpu-aot-smoke.mlir"
                    --compute-bundle "${compiler_cpu_bundle}" --host-runtime-bundle
            DEPENDS vernon-compile "${VERNON_TEST_ROOT}/integration/cpu-aot-smoke.mlir"
            VERBATIM)
        add_custom_target(vernon-compiler-cpu-aot-test-bundle DEPENDS "${compiler_cpu_bundle}/compute.json")
        target_sources(VernonRuntimeTests PRIVATE runtime/runtime_compiler_cpu_aot_test.cpp)
        target_compile_definitions(VernonRuntimeTests PRIVATE VERNON_COMPILER_CPU_BUNDLE_PATH="${compiler_cpu_bundle}")
        add_dependencies(VernonRuntimeTests vernon-compiler-cpu-aot-test-bundle)
    endif()

    if(VERNON_ENABLE_CUDA_RUNTIME)
        target_sources(VernonRuntimeTests PRIVATE runtime/runtime_cuda_test.cpp runtime/runtime_cuda_pipeline_test.cpp)
    endif()

    if(VERNON_ENABLE_VULKAN_RUNTIME AND TARGET VernonDSLCompiler)
        target_sources(VernonRuntimeTests PRIVATE runtime/runtime_vulkan_test.cpp)
        target_link_libraries(VernonRuntimeTests PRIVATE VernonDSLCompiler)
    endif()

    if(VERNON_ENABLE_VULKAN_RUNTIME)
        target_sources(VernonRuntimeTests PRIVATE rhi/vulkan_native_interop_test.cpp)
        target_link_libraries(VernonRuntimeTests PRIVATE Vernon::RHI Vulkan::Headers)
    endif()

    if(VERNON_ENABLE_DIRECTX12_RUNTIME)
        target_sources(VernonRuntimeTests PRIVATE runtime/runtime_directx12_test.cpp
                                                  rhi/directx12_native_interop_test.cpp)
        target_link_libraries(VernonRuntimeTests PRIVATE Vernon::RHI d3d12 dxgi)
        if(TARGET vernon-dsl-native AND Python_FOUND)
            set(directx_pipeline_test_dir "${VERNON_TEST_BINARY_ROOT}/directx_pipeline_bundle")
            set(directx_pipeline_test_manifest "${directx_pipeline_test_dir}/directx_pipeline_bundle.pipeline.json")
            set(directx_compute_pipeline_test_dir "${VERNON_TEST_BINARY_ROOT}/directx_compute_pipeline_bundle")
            set(directx_compute_pipeline_test_manifest
                "${directx_compute_pipeline_test_dir}/directx_compute_pipeline_bundle.pipeline.json")
            add_custom_command(
                OUTPUT "${directx_pipeline_test_manifest}"
                COMMAND
                    ${CMAKE_COMMAND} -E env "PYTHONPATH=${VERNON_REPOSITORY_ROOT}/python" ${Python_EXECUTABLE} -m
                    vernon_dsl.pipeline_asset_cli
                    "${VERNON_REPOSITORY_ROOT}/python/tests/pipeline_asset_fixture.py:sampled_asset" --target directx
                    --output "${directx_pipeline_test_dir}"
                DEPENDS vernon-dsl-native
                        "${VERNON_REPOSITORY_ROOT}/python/tests/pipeline_asset_fixture.py"
                        "${VERNON_REPOSITORY_ROOT}/python/vernon_dsl/bundle/parameters.py"
                        "${VERNON_REPOSITORY_ROOT}/python/vernon_dsl/bundle/requirements.py"
                VERBATIM)
            add_custom_target(vernon-directx-pipeline-test-bundle DEPENDS "${directx_pipeline_test_manifest}")
            add_custom_command(
                OUTPUT "${directx_compute_pipeline_test_manifest}"
                COMMAND
                    ${CMAKE_COMMAND} -E env "PYTHONPATH=${VERNON_REPOSITORY_ROOT}/python" ${Python_EXECUTABLE} -m
                    vernon_dsl.pipeline_asset_cli
                    "${VERNON_REPOSITORY_ROOT}/python/tests/pipeline_asset_fixture.py:scale_asset" --target directx
                    --output "${directx_compute_pipeline_test_dir}"
                DEPENDS vernon-dsl-native
                        "${VERNON_REPOSITORY_ROOT}/python/tests/pipeline_asset_fixture.py"
                        "${VERNON_REPOSITORY_ROOT}/python/vernon_dsl/bundle/parameters.py"
                        "${VERNON_REPOSITORY_ROOT}/python/vernon_dsl/bundle/requirements.py"
                VERBATIM)
            add_custom_target(vernon-directx-compute-pipeline-test-bundle
                              DEPENDS "${directx_compute_pipeline_test_manifest}")
            target_sources(VernonRuntimeTests PRIVATE runtime/runtime_directx12_pipeline_test.cpp)
            target_compile_definitions(
                VernonRuntimeTests
                PRIVATE VERNON_DIRECTX_PIPELINE_BUNDLE="${directx_pipeline_test_manifest}"
                        VERNON_DIRECTX_COMPUTE_PIPELINE_BUNDLE="${directx_compute_pipeline_test_manifest}")
            target_link_libraries(VernonRuntimeTests PRIVATE VernonDSLCompiler)
            vernon_stage_target_files(VernonRuntimeTests VernonDSLCompiler)
            add_dependencies(VernonRuntimeTests vernon-directx-pipeline-test-bundle
                             vernon-directx-compute-pipeline-test-bundle)
        endif()
    endif()

    if(VERNON_ENABLE_VULKAN_RUNTIME
       AND TARGET vernon-dsl-native
       AND Python_FOUND)
        set(vulkan_pipeline_test_dir "${VERNON_TEST_BINARY_ROOT}/vulkan_pipeline_bundle")
        set(vulkan_pipeline_test_manifest "${vulkan_pipeline_test_dir}/vulkan_pipeline_bundle.pipeline.json")
        add_custom_command(
            OUTPUT "${vulkan_pipeline_test_manifest}"
            COMMAND
                ${CMAKE_COMMAND} -E env "PYTHONPATH=${VERNON_REPOSITORY_ROOT}/python" ${Python_EXECUTABLE} -m
                vernon_dsl.pipeline_asset_cli
                "${VERNON_REPOSITORY_ROOT}/python/tests/pipeline_asset_fixture.py:sampled_asset" --target vulkan
                --output "${vulkan_pipeline_test_dir}"
            DEPENDS vernon-dsl-native
                    "${VERNON_REPOSITORY_ROOT}/python/tests/pipeline_asset_fixture.py"
                    "${VERNON_REPOSITORY_ROOT}/python/vernon_dsl/bundle/parameters.py"
                    "${VERNON_REPOSITORY_ROOT}/python/vernon_dsl/bundle/requirements.py"
                    "${VERNON_REPOSITORY_ROOT}/python/vernon_dsl/bundle/types.py"
                    "${VERNON_REPOSITORY_ROOT}/python/vernon_dsl/pipeline_assets.py"
            VERBATIM)
        add_custom_target(vernon-vulkan-pipeline-test-bundle DEPENDS "${vulkan_pipeline_test_manifest}")
        set(vulkan_compute_pipeline_test_dir "${VERNON_TEST_BINARY_ROOT}/vulkan_compute_pipeline_bundle")
        set(vulkan_compute_pipeline_test_manifest
            "${vulkan_compute_pipeline_test_dir}/vulkan_compute_pipeline_bundle.pipeline.json")
        add_custom_command(
            OUTPUT "${vulkan_compute_pipeline_test_manifest}"
            COMMAND
                ${CMAKE_COMMAND} -E env "PYTHONPATH=${VERNON_REPOSITORY_ROOT}/python" ${Python_EXECUTABLE} -m
                vernon_dsl.pipeline_asset_cli
                "${VERNON_REPOSITORY_ROOT}/python/tests/pipeline_asset_fixture.py:scale_asset" --target vulkan --output
                "${vulkan_compute_pipeline_test_dir}"
            DEPENDS vernon-dsl-native "${VERNON_REPOSITORY_ROOT}/python/tests/pipeline_asset_fixture.py"
                    "${VERNON_REPOSITORY_ROOT}/python/vernon_dsl/bundle/parameters.py"
            VERBATIM)
        add_custom_target(vernon-vulkan-compute-pipeline-test-bundle DEPENDS "${vulkan_compute_pipeline_test_manifest}")
        set(vulkan_cube_map_test_dir "${VERNON_TEST_BINARY_ROOT}/vulkan_cube_map_bundle")
        set(vulkan_cube_map_test_manifest "${vulkan_cube_map_test_dir}/vulkan_cube_map_bundle.pipeline.json")
        add_custom_command(
            OUTPUT "${vulkan_cube_map_test_manifest}"
            COMMAND
                ${CMAKE_COMMAND} -E env "PYTHONPATH=${VERNON_REPOSITORY_ROOT}/python" ${Python_EXECUTABLE} -m
                vernon_dsl.pipeline_asset_cli "${VERNON_REPOSITORY_ROOT}/python/tests/cube_map_shader.py:cube_map_asset"
                --target vulkan --output "${vulkan_cube_map_test_dir}"
            DEPENDS vernon-dsl-native
                    "${VERNON_REPOSITORY_ROOT}/python/tests/cube_map_shader.py"
                    "${VERNON_REPOSITORY_ROOT}/python/vernon_dsl/bundle/parameters.py"
                    "${VERNON_REPOSITORY_ROOT}/python/vernon_dsl/bundle/requirements.py"
                    "${VERNON_REPOSITORY_ROOT}/python/vernon_dsl/bundle/types.py"
                    "${VERNON_REPOSITORY_ROOT}/python/vernon_dsl/pipeline_assets.py"
            VERBATIM)
        add_custom_target(vernon-vulkan-cube-map-test-bundle DEPENDS "${vulkan_cube_map_test_manifest}")
        target_sources(VernonRuntimeTests PRIVATE runtime/runtime_vulkan_pipeline_test.cpp
                                                  runtime/runtime_vulkan_cube_map_test.cpp)
        target_compile_definitions(
            VernonRuntimeTests
            PRIVATE VERNON_VULKAN_PIPELINE_BUNDLE="${vulkan_pipeline_test_manifest}"
                    VERNON_VULKAN_CUBE_MAP_BUNDLE="${vulkan_cube_map_test_manifest}"
                    VERNON_VULKAN_COMPUTE_PIPELINE_BUNDLE="${vulkan_compute_pipeline_test_manifest}")
        add_dependencies(
            VernonRuntimeTests
            vernon-vulkan-pipeline-test-bundle
            vernon-vulkan-cube-map-test-bundle
            vernon-vulkan-compute-pipeline-test-bundle)
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

if(TARGET VernonDSLCompiler)
    add_executable(
        VernonCompilerTests
        compiler/compiled_program_ownership_test.cpp
        compiler/compiler_c_api_test.cpp
        compiler/compiler_cpp_api_test.cpp
        compiler/compiler_graphics_output_test.cpp
        compiler/compiler_synchronization_test.cpp
        compiler/spirv_image_query_size_lod_materialization_test.cpp
        compiler/vernon_compile_packaging_test.cpp)
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
    if(MSVC)
        target_compile_options(VernonCompilerTests PRIVATE /EHsc)
    endif()
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
        add_test(NAME vernon-python-native-compiled-program
                 COMMAND ${Python_EXECUTABLE} "${VERNON_REPOSITORY_ROOT}/python/tests/test_native_compiled_program.py")
        add_test(NAME vernon-python-compile-surface-parity
                 COMMAND ${Python_EXECUTABLE} "${VERNON_TEST_ROOT}/compiler/compile_surface_parity_test.py"
                         $<TARGET_FILE:VernonDSLCompiler> $<TARGET_FILE:vernon-compile>)
        set_tests_properties(vernon-python-native-compiled-program vernon-python-compile-surface-parity
                             PROPERTIES ENVIRONMENT "PYTHONPATH=${VERNON_REPOSITORY_ROOT}/python")
    endif()

    add_test(
        NAME vernon-cpu-lowering-integration
        COMMAND ${CMAKE_COMMAND} "-DVERNON_OPT=$<TARGET_FILE:vernon-opt>" "-DTEST_ROOT=${VERNON_TEST_ROOT}/integration"
                -P "${VERNON_TEST_ROOT}/scripts/run_cpu_lowering_tests.cmake")
    add_test(
        NAME vernon-gpu-atomic-exchange-to-spirv
        COMMAND
            ${CMAKE_COMMAND} "-DVERNON_OPT=$<TARGET_FILE:vernon-opt>"
            "-DINPUT=${VERNON_TEST_ROOT}/integration/gpu-atomic-exchange-to-spirv.mlir" -P
            "${VERNON_TEST_ROOT}/scripts/run_gpu_to_spirv_atomic_exchange_test.cmake")

    if(WIN32)
        add_test(NAME vernon-validation-integration
                 COMMAND powershell -ExecutionPolicy Bypass -File
                         "${VERNON_TEST_ROOT}/scripts/run_validation_tests.ps1" -VernonOpt $<TARGET_FILE:vernon-opt>)
        add_test(
            NAME vernon-python-native-integration
            COMMAND
                powershell -ExecutionPolicy Bypass -File "${VERNON_TEST_ROOT}/scripts/run_frontend_integration.ps1"
                -ProjectRoot "${VERNON_REPOSITORY_ROOT}" -BinaryDir "${VERNON_TEST_BINARY_ROOT}" -VernonOpt
                $<TARGET_FILE:vernon-opt> -VernonCompiler $<TARGET_FILE:vernon-compile>)
    endif()
endif()
