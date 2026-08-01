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

if(Python_EXECUTABLE)
    add_test(NAME VernonGeneratedVersionsCheck COMMAND "${Python_EXECUTABLE}"
                                                       "${VERNON_REPOSITORY_ROOT}/tools/generate_versions.py" --check)
endif()

function(vernon_discover_gtests target)
    gtest_discover_tests(${target} DISCOVERY_MODE POST_BUILD WORKING_DIRECTORY "${VERNON_TEST_BINARY_ROOT}")
endfunction()

function(
    vernon_add_pipeline_test_bundle
    name
    architecture
    asset_reference
    output_stem
    output_variable)
    set(_vernon_bundle_directory "${VERNON_TEST_BINARY_ROOT}/${output_stem}")
    set(_vernon_bundle_manifest "${_vernon_bundle_directory}/${output_stem}.pipeline.json")
    string(
        REGEX
        REPLACE ":[^:]+$"
                ""
                _vernon_asset_source
                "${asset_reference}")
    add_custom_command(
        OUTPUT "${_vernon_bundle_manifest}"
        COMMAND
            ${CMAKE_COMMAND} -E env "PYTHONPATH=${VERNON_REPOSITORY_ROOT}/python" ${Python_EXECUTABLE} -m
            vernon_dsl.pipeline_asset_cli "${asset_reference}" --target "${architecture}" --output
            "${_vernon_bundle_directory}"
        DEPENDS vernon-dsl-native
                "${_vernon_asset_source}"
                "${VERNON_REPOSITORY_ROOT}/python/vernon_dsl/bundle/parameters.py"
                "${VERNON_REPOSITORY_ROOT}/python/vernon_dsl/bundle/requirements.py"
                "${VERNON_REPOSITORY_ROOT}/python/vernon_dsl/bundle/types.py"
                "${VERNON_REPOSITORY_ROOT}/python/vernon_dsl/pipeline_assets.py"
        VERBATIM)
    add_custom_target("vernon-${name}-test-bundle" DEPENDS "${_vernon_bundle_manifest}")
    set(${output_variable}
        "${_vernon_bundle_manifest}"
        PARENT_SCOPE)
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

    if(VERNON_ENABLE_METAL_RUNTIME)
        target_sources(VernonRuntimeTests PRIVATE runtime/runtime_metal_test.cpp)
        if(TARGET vernon-dsl-native AND Python_FOUND)
            vernon_add_pipeline_test_bundle(
                metal-compute-pipeline
                metal
                "${VERNON_REPOSITORY_ROOT}/python/tests/pipeline_asset_fixture.py:scale_asset"
                metal_compute_pipeline_bundle
                metal_compute_pipeline_test_manifest)
            vernon_add_pipeline_test_bundle(
                metal-graphics-pipeline
                metal
                "${VERNON_REPOSITORY_ROOT}/python/tests/pipeline_asset_fixture.py:sampled_asset"
                metal_graphics_pipeline_bundle
                metal_graphics_pipeline_test_manifest)
            vernon_add_pipeline_test_bundle(
                metal-resolution-pipeline
                metal
                "${VERNON_REPOSITORY_ROOT}/python/tests/pipeline_asset_fixture.py:resolution_asset"
                metal_resolution_pipeline_bundle
                metal_resolution_pipeline_test_manifest)
            target_compile_definitions(
                VernonRuntimeTests
                PRIVATE VERNON_METAL_COMPUTE_PIPELINE_BUNDLE="${metal_compute_pipeline_test_manifest}"
                        VERNON_METAL_GRAPHICS_PIPELINE_BUNDLE="${metal_graphics_pipeline_test_manifest}"
                        VERNON_METAL_RESOLUTION_PIPELINE_BUNDLE="${metal_resolution_pipeline_test_manifest}")
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
                PROPERTIES ENVIRONMENT "PYTHONPATH=${VERNON_REPOSITORY_ROOT}/python")
        endif()
    endif()

    if(VERNON_ENABLE_DIRECTX12_RUNTIME)
        target_sources(VernonRuntimeTests PRIVATE runtime/runtime_directx12_test.cpp
                                                  rhi/directx12_native_interop_test.cpp)
        target_link_libraries(VernonRuntimeTests PRIVATE Vernon::RHI d3d12 dxgi)
        if(TARGET vernon-dsl-native AND Python_FOUND)
            vernon_add_pipeline_test_bundle(
                directx-pipeline
                directx
                "${VERNON_REPOSITORY_ROOT}/python/tests/pipeline_asset_fixture.py:sampled_asset"
                directx_pipeline_bundle
                directx_pipeline_test_manifest)
            vernon_add_pipeline_test_bundle(
                directx-compute-pipeline
                directx
                "${VERNON_REPOSITORY_ROOT}/python/tests/pipeline_asset_fixture.py:scale_asset"
                directx_compute_pipeline_bundle
                directx_compute_pipeline_test_manifest)
            vernon_add_pipeline_test_bundle(
                directx-resolution-pipeline
                directx
                "${VERNON_REPOSITORY_ROOT}/python/tests/pipeline_asset_fixture.py:resolution_asset"
                directx_resolution_pipeline_bundle
                directx_resolution_pipeline_test_manifest)
            target_sources(VernonRuntimeTests PRIVATE runtime/runtime_directx12_pipeline_test.cpp)
            target_compile_definitions(
                VernonRuntimeTests
                PRIVATE VERNON_DIRECTX_PIPELINE_BUNDLE="${directx_pipeline_test_manifest}"
                        VERNON_DIRECTX_COMPUTE_PIPELINE_BUNDLE="${directx_compute_pipeline_test_manifest}"
                        VERNON_DIRECTX_RESOLUTION_PIPELINE_BUNDLE="${directx_resolution_pipeline_test_manifest}")
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
        vernon_add_pipeline_test_bundle(
            vulkan-pipeline
            vulkan
            "${VERNON_REPOSITORY_ROOT}/python/tests/pipeline_asset_fixture.py:sampled_asset"
            vulkan_pipeline_bundle
            vulkan_pipeline_test_manifest)
        vernon_add_pipeline_test_bundle(
            vulkan-compute-pipeline
            vulkan
            "${VERNON_REPOSITORY_ROOT}/python/tests/pipeline_asset_fixture.py:scale_asset"
            vulkan_compute_pipeline_bundle
            vulkan_compute_pipeline_test_manifest)
        vernon_add_pipeline_test_bundle(
            vulkan-cube-map
            vulkan
            "${VERNON_REPOSITORY_ROOT}/python/tests/cube_map_shader.py:cube_map_asset"
            vulkan_cube_map_bundle
            vulkan_cube_map_test_manifest)
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

    if(TARGET vernon-dsl-native AND Python_FOUND)
        vernon_add_pipeline_test_bundle(
            opengl-runtime-acceptance
            opengl
            "${VERNON_REPOSITORY_ROOT}/python/tests/pipeline_asset_fixture.py:opengl_runtime_acceptance_asset"
            opengl_runtime_acceptance_bundle
            opengl_runtime_acceptance_manifest)
        target_compile_definitions(
            VernonRuntimeTests PRIVATE VERNON_OPENGL_RUNTIME_ACCEPTANCE_BUNDLE="${opengl_runtime_acceptance_manifest}")
        add_dependencies(VernonRuntimeTests vernon-opengl-runtime-acceptance-test-bundle)
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
    add_test(
        NAME vernon-to-gpu-tensor-view-1-to-n
        COMMAND
            ${CMAKE_COMMAND} "-DVERNON_OPT=$<TARGET_FILE:vernon-opt>"
            "-DINPUT=${VERNON_TEST_ROOT}/integration/vernon-to-gpu-tensor-view-1-to-n.mlir" -P
            "${VERNON_TEST_ROOT}/scripts/run_vernon_to_gpu_tensor_view_test.cmake")

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
