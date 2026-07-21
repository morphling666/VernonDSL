include(FetchContent)

find_package(GTest CONFIG QUIET)
if(NOT GTest_FOUND)
  set(gtest_force_shared_crt ON CACHE BOOL
    "Use the shared MSVC runtime in GoogleTest" FORCE)
  FetchContent_Declare(googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.tar.gz
    URL_HASH
      SHA256=65fab701d9829d38cb77c14acdc431d2108bfdbf8979e40eb8ae567edf10b27c
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )
  FetchContent_MakeAvailable(googletest)
endif()
include(GoogleTest)

set(VERNON_TEST_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
set(VERNON_TEST_BINARY_ROOT "${CMAKE_CURRENT_BINARY_DIR}")

function(vernon_discover_gtests target)
  gtest_discover_tests(${target}
    DISCOVERY_MODE PRE_TEST
    WORKING_DIRECTORY "${VERNON_TEST_BINARY_ROOT}"
  )
endfunction()

if(VERNON_ENABLE_RUNTIME)
  add_library(VernonRuntimePublicCHeaderCheck OBJECT
    runtime/runtime_public_header_compile.c
  )
  target_link_libraries(VernonRuntimePublicCHeaderCheck PRIVATE Vernon::Runtime)

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
  target_include_directories(vernon-cpu-aot-fixture PRIVATE
    "${PROJECT_SOURCE_DIR}/source/include")
  set(cpu_bundle "${VERNON_TEST_BINARY_ROOT}/cpu_aot_bundle")
  add_custom_command(
    OUTPUT "${cpu_bundle}/compute.json" "${cpu_bundle}/pipeline.bundle"
    COMMAND ${CMAKE_COMMAND}
      "-DARTIFACT=$<TARGET_FILE:vernon-cpu-aot-fixture>"
      "-DOUTPUT=${cpu_bundle}"
      "-DOPERATING_SYSTEM=${runtime_test_os}"
      "-DARCHITECTURE=${runtime_test_arch}"
      -P "${VERNON_TEST_ROOT}/scripts/write_cpu_aot_bundle.cmake"
    DEPENDS vernon-cpu-aot-fixture
      "${VERNON_TEST_ROOT}/scripts/write_cpu_aot_bundle.cmake"
    VERBATIM
  )
  add_custom_target(vernon-cpu-aot-test-bundle
    DEPENDS "${cpu_bundle}/compute.json" "${cpu_bundle}/pipeline.bundle")

  add_executable(VernonRuntimeTests
    runtime/runtime_c_api_test.cpp
    runtime/runtime_common_api_test.cpp
    runtime/runtime_cpu_pipeline_test.cpp
    runtime/runtime_external_gl_test.cpp
    "${PROJECT_SOURCE_DIR}/source/lib/runtime/content_hash.cpp"
  )
  set_target_properties(VernonRuntimeTests PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/source")
  target_include_directories(VernonRuntimeTests PRIVATE
    "${PROJECT_SOURCE_DIR}/source/lib"
    "${VERNON_TEST_ROOT}/support"
  )
  target_link_libraries(VernonRuntimeTests PRIVATE
    GTest::gtest_main
    Vernon::Runtime
    nlohmann_json::nlohmann_json
  )
  target_compile_definitions(VernonRuntimeTests PRIVATE
    VERNON_CPU_BUNDLE_PATH="${cpu_bundle}"
    VERNON_RUNTIME_TEST_OS="${runtime_test_os}"
    VERNON_RUNTIME_TEST_ARCH="${runtime_test_arch}"
  )
  add_dependencies(VernonRuntimeTests vernon-cpu-aot-test-bundle)
  add_dependencies(VernonRuntimeTests VernonRuntimePublicCHeaderCheck)
  if(MSVC)
    target_compile_options(VernonRuntimeTests PRIVATE /EHsc)
  endif()

  if(TARGET vernon-compile)
    set(compiler_cpu_bundle
      "${VERNON_TEST_BINARY_ROOT}/compiler_cpu_aot_bundle")
    add_custom_command(
      OUTPUT "${compiler_cpu_bundle}/compute.json"
      COMMAND $<TARGET_FILE:vernon-compile> --target cpu
        "${VERNON_TEST_ROOT}/integration/cpu-aot-smoke.mlir"
        --compute-bundle "${compiler_cpu_bundle}"
        --host-runtime-bundle
      DEPENDS vernon-compile
        "${VERNON_TEST_ROOT}/integration/cpu-aot-smoke.mlir"
      VERBATIM
    )
    add_custom_target(vernon-compiler-cpu-aot-test-bundle
      DEPENDS "${compiler_cpu_bundle}/compute.json")
    target_sources(VernonRuntimeTests PRIVATE
      runtime/runtime_compiler_cpu_aot_test.cpp)
    target_compile_definitions(VernonRuntimeTests PRIVATE
      VERNON_COMPILER_CPU_BUNDLE_PATH="${compiler_cpu_bundle}")
    add_dependencies(VernonRuntimeTests
      vernon-compiler-cpu-aot-test-bundle)
  endif()

  if(VERNON_ENABLE_CUDA_RUNTIME)
    target_sources(VernonRuntimeTests PRIVATE
      runtime/runtime_cuda_test.cpp
      runtime/runtime_cuda_pipeline_test.cpp
    )
  endif()

  if(VERNON_ENABLE_VULKAN_RUNTIME AND TARGET VernonDSLCompiler)
    target_sources(VernonRuntimeTests PRIVATE runtime/runtime_vulkan_test.cpp)
    target_link_libraries(VernonRuntimeTests PRIVATE VernonDSLCompiler)
  endif()

  if(VERNON_ENABLE_VULKAN_RUNTIME AND TARGET vernon-dsl-native
     AND Python_FOUND)
    set(vulkan_pipeline_test_dir
      "${VERNON_TEST_BINARY_ROOT}/vulkan_pipeline_bundle")
    set(vulkan_pipeline_test_manifest
      "${vulkan_pipeline_test_dir}/vulkan_pipeline_bundle.pipeline.json")
    add_custom_command(
      OUTPUT "${vulkan_pipeline_test_manifest}"
      COMMAND ${CMAKE_COMMAND} -E env
        "PYTHONPATH=${PROJECT_SOURCE_DIR}/python"
        ${Python_EXECUTABLE} -m vernon_dsl.shader_asset_cli
        "${PROJECT_SOURCE_DIR}/python/tests/pipeline_asset_fixture.py:sampled_asset"
        --target vulkan
        --output "${vulkan_pipeline_test_dir}"
      DEPENDS vernon-dsl-native
        "${PROJECT_SOURCE_DIR}/python/tests/pipeline_asset_fixture.py"
        "${PROJECT_SOURCE_DIR}/python/vernon_dsl/pipeline_compile.py"
        "${PROJECT_SOURCE_DIR}/python/vernon_dsl/shader_assets.py"
      VERBATIM
    )
    add_custom_target(vernon-vulkan-pipeline-test-bundle
      DEPENDS "${vulkan_pipeline_test_manifest}")
    target_sources(VernonRuntimeTests PRIVATE
      runtime/runtime_vulkan_pipeline_test.cpp)
    target_compile_definitions(VernonRuntimeTests PRIVATE
      VERNON_VULKAN_PIPELINE_BUNDLE="${vulkan_pipeline_test_manifest}")
    add_dependencies(VernonRuntimeTests vernon-vulkan-pipeline-test-bundle)
  endif()

  vernon_discover_gtests(VernonRuntimeTests)
endif()

if(VERNON_ENABLE_COMPILER)
  add_executable(VernonCompilerTests
    compiler/compiler_c_api_test.cpp
    compiler/compiler_cpp_api_test.cpp
    compiler/compiler_graphics_output_test.cpp
    compiler/compiled_program_ownership_test.cpp
    compiler/vernon_compile_packaging_test.cpp
  )
  set_target_properties(VernonCompilerTests PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/source")
  target_include_directories(VernonCompilerTests PRIVATE
    "${PROJECT_SOURCE_DIR}/source/tools"
    "${VERNON_TEST_ROOT}/support"
  )
  target_link_libraries(VernonCompilerTests PRIVATE
    GTest::gtest_main
    VernonDSLCompiler
    vernon-compile-packaging
    nlohmann_json::nlohmann_json
  )
  if(MSVC)
    target_compile_options(VernonCompilerTests PRIVATE /EHsc)
  endif()
  add_dependencies(VernonCompilerTests vernon-opt vernon-compile)
  vernon_discover_gtests(VernonCompilerTests)

  if(Python_Interpreter_FOUND)
    foreach(test_module IN ITEMS test_compiler.py test_shared_definitions.py
                                 test_blinn_phong_numeric.py)
      string(REPLACE ".py" "" test_name "${test_module}")
      add_test(NAME "vernon-python-${test_name}"
        COMMAND ${Python_EXECUTABLE} -m unittest discover
          -s "${PROJECT_SOURCE_DIR}/python/tests" -p "${test_module}")
      set_tests_properties("vernon-python-${test_name}" PROPERTIES
        ENVIRONMENT "PYTHONPATH=${PROJECT_SOURCE_DIR}/python")
    endforeach()
  endif()

  if(TARGET vernon-dsl-native)
    add_test(NAME vernon-python-native-compiled-program
      COMMAND ${Python_EXECUTABLE}
        "${PROJECT_SOURCE_DIR}/python/tests/test_native_compiled_program.py"
        $<TARGET_FILE:vernon-dsl-native>)
    add_test(NAME vernon-python-compile-surface-parity
      COMMAND ${Python_EXECUTABLE}
        "${VERNON_TEST_ROOT}/compiler/compile_surface_parity_test.py"
        $<TARGET_FILE:vernon-dsl-native>
        $<TARGET_FILE:VernonDSLCompiler>
        $<TARGET_FILE:vernon-compile>)
    set_tests_properties(vernon-python-compile-surface-parity PROPERTIES
      ENVIRONMENT "PYTHONPATH=${PROJECT_SOURCE_DIR}/python")
  endif()

  add_test(NAME vernon-cpu-lowering-integration
    COMMAND ${CMAKE_COMMAND}
      "-DVERNON_OPT=$<TARGET_FILE:vernon-opt>"
      "-DTEST_ROOT=${VERNON_TEST_ROOT}/integration"
      -P "${VERNON_TEST_ROOT}/scripts/run_cpu_lowering_tests.cmake")

  if(WIN32)
    add_test(NAME vernon-validation-integration
      COMMAND powershell -ExecutionPolicy Bypass
        -File "${VERNON_TEST_ROOT}/scripts/run_validation_tests.ps1"
        -VernonOpt $<TARGET_FILE:vernon-opt>)
    add_test(NAME vernon-python-native-integration
      COMMAND powershell -ExecutionPolicy Bypass
        -File "${VERNON_TEST_ROOT}/scripts/run_frontend_integration.ps1"
        -ProjectRoot "${PROJECT_SOURCE_DIR}"
        -BinaryDir "${VERNON_TEST_BINARY_ROOT}"
        -VernonOpt $<TARGET_FILE:vernon-opt>
        -VernonCompiler $<TARGET_FILE:vernon-compile>)
  endif()
endif()
