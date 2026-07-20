function(vernon_add_runtime_tests)
  add_executable(vernon-runtime-public-header-test
    ${_VERNON_RUNTIME_SOURCE_DIR}/tests/runtime_public_header_test.c)
  target_link_libraries(vernon-runtime-public-header-test PRIVATE Vernon::Runtime)
  add_test(NAME vernon-runtime-public-header-test
    COMMAND vernon-runtime-public-header-test)

  add_executable(vernon-runtime-external-gl-test
    ${_VERNON_RUNTIME_SOURCE_DIR}/tests/runtime_external_gl_test.cpp)
  target_link_libraries(vernon-runtime-external-gl-test PRIVATE Vernon::Runtime)
  add_test(NAME vernon-runtime-external-gl-test
    COMMAND vernon-runtime-external-gl-test)

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
  add_library(vernon-cpu-aot-fixture SHARED
    ${_VERNON_RUNTIME_SOURCE_DIR}/tests/cpu_aot_fixture.c)
  target_include_directories(vernon-cpu-aot-fixture PRIVATE
    ${_VERNON_RUNTIME_SOURCE_DIR}/include)
  set(cpu_bundle ${CMAKE_CURRENT_BINARY_DIR}/cpu_aot_bundle)
  add_custom_command(
    OUTPUT ${cpu_bundle}/compute.json
    COMMAND ${CMAKE_COMMAND}
      "-DARTIFACT=$<TARGET_FILE:vernon-cpu-aot-fixture>"
      "-DOUTPUT=${cpu_bundle}"
      "-DOPERATING_SYSTEM=${runtime_test_os}"
      "-DARCHITECTURE=${runtime_test_arch}"
      -P ${_VERNON_RUNTIME_SOURCE_DIR}/tests/write_cpu_aot_bundle.cmake
    DEPENDS vernon-cpu-aot-fixture
      ${_VERNON_RUNTIME_SOURCE_DIR}/tests/write_cpu_aot_bundle.cmake
    VERBATIM)
  add_custom_target(vernon-cpu-aot-test-bundle
    DEPENDS ${cpu_bundle}/compute.json)
  add_executable(vernon-runtime-cpu-aot-test
    ${_VERNON_RUNTIME_SOURCE_DIR}/tests/runtime_c_api_test.c)
  target_link_libraries(vernon-runtime-cpu-aot-test PRIVATE Vernon::Runtime)
  target_compile_definitions(vernon-runtime-cpu-aot-test PRIVATE
    VERNON_CPU_BUNDLE_PATH="${cpu_bundle}")
  add_dependencies(vernon-runtime-cpu-aot-test vernon-cpu-aot-test-bundle)
  add_test(NAME vernon-runtime-cpu-aot-test COMMAND vernon-runtime-cpu-aot-test)

  if(VERNON_ENABLE_CUDA_RUNTIME)
    add_executable(vernon-runtime-cuda-test
      ${_VERNON_RUNTIME_SOURCE_DIR}/tests/runtime_cuda_test.cpp)
    target_link_libraries(vernon-runtime-cuda-test PRIVATE Vernon::Runtime)
    add_test(NAME vernon-runtime-cuda-test COMMAND vernon-runtime-cuda-test)
  endif()
endfunction()
