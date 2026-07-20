if(VERNON_ENABLE_RUNTIME)
  include(VernonRuntimeTarget.cmake)
  vernon_add_runtime()
endif()

if(NOT VERNON_ENABLE_COMPILER)
  if(BUILD_TESTING AND VERNON_ENABLE_RUNTIME)
    include(VernonRuntimeTests.cmake)
    vernon_add_runtime_tests()
  endif()
  return()
endif()

add_subdirectory(include/mlir/Dialect/Vernon)
add_subdirectory(lib/Dialect/Vernon)

add_library(VernonDSLCompiler SHARED
  lib/VernonCompiler.cpp
)
llvm_map_components_to_libnames(VERNON_LLVM_JIT_LIBS
  AsmParser
  Core
  OrcJIT
  native
)
target_compile_definitions(VernonDSLCompiler PRIVATE
  VERNON_DSL_COMPILER_BUILD
)
if(VERNON_ENABLE_SPIRV_CROSS)
  target_compile_definitions(VernonDSLCompiler PRIVATE
    VERNON_HAS_SPIRV_CROSS=1
  )
  if(MSVC)
    target_compile_options(VernonDSLCompiler PRIVATE /EHsc)
  else()
    target_compile_options(VernonDSLCompiler PRIVATE -fexceptions)
  endif()
endif()
target_include_directories(VernonDSLCompiler
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(VernonDSLCompiler PRIVATE
  MLIRBufferizationTransforms
  MLIRVernonDialect
  MLIRVernonTransforms
  MLIRFuncDialect
  MLIRGPUToSPIRV
  MLIRGPUPipelines
  MLIRIR
  MLIRLinalgTransforms
  MLIRMathToSPIRV
  MLIRParser
  MLIRPass
  MLIRRegisterAllDialects
  MLIRRegisterAllExtensions
  MLIRSCFToControlFlow
  MLIRSPIRVDialect
  MLIRSPIRVSerialization
  MLIRSPIRVTransforms
  MLIRSupport
  MLIRToLLVMIRTranslationRegistration
  ${VERNON_LLVM_JIT_LIBS}
)
if(VERNON_ENABLE_SPIRV_CROSS)
  target_link_libraries(VernonDSLCompiler PRIVATE
    spirv-cross-glsl
    spirv-cross-hlsl
    spirv-cross-msl
  )
endif()

install(TARGETS VernonDSLCompiler
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)
install(FILES
  include/VernonCommon.h
  include/VernonCompiler.h
  DESTINATION include)
install(FILES
  include/vernon-c/Common.h
  include/vernon-c/Compiler.h
  DESTINATION include/vernon-c)

if(VERNON_ENABLE_RUNTIME)
  if(VERNON_ENABLE_PYTHON_BINDINGS)
    find_package(Python 3.11 EXACT COMPONENTS Interpreter Development.Module QUIET)
    if(Python_FOUND)
      execute_process(
        COMMAND ${Python_EXECUTABLE} -m nanobind --cmake_dir
        OUTPUT_VARIABLE VERNON_NANOBIND_CMAKE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE VERNON_NANOBIND_RESULT
      )
      if(VERNON_NANOBIND_RESULT EQUAL 0)
        list(APPEND CMAKE_PREFIX_PATH "${VERNON_NANOBIND_CMAKE_DIR}")
        find_package(nanobind CONFIG QUIET)
      endif()
    endif()
    if(nanobind_FOUND)
      nanobind_add_module(vernon-dsl-native python/native_module.cpp)
      target_link_libraries(vernon-dsl-native PRIVATE
        VernonDSLCompiler Vernon::Runtime)
      if(MSVC)
        target_compile_options(vernon-dsl-native PRIVATE /EHsc)
      endif()
      set_target_properties(vernon-dsl-native PROPERTIES OUTPUT_NAME "_native")
      install(TARGETS vernon-dsl-native
        RUNTIME DESTINATION vernon_dsl
        LIBRARY DESTINATION vernon_dsl
      )
    else()
      message(STATUS "nanobind not found; skipping vernon_dsl._native")
    endif()
  endif()
endif()

add_executable(vernon-compile
  tools/vernon_compile.cpp
)
target_link_libraries(vernon-compile PRIVATE VernonDSLCompiler LLVMSupport)
if(MSVC)
  target_compile_options(vernon-compile PRIVATE /EHsc)
endif()
install(TARGETS vernon-compile RUNTIME DESTINATION bin)

if(BUILD_TESTING)
  if(VERNON_ENABLE_RUNTIME)
    include(VernonRuntimeTests.cmake)
    vernon_add_runtime_tests()
  endif()
  add_executable(vernon-compiler-c-api-test
    tests/compiler_c_api_test.c
  )
  target_link_libraries(vernon-compiler-c-api-test PRIVATE VernonDSLCompiler)
  add_test(NAME vernon-compiler-c-api-test
    COMMAND vernon-compiler-c-api-test
  )

  add_executable(vernon-compiler-cpp-api-test
    tests/compiler_cpp_api_test.cpp
  )
  target_link_libraries(vernon-compiler-cpp-api-test PRIVATE VernonDSLCompiler)
  add_test(NAME vernon-compiler-cpp-api-test
    COMMAND vernon-compiler-cpp-api-test
  )
  if(VERNON_ENABLE_RUNTIME AND VERNON_ENABLE_VULKAN_RUNTIME)
    find_package(Python 3.11 EXACT COMPONENTS Interpreter QUIET)
    if(Python_FOUND)
      set(VERNON_VULKAN_PIPELINE_TEST_DIR
        ${CMAKE_CURRENT_BINARY_DIR}/vulkan_pipeline_bundle)
      add_custom_command(
        OUTPUT ${VERNON_VULKAN_PIPELINE_TEST_DIR}/pipeline.bundle
        COMMAND ${CMAKE_COMMAND} -E env
          "PYTHONPATH=${CMAKE_SOURCE_DIR}/python"
          ${Python_EXECUTABLE} -m vernon_dsl.shader_asset_cli
          ${CMAKE_SOURCE_DIR}/examples/variant_mesh.shader-pipeline.json
          --asset-root ${CMAKE_SOURCE_DIR}/examples
          --compiler $<TARGET_FILE:vernon-compile>
          --target vulkan
          --output ${VERNON_VULKAN_PIPELINE_TEST_DIR}
        DEPENDS vernon-compile
          ${CMAKE_SOURCE_DIR}/examples/variant_mesh.shader-pipeline.json
          ${CMAKE_SOURCE_DIR}/examples/variant_mesh.shader-module.json
          ${CMAKE_SOURCE_DIR}/examples/variant_mesh.py
          ${CMAKE_SOURCE_DIR}/python/vernon_dsl/shader_assets.py
        VERBATIM)
      add_custom_target(vernon-vulkan-pipeline-test-bundle
        DEPENDS ${VERNON_VULKAN_PIPELINE_TEST_DIR}/pipeline.bundle)
      add_executable(vernon-runtime-vulkan-pipeline-test
        tests/runtime_vulkan_pipeline_test.cpp)
      target_link_libraries(vernon-runtime-vulkan-pipeline-test PRIVATE
        Vernon::Runtime)
      target_compile_definitions(vernon-runtime-vulkan-pipeline-test
        PRIVATE
          VERNON_VULKAN_PIPELINE_BUNDLE="${VERNON_VULKAN_PIPELINE_TEST_DIR}/pipeline.bundle")
      add_dependencies(vernon-runtime-vulkan-pipeline-test
        vernon-vulkan-pipeline-test-bundle)
      add_test(NAME vernon-runtime-vulkan-pipeline-test
        COMMAND vernon-runtime-vulkan-pipeline-test)
    endif()
  endif()
  if(VERNON_ENABLE_RUNTIME)
    if(VERNON_ENABLE_VULKAN_RUNTIME)
      add_executable(vernon-runtime-vulkan-test tests/runtime_vulkan_test.cpp)
      target_link_libraries(vernon-runtime-vulkan-test PRIVATE
        VernonDSLCompiler Vernon::Runtime)
      add_test(NAME vernon-runtime-vulkan-test
        COMMAND vernon-runtime-vulkan-test)
    endif()
  endif()
endif()

add_subdirectory(tools/vernon_opt)

if(BUILD_TESTING AND WIN32)
  add_test(NAME vernon-python-native-integration
    COMMAND powershell -ExecutionPolicy Bypass
      -File ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_frontend_integration.ps1
      -ProjectRoot ${CMAKE_SOURCE_DIR}
      -BinaryDir ${CMAKE_CURRENT_BINARY_DIR}
      -VernonOpt $<TARGET_FILE:vernon-opt>
      -VernonCompiler $<TARGET_FILE:vernon-compile>
  )
endif()