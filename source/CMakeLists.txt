add_subdirectory(include/mlir/Dialect/Vernon)
add_subdirectory(lib/Dialect/Vernon)

add_library(VernonDSLCompiler SHARED
  lib/VernonCompiler.cpp
)
llvm_map_components_to_libnames(VERNON_LLVM_JIT_LIBS
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
  MLIRVernonDialect
  MLIRVernonTransforms
  MLIRFuncDialect
  MLIRGPUToSPIRV
  MLIRGPUPipelines
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRRegisterAllDialects
  MLIRRegisterAllExtensions
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
install(FILES include/VernonCompiler.h DESTINATION include)
install(FILES include/vernon-c/Compiler.h DESTINATION include/vernon-c)

add_executable(vernon-compile
  tools/vernon_compile.cpp
)
target_link_libraries(vernon-compile PRIVATE VernonDSLCompiler)
install(TARGETS vernon-compile RUNTIME DESTINATION bin)

if(BUILD_TESTING)
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
endif()

add_subdirectory(tools/vernon_opt)

if(BUILD_TESTING AND WIN32)
  add_test(NAME vernon-python-native-integration
    COMMAND powershell -ExecutionPolicy Bypass
      -File ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_frontend_integration.ps1
      -ProjectRoot ${CMAKE_SOURCE_DIR}
      -BinaryDir ${CMAKE_CURRENT_BINARY_DIR}
      -VernonOpt $<TARGET_FILE:vernon-opt>
  )
endif()