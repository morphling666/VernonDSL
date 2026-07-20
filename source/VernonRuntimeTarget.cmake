set(_VERNON_RUNTIME_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}")

function(vernon_add_runtime)
  add_library(VernonRuntime SHARED
    ${_VERNON_RUNTIME_SOURCE_DIR}/lib/VernonRuntime.cpp
    ${_VERNON_RUNTIME_SOURCE_DIR}/lib/runtime/backend_cuda_driver.cpp
    ${_VERNON_RUNTIME_SOURCE_DIR}/lib/runtime/backend_opengl_driver.cpp
    ${_VERNON_RUNTIME_SOURCE_DIR}/lib/runtime/content_hash.cpp
    ${_VERNON_RUNTIME_SOURCE_DIR}/lib/runtime/platform_library.cpp)
  add_library(Vernon::Runtime ALIAS VernonRuntime)
  target_compile_definitions(VernonRuntime PRIVATE VERNON_RUNTIME_BUILD)
  if(VERNON_ENABLE_CUDA_RUNTIME)
    target_compile_definitions(VernonRuntime PRIVATE VERNON_HAS_CUDA_RUNTIME=1)
  endif()
  if(VERNON_ENABLE_VULKAN_RUNTIME)
    target_sources(VernonRuntime PRIVATE
      ${_VERNON_RUNTIME_SOURCE_DIR}/lib/runtime/backend_vulkan_driver.cpp)
    target_compile_definitions(VernonRuntime PRIVATE
      VERNON_HAS_VULKAN_RUNTIME=1 VK_NO_PROTOTYPES=1)
    target_link_libraries(VernonRuntime PRIVATE Vulkan::Headers)
  endif()
  target_include_directories(VernonRuntime PUBLIC
    $<BUILD_INTERFACE:${_VERNON_RUNTIME_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
  target_link_libraries(VernonRuntime PRIVATE
    nlohmann_json::nlohmann_json ${CMAKE_DL_LIBS})
  if(MSVC)
    target_compile_options(VernonRuntime PRIVATE /EHsc)
  endif()
  install(TARGETS VernonRuntime EXPORT VernonRuntimeTargets
    RUNTIME DESTINATION bin LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
  install(FILES
    ${_VERNON_RUNTIME_SOURCE_DIR}/include/VernonCommon.h
    ${_VERNON_RUNTIME_SOURCE_DIR}/include/VernonRuntime.h
    DESTINATION include)
  install(FILES
    ${_VERNON_RUNTIME_SOURCE_DIR}/include/vernon-c/Common.h
    ${_VERNON_RUNTIME_SOURCE_DIR}/include/vernon-c/Runtime.h
    DESTINATION include/vernon-c)
  install(EXPORT VernonRuntimeTargets FILE VernonRuntimeTargets.cmake
    NAMESPACE Vernon:: DESTINATION lib/cmake/VernonRuntime)
endfunction()
