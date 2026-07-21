if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/lib/VernonRuntime.cpp")
  set(_VERNON_RUNTIME_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}")
else()
  get_filename_component(_VERNON_RUNTIME_SOURCE_DIR
    "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
endif()

function(vernon_add_runtime)
  if(NOT DEFINED VERNON_RUNTIME_LIBRARY_TYPE)
    set(VERNON_RUNTIME_LIBRARY_TYPE SHARED)
  endif()
  add_library(VernonRuntime ${VERNON_RUNTIME_LIBRARY_TYPE}
    ${_VERNON_RUNTIME_SOURCE_DIR}/lib/VernonRuntime.cpp
    ${_VERNON_RUNTIME_SOURCE_DIR}/lib/runtime/backend_cuda_driver.cpp
    ${_VERNON_RUNTIME_SOURCE_DIR}/lib/runtime/backend_opengl_driver.cpp
    ${_VERNON_RUNTIME_SOURCE_DIR}/lib/runtime/content_hash.cpp
    ${_VERNON_RUNTIME_SOURCE_DIR}/lib/runtime/platform_library.cpp)
  add_library(Vernon::Runtime ALIAS VernonRuntime)
  target_compile_definitions(VernonRuntime PRIVATE VERNON_RUNTIME_BUILD)
  if(VERNON_RUNTIME_LIBRARY_TYPE STREQUAL "STATIC")
    target_compile_definitions(VernonRuntime PUBLIC VERNON_RUNTIME_STATIC)
  endif()
  if(VERNON_ENABLE_CUDA_RUNTIME)
    target_compile_definitions(VernonRuntime PUBLIC VERNON_HAS_CUDA_RUNTIME=1)
  endif()
  if(VERNON_ENABLE_VULKAN_RUNTIME)
    target_sources(VernonRuntime PRIVATE
      ${_VERNON_RUNTIME_SOURCE_DIR}/lib/runtime/backend_vulkan_driver.cpp)
    target_compile_definitions(VernonRuntime
      PUBLIC VERNON_HAS_VULKAN_RUNTIME=1
      PRIVATE VK_NO_PROTOTYPES=1)
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
  if(SKBUILD)
    set(_vernon_runtime_bin_destination vernon_dsl)
    set(_vernon_runtime_lib_destination vernon_dsl)
    set_target_properties(VernonRuntime PROPERTIES
      OUTPUT_NAME VernonDSLHostRuntime)
  else()
    set(_vernon_runtime_bin_destination bin)
    set(_vernon_runtime_lib_destination lib)
  endif()
  install(TARGETS VernonRuntime EXPORT VernonRuntimeTargets
    RUNTIME DESTINATION ${_vernon_runtime_bin_destination}
      COMPONENT VernonWheel
    LIBRARY DESTINATION ${_vernon_runtime_lib_destination}
      COMPONENT VernonWheel
    ARCHIVE DESTINATION lib
      COMPONENT VernonDevelopment)
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
