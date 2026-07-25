if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/VernonRuntime.cpp")
    set(_VERNON_RUNTIME_IMPL_DIR "${CMAKE_CURRENT_LIST_DIR}")
    get_filename_component(_VERNON_RUNTIME_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../include" ABSOLUTE)
elseif(EXISTS "${CMAKE_CURRENT_LIST_DIR}/../lib/runtime/VernonRuntime.cpp")
    get_filename_component(_VERNON_RUNTIME_PACKAGE_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
    set(_VERNON_RUNTIME_IMPL_DIR "${_VERNON_RUNTIME_PACKAGE_ROOT}/lib/runtime")
    set(_VERNON_RUNTIME_INCLUDE_DIR "${_VERNON_RUNTIME_PACKAGE_ROOT}/include")
else()
    message(FATAL_ERROR "Cannot locate VernonRuntime implementation sources")
endif()

function(vernon_add_runtime)
    if(NOT DEFINED VERNON_RUNTIME_LIBRARY_TYPE)
        set(VERNON_RUNTIME_LIBRARY_TYPE SHARED)
    endif()
    add_library(
        VernonRuntimeInternals STATIC
        ${_VERNON_RUNTIME_IMPL_DIR}/compute_launch_planner.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/content_hash.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/graphics_invocation_planner.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/pipeline_bundle.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/pipeline_manifest.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/pipeline_metadata.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/tensor_bridge.cpp)
    target_include_directories(VernonRuntimeInternals PRIVATE ${_VERNON_RUNTIME_INCLUDE_DIR})
    target_link_libraries(VernonRuntimeInternals PRIVATE nlohmann_json::nlohmann_json)
    set_target_properties(VernonRuntimeInternals PROPERTIES POSITION_INDEPENDENT_CODE ON)
    if(MSVC)
        target_compile_options(VernonRuntimeInternals PRIVATE /EHsc)
    endif()
    add_library(
        VernonRuntime
        ${VERNON_RUNTIME_LIBRARY_TYPE}
        ${_VERNON_RUNTIME_IMPL_DIR}/VernonRuntime.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/backend_cpu.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/backend_opengl.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/backend_opengl_driver.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/graphics_opengl_encoder.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/platform_library.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/runtime_dispatch.cpp)
    add_library(Vernon::Runtime ALIAS VernonRuntime)
    set_target_properties(VernonRuntime PROPERTIES EXPORT_NAME Runtime)
    target_compile_definitions(VernonRuntime PRIVATE VERNON_RUNTIME_BUILD)
    if(VERNON_RUNTIME_LIBRARY_TYPE STREQUAL "STATIC")
        target_compile_definitions(VernonRuntime PUBLIC VERNON_RUNTIME_STATIC)
    endif()
    if(VERNON_ENABLE_CUDA_RUNTIME)
        target_sources(VernonRuntime PRIVATE ${_VERNON_RUNTIME_IMPL_DIR}/backend_cuda.cpp
                                             ${_VERNON_RUNTIME_IMPL_DIR}/backend_cuda_driver.cpp)
        target_compile_definitions(VernonRuntime PUBLIC VERNON_HAS_CUDA_RUNTIME=1)
    endif()
    if(VERNON_ENABLE_VULKAN_RUNTIME)
        target_sources(
            VernonRuntime
            PRIVATE ${_VERNON_RUNTIME_IMPL_DIR}/backend_vulkan.cpp
                    ${_VERNON_RUNTIME_IMPL_DIR}/backend_vulkan_driver.cpp
                    ${_VERNON_RUNTIME_IMPL_DIR}/graphics_vulkan_encoder.cpp)
        target_compile_definitions(
            VernonRuntime
            PUBLIC VERNON_HAS_VULKAN_RUNTIME=1
            PRIVATE VK_NO_PROTOTYPES=1)
        target_link_libraries(VernonRuntime PRIVATE Vulkan::Headers)
    endif()
    target_include_directories(
        VernonRuntime
        PUBLIC $<BUILD_INTERFACE:${_VERNON_RUNTIME_INCLUDE_DIR}> $<INSTALL_INTERFACE:include>
        PRIVATE ${_VERNON_RUNTIME_IMPL_DIR}/..)
    target_link_libraries(VernonRuntime PRIVATE VernonRuntimeInternals nlohmann_json::nlohmann_json ${CMAKE_DL_LIBS})
    if(BUILD_TESTING)
        target_compile_definitions(VernonRuntime PRIVATE VERNON_RUNTIME_TESTING=1)
    endif()
    if(MSVC)
        target_compile_options(VernonRuntime PRIVATE /EHsc)
    endif()
    if(SKBUILD)
        set(_vernon_runtime_bin_destination vernon_dsl)
        set(_vernon_runtime_lib_destination vernon_dsl)
        set_target_properties(VernonRuntime PROPERTIES OUTPUT_NAME VernonDSLHostRuntime)
    else()
        set(_vernon_runtime_bin_destination bin)
        set(_vernon_runtime_lib_destination lib)
    endif()
    install(
        TARGETS VernonRuntime
        EXPORT VernonRuntimeTargets
        RUNTIME DESTINATION ${_vernon_runtime_bin_destination} COMPONENT VernonWheel
        LIBRARY DESTINATION ${_vernon_runtime_lib_destination} COMPONENT VernonWheel
        ARCHIVE DESTINATION lib COMPONENT VernonDevelopment)
    install(FILES ${_VERNON_RUNTIME_INCLUDE_DIR}/VernonCommon.h ${_VERNON_RUNTIME_INCLUDE_DIR}/VernonRuntime.h
            DESTINATION include)
    install(FILES ${_VERNON_RUNTIME_INCLUDE_DIR}/vernon-c/Common.h ${_VERNON_RUNTIME_INCLUDE_DIR}/vernon-c/Runtime.h
            DESTINATION include/vernon-c)
    install(
        EXPORT VernonRuntimeTargets
        FILE VernonRuntimeTargets.cmake
        NAMESPACE Vernon::
        DESTINATION lib/cmake/VernonRuntime)
endfunction()
