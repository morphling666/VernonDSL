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

    add_library(VernonPlatform STATIC ${_VERNON_RUNTIME_IMPL_DIR}/../platform/platform_library.cpp)
    set_target_properties(VernonPlatform PROPERTIES EXPORT_NAME Platform POSITION_INDEPENDENT_CODE ON)

    add_library(
        VernonRHI
        ${VERNON_RUNTIME_LIBRARY_TYPE}
        ${_VERNON_RUNTIME_IMPL_DIR}/../rhi/opengl_backend.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/../rhi/opengl_driver.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/../rhi/rhi_command.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/../rhi/rhi_device.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/../rhi/rhi.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/../rhi/rhi_opengl.cpp)
    add_library(Vernon::RHI ALIAS VernonRHI)
    set_target_properties(VernonRHI PROPERTIES EXPORT_NAME RHI POSITION_INDEPENDENT_CODE ON)
    if(VERNON_RUNTIME_LIBRARY_TYPE STREQUAL "STATIC")
        target_compile_definitions(VernonRHI PUBLIC VERNON_RHI_STATIC)
    else()
        target_compile_definitions(VernonRHI PRIVATE VERNON_RHI_BUILD)
    endif()
    if(WIN32)
        target_compile_definitions(VernonRHI PRIVATE NOMINMAX)
    endif()
    target_include_directories(VernonRHI PUBLIC $<BUILD_INTERFACE:${_VERNON_RUNTIME_INCLUDE_DIR}>
                                                $<INSTALL_INTERFACE:include>)
    target_link_libraries(VernonRHI PRIVATE VernonPlatform ${CMAKE_DL_LIBS})
    if(VERNON_ENABLE_CUDA_RUNTIME)
        target_sources(
            VernonRHI
            PRIVATE ${_VERNON_RUNTIME_IMPL_DIR}/../rhi/cuda_backend.cpp
                    ${_VERNON_RUNTIME_IMPL_DIR}/../rhi/cuda_driver.cpp ${_VERNON_RUNTIME_IMPL_DIR}/../rhi/rhi_cuda.cpp)
        target_compile_definitions(VernonRHI PRIVATE VERNON_HAS_CUDA_RHI=1)
    endif()
    if(VERNON_ENABLE_DIRECTX12_RUNTIME)
        target_sources(VernonRHI PRIVATE ${_VERNON_RUNTIME_IMPL_DIR}/../rhi/directx12_backend.cpp
                                         ${_VERNON_RUNTIME_IMPL_DIR}/../rhi/rhi_directx12.cpp)
        target_compile_definitions(VernonRHI PRIVATE VERNON_HAS_DIRECTX12_RHI=1)
        target_link_libraries(VernonRHI PRIVATE d3d12 dxgi dxguid)
    endif()
    if(VERNON_ENABLE_VULKAN_RUNTIME)
        target_sources(
            VernonRHI
            PRIVATE ${_VERNON_RUNTIME_IMPL_DIR}/../rhi/vulkan_backend.cpp
                    ${_VERNON_RUNTIME_IMPL_DIR}/../rhi/vulkan_driver.cpp
                    ${_VERNON_RUNTIME_IMPL_DIR}/../rhi/rhi_vulkan.cpp)
        target_compile_definitions(VernonRHI PRIVATE VERNON_HAS_VULKAN_RHI=1 VK_NO_PROTOTYPES=1)
        target_link_libraries(VernonRHI PRIVATE $<BUILD_INTERFACE:Vulkan::Headers>)
    endif()

    add_library(VernonExecutionGraph STATIC ${_VERNON_RUNTIME_IMPL_DIR}/../execution_graph/execution_graph.cpp)
    add_library(Vernon::ExecutionGraph ALIAS VernonExecutionGraph)
    set_target_properties(VernonExecutionGraph PROPERTIES EXPORT_NAME ExecutionGraph POSITION_INDEPENDENT_CODE ON)
    target_include_directories(VernonExecutionGraph PUBLIC $<BUILD_INTERFACE:${_VERNON_RUNTIME_INCLUDE_DIR}>
                                                           $<INSTALL_INTERFACE:include>)
    target_link_libraries(VernonExecutionGraph PUBLIC Vernon::RHI)
    if(MSVC)
        target_compile_options(VernonExecutionGraph PRIVATE /EHsc)
    endif()

    add_library(
        VernonRuntimeCore STATIC
        ${_VERNON_RUNTIME_IMPL_DIR}/compute_launch_planner.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/content_hash.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/graphics_invocation_planner.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/pipeline_bundle.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/pipeline_manifest.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/pipeline_metadata.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/runtime_core.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/tensor_bridge.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/vertex_attribute_capabilities.cpp)
    add_library(Vernon::RuntimeCore ALIAS VernonRuntimeCore)
    add_library(VernonRuntimeInternals ALIAS VernonRuntimeCore)
    set_target_properties(VernonRuntimeCore PROPERTIES EXPORT_NAME RuntimeCore POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(VernonRuntimeCore PUBLIC VERNON_RUNTIME_CORE_STATIC)
    target_include_directories(VernonRuntimeCore PUBLIC $<BUILD_INTERFACE:${_VERNON_RUNTIME_INCLUDE_DIR}>
                                                        $<INSTALL_INTERFACE:include>)
    target_link_libraries(VernonRuntimeCore PRIVATE $<BUILD_INTERFACE:nlohmann_json::nlohmann_json>)
    if(MSVC)
        target_compile_options(VernonRuntimeCore PRIVATE /EHsc)
    endif()

    add_library(
        VernonRuntimeRHIAdapter STATIC
        ${_VERNON_RUNTIME_IMPL_DIR}/rhi_adapter/adapter_common.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/rhi_adapter/adapter_cuda.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/rhi_adapter/adapter_directx12.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/rhi_adapter/adapter_opengl.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/rhi_adapter/adapter_vulkan.cpp)
    add_library(Vernon::RuntimeRHIAdapter ALIAS VernonRuntimeRHIAdapter)
    set_target_properties(VernonRuntimeRHIAdapter PROPERTIES EXPORT_NAME RuntimeRHIAdapter POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(VernonRuntimeRHIAdapter PUBLIC VERNON_RUNTIME_RHI_ADAPTER_STATIC)
    if(WIN32)
        target_compile_definitions(VernonRuntimeRHIAdapter PRIVATE NOMINMAX)
    endif()
    if(VERNON_ENABLE_CUDA_RUNTIME)
        target_compile_definitions(VernonRuntimeRHIAdapter PRIVATE VERNON_HAS_CUDA_RHI=1)
    endif()
    if(VERNON_ENABLE_DIRECTX12_RUNTIME)
        target_compile_definitions(VernonRuntimeRHIAdapter PRIVATE VERNON_HAS_DIRECTX12_RHI=1)
        target_link_libraries(VernonRuntimeRHIAdapter PRIVATE d3d12)
    endif()
    if(VERNON_ENABLE_VULKAN_RUNTIME)
        target_compile_definitions(VernonRuntimeRHIAdapter PRIVATE VERNON_HAS_VULKAN_RHI=1 VK_NO_PROTOTYPES=1)
        target_link_libraries(VernonRuntimeRHIAdapter PRIVATE $<BUILD_INTERFACE:Vulkan::Headers>)
    endif()
    target_link_libraries(VernonRuntimeRHIAdapter PUBLIC Vernon::RuntimeCore Vernon::RHI)

    add_library(
        VernonRuntime
        ${VERNON_RUNTIME_LIBRARY_TYPE}
        ${_VERNON_RUNTIME_IMPL_DIR}/VernonRuntime.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/backend_cpu.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/backend_opengl.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/runtime_backend_dispatch.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/runtime_pipeline_cpu.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/runtime_pipeline_cuda.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/runtime_pipeline_direct.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/runtime_pipeline_directx12.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/runtime_pipeline_dispatch.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/runtime_pipeline_opengl.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/runtime_pipeline_vulkan.cpp)
    add_library(Vernon::Runtime ALIAS VernonRuntime)
    set_target_properties(VernonRuntime PROPERTIES EXPORT_NAME Runtime)
    target_compile_definitions(VernonRuntime PRIVATE VERNON_RUNTIME_BUILD)
    if(WIN32)
        target_compile_definitions(VernonRuntime PRIVATE NOMINMAX)
    endif()
    if(VERNON_RUNTIME_LIBRARY_TYPE STREQUAL "STATIC")
        target_compile_definitions(VernonRuntime PUBLIC VERNON_RUNTIME_STATIC)
    endif()
    if(VERNON_ENABLE_CUDA_RUNTIME)
        target_compile_definitions(VernonRuntime PUBLIC VERNON_HAS_CUDA_RUNTIME=1)
    endif()
    if(VERNON_ENABLE_VULKAN_RUNTIME)
        target_compile_definitions(
            VernonRuntime
            PUBLIC VERNON_HAS_VULKAN_RUNTIME=1
            PRIVATE VK_NO_PROTOTYPES=1)
        target_link_libraries(VernonRuntime PRIVATE $<BUILD_INTERFACE:Vulkan::Headers>)
    endif()
    if(VERNON_ENABLE_DIRECTX12_RUNTIME)
        if(NOT WIN32)
            message(FATAL_ERROR "VERNON_ENABLE_DIRECTX12_RUNTIME is supported only on Windows")
        endif()
        target_compile_definitions(VernonRuntime PUBLIC VERNON_HAS_DIRECTX12_RUNTIME=1)
        target_link_libraries(VernonRuntime PRIVATE d3d12 dxgi dxguid)
    endif()
    target_include_directories(
        VernonRuntime
        PUBLIC $<BUILD_INTERFACE:${_VERNON_RUNTIME_INCLUDE_DIR}> $<INSTALL_INTERFACE:include>
        PRIVATE ${_VERNON_RUNTIME_IMPL_DIR}/..)
    target_link_libraries(
        VernonRuntime
        PRIVATE VernonRuntimeCore
                VernonRuntimeRHIAdapter
                VernonPlatform
                VernonRHI
                $<BUILD_INTERFACE:nlohmann_json::nlohmann_json>
                ${CMAKE_DL_LIBS})
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
    set(_vernon_runtime_install_targets VernonRuntime VernonRHI VernonExecutionGraph)
    if(VERNON_RUNTIME_LIBRARY_TYPE STREQUAL "STATIC")
        list(
            APPEND
            _vernon_runtime_install_targets
            VernonRuntimeCore
            VernonRuntimeRHIAdapter
            VernonPlatform)
    endif()
    install(
        TARGETS ${_vernon_runtime_install_targets}
        EXPORT VernonRuntimeTargets
        RUNTIME DESTINATION ${_vernon_runtime_bin_destination} COMPONENT VernonWheel
        LIBRARY DESTINATION ${_vernon_runtime_lib_destination} COMPONENT VernonWheel
        ARCHIVE DESTINATION lib COMPONENT VernonDevelopment)
    install(
        FILES ${_VERNON_RUNTIME_INCLUDE_DIR}/VernonCommon.h
              ${_VERNON_RUNTIME_INCLUDE_DIR}/VernonOpenGLContext.h
              ${_VERNON_RUNTIME_INCLUDE_DIR}/VernonRHI.h
              ${_VERNON_RUNTIME_INCLUDE_DIR}/VernonRHI.hpp
              ${_VERNON_RUNTIME_INCLUDE_DIR}/VernonExecutionGraph.h
              ${_VERNON_RUNTIME_INCLUDE_DIR}/VernonRuntime.h
              ${_VERNON_RUNTIME_INCLUDE_DIR}/VernonRuntime.hpp
              ${_VERNON_RUNTIME_INCLUDE_DIR}/VernonRuntimeCore.h
              ${_VERNON_RUNTIME_INCLUDE_DIR}/VernonRuntimeProvider.h
              ${_VERNON_RUNTIME_INCLUDE_DIR}/VernonRuntimeRHIAdapter.h
        DESTINATION include
        COMPONENT VernonDevelopment)
    install(
        FILES ${_VERNON_RUNTIME_INCLUDE_DIR}/vernon-c/Common.h ${_VERNON_RUNTIME_INCLUDE_DIR}/vernon-c/Runtime.h
        DESTINATION include/vernon-c
        COMPONENT VernonDevelopment)
    install(
        EXPORT VernonRuntimeTargets
        FILE VernonRuntimeTargets.cmake
        NAMESPACE Vernon::
        DESTINATION lib/cmake/VernonRuntime
        COMPONENT VernonDevelopment)
endfunction()
