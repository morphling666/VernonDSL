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

    if(VERNON_RUNTIME_PROFILE STREQUAL "web")
        set(_vernon_platform_library_source ${_VERNON_RUNTIME_IMPL_DIR}/../platform/platform_library_web.cpp)
    else()
        set(_vernon_platform_library_source ${_VERNON_RUNTIME_IMPL_DIR}/../platform/platform_library.cpp)
    endif()
    add_library(VernonPlatform STATIC ${_vernon_platform_library_source})
    set_target_properties(VernonPlatform PROPERTIES EXPORT_NAME Platform POSITION_INDEPENDENT_CODE ON)
    if(VERNON_RUNTIME_PROFILE STREQUAL "web")
        target_compile_definitions(VernonPlatform PRIVATE VERNON_RUNTIME_PROFILE_WEB=1)
    endif()

    add_library(
        VernonRHI
        ${VERNON_RUNTIME_LIBRARY_TYPE}
        ${_VERNON_RUNTIME_IMPL_DIR}/../rhi/opengl_backend.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/../rhi/opengl_driver.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/../rhi/image_descriptor_validation.cpp
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
    target_include_directories(
        VernonRHI
        PUBLIC $<BUILD_INTERFACE:${_VERNON_RUNTIME_INCLUDE_DIR}> $<INSTALL_INTERFACE:include>
        PRIVATE $<BUILD_INTERFACE:${_VERNON_RUNTIME_IMPL_DIR}/..>)
    target_link_libraries(VernonRHI PRIVATE VernonPlatform)
    if(NOT
       VERNON_RUNTIME_PROFILE
       STREQUAL
       "web")
        target_link_libraries(VernonRHI PRIVATE ${CMAKE_DL_LIBS})
    endif()
    if(VERNON_ENABLE_CUDA_RUNTIME)
        target_sources(
            VernonRHI
            PRIVATE ${_VERNON_RUNTIME_IMPL_DIR}/../rhi/cuda_backend.cpp
                    ${_VERNON_RUNTIME_IMPL_DIR}/../rhi/cuda_driver.cpp ${_VERNON_RUNTIME_IMPL_DIR}/../rhi/rhi_cuda.cpp)
        target_compile_definitions(VernonRHI PRIVATE VERNON_HAS_CUDA_RHI=1)
    endif()
    if(VERNON_ENABLE_DIRECTX12_RUNTIME)
        find_program(_vernon_hlsl_compiler NAMES dxc fxc REQUIRED)
        get_filename_component(_vernon_hlsl_compiler_name "${_vernon_hlsl_compiler}" NAME_WE)
        set(_vernon_directx12_generated_dir "${CMAKE_CURRENT_BINARY_DIR}/directx12_generated")
        file(MAKE_DIRECTORY "${_vernon_directx12_generated_dir}")
        set(_vernon_mipmap_shader "${_VERNON_RUNTIME_IMPL_DIR}/../rhi/directx12_mipmap.hlsl")
        foreach(_vernon_mipmap_dimension IN ITEMS 2d 3d)
            set(_vernon_mipmap_blob
                "${_vernon_directx12_generated_dir}/directx12_mipmap_${_vernon_mipmap_dimension}.bin")
            set(_vernon_mipmap_header
                "${_vernon_directx12_generated_dir}/directx12_mipmap_${_vernon_mipmap_dimension}.h")
            if(_vernon_hlsl_compiler_name STREQUAL "dxc")
                set(_vernon_hlsl_args
                    -T
                    cs_6_0
                    -E
                    main
                    -Fo
                    "${_vernon_mipmap_blob}")
                if(_vernon_mipmap_dimension STREQUAL "3d")
                    list(
                        APPEND
                        _vernon_hlsl_args
                        -D
                        VERNON_MIPMAP_3D=1)
                else()
                    list(
                        APPEND
                        _vernon_hlsl_args
                        -D
                        VERNON_MIPMAP_3D=0)
                endif()
            else()
                set(_vernon_hlsl_args
                    /nologo
                    /T
                    cs_5_1
                    /E
                    main
                    /Fo
                    "${_vernon_mipmap_blob}")
                if(_vernon_mipmap_dimension STREQUAL "3d")
                    list(
                        APPEND
                        _vernon_hlsl_args
                        /D
                        VERNON_MIPMAP_3D=1)
                else()
                    list(
                        APPEND
                        _vernon_hlsl_args
                        /D
                        VERNON_MIPMAP_3D=0)
                endif()
            endif()
            add_custom_command(
                OUTPUT "${_vernon_mipmap_header}"
                COMMAND "${_vernon_hlsl_compiler}" ${_vernon_hlsl_args} "${_vernon_mipmap_shader}"
                COMMAND
                    "${CMAKE_COMMAND}" -DINPUT=${_vernon_mipmap_blob} -DOUTPUT=${_vernon_mipmap_header}
                    -DSYMBOL=vernon_directx12_mipmap_${_vernon_mipmap_dimension} -P
                    "${_vernon_repository_root}/cmake/EmbedBinary.cmake"
                DEPENDS "${_vernon_mipmap_shader}" "${_vernon_repository_root}/cmake/EmbedBinary.cmake"
                VERBATIM)
            list(APPEND _vernon_mipmap_headers "${_vernon_mipmap_header}")
        endforeach()
        target_sources(VernonRHI PRIVATE ${_VERNON_RUNTIME_IMPL_DIR}/../rhi/directx12_backend.cpp
                                         ${_VERNON_RUNTIME_IMPL_DIR}/../rhi/rhi_directx12.cpp ${_vernon_mipmap_headers})
        target_include_directories(VernonRHI PRIVATE "${_vernon_directx12_generated_dir}")
        target_compile_definitions(VernonRHI PRIVATE VERNON_HAS_DIRECTX12_RHI=1)
        target_link_libraries(VernonRHI PRIVATE d3d12 dxgi dxguid)
    endif()
    if(VERNON_ENABLE_VULKAN_RUNTIME)
        target_sources(
            VernonRHI
            PRIVATE ${_VERNON_RUNTIME_IMPL_DIR}/../rhi/vulkan_backend.cpp
                    ${_VERNON_RUNTIME_IMPL_DIR}/../rhi/vulkan_driver.cpp
                    ${_VERNON_RUNTIME_IMPL_DIR}/../rhi/rhi_vulkan.cpp)
        target_compile_definitions(VernonRHI PRIVATE VERNON_HAS_VULKAN_RHI=1 VK_ENABLE_BETA_EXTENSIONS=1
                                                     VK_NO_PROTOTYPES=1)
        target_link_libraries(VernonRHI PRIVATE $<BUILD_INTERFACE:Vulkan::Headers>)
    endif()
    if(VERNON_ENABLE_METAL_RUNTIME)
        if(NOT APPLE
           AND NOT
               CMAKE_SYSTEM_NAME
               STREQUAL
               "iOS")
            message(FATAL_ERROR "VERNON_ENABLE_METAL_RUNTIME is supported only on Apple platforms")
        endif()
        if(CMAKE_SYSTEM_NAME STREQUAL "iOS"
           AND NOT
               VERNON_RUNTIME_LIBRARY_TYPE
               STREQUAL
               "STATIC")
            message(FATAL_ERROR "The Metal Runtime must be static on iOS")
        endif()
        target_sources(VernonRHI PRIVATE ${_VERNON_RUNTIME_IMPL_DIR}/../rhi/metal_backend.mm
                                         ${_VERNON_RUNTIME_IMPL_DIR}/../rhi/rhi_metal.mm)
        set_target_properties(VernonRHI PROPERTIES OBJCXX_STANDARD 17 OBJCXX_STANDARD_REQUIRED ON)
        target_compile_definitions(VernonRHI PRIVATE VERNON_HAS_METAL_RHI=1)
        target_compile_options(VernonRHI PRIVATE "$<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>")
        find_library(_vernon_metal_framework Metal REQUIRED)
        find_library(_vernon_foundation_framework Foundation REQUIRED)
        target_link_libraries(VernonRHI PRIVATE ${_vernon_metal_framework} ${_vernon_foundation_framework})
    endif()

    add_library(
        VernonExecutionGraph STATIC
        ${_VERNON_RUNTIME_IMPL_DIR}/../execution_graph/execution_command_model.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/../execution_graph/execution_command_plan.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/../execution_graph/execution_graph.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/../execution_graph/execution_graph_autodiff.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/../execution_graph/execution_graph_checkpoint_planner.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/../execution_graph/execution_graph_submission.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/../execution_graph/execution_graph_validation.cpp)
    add_library(Vernon::ExecutionGraph ALIAS VernonExecutionGraph)
    set_target_properties(VernonExecutionGraph PROPERTIES EXPORT_NAME ExecutionGraph POSITION_INDEPENDENT_CODE ON)
    target_include_directories(
        VernonExecutionGraph
        PUBLIC $<BUILD_INTERFACE:${_VERNON_RUNTIME_INCLUDE_DIR}> $<INSTALL_INTERFACE:include>
        PRIVATE $<BUILD_INTERFACE:${_VERNON_RUNTIME_IMPL_DIR}/..>)
    target_link_libraries(VernonExecutionGraph PUBLIC Vernon::RHI)
    if(MSVC)
        target_compile_options(VernonExecutionGraph PRIVATE /EHsc)
    endif()

    add_library(VernonRuntimeCore STATIC ${_VERNON_RUNTIME_IMPL_DIR}/graphics_variant_key.cpp
                                         ${_VERNON_RUNTIME_IMPL_DIR}/runtime_core.cpp)
    add_library(Vernon::RuntimeCore ALIAS VernonRuntimeCore)
    set_target_properties(VernonRuntimeCore PROPERTIES EXPORT_NAME RuntimeCore POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(VernonRuntimeCore PUBLIC VERNON_RUNTIME_CORE_STATIC)
    target_include_directories(
        VernonRuntimeCore
        PUBLIC $<BUILD_INTERFACE:${_VERNON_RUNTIME_INCLUDE_DIR}> $<INSTALL_INTERFACE:include>
        PRIVATE $<BUILD_INTERFACE:${_VERNON_RUNTIME_IMPL_DIR}/..>)
    if(MSVC)
        target_compile_options(VernonRuntimeCore PRIVATE /EHsc)
    endif()

    add_library(
        VernonRuntimeInternals OBJECT
        ${_VERNON_RUNTIME_IMPL_DIR}/autodiff/autodiff_metadata.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/compute_launch_planner.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/content_hash.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/dirty_index_set.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/dirty_range_set.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/graphics_invocation_planner.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/graphics_scope_planner.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/target_implementation_metadata.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/prepared_graphics_draw.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/pipeline_bundle.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/pipeline_manifest.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/program_execution_backend.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/program_execution_manifest.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/program_instance.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/resolved_execution_plan.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/shape_layout.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/target_binding_plan.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/pipeline_metadata.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/tensor_bridge.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/vertex_attribute_capabilities.cpp)
    set_target_properties(VernonRuntimeInternals PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_include_directories(VernonRuntimeInternals PRIVATE $<BUILD_INTERFACE:${_VERNON_RUNTIME_INCLUDE_DIR}>
                                                              $<BUILD_INTERFACE:${_VERNON_RUNTIME_IMPL_DIR}/..>)
    target_link_libraries(VernonRuntimeInternals PRIVATE $<BUILD_INTERFACE:nlohmann_json::nlohmann_json>)
    if(MSVC)
        target_compile_options(VernonRuntimeInternals PRIVATE /EHsc)
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
    if(VERNON_ENABLE_METAL_RUNTIME)
        target_sources(VernonRuntimeRHIAdapter PRIVATE ${_VERNON_RUNTIME_IMPL_DIR}/rhi_adapter/adapter_metal.mm)
        set_target_properties(VernonRuntimeRHIAdapter PROPERTIES OBJCXX_STANDARD 17 OBJCXX_STANDARD_REQUIRED ON)
        target_compile_definitions(VernonRuntimeRHIAdapter PRIVATE VERNON_HAS_METAL_RHI=1)
        target_compile_options(VernonRuntimeRHIAdapter PRIVATE "$<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>")
        target_link_libraries(VernonRuntimeRHIAdapter PRIVATE ${_vernon_metal_framework}
                                                              ${_vernon_foundation_framework})
    endif()
    target_include_directories(VernonRuntimeRHIAdapter PRIVATE $<BUILD_INTERFACE:${_VERNON_RUNTIME_IMPL_DIR}/..>)
    target_link_libraries(VernonRuntimeRHIAdapter PUBLIC Vernon::RHI)

    add_library(
        VernonRuntime
        ${VERNON_RUNTIME_LIBRARY_TYPE}
        ${_VERNON_RUNTIME_IMPL_DIR}/VernonRuntime.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/program_graphics_executor.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/program_execution/program_invocation_state.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/program_execution/materialized_node_frame.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/program_execution/program_tensor_copy.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/program_execution/publication_transaction.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/program_execution/publication_executor.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/program_execution/resolved_transfer_executor.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/program_execution/device_buffer.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/program_execution/failure_injection.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/program_execution/physical_buffer_view.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/autodiff/host_tape_allocator.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/autodiff/program_tape_scratch.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/autodiff/retained_pullback_state.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/autodiff/program_boundary_binder.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/autodiff/program_invocation_builder.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/autodiff/program_residual_planner.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/autodiff/program_shape_resolver.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/autodiff/program_tape_lifecycle.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/autodiff/program_value_materializer.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/autodiff/runtime_autodiff.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/autodiff/runtime_autodiff_policy.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/autodiff/runtime_gpu_commands.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/autodiff/runtime_gpu_replay.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/autodiff/runtime_program_autodiff.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/backend_cpu.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/backend_opengl.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/cpu_workgroup_dispatch.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/program_boundary_view.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/runtime_backend_dispatch.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/runtime_pipeline_cpu.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/runtime_pipeline_cuda.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/runtime_pipeline_direct.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/runtime_pipeline_directx12.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/runtime_pipeline_dispatch.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/runtime_pipeline_metal.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/runtime_pipeline_opengl.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/runtime_pipeline_vulkan.cpp)
    add_library(Vernon::Runtime ALIAS VernonRuntime)
    target_sources(VernonRuntime PRIVATE $<TARGET_OBJECTS:VernonRuntimeInternals>)
    set_target_properties(VernonRuntime PROPERTIES EXPORT_NAME Runtime)
    target_compile_definitions(VernonRuntime PRIVATE VERNON_RUNTIME_BUILD)
    if(VERNON_RUNTIME_PROFILE STREQUAL "web")
        target_compile_definitions(VernonRuntime PUBLIC VERNON_RUNTIME_PROFILE_WEB=1)
        target_compile_definitions(VernonRuntimeInternals PRIVATE VERNON_RUNTIME_PROFILE_WEB=1)
    endif()
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
    if(VERNON_ENABLE_METAL_RUNTIME)
        target_compile_definitions(VernonRuntime PUBLIC VERNON_HAS_METAL_RUNTIME=1)
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
                VernonExecutionGraph
                $<BUILD_INTERFACE:nlohmann_json::nlohmann_json>)
    if(NOT
       VERNON_RUNTIME_PROFILE
       STREQUAL
       "web")
        target_link_libraries(VernonRuntime PRIVATE ${CMAKE_DL_LIBS})
    endif()
    # Runtime deployment is compiler-independent. CPU AOT entries are linked by the embedding application and resolved
    # through the static registry; LLVM, MLIR, and ORC must never become runtime link dependencies.
    foreach(_vernon_runtime_only_target IN ITEMS VernonRuntime VernonRuntimeInternals VernonRuntimeCore)
        get_target_property(_vernon_runtime_links ${_vernon_runtime_only_target} LINK_LIBRARIES)
        if(_vernon_runtime_links MATCHES "(^|;)(LLVM[^;]*|MLIR[^;]*|[^;]*ORC[^;]*)")
            message(FATAL_ERROR "${_vernon_runtime_only_target} must not link LLVM, MLIR, or ORC")
        endif()
    endforeach()
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
        FILES ${_VERNON_RUNTIME_INCLUDE_DIR}/VernonVersions.h
              ${_VERNON_RUNTIME_INCLUDE_DIR}/VernonCommon.h
              ${_VERNON_RUNTIME_INCLUDE_DIR}/VernonGraphicsState.h
              ${_VERNON_RUNTIME_INCLUDE_DIR}/VernonOpenGLContext.h
              ${_VERNON_RUNTIME_INCLUDE_DIR}/VernonRHI.h
              ${_VERNON_RUNTIME_INCLUDE_DIR}/VernonRHI.hpp
              ${_VERNON_RUNTIME_INCLUDE_DIR}/VernonTextureTypes.h
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
