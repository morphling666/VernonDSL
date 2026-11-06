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

    add_library(VernonRuntimeUtilities STATIC ${_VERNON_RUNTIME_IMPL_DIR}/dirty_range_set.cpp
                                              ${_VERNON_RUNTIME_IMPL_DIR}/tensor_bridge.cpp)
    set_target_properties(VernonRuntimeUtilities PROPERTIES EXPORT_NAME RuntimeUtilities POSITION_INDEPENDENT_CODE ON)
    target_include_directories(VernonRuntimeUtilities PRIVATE $<BUILD_INTERFACE:${_VERNON_RUNTIME_INCLUDE_DIR}>
                                                              $<BUILD_INTERFACE:${_VERNON_RUNTIME_IMPL_DIR}/..>)
    if(MSVC)
        target_compile_options(VernonRuntimeUtilities PRIVATE /EHsc)
    endif()

    add_library(
        VernonRuntimeInternals OBJECT
        ${_VERNON_RUNTIME_IMPL_DIR}/autodiff/autodiff_metadata.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/compute_launch_planner.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/content_hash.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/graphics_invocation_planner.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/graphics_scope_materializer.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/graphics_variant_key.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/target_implementation_metadata.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/prepared_graphics_draw.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/recorded_graphics.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/stage_artifact.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/stage_binding_plan.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/program_graph_composition.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/program_execution_backend.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/program_execution_manifest.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/resolved_execution_plan.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/shape_layout.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/target_binding_plan.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/prepared_binding_plan.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/pipeline_metadata.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/vertex_attribute_capabilities.cpp)
    set_target_properties(VernonRuntimeInternals PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(VernonRuntimeInternals PRIVATE VERNON_RUNTIME_STATIC)
    target_include_directories(VernonRuntimeInternals PRIVATE $<BUILD_INTERFACE:${_VERNON_RUNTIME_INCLUDE_DIR}>
                                                              $<BUILD_INTERFACE:${_VERNON_RUNTIME_IMPL_DIR}/..>)
    target_link_libraries(VernonRuntimeInternals PRIVATE VernonRuntimeUtilities
                                                         $<BUILD_INTERFACE:nlohmann_json::nlohmann_json>)
    if(MSVC)
        target_compile_options(VernonRuntimeInternals PRIVATE /EHsc)
    endif()

    set(_vernon_runtime_implementation_sources
        ${_VERNON_RUNTIME_IMPL_DIR}/program_graphics_executor.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/physical_buffer_view.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/set_frame.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/set_values.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/tensor_copy_plan.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/autodiff/derivative_groups.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/autodiff/host_tape_allocator.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/autodiff/runtime_gpu_replay.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/backend_cpu.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/cpu_workgroup_dispatch.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/program_boundary_view.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/runtime_backend_dispatch.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/runtime_pipeline_cuda.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/runtime_pipeline_resolved_compute.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/runtime_pipeline_dispatch.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/runtime_pipeline_opengl.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/runtime_pipeline_vulkan.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/schedule/executor.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/schedule/fragment.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/schedule/linker.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/schedule/pass_fragment.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/schedule/pass_plan.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/schedule/program_arguments.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/schedule/vjp_plan.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/../hal/cpu/cpu_device.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/../hal/cpu/cpu_resources.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/../hal/cuda/cuda_device.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/../hal/cuda/cuda_resources.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/../hal/native_images.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/../hal/opengl/opengl_device.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/../hal/opengl/opengl_driver.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/../hal/opengl/opengl_resources.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/../hal/opengl/opengl_state.cpp)

    add_library(VernonRuntimeImplementation OBJECT ${_vernon_runtime_implementation_sources})
    set_target_properties(VernonRuntimeImplementation PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(VernonRuntimeImplementation PRIVATE VERNON_RUNTIME_STATIC)
    if(VERNON_RUNTIME_LIBRARY_TYPE STREQUAL "SHARED")
        target_compile_definitions(VernonRuntimeImplementation PRIVATE VERNON_CPU_WORKGROUP_ABI_BUILD)
    endif()
    target_include_directories(VernonRuntimeImplementation PRIVATE ${_VERNON_RUNTIME_IMPL_DIR}/..
                                                                   ${_VERNON_RUNTIME_INCLUDE_DIR})
    target_link_libraries(VernonRuntimeImplementation PRIVATE VernonRuntimeUtilities VernonPlatform
                                                              $<BUILD_INTERFACE:nlohmann_json::nlohmann_json>)

    # The translation units that define the public C API; each target compiles them with its own export visibility.
    set(_vernon_runtime_api_sources
        ${_VERNON_RUNTIME_IMPL_DIR}/VernonRuntime.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/capi/boundary.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/capi/graphics_recording.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/capi/program_executable.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/capi/program_loading.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/capi/program_state.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/capi/schedule.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/host_containers.cpp
        ${_VERNON_RUNTIME_IMPL_DIR}/../hal/capi.cpp)

    add_library(VernonRuntime ${VERNON_RUNTIME_LIBRARY_TYPE} ${_vernon_runtime_api_sources}
                              ${_VERNON_RUNTIME_IMPL_DIR}/runtime_python_bridge.cpp)
    target_compile_definitions(VernonRuntime PRIVATE VERNON_RUNTIME_BUILD)
    target_sources(VernonRuntime PRIVATE $<TARGET_OBJECTS:VernonRuntimeImplementation>
                                         $<TARGET_OBJECTS:VernonRuntimeInternals>)
    add_library(Vernon::Runtime ALIAS VernonRuntime)
    set_target_properties(VernonRuntime PROPERTIES EXPORT_NAME Runtime)
    if(VERNON_RUNTIME_PROFILE STREQUAL "web")
        target_compile_definitions(VernonRuntime PUBLIC VERNON_RUNTIME_PROFILE_WEB=1)
        target_compile_definitions(VernonRuntimeInternals PRIVATE VERNON_RUNTIME_PROFILE_WEB=1)
        target_compile_definitions(VernonRuntimeImplementation PRIVATE VERNON_RUNTIME_PROFILE_WEB=1)
    endif()
    if(WIN32)
        target_compile_definitions(VernonRuntimeImplementation PRIVATE NOMINMAX)
        target_compile_definitions(VernonRuntime PRIVATE NOMINMAX)
    endif()
    if(VERNON_RUNTIME_LIBRARY_TYPE STREQUAL "STATIC")
        target_compile_definitions(VernonRuntime PUBLIC VERNON_RUNTIME_STATIC)
    endif()
    if(VERNON_ENABLE_CUDA_RUNTIME)
        target_compile_definitions(VernonRuntime PUBLIC VERNON_HAS_CUDA_RUNTIME=1)
        target_compile_definitions(VernonRuntimeImplementation PRIVATE VERNON_HAS_CUDA_RUNTIME=1)
        target_sources(VernonRuntimeImplementation PRIVATE ${_VERNON_RUNTIME_IMPL_DIR}/../hal/cuda/cuda_driver.cpp
                                                           ${_VERNON_RUNTIME_IMPL_DIR}/../hal/cuda/cuda_state.cpp)
        target_link_libraries(VernonRuntimeImplementation PRIVATE $<BUILD_INTERFACE:nvtx3::nvtx3-c>)
    endif()
    if(VERNON_ENABLE_VULKAN_RUNTIME)
        target_compile_definitions(VernonRuntime PUBLIC VERNON_HAS_VULKAN_RUNTIME=1)
        target_compile_definitions(VernonRuntimeImplementation PRIVATE VERNON_HAS_VULKAN_RUNTIME=1 VK_NO_PROTOTYPES=1
                                                                       VK_ENABLE_BETA_EXTENSIONS=1)
        target_sources(
            VernonRuntimeImplementation
            PRIVATE ${_VERNON_RUNTIME_IMPL_DIR}/../hal/vulkan/vulkan_device.cpp
                    ${_VERNON_RUNTIME_IMPL_DIR}/../hal/vulkan/vulkan_driver.cpp
                    ${_VERNON_RUNTIME_IMPL_DIR}/../hal/vulkan/vulkan_platform_surface.cpp
                    ${_VERNON_RUNTIME_IMPL_DIR}/../hal/vulkan/vulkan_resources.cpp
                    ${_VERNON_RUNTIME_IMPL_DIR}/../hal/vulkan/vulkan_state.cpp)
        target_link_libraries(VernonRuntimeImplementation PRIVATE $<BUILD_INTERFACE:Vulkan::Headers>)
        target_link_libraries(VernonRuntime PRIVATE $<BUILD_INTERFACE:Vulkan::Headers>)
    endif()
    if(VERNON_ENABLE_DIRECTX12_RUNTIME)
        target_compile_definitions(VernonRuntime PUBLIC VERNON_HAS_DIRECTX12_RUNTIME=1)
        target_compile_definitions(VernonRuntimeImplementation PRIVATE VERNON_HAS_DIRECTX12_RUNTIME=1)
        target_sources(
            VernonRuntimeImplementation
            PRIVATE ${_VERNON_RUNTIME_IMPL_DIR}/runtime_pipeline_directx12.cpp
                    ${_VERNON_RUNTIME_IMPL_DIR}/../hal/d3d12/d3d12_device.cpp
                    ${_VERNON_RUNTIME_IMPL_DIR}/../hal/d3d12/d3d12_resources.cpp
                    ${_VERNON_RUNTIME_IMPL_DIR}/../hal/d3d12/d3d12_state.cpp)
        target_link_libraries(VernonRuntimeImplementation PRIVATE d3d12 dxgi dxguid)
        target_link_libraries(VernonRuntime PRIVATE d3d12 dxgi dxguid)
        # The mipmap box filters are compiled to DXIL at build time; without DXC, D3D12 images report mipmap generation
        # as unsupported.
        set(_vernon_dxc_hints)
        if(VERNON_DXC_EXECUTABLE)
            get_filename_component(_vernon_dxc_directory "${VERNON_DXC_EXECUTABLE}" DIRECTORY)
            list(APPEND _vernon_dxc_hints "${_vernon_dxc_directory}")
        endif()
        file(
            GLOB _vernon_windows_sdk_versions
            LIST_DIRECTORIES TRUE
            "$ENV{ProgramFiles\(x86\)}/Windows Kits/10/bin/10.*")
        list(
            SORT _vernon_windows_sdk_versions
            COMPARE NATURAL
            ORDER DESCENDING)
        foreach(_vernon_windows_sdk_version IN LISTS _vernon_windows_sdk_versions)
            list(APPEND _vernon_dxc_hints "${_vernon_windows_sdk_version}/x64")
        endforeach()
        find_program(
            VERNON_MIPMAP_DXC
            NAMES dxc
            HINTS ${_vernon_dxc_hints})
        if(VERNON_MIPMAP_DXC)
            set(_vernon_d3d12_generated_dir "${CMAKE_CURRENT_BINARY_DIR}/d3d12_generated")
            file(MAKE_DIRECTORY "${_vernon_d3d12_generated_dir}")
            set(_vernon_mipmap_shader "${_VERNON_RUNTIME_IMPL_DIR}/../hal/d3d12/d3d12_mipmap.hlsl")
            set(_vernon_mipmap_headers)
            foreach(_vernon_mipmap_dimension IN ITEMS 2d 3d)
                set(_vernon_mipmap_blob "${_vernon_d3d12_generated_dir}/d3d12_mipmap_${_vernon_mipmap_dimension}.bin")
                set(_vernon_mipmap_header "${_vernon_d3d12_generated_dir}/d3d12_mipmap_${_vernon_mipmap_dimension}.h")
                if(_vernon_mipmap_dimension STREQUAL "3d")
                    set(_vernon_mipmap_volume 1)
                else()
                    set(_vernon_mipmap_volume 0)
                endif()
                add_custom_command(
                    OUTPUT "${_vernon_mipmap_header}"
                    COMMAND
                        "${VERNON_MIPMAP_DXC}" -nologo -T cs_6_0 -E main -D VERNON_MIPMAP_3D=${_vernon_mipmap_volume}
                        -Fo "${_vernon_mipmap_blob}" "${_vernon_mipmap_shader}"
                    COMMAND
                        "${CMAKE_COMMAND}" -DINPUT=${_vernon_mipmap_blob} -DOUTPUT=${_vernon_mipmap_header}
                        -DSYMBOL=vernon_d3d12_mipmap_${_vernon_mipmap_dimension} -P
                        "${VERNON_REPOSITORY_ROOT}/cmake/EmbedBinary.cmake"
                    DEPENDS "${_vernon_mipmap_shader}" "${VERNON_REPOSITORY_ROOT}/cmake/EmbedBinary.cmake"
                    VERBATIM)
                list(APPEND _vernon_mipmap_headers "${_vernon_mipmap_header}")
            endforeach()
            set(_vernon_stream_shader "${_VERNON_RUNTIME_IMPL_DIR}/../hal/d3d12/d3d12_draw_stream.hlsl")
            set(_vernon_stream_blob "${_vernon_d3d12_generated_dir}/d3d12_draw_stream.bin")
            set(_vernon_stream_header "${_vernon_d3d12_generated_dir}/d3d12_draw_stream.h")
            add_custom_command(
                OUTPUT "${_vernon_stream_header}"
                COMMAND "${VERNON_MIPMAP_DXC}" -nologo -T cs_6_0 -E main -Fo "${_vernon_stream_blob}"
                        "${_vernon_stream_shader}"
                COMMAND "${CMAKE_COMMAND}" -DINPUT=${_vernon_stream_blob} -DOUTPUT=${_vernon_stream_header}
                        -DSYMBOL=vernon_d3d12_draw_stream -P "${VERNON_REPOSITORY_ROOT}/cmake/EmbedBinary.cmake"
                DEPENDS "${_vernon_stream_shader}" "${VERNON_REPOSITORY_ROOT}/cmake/EmbedBinary.cmake"
                VERBATIM)
            list(APPEND _vernon_mipmap_headers "${_vernon_stream_header}")
            target_sources(VernonRuntimeImplementation PRIVATE ${_vernon_mipmap_headers})
            target_include_directories(VernonRuntimeImplementation PRIVATE "${_vernon_d3d12_generated_dir}")
            target_compile_definitions(VernonRuntimeImplementation PRIVATE VERNON_DIRECTX12_MIPMAPS=1
                                                                           VERNON_DIRECTX12_DRAW_STREAMS=1)
        else()
            message(WARNING "DXC was not found; DirectX 12 images cannot generate mipmaps and DirectX 12 devices "
                            "draw no device draw streams")
        endif()
    endif()
    if(VERNON_ENABLE_METAL_RUNTIME)
        target_compile_definitions(VernonRuntime PUBLIC VERNON_HAS_METAL_RUNTIME=1)
        target_compile_definitions(VernonRuntimeImplementation PRIVATE VERNON_HAS_METAL_RUNTIME=1)
        set(_vernon_metal_sources
            ${_VERNON_RUNTIME_IMPL_DIR}/../hal/metal/metal_device.mm
            ${_VERNON_RUNTIME_IMPL_DIR}/../hal/metal/metal_resources.mm
            ${_VERNON_RUNTIME_IMPL_DIR}/../hal/metal/metal_state.mm)
        target_sources(VernonRuntimeImplementation PRIVATE ${_VERNON_RUNTIME_IMPL_DIR}/runtime_pipeline_metal.cpp
                                                           ${_vernon_metal_sources})
        # metal_native.h holds ARC-managed ids inside C++ objects and refuses to compile without ARC.
        set_source_files_properties(${_vernon_metal_sources} PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
        set_target_properties(VernonRuntimeImplementation PROPERTIES OBJCXX_STANDARD 17 OBJCXX_STANDARD_REQUIRED ON)
        find_library(VERNON_METAL_FRAMEWORK Metal REQUIRED)
        find_library(VERNON_QUARTZCORE_FRAMEWORK QuartzCore REQUIRED)
        find_library(VERNON_FOUNDATION_FRAMEWORK Foundation REQUIRED)
        set(VERNON_METAL_FRAMEWORKS ${VERNON_METAL_FRAMEWORK} ${VERNON_QUARTZCORE_FRAMEWORK}
                                    ${VERNON_FOUNDATION_FRAMEWORK})
        target_link_libraries(VernonRuntimeImplementation PRIVATE ${VERNON_METAL_FRAMEWORKS})
        target_link_libraries(VernonRuntime PRIVATE ${VERNON_METAL_FRAMEWORKS})
    endif()
    target_include_directories(
        VernonRuntime
        PUBLIC $<BUILD_INTERFACE:${_VERNON_RUNTIME_INCLUDE_DIR}> $<INSTALL_INTERFACE:include>
        PRIVATE ${_VERNON_RUNTIME_IMPL_DIR}/..)
    target_link_libraries(VernonRuntime PRIVATE VernonRuntimeUtilities VernonPlatform
                                                $<BUILD_INTERFACE:nlohmann_json::nlohmann_json>)
    if(NOT
       VERNON_RUNTIME_PROFILE
       STREQUAL
       "web")
        target_link_libraries(VernonRuntime PRIVATE ${CMAKE_DL_LIBS})
    endif()
    # Runtime deployment is compiler-independent. CPU AOT entries are linked by the embedding application and resolved
    # through the static registry; LLVM, MLIR, and ORC must never become runtime link dependencies.
    foreach(_vernon_runtime_only_target IN ITEMS VernonRuntime VernonRuntimeInternals)
        get_target_property(_vernon_runtime_links ${_vernon_runtime_only_target} LINK_LIBRARIES)
        if(_vernon_runtime_links MATCHES "(^|;)(LLVM[^;]*|MLIR[^;]*|[^;]*ORC[^;]*)")
            message(FATAL_ERROR "${_vernon_runtime_only_target} must not link LLVM, MLIR, or ORC")
        endif()
    endforeach()
    if(BUILD_TESTING)
        target_compile_definitions(VernonRuntimeImplementation PRIVATE VERNON_RUNTIME_TESTING=1)
        add_library(VernonRuntimeTestHost STATIC ${_vernon_runtime_api_sources})
        target_sources(VernonRuntimeTestHost PRIVATE $<TARGET_OBJECTS:VernonRuntimeImplementation>
                                                     $<TARGET_OBJECTS:VernonRuntimeInternals>)
        add_library(Vernon::RuntimeTesting ALIAS VernonRuntimeTestHost)
        target_compile_definitions(VernonRuntimeTestHost PUBLIC VERNON_RUNTIME_STATIC VERNON_RUNTIME_TESTING=1)
        if(WIN32)
            target_compile_definitions(VernonRuntimeTestHost PRIVATE NOMINMAX)
        endif()
        target_include_directories(VernonRuntimeTestHost PUBLIC $<BUILD_INTERFACE:${_VERNON_RUNTIME_INCLUDE_DIR}>
                                                                $<BUILD_INTERFACE:${_VERNON_RUNTIME_IMPL_DIR}/..>)
        target_link_libraries(VernonRuntimeTestHost PUBLIC VernonRuntimeUtilities VernonPlatform
                                                           $<BUILD_INTERFACE:nlohmann_json::nlohmann_json>)
        if(NOT
           VERNON_RUNTIME_PROFILE
           STREQUAL
           "web")
            target_link_libraries(VernonRuntimeTestHost PUBLIC ${CMAKE_DL_LIBS})
        endif()
        if(VERNON_ENABLE_CUDA_RUNTIME)
            target_compile_definitions(VernonRuntimeTestHost PRIVATE VERNON_HAS_CUDA_RUNTIME=1)
        endif()
        if(VERNON_ENABLE_VULKAN_RUNTIME)
            target_compile_definitions(VernonRuntimeTestHost PRIVATE VERNON_HAS_VULKAN_RUNTIME=1)
            target_link_libraries(VernonRuntimeTestHost PUBLIC $<BUILD_INTERFACE:Vulkan::Headers>)
        endif()
        if(VERNON_ENABLE_DIRECTX12_RUNTIME)
            target_compile_definitions(VernonRuntimeTestHost PRIVATE VERNON_HAS_DIRECTX12_RUNTIME=1)
            target_link_libraries(VernonRuntimeTestHost PUBLIC d3d12 dxgi dxguid)
        endif()
        if(VERNON_ENABLE_METAL_RUNTIME)
            target_compile_definitions(VernonRuntimeTestHost PRIVATE VERNON_HAS_METAL_RUNTIME=1)
            target_link_libraries(VernonRuntimeTestHost PUBLIC ${VERNON_METAL_FRAMEWORKS})
        endif()
    endif()
    if(MSVC)
        target_compile_options(VernonRuntimeImplementation PRIVATE /EHsc)
    endif()
    if(SKBUILD)
        set(_vernon_runtime_bin_destination vernon_dsl)
        set(_vernon_runtime_lib_destination vernon_dsl)
        set_target_properties(VernonRuntime PROPERTIES OUTPUT_NAME VernonDSLHostRuntime)
    else()
        set(_vernon_runtime_bin_destination bin)
        set(_vernon_runtime_lib_destination lib)
    endif()
    set(_vernon_runtime_install_targets VernonRuntime)
    if(VERNON_RUNTIME_LIBRARY_TYPE STREQUAL "STATIC")
        list(
            APPEND
            _vernon_runtime_install_targets
            VernonRuntimeUtilities
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
              ${_VERNON_RUNTIME_INCLUDE_DIR}/VernonFormats.def
              ${_VERNON_RUNTIME_INCLUDE_DIR}/VernonFormats.hpp
              ${_VERNON_RUNTIME_INCLUDE_DIR}/VernonResult.hpp
              ${_VERNON_RUNTIME_INCLUDE_DIR}/VernonError.hpp
              ${_VERNON_RUNTIME_INCLUDE_DIR}/VernonLifecycle.hpp
              ${_VERNON_RUNTIME_INCLUDE_DIR}/VernonProgramSemanticTypes.h
              ${_VERNON_RUNTIME_INCLUDE_DIR}/VernonGraphicsState.h
              ${_VERNON_RUNTIME_INCLUDE_DIR}/VernonOpenGLContext.h
              ${_VERNON_RUNTIME_INCLUDE_DIR}/VernonTextureTypes.h
              ${_VERNON_RUNTIME_INCLUDE_DIR}/VernonHal.h
              ${_VERNON_RUNTIME_INCLUDE_DIR}/VernonSchedule.h
              ${_VERNON_RUNTIME_INCLUDE_DIR}/VernonRuntime.h
              ${_VERNON_RUNTIME_INCLUDE_DIR}/VernonRuntime.hpp
              ${_VERNON_RUNTIME_INCLUDE_DIR}/VernonRuntimeProvider.h
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
