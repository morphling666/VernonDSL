include_guard(GLOBAL)

include("${CMAKE_CURRENT_LIST_DIR}/VernonTargetHelpers.cmake")

function(_vernon_set_python_module_output target package_directory)
    set_target_properties(
        ${target}
        PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${package_directory}"
                   RUNTIME_OUTPUT_DIRECTORY "${package_directory}"
                   LIBRARY_OUTPUT_DIRECTORY_DEBUG "${package_directory}"
                   LIBRARY_OUTPUT_DIRECTORY_RELEASE "${package_directory}"
                   LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO "${package_directory}"
                   LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL "${package_directory}"
                   RUNTIME_OUTPUT_DIRECTORY_DEBUG "${package_directory}"
                   RUNTIME_OUTPUT_DIRECTORY_RELEASE "${package_directory}"
                   RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${package_directory}"
                   RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${package_directory}")
endfunction()

function(vernon_add_python_bindings package_directory source_directory)
    if(NOT TARGET VernonRuntime OR NOT TARGET VernonDSLCompiler)
        message(FATAL_ERROR "Python bindings require VernonRuntime and VernonDSLCompiler")
    endif()

    find_package(
        Python 3.11...<3.15
        COMPONENTS Interpreter Development.Module
        QUIET)
    if(NOT Python_FOUND)
        message(STATUS "Python development files not found; skipping vernon_dsl._native")
        return()
    endif()

    execute_process(
        COMMAND ${Python_EXECUTABLE} -m nanobind --cmake_dir
        OUTPUT_VARIABLE VERNON_NANOBIND_CMAKE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE VERNON_NANOBIND_RESULT)
    if(NOT
       VERNON_NANOBIND_RESULT
       EQUAL
       0)
        message(STATUS "nanobind not found; skipping vernon_dsl._native")
        return()
    endif()

    list(APPEND CMAKE_PREFIX_PATH "${VERNON_NANOBIND_CMAKE_DIR}")
    find_package(nanobind CONFIG QUIET)
    if(NOT nanobind_FOUND)
        message(STATUS "nanobind not found; skipping vernon_dsl._native")
        return()
    endif()

    target_sources(VernonDSLCompiler PRIVATE "${source_directory}/lib/compiler/compiler_python_bridge.cpp")
    nanobind_add_module(
        vernon-dsl-native
        "${source_directory}/python/native_command_retention.cpp"
        "${source_directory}/python/native_compiler.cpp"
        "${source_directory}/python/native_module.cpp"
        "${source_directory}/python/native_pipeline.cpp"
        "${source_directory}/python/native_pipeline_autodiff.cpp"
        "${source_directory}/python/native_rhi.cpp"
        "${source_directory}/python/native_runtime.cpp")
    _vernon_set_python_module_output(vernon-dsl-native "${package_directory}")
    target_link_libraries(
        vernon-dsl-native
        PRIVATE VernonDSLCompiler
                Vernon::Runtime
                Vernon::RHI
                Vernon::ExecutionGraph)
    target_include_directories(vernon-dsl-native PRIVATE "${source_directory}/lib/compiler" "${source_directory}/lib")
    if(MSVC)
        target_compile_options(vernon-dsl-native PRIVATE /EHsc)
    endif()
    vernon_stage_target_files(
        vernon-dsl-native
        VernonDSLCompiler
        VernonRuntime
        VernonRHI)
    set_target_properties(vernon-dsl-native PROPERTIES OUTPUT_NAME "_native")
    install(
        TARGETS vernon-dsl-native
        RUNTIME DESTINATION vernon_dsl COMPONENT VernonWheel
        LIBRARY DESTINATION vernon_dsl COMPONENT VernonWheel)

    if(VERNON_ENABLE_GLFW_CONTEXT_OWNER AND TARGET glfw)
        nanobind_add_module(vernon-dsl-gl-context "${source_directory}/python/gl_context_module.cpp")
        _vernon_set_python_module_output(vernon-dsl-gl-context "${package_directory}")
        target_link_libraries(vernon-dsl-gl-context PRIVATE glfw)
        if(MSVC)
            target_compile_options(vernon-dsl-gl-context PRIVATE /EHsc)
        endif()
        set_target_properties(vernon-dsl-gl-context PROPERTIES OUTPUT_NAME "_gl_context")
        install(
            TARGETS vernon-dsl-gl-context
            RUNTIME DESTINATION vernon_dsl COMPONENT VernonWheel
            LIBRARY DESTINATION vernon_dsl COMPONENT VernonWheel)
    endif()
endfunction()
