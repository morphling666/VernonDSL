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

# The tally sample library's generated nanobind module (python/tests/bindgen_sample), which the bindgen tests bind
# Python-mode Host code to. It builds next to the sample, where the tests import it as tally_api.
function(_vernon_add_bindgen_sample package_directory)
    set(sample "${VERNON_REPOSITORY_ROOT}/python/tests/bindgen_sample")
    nanobind_add_module(
        vernon-bindgen-tally-api
        "${sample}/generated/tally_api_nanobind.cpp"
        "${sample}/generated/tally_api_thunks.cpp"
        "${sample}/cpp/tally_api.cpp"
        "${sample}/cpp/counter.cpp")
    target_include_directories(vernon-bindgen-tally-api PRIVATE "${sample}/generated" "${sample}/cpp"
                                                                "${package_directory}/hostlang/include")
    if(MSVC)
        target_compile_options(vernon-bindgen-tally-api PRIVATE /EHsc)
    endif()
    set_target_properties(vernon-bindgen-tally-api PROPERTIES OUTPUT_NAME "tally_api")
    _vernon_set_python_module_output(vernon-bindgen-tally-api "${sample}")
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
        "${source_directory}/python/native_compiler.cpp"
        "${source_directory}/python/native_host.cpp"
        "${source_directory}/python/native_module.cpp"
        "${source_directory}/python/native_program.cpp"
        "${source_directory}/python/native_program_autodiff.cpp"
        "${source_directory}/python/native_rhi.cpp"
        "${source_directory}/python/native_runtime.cpp")
    if(NOT SKBUILD)
        _vernon_set_python_module_output(vernon-dsl-native "${package_directory}")
    endif()
    target_link_libraries(
        vernon-dsl-native
        PRIVATE VernonRuntimeUtilities
                VernonDSLCompiler
                Vernon::Runtime
                Vernon::RHI)
    target_include_directories(vernon-dsl-native PRIVATE "${source_directory}/lib/compiler" "${source_directory}/lib")
    if(BUILD_TESTING)
        target_compile_definitions(vernon-dsl-native PRIVATE VERNON_ENABLE_LIFECYCLE_TEST_HOOKS=1)
    endif()
    if(TARGET VernonGlfwContextOwner)
        target_link_libraries(vernon-dsl-native PRIVATE Vernon::GlfwContextOwner)
        target_compile_definitions(vernon-dsl-native PRIVATE VERNON_HAS_GLFW_CONTEXT_OWNER=1)
    endif()
    if(MSVC)
        target_compile_options(vernon-dsl-native PRIVATE /EHsc)
    endif()
    vernon_stage_target_files(
        vernon-dsl-native
        VernonDSLCompiler
        VernonRuntime
        VernonRHI)
    set_target_properties(vernon-dsl-native PROPERTIES OUTPUT_NAME "_native")
    if(BUILD_TESTING)
        _vernon_add_bindgen_sample("${package_directory}")
    endif()
    install(
        TARGETS vernon-dsl-native
        RUNTIME DESTINATION vernon_dsl COMPONENT VernonWheel
        LIBRARY DESTINATION vernon_dsl COMPONENT VernonWheel)

endfunction()
