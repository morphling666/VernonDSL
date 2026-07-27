option(VERNON_ENABLE_PYTHON_BINDINGS "Build the nanobind Python module" ON)
set(_vernon_gl_context_default OFF)
if(VERNON_BUILD_RUNTIME
   AND VERNON_BUILD_COMPILER
   AND VERNON_ENABLE_PYTHON_BINDINGS)
    set(_vernon_gl_context_default ON)
endif()
option(VERNON_ENABLE_GLFW_CONTEXT_OWNER "Build the Python GLFW OpenGL context-owner module"
       ${_vernon_gl_context_default})

set(VERNON_PYTHON_PACKAGE_DIR "${VERNON_REPOSITORY_ROOT}/python/vernon_dsl")

function(vernon_set_python_module_output target)
    set_target_properties(
        ${target}
        PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${VERNON_PYTHON_PACKAGE_DIR}"
                   RUNTIME_OUTPUT_DIRECTORY "${VERNON_PYTHON_PACKAGE_DIR}"
                   LIBRARY_OUTPUT_DIRECTORY_DEBUG "${VERNON_PYTHON_PACKAGE_DIR}"
                   LIBRARY_OUTPUT_DIRECTORY_RELEASE "${VERNON_PYTHON_PACKAGE_DIR}"
                   LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO "${VERNON_PYTHON_PACKAGE_DIR}"
                   LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL "${VERNON_PYTHON_PACKAGE_DIR}"
                   RUNTIME_OUTPUT_DIRECTORY_DEBUG "${VERNON_PYTHON_PACKAGE_DIR}"
                   RUNTIME_OUTPUT_DIRECTORY_RELEASE "${VERNON_PYTHON_PACKAGE_DIR}"
                   RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${VERNON_PYTHON_PACKAGE_DIR}"
                   RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${VERNON_PYTHON_PACKAGE_DIR}")
endfunction()

if(VERNON_ENABLE_GLFW_CONTEXT_OWNER)
    include("${VERNON_REPOSITORY_ROOT}/cmake/IncludeGlfw.cmake")
    include_glfw()
endif()

if(VERNON_BUILD_RUNTIME)
    add_subdirectory(lib/runtime)
endif()
if(VERNON_BUILD_COMPILER)
    add_subdirectory(lib/compiler)
endif()

if(VERNON_ENABLE_PYTHON_BINDINGS
   AND TARGET VernonRuntime
   AND TARGET VernonDSLCompiler)
    find_package(
        Python 3.11.9...<3.12
        COMPONENTS Interpreter Development.Module
        QUIET)
endif()

if(TARGET VernonRuntime AND TARGET VernonDSLCompiler)
    if(VERNON_ENABLE_PYTHON_BINDINGS)
        if(Python_FOUND)
            execute_process(
                COMMAND ${Python_EXECUTABLE} -m nanobind --cmake_dir
                OUTPUT_VARIABLE VERNON_NANOBIND_CMAKE_DIR
                OUTPUT_STRIP_TRAILING_WHITESPACE
                RESULT_VARIABLE VERNON_NANOBIND_RESULT)
            if(VERNON_NANOBIND_RESULT EQUAL 0)
                list(APPEND CMAKE_PREFIX_PATH "${VERNON_NANOBIND_CMAKE_DIR}")
                find_package(nanobind CONFIG QUIET)
            endif()
        endif()
        if(nanobind_FOUND)
            nanobind_add_module(vernon-dsl-native python/native_module.cpp)
            vernon_set_python_module_output(vernon-dsl-native)
            target_link_libraries(
                vernon-dsl-native
                PRIVATE VernonDSLCompiler
                        Vernon::Runtime
                        Vernon::RHI
                        Vernon::ExecutionGraph)
            if(MSVC)
                target_compile_options(vernon-dsl-native PRIVATE /EHsc)
            endif()
            add_custom_command(
                TARGET vernon-dsl-native
                POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:VernonDSLCompiler>
                        $<TARGET_FILE_DIR:vernon-dsl-native>
                COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:VernonRuntime>
                        $<TARGET_FILE_DIR:vernon-dsl-native>
                COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:VernonRHI>
                        $<TARGET_FILE_DIR:vernon-dsl-native>)
            set_target_properties(vernon-dsl-native PROPERTIES OUTPUT_NAME "_native")
            if(APPLE)
                set_target_properties(vernon-dsl-native PROPERTIES INSTALL_RPATH "@loader_path")
            elseif(UNIX)
                set_target_properties(vernon-dsl-native PROPERTIES INSTALL_RPATH "$ORIGIN")
            endif()
            install(
                TARGETS vernon-dsl-native
                RUNTIME DESTINATION vernon_dsl COMPONENT VernonWheel
                LIBRARY DESTINATION vernon_dsl COMPONENT VernonWheel)
            if(VERNON_ENABLE_GLFW_CONTEXT_OWNER AND TARGET glfw)
                nanobind_add_module(vernon-dsl-gl-context python/gl_context_module.cpp)
                vernon_set_python_module_output(vernon-dsl-gl-context)
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
        else()
            message(STATUS "nanobind not found; skipping vernon_dsl._native")
        endif()
    endif()
endif()

if(BUILD_TESTING)
    find_package(
        Python 3.11.9...<3.12
        COMPONENTS Interpreter
        QUIET)
    add_subdirectory(tests)
endif()
