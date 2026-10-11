set(_vernon_gl_context_default OFF)
if(VERNON_BUILD_RUNTIME
   AND VERNON_BUILD_COMPILER
   AND VERNON_ENABLE_PYTHON_BINDINGS)
    set(_vernon_gl_context_default ON)
endif()
option(VERNON_ENABLE_GLFW_CONTEXT_OWNER "Build the GLFW OpenGL context-owner host component"
       ${_vernon_gl_context_default})

set(VERNON_PYTHON_PACKAGE_DIR "${VERNON_REPOSITORY_ROOT}/python/vernon_dsl")

if(VERNON_ENABLE_GLFW_CONTEXT_OWNER)
    include("${VERNON_REPOSITORY_ROOT}/cmake/IncludeGlfw.cmake")
    include_glfw()
    add_library(VernonGlfwContextOwner STATIC host/glfw_context_owner.cpp)
    add_library(Vernon::GlfwContextOwner ALIAS VernonGlfwContextOwner)
    target_include_directories(VernonGlfwContextOwner PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/host"
                                                             "${CMAKE_CURRENT_SOURCE_DIR}/include")
    target_link_libraries(VernonGlfwContextOwner PRIVATE glfw)
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
    include("${VERNON_REPOSITORY_ROOT}/cmake/VernonPythonBindings.cmake")
    vernon_add_python_bindings("${VERNON_PYTHON_PACKAGE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
endif()

if(TARGET vernon-dsl-native)
    find_package(
        Python 3.11...<3.15
        COMPONENTS Interpreter
        QUIET)
    if(Python_EXECUTABLE)
        file(
            GLOB_RECURSE
            _vernon_cooker_python_sources
            CONFIGURE_DEPENDS
            "${VERNON_PYTHON_PACKAGE_DIR}/*.py")
        set(_vernon_cooker_dependencies vernon-dsl-native "$<TARGET_FILE:vernon-dsl-native>")
        if(TARGET VernonDSLCompiler)
            list(
                APPEND
                _vernon_cooker_dependencies
                VernonDSLCompiler
                "$<TARGET_FILE:VernonDSLCompiler>")
        endif()
        vernon_configure_program_cooker(
            PYTHON
            "${Python_EXECUTABLE}"
            PYTHON_PATH
            "${VERNON_REPOSITORY_ROOT}/python"
            DEPENDS
            ${_vernon_cooker_dependencies}
            ${_vernon_cooker_python_sources})
    endif()
endif()

if(BUILD_TESTING)
    find_package(
        Python 3.11...<3.15
        COMPONENTS Interpreter
        QUIET)
    include("${VERNON_REPOSITORY_ROOT}/cmake/VernonTests.cmake")
    vernon_add_tests()
endif()
