set(_vernon_gl_context_default OFF)
if(VERNON_BUILD_RUNTIME
   AND VERNON_BUILD_COMPILER
   AND VERNON_ENABLE_PYTHON_BINDINGS)
    set(_vernon_gl_context_default ON)
endif()
option(VERNON_ENABLE_GLFW_CONTEXT_OWNER "Build the Python GLFW OpenGL context-owner module"
       ${_vernon_gl_context_default})

set(VERNON_PYTHON_PACKAGE_DIR "${VERNON_REPOSITORY_ROOT}/python/vernon_dsl")

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
    include("${VERNON_REPOSITORY_ROOT}/cmake/VernonPythonBindings.cmake")
    vernon_add_python_bindings("${VERNON_PYTHON_PACKAGE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
endif()

if(BUILD_TESTING)
    find_package(
        Python 3.11...<3.15
        COMPONENTS Interpreter
        QUIET)
    include("${VERNON_REPOSITORY_ROOT}/cmake/VernonTests.cmake")
    vernon_add_tests()
endif()
