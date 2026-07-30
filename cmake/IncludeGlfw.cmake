include_guard(GLOBAL)

set(GLFW_REPOSITORY
    "https://github.com/glfw/glfw.git"
    CACHE STRING "GLFW repository to fetch")
set(GLFW_REVISION
    "3.4"
    CACHE STRING "GLFW dependency revision")

function(include_glfw)
    include(FetchContent)
    set(GLFW_BUILD_DOCS
        OFF
        CACHE BOOL "" FORCE)
    set(GLFW_BUILD_EXAMPLES
        OFF
        CACHE BOOL "" FORCE)
    set(GLFW_BUILD_TESTS
        OFF
        CACHE BOOL "" FORCE)
    set(GLFW_INSTALL
        OFF
        CACHE BOOL "" FORCE)
    FetchContent_Declare(
        glfw
        GIT_REPOSITORY "${GLFW_REPOSITORY}"
        GIT_TAG "${GLFW_REVISION}"
        GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(glfw)
endfunction()
