set(SPIRV_CROSS_REPOSITORY
    "https://github.com/KhronosGroup/SPIRV-Cross.git"
    CACHE STRING "SPIRV-Cross repository to fetch")
set(SPIRV_CROSS_REVISION
    "vulkan-sdk-1.4.350.1"
    CACHE STRING "SPIRV-Cross dependency revision")

function(include_spirv_cross)
    include(FetchContent)
    set(SPIRV_CROSS_CLI
        OFF
        CACHE BOOL "" FORCE)
    set(SPIRV_CROSS_ENABLE_TESTS
        OFF
        CACHE BOOL "" FORCE)
    set(SPIRV_CROSS_ENABLE_C_API
        OFF
        CACHE BOOL "" FORCE)
    set(SPIRV_CROSS_SKIP_INSTALL
        ON
        CACHE BOOL "Do not install embedded SPIRV-Cross targets" FORCE)
    FetchContent_Declare(
        spirv_cross
        GIT_REPOSITORY "${SPIRV_CROSS_REPOSITORY}"
        GIT_TAG "${SPIRV_CROSS_REVISION}"
        GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(spirv_cross)
endfunction()
