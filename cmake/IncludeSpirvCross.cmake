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
    FetchContent_Declare(
        spirv_cross
        GIT_REPOSITORY "${SPIRV_CROSS_REPOSITORY}"
        GIT_TAG "${SPIRV_CROSS_REVISION}"
        GIT_SHALLOW TRUE)
    FetchContent_GetProperties(spirv_cross)
    if(spirv_cross_POPULATED)
        return()
    endif()

    if(POLICY CMP0169)
        cmake_policy(PUSH)
        cmake_policy(SET CMP0169 OLD)
    endif()
    FetchContent_Populate(spirv_cross)
    if(POLICY CMP0169)
        cmake_policy(POP)
    endif()
    # SPIRV-Cross is statically linked into the compiler. Excluding its directory keeps unrelated install rules out of
    # the Vernon SDK.
    add_subdirectory("${spirv_cross_SOURCE_DIR}" "${spirv_cross_BINARY_DIR}" EXCLUDE_FROM_ALL)
endfunction()
