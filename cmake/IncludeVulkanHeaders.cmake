set(VULKAN_HEADERS_REPOSITORY
    "https://github.com/KhronosGroup/Vulkan-Headers.git"
    CACHE STRING "Vulkan-Headers repository to fetch")
set(VULKAN_HEADERS_REVISION
    "v1.4.357"
    CACHE STRING "Vulkan-Headers dependency revision")

function(include_vulkan_headers)
    include(FetchContent)
    FetchContent_Declare(
        vulkan_headers
        GIT_REPOSITORY "${VULKAN_HEADERS_REPOSITORY}"
        GIT_TAG "${VULKAN_HEADERS_REVISION}"
        GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(vulkan_headers)
endfunction()
