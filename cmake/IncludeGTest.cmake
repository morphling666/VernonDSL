include_guard(GLOBAL)

set(GTEST_ARCHIVE_URL
    "https://github.com/google/googletest/archive/refs/tags/v1.17.0.tar.gz"
    CACHE STRING "GoogleTest archive to fetch")
set(GTEST_ARCHIVE_SHA256
    "65fab701d9829d38cb77c14acdc431d2108bfdbf8979e40eb8ae567edf10b27c"
    CACHE STRING "GoogleTest archive SHA-256")

function(include_gtest)
    find_package(GTest CONFIG QUIET)
    if(GTest_FOUND)
        return()
    endif()

    include(FetchContent)
    set(gtest_force_shared_crt
        ON
        CACHE BOOL "Use the shared MSVC runtime in GoogleTest" FORCE)
    FetchContent_Declare(
        googletest
        URL "${GTEST_ARCHIVE_URL}"
        URL_HASH "SHA256=${GTEST_ARCHIVE_SHA256}"
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    FetchContent_MakeAvailable(googletest)
endfunction()
