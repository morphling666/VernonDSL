include_guard(GLOBAL)

set(NVTX_REPOSITORY
    "https://github.com/NVIDIA/NVTX.git"
    CACHE STRING "NVTX repository to fetch")
set(NVTX_REVISION
    "v3.6.0-c-cpp"
    CACHE STRING "NVTX dependency revision")

# NVTX asks libraries to include its imported-target script rather than add its project, which would also build its
# writer loader. SOURCE_SUBDIR names a directory without a CMakeLists.txt so that FetchContent only populates.
function(include_nvtx)
    include(FetchContent)
    FetchContent_Declare(
        nvtx
        GIT_REPOSITORY "${NVTX_REPOSITORY}"
        GIT_TAG "${NVTX_REVISION}"
        GIT_SHALLOW TRUE
        SOURCE_SUBDIR include)
    FetchContent_MakeAvailable(nvtx)
    include("${nvtx_SOURCE_DIR}/nvtxImportedTargets.cmake")
endfunction()
