set(NLOHMANN_JSON_REPOSITORY
    "https://github.com/nlohmann/json.git"
    CACHE STRING "nlohmann/json repository to fetch")
set(NLOHMANN_JSON_REVISION
    "v3.12.0"
    CACHE STRING "nlohmann/json dependency revision")

function(include_nlohmann_json)
    find_package(nlohmann_json CONFIG QUIET)
    if(nlohmann_json_FOUND)
        return()
    endif()

    include(FetchContent)
    FetchContent_Declare(
        nlohmann_json
        GIT_REPOSITORY "${NLOHMANN_JSON_REPOSITORY}"
        GIT_TAG "${NLOHMANN_JSON_REVISION}"
        GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(nlohmann_json)
endfunction()
