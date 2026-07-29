find_package(Git REQUIRED)
find_program(VERNON_UV_EXECUTABLE uv REQUIRED)

execute_process(
    COMMAND "${GIT_EXECUTABLE}" rev-parse --git-common-dir
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    RESULT_VARIABLE _vernon_git_dir_result
    OUTPUT_VARIABLE _vernon_git_dir
    ERROR_VARIABLE _vernon_git_dir_error
    OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT
   _vernon_git_dir_result
   EQUAL
   0)
    message(FATAL_ERROR "Cannot locate the Git directory: ${_vernon_git_dir_error}")
endif()

get_filename_component(
    _vernon_git_dir
    "${_vernon_git_dir}"
    ABSOLUTE
    BASE_DIR
    "${CMAKE_SOURCE_DIR}")
file(TO_CMAKE_PATH "${VERNON_UV_EXECUTABLE}" _vernon_hook_uv)
file(TO_CMAKE_PATH "${CMAKE_SOURCE_DIR}" _vernon_hook_source)
set(_vernon_pre_commit_hook "${_vernon_git_dir}/hooks/pre-commit")
file(MAKE_DIRECTORY "${_vernon_git_dir}/hooks")
file(WRITE "${_vernon_pre_commit_hook}" "#!/bin/sh\n" "exec \"${_vernon_hook_uv}\" run --frozen --all-extras python "
                                        "\"${_vernon_hook_source}/scripts/fix_code_style.py\"\n")
file(
    CHMOD
    "${_vernon_pre_commit_hook}"
    PERMISSIONS
    OWNER_READ
    OWNER_WRITE
    OWNER_EXECUTE
    GROUP_READ
    GROUP_EXECUTE
    WORLD_READ
    WORLD_EXECUTE)
message(STATUS "Installed VernonDSL pre-commit formatting hook: ${_vernon_pre_commit_hook}")
