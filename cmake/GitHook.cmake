execute_process(
    COMMAND git --version
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    RESULT_VARIABLE GIT_VERSION_RESULT
    ERROR_VARIABLE GIT_VERSION_ERROR
    OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT
   GIT_VERSION_RESULT
   EQUAL
   0)
    message(FATAL_ERROR "Error running 'git --version': ${GIT_VERSION_ERROR}")
endif()

execute_process(
    COMMAND git rev-parse --git-common-dir
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    RESULT_VARIABLE GIT_DIR_RESULT
    OUTPUT_VARIABLE GIT_DIR
    ERROR_VARIABLE GIT_DIR_ERROR
    OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT
   GIT_DIR_RESULT
   EQUAL
   0)
    message(FATAL_ERROR "Cannot locate the Git directory: ${GIT_DIR_ERROR}")
endif()

get_filename_component(
    GIT_DIR
    "${GIT_DIR}"
    ABSOLUTE
    BASE_DIR
    "${CMAKE_SOURCE_DIR}")
include("${CMAKE_CURRENT_LIST_DIR}/VernonPythonVenv.cmake")
vernon_prepare_python_venv(FORMAT_PYTHON)

file(MAKE_DIRECTORY "${GIT_DIR}/hooks")
set(PRE_COMMIT_HOOK "${GIT_DIR}/hooks/pre-commit")
file(WRITE "${PRE_COMMIT_HOOK}" "#!/bin/sh\n"
                                "\"${FORMAT_PYTHON}\" \"${CMAKE_SOURCE_DIR}/scripts/fix_code_style.py\"\n")
file(
    CHMOD
    "${PRE_COMMIT_HOOK}"
    PERMISSIONS
    OWNER_READ
    OWNER_WRITE
    OWNER_EXECUTE
    GROUP_READ
    GROUP_EXECUTE
    WORLD_READ
    WORLD_EXECUTE)
message(STATUS "Installed VernonDSL pre-commit formatting hook: ${PRE_COMMIT_HOOK}")
