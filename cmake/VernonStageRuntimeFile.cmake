if(NOT DEFINED VERNON_STAGE_SOURCE
   OR NOT DEFINED VERNON_STAGE_DESTINATION
   OR NOT DEFINED VERNON_STAGE_LOCK_ROOT)
    message(FATAL_ERROR "Runtime file staging requires source, destination, and lock root")
endif()

file(MAKE_DIRECTORY "${VERNON_STAGE_DESTINATION}" "${VERNON_STAGE_LOCK_ROOT}")
string(SHA256 _vernon_stage_destination_hash "${VERNON_STAGE_DESTINATION}")
file(
    LOCK "${VERNON_STAGE_LOCK_ROOT}/${_vernon_stage_destination_hash}.lock"
    GUARD PROCESS
    TIMEOUT 120
    RESULT_VARIABLE _vernon_stage_lock_result)
if(NOT
   _vernon_stage_lock_result
   EQUAL
   0)
    message(FATAL_ERROR "Cannot lock runtime staging destination: ${_vernon_stage_lock_result}")
endif()

get_filename_component(_vernon_stage_filename "${VERNON_STAGE_SOURCE}" NAME)
execute_process(
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${VERNON_STAGE_SOURCE}"
            "${VERNON_STAGE_DESTINATION}/${_vernon_stage_filename}"
    RESULT_VARIABLE _vernon_stage_copy_result
    ERROR_VARIABLE _vernon_stage_copy_error)
if(NOT
   _vernon_stage_copy_result
   EQUAL
   0)
    message(FATAL_ERROR "Cannot stage runtime file '${VERNON_STAGE_SOURCE}': ${_vernon_stage_copy_error}")
endif()
