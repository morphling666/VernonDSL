include_guard(GLOBAL)

function(vernon_prepare_python_venv output_variable)
    find_program(VERNON_UV_EXECUTABLE uv REQUIRED)
    set(VERNON_VENV_DIR
        "${VERNON_REPOSITORY_ROOT}/.venv"
        CACHE PATH "VernonDSL virtual environment")
    execute_process(
        COMMAND "${VERNON_UV_EXECUTABLE}" sync --extra build --frozen
        WORKING_DIRECTORY "${VERNON_REPOSITORY_ROOT}"
        RESULT_VARIABLE _vernon_uv_sync_result COMMAND_ERROR_IS_FATAL ANY)
    if(NOT
       _vernon_uv_sync_result
       EQUAL
       0)
        message(FATAL_ERROR "uv sync failed with exit code ${_vernon_uv_sync_result}")
    endif()
    if(WIN32)
        set(_vernon_venv_python "${VERNON_VENV_DIR}/Scripts/python.exe")
    else()
        set(_vernon_venv_python "${VERNON_VENV_DIR}/bin/python")
    endif()
    if(NOT EXISTS "${_vernon_venv_python}")
        message(FATAL_ERROR "uv sync did not create the expected Python executable: ${_vernon_venv_python}")
    endif()
    set(${output_variable}
        "${_vernon_venv_python}"
        PARENT_SCOPE)
endfunction()
