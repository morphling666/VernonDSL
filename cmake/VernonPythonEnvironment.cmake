include_guard(GLOBAL)

function(vernon_prepare_python_environment repository_root)
    option(VERNON_BOOTSTRAP_UV "Synchronize the development environment during configuration" ON)
    if(NOT VERNON_BOOTSTRAP_UV OR SKBUILD)
        return()
    endif()

    find_program(VERNON_UV_EXECUTABLE uv REQUIRED)
    execute_process(COMMAND ${VERNON_UV_EXECUTABLE} sync --extra build --extra examples --frozen
                    WORKING_DIRECTORY "${repository_root}" COMMAND_ERROR_IS_FATAL ANY)
    execute_process(
        COMMAND ${VERNON_UV_EXECUTABLE} run --frozen --no-sync python -c "import sys; print(sys.executable)"
        WORKING_DIRECTORY "${repository_root}"
        OUTPUT_VARIABLE _vernon_python_executable
        OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
    set(Python_EXECUTABLE
        "${_vernon_python_executable}"
        CACHE FILEPATH "Python interpreter" FORCE)
endfunction()
