foreach(
    required_variable IN
    ITEMS VERNON_BUILD_DIR
          VERNON_SMOKE_SOURCE_DIR
          VERNON_SMOKE_BUILD_DIR
          VERNON_INSTALL_PREFIX)
    if(NOT DEFINED ${required_variable} OR "${${required_variable}}" STREQUAL "")
        message(FATAL_ERROR "${required_variable} is required")
    endif()
endforeach()

file(REMOVE_RECURSE "${VERNON_SMOKE_BUILD_DIR}" "${VERNON_INSTALL_PREFIX}")

foreach(_vernon_install_component IN ITEMS VernonWheel VernonDevelopment)
    set(_vernon_install_command "${CMAKE_COMMAND}" --install "${VERNON_BUILD_DIR}" --prefix "${VERNON_INSTALL_PREFIX}"
                                --component "${_vernon_install_component}")
    if(VERNON_TEST_CONFIG)
        list(
            APPEND
            _vernon_install_command
            --config
            "${VERNON_TEST_CONFIG}")
    endif()
    execute_process(COMMAND ${_vernon_install_command} COMMAND_ERROR_IS_FATAL ANY)
endforeach()

set(_vernon_configure_command "${CMAKE_COMMAND}" -S "${VERNON_SMOKE_SOURCE_DIR}" -B "${VERNON_SMOKE_BUILD_DIR}"
                              "-DCMAKE_PREFIX_PATH=${VERNON_INSTALL_PREFIX}")
if(VERNON_TEST_GENERATOR)
    list(
        APPEND
        _vernon_configure_command
        -G
        "${VERNON_TEST_GENERATOR}")
endif()
if(VERNON_TEST_PLATFORM)
    list(
        APPEND
        _vernon_configure_command
        -A
        "${VERNON_TEST_PLATFORM}")
endif()
execute_process(COMMAND ${_vernon_configure_command} COMMAND_ERROR_IS_FATAL ANY)

set(_vernon_build_command "${CMAKE_COMMAND}" --build "${VERNON_SMOKE_BUILD_DIR}")
if(VERNON_TEST_CONFIG)
    list(
        APPEND
        _vernon_build_command
        --config
        "${VERNON_TEST_CONFIG}")
endif()
execute_process(COMMAND ${_vernon_build_command} COMMAND_ERROR_IS_FATAL ANY)
