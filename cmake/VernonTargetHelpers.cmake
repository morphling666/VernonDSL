include_guard(GLOBAL)

function(vernon_register_runtime_files target)
    if(NOT TARGET ${target})
        message(FATAL_ERROR "Cannot register runtime files for unknown target '${target}'")
    endif()
    set_property(
        TARGET ${target}
        APPEND
        PROPERTY VERNON_RUNTIME_FILES ${ARGN})
endfunction()

function(vernon_stage_target_files destination_target)
    if(NOT WIN32)
        return()
    endif()
    if(NOT TARGET ${destination_target})
        message(FATAL_ERROR "Cannot stage runtime files for unknown target '${destination_target}'")
    endif()
    foreach(dependency_target IN LISTS ARGN)
        if(NOT TARGET ${dependency_target})
            message(FATAL_ERROR "Cannot stage unknown dependency target '${dependency_target}'")
        endif()

        string(MAKE_C_IDENTIFIER "${destination_target}_${dependency_target}_runtime_files" _vernon_stage_target)
        set(_vernon_stage_files "$<TARGET_FILE:${dependency_target}>")
        get_target_property(_vernon_runtime_files ${dependency_target} VERNON_RUNTIME_FILES)
        if(_vernon_runtime_files)
            list(APPEND _vernon_stage_files ${_vernon_runtime_files})
        endif()
        set(_vernon_stage_commands)
        foreach(_vernon_stage_file IN LISTS _vernon_stage_files)
            list(
                APPEND
                _vernon_stage_commands
                COMMAND
                ${CMAKE_COMMAND}
                "-DVERNON_STAGE_SOURCE=${_vernon_stage_file}"
                "-DVERNON_STAGE_DESTINATION=$<TARGET_FILE_DIR:${destination_target}>"
                "-DVERNON_STAGE_LOCK_ROOT=${CMAKE_BINARY_DIR}/CMakeFiles/vernon-runtime-stage-locks"
                -P
                "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/VernonStageRuntimeFile.cmake")
        endforeach()
        add_custom_target(
            ${_vernon_stage_target} ALL
            ${_vernon_stage_commands}
            DEPENDS ${_vernon_stage_files}
            VERBATIM)
        add_dependencies(${destination_target} ${_vernon_stage_target})
    endforeach()
endfunction()
