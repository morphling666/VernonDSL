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
        add_custom_command(
            TARGET ${destination_target}
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:${dependency_target}>
                    $<TARGET_FILE_DIR:${destination_target}>
            VERBATIM)
        get_target_property(_vernon_runtime_files ${dependency_target} VERNON_RUNTIME_FILES)
        if(_vernon_runtime_files)
            foreach(_vernon_runtime_file IN LISTS _vernon_runtime_files)
                add_custom_command(
                    TARGET ${destination_target}
                    POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_vernon_runtime_file}"
                            $<TARGET_FILE_DIR:${destination_target}>
                    VERBATIM)
            endforeach()
        endif()
    endforeach()
endfunction()
