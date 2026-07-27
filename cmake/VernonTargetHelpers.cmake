include_guard(GLOBAL)

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
    endforeach()
endfunction()
