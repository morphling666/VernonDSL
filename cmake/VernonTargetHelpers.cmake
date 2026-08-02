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
        add_custom_target(
            ${_vernon_stage_target} ALL
            COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${destination_target}>
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${_vernon_stage_files}
                    $<TARGET_FILE_DIR:${destination_target}>
            DEPENDS ${_vernon_stage_files}
            VERBATIM)
        add_dependencies(${destination_target} ${_vernon_stage_target})

        # Retain the post-build copy for generators that build only the destination target and do not revisit already
        # up-to-date ALL targets.
        add_custom_command(
            TARGET ${destination_target}
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${_vernon_stage_files}
                    $<TARGET_FILE_DIR:${destination_target}>
            VERBATIM)
    endforeach()
endfunction()
