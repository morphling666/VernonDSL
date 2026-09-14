include_guard(GLOBAL)

function(
    vernon_add_program_asset_bundle
    name
    architecture
    asset_reference
    output_directory
    output_stem
    output_variable)
    set(_vernon_bundle_directory "${output_directory}/${output_stem}")
    set(_vernon_bundle_manifest "${_vernon_bundle_directory}/${output_stem}.program.json")
    string(
        REGEX
        REPLACE ":[^:]+$"
                ""
                _vernon_asset_source
                "${asset_reference}")
    add_custom_command(
        OUTPUT "${_vernon_bundle_manifest}"
        COMMAND ${CMAKE_COMMAND} -E rm -rf "${_vernon_bundle_directory}"
        COMMAND
            ${CMAKE_COMMAND} -E env "PYTHONPATH=${VERNON_REPOSITORY_ROOT}/python" ${Python_EXECUTABLE} -m
            vernon_dsl.program_asset_cli "${asset_reference}" --target "${architecture}" --output
            "${_vernon_bundle_directory}"
        DEPENDS vernon-dsl-native
                "$<TARGET_FILE:vernon-dsl-native>"
                "${_vernon_asset_source}"
                ${VERNON_PROGRAM_FIXTURE_PYTHON_SOURCES}
        VERBATIM)
    add_custom_target("vernon-${name}-bundle" DEPENDS "${_vernon_bundle_manifest}")
    set(${output_variable}
        "${_vernon_bundle_manifest}"
        PARENT_SCOPE)
endfunction()

function(
    vernon_add_cpu_program_fixture_library
    name
    output_directory
    manifest_name
    asset_reference
    wrapper_function
    prepare_script
    manifest_variable
    target_variable)
    cmake_parse_arguments(
        FIXTURE
        ""
        "COOK_SCRIPT"
        ""
        ${ARGN})
    if(FIXTURE_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "unknown CPU Program fixture arguments: ${FIXTURE_UNPARSED_ARGUMENTS}")
    endif()
    set(_vernon_manifest "${output_directory}/${manifest_name}")
    set(_vernon_registration "${output_directory}/program_registration.c")
    set(_vernon_wrapper "${output_directory}/program_registration_wrapper.c")
    set(_vernon_artifact_archive "${output_directory}/program_artifacts.a")
    string(
        REGEX
        REPLACE ":[^:]+$"
                ""
                _vernon_asset_source
                "${asset_reference}")
    if(FIXTURE_COOK_SCRIPT)
        set(_vernon_cook_command ${Python_EXECUTABLE} "${FIXTURE_COOK_SCRIPT}" "${asset_reference}"
                                 "${output_directory}")
    else()
        set(_vernon_cook_command ${Python_EXECUTABLE} -m vernon_dsl.program_asset_cli "${asset_reference}" --target cpu
                                 --output "${output_directory}")
    endif()
    add_custom_command(
        OUTPUT "${_vernon_manifest}"
               "${_vernon_registration}"
               "${_vernon_wrapper}"
               "${_vernon_artifact_archive}"
        COMMAND ${CMAKE_COMMAND} -E rm -rf "${output_directory}"
        COMMAND ${CMAKE_COMMAND} -E env "PYTHONPATH=${VERNON_REPOSITORY_ROOT}/python" ${_vernon_cook_command}
        COMMAND ${Python_EXECUTABLE} "${prepare_script}" "${_vernon_manifest}" "${CMAKE_C_OUTPUT_EXTENSION}"
                --wrapper-function "${wrapper_function}" --archiver "${CMAKE_AR}"
        DEPENDS vernon-dsl-native
                "$<TARGET_FILE:vernon-dsl-native>"
                "${_vernon_asset_source}"
                ${FIXTURE_COOK_SCRIPT}
                ${VERNON_PROGRAM_FIXTURE_PYTHON_SOURCES}
                "${prepare_script}"
        VERBATIM)
    set_source_files_properties("${_vernon_registration}" "${_vernon_wrapper}" PROPERTIES GENERATED TRUE)
    set(_vernon_target "vernon-${name}-fixture")
    add_library(${_vernon_target} STATIC "${_vernon_registration}" "${_vernon_wrapper}")
    add_custom_target(${_vernon_target}-artifacts DEPENDS "${_vernon_artifact_archive}")
    add_dependencies(${_vernon_target} ${_vernon_target}-artifacts)
    target_link_libraries(${_vernon_target} PUBLIC "${_vernon_artifact_archive}")
    set_target_properties(${_vernon_target} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_include_directories(${_vernon_target} PRIVATE "${VERNON_REPOSITORY_ROOT}/source/include")
    set(${manifest_variable}
        "${_vernon_manifest}"
        PARENT_SCOPE)
    set(${target_variable}
        "${_vernon_target}"
        PARENT_SCOPE)
endfunction()
