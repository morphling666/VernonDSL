if(NOT DEFINED GENERATOR OR NOT DEFINED TEST_OUTPUT_DIRECTORY)
    message(FATAL_ERROR "CPU AOT bundle platform mapping test arguments are incomplete")
endif()

file(REMOVE_RECURSE "${TEST_OUTPUT_DIRECTORY}")

function(
    check_platform_mapping
    operating_system
    expected_triple
    expected_format)
    set(output_directory "${TEST_OUTPUT_DIRECTORY}/${operating_system}")
    execute_process(
        COMMAND "${CMAKE_COMMAND}" "-DARTIFACT=${CMAKE_CURRENT_LIST_FILE}" "-DOUTPUT=${output_directory}"
                "-DOPERATING_SYSTEM=${operating_system}" "-DARCHITECTURE=x86_64" -P "${GENERATOR}"
        RESULT_VARIABLE generator_result
        OUTPUT_VARIABLE generator_stdout
        ERROR_VARIABLE generator_stderr)
    if(NOT
       generator_result
       EQUAL
       0)
        message(
            FATAL_ERROR
                "Generating the ${operating_system} CPU AOT test bundle failed:\n${generator_stdout}${generator_stderr}"
        )
    endif()

    file(READ "${output_directory}/cpu_fill.program.json" manifest)
    string(
        JSON
        envelope_type
        GET
        "${manifest}"
        type)
    string(
        JSON
        target_kind
        GET
        "${manifest}"
        target
        kind)
    string(
        JSON
        actual_triple
        GET
        "${manifest}"
        target
        options
        triple)
    string(
        JSON
        actual_format
        GET
        "${manifest}"
        variants
        0
        artifact_system
        object_format)
    if(NOT
       envelope_type
       STREQUAL
       "program"
       OR NOT
          target_kind
          STREQUAL
          "cpu")
        message(FATAL_ERROR "CPU AOT mapping fixture is not an exact type:program CPU envelope")
    endif()
    if(NOT
       actual_triple
       STREQUAL
       expected_triple)
        message(FATAL_ERROR "${operating_system} mapped to '${actual_triple}', expected '${expected_triple}'")
    endif()
    if(NOT
       actual_format
       STREQUAL
       expected_format)
        message(FATAL_ERROR "${operating_system} mapped to '${actual_format}', expected '${expected_format}'")
    endif()
endfunction()

check_platform_mapping(windows x86_64-pc-windows-msvc coff)
check_platform_mapping(macos x86_64-apple-darwin macho)
check_platform_mapping(linux x86_64-unknown-linux-gnu elf)

file(REMOVE_RECURSE "${TEST_OUTPUT_DIRECTORY}")
