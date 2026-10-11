include_guard(GLOBAL)

# Build-time Program cooking for any CMake project that consumes VernonRuntime. Cooked Programs are target-specific (CPU
# objects are native code for one triple), so projects keep only DSL sources and cook into their binary tree.
#
# vernon_configure_program_cooker(PYTHON <interpreter> [PYTHON_PATH <dirs>...] [DEPENDS <files or targets>...])
#
# Selects the Python environment whose vernon_dsl cooks every asset. PYTHON_PATH entries precede each asset's own and
# replace any inherited PYTHONPATH, so cooks do not depend on the calling shell. DEPENDS re-cooks every asset when the
# cooker itself changes.
#
# vernon_add_program_asset(<name> SOURCE <file.py:declaration> TARGET <cooker target> OUTPUT_DIRECTORY <dir>
# [COOK_OPTIONS <cooker arguments>...] [PYTHON_PATH <dirs>...] [DEPENDS <files>...] [MANIFEST_VARIABLE <variable>])
#
# Cooks into OUTPUT_DIRECTORY, which then holds <directory name>.program.json and its artifacts, and creates the
# INTERFACE library <name>. Linking <name> orders the consumer after the cook; for a CPU Program it also links the
# cooked objects, which register their entries from static constructors.

if(EMSCRIPTEN AND NOT DEFINED CMAKE_LINK_LIBRARY_USING_WHOLE_ARCHIVE)
    set(CMAKE_LINK_LIBRARY_USING_WHOLE_ARCHIVE
        "LINKER:--whole-archive" "<LINK_ITEM>" "LINKER:--no-whole-archive"
        CACHE INTERNAL "wasm-ld whole-archive link feature")
    set(CMAKE_LINK_LIBRARY_USING_WHOLE_ARCHIVE_SUPPORTED
        TRUE
        CACHE INTERNAL "")
endif()

function(vernon_configure_program_cooker)
    cmake_parse_arguments(
        COOKER
        ""
        "PYTHON"
        "PYTHON_PATH;DEPENDS"
        ${ARGN})
    if(COOKER_UNPARSED_ARGUMENTS OR NOT COOKER_PYTHON)
        message(FATAL_ERROR "vernon_configure_program_cooker requires PYTHON")
    endif()
    set_property(GLOBAL PROPERTY VERNON_PROGRAM_COOK_PYTHON "${COOKER_PYTHON}")
    set_property(GLOBAL PROPERTY VERNON_PROGRAM_COOK_PYTHON_PATH "${COOKER_PYTHON_PATH}")
    set_property(GLOBAL PROPERTY VERNON_PROGRAM_COOK_DEPENDS "${COOKER_DEPENDS}")
endfunction()

function(vernon_add_program_asset name)
    cmake_parse_arguments(
        ASSET
        ""
        "SOURCE;TARGET;OUTPUT_DIRECTORY;MANIFEST_VARIABLE"
        "COOK_OPTIONS;PYTHON_PATH;DEPENDS"
        ${ARGN})
    if(ASSET_UNPARSED_ARGUMENTS
       OR NOT ASSET_SOURCE
       OR NOT ASSET_TARGET
       OR NOT ASSET_OUTPUT_DIRECTORY)
        message(FATAL_ERROR "vernon_add_program_asset(${name}) requires SOURCE, TARGET and OUTPUT_DIRECTORY")
    endif()
    get_property(python GLOBAL PROPERTY VERNON_PROGRAM_COOK_PYTHON)
    if(NOT python)
        message(FATAL_ERROR "Call vernon_configure_program_cooker before vernon_add_program_asset(${name})")
    endif()
    set(archive_script "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/VernonArchiveCpuProgram.cmake")
    get_property(configured_python_path GLOBAL PROPERTY VERNON_PROGRAM_COOK_PYTHON_PATH)
    get_property(cooker_dependencies GLOBAL PROPERTY VERNON_PROGRAM_COOK_DEPENDS)

    if(NOT
       ASSET_SOURCE
       MATCHES
       "^(.+\\.py):([A-Za-z_][A-Za-z0-9_]*)$")
        message(FATAL_ERROR "vernon_add_program_asset(${name}) SOURCE must be <file.py>:<declaration>")
    endif()
    set(declaration "${CMAKE_MATCH_2}")
    get_filename_component(source_file "${CMAKE_MATCH_1}" ABSOLUTE)
    set(source_reference "${source_file}:${declaration}")
    get_filename_component(
        output_directory
        "${ASSET_OUTPUT_DIRECTORY}"
        ABSOLUTE
        BASE_DIR
        "${CMAKE_CURRENT_BINARY_DIR}")
    get_filename_component(output_name "${output_directory}" NAME)
    set(manifest "${output_directory}/${output_name}.program.json")

    set(python_path ${ASSET_PYTHON_PATH} ${configured_python_path})
    if(python_path)
        if(CMAKE_HOST_WIN32)
            list(
                JOIN
                python_path
                "$<SEMICOLON>"
                python_path)
        else()
            list(
                JOIN
                python_path
                ":"
                python_path)
        endif()
        set(environment "PYTHONPATH=${python_path}")
    else()
        set(environment "--unset=PYTHONPATH")
    endif()

    set(outputs "${manifest}")
    set(archive_command)
    if(ASSET_TARGET STREQUAL "cpu")
        set(archive
            "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}${name}_program_objects${CMAKE_STATIC_LIBRARY_SUFFIX}"
        )
        if(MSVC)
            set(archiver_flavor msvc)
        else()
            set(archiver_flavor ar)
        endif()
        list(APPEND outputs "${archive}")
        set(archive_command COMMAND "${CMAKE_COMMAND}" "-DMANIFEST=${manifest}" "-DARCHIVE=${archive}"
                            "-DARCHIVER=${CMAKE_AR}" "-DARCHIVER_FLAVOR=${archiver_flavor}" -P "${archive_script}")
    endif()

    add_custom_command(
        OUTPUT ${outputs}
        COMMAND "${CMAKE_COMMAND}" -E rm -rf "${output_directory}"
        COMMAND
            "${CMAKE_COMMAND}" -E env "${environment}" "${python}" -m vernon_dsl.program_asset_cli "${source_reference}"
            --target "${ASSET_TARGET}" ${ASSET_COOK_OPTIONS} --output "${output_directory}" ${archive_command}
        DEPENDS "${source_file}"
                ${ASSET_DEPENDS}
                ${cooker_dependencies}
                "${archive_script}"
        COMMENT "Cooking ${ASSET_TARGET} Program ${ASSET_SOURCE}"
        VERBATIM)
    add_custom_target(${name}-cook DEPENDS ${outputs})
    add_library(${name} INTERFACE)
    add_dependencies(${name} ${name}-cook)
    if(ASSET_TARGET STREQUAL "cpu")
        if(CMAKE_VERSION VERSION_LESS 3.24)
            message(FATAL_ERROR "Linking CPU Program ${name} requires CMake 3.24 for the WHOLE_ARCHIVE link feature")
        endif()
        # Nothing references the objects' registration constructors, so a plain archive link would drop every member.
        target_link_libraries(${name} INTERFACE "$<LINK_LIBRARY:WHOLE_ARCHIVE,${archive}>")
    endif()
    if(ASSET_MANIFEST_VARIABLE)
        set(${ASSET_MANIFEST_VARIABLE}
            "${manifest}"
            PARENT_SCOPE)
    endif()
endfunction()
