# Packs the relocatable objects referenced by a cooked CPU Program manifest into one archive. Object names are content
# hashes known only after cooking, so build rules link this fixed-name archive instead.
#
# Inputs: MANIFEST, ARCHIVE, ARCHIVER, ARCHIVER_FLAVOR (msvc or ar).
foreach(
    input
    MANIFEST
    ARCHIVE
    ARCHIVER
    ARCHIVER_FLAVOR)
    if(NOT ${input})
        message(FATAL_ERROR "VernonArchiveCpuProgram.cmake requires -D${input}")
    endif()
endforeach()

get_filename_component(asset_dir "${MANIFEST}" DIRECTORY)
file(READ "${MANIFEST}" document)
string(
    JSON
    blob_count
    ERROR_VARIABLE
    blob_error
    LENGTH "${document}" blobs)
if(blob_error OR blob_count EQUAL 0)
    message(FATAL_ERROR "CPU Program manifest has no blobs: ${MANIFEST}")
endif()

set(objects)
math(EXPR last_blob_index "${blob_count} - 1")
foreach(blob_index RANGE 0 ${last_blob_index})
    string(
        JSON
        blob_id
        MEMBER
        "${document}"
        blobs
        ${blob_index})
    string(
        JSON
        blob_uri
        ERROR_VARIABLE
        uri_error
        GET
        "${document}"
        blobs
        "${blob_id}"
        location
        uri)
    if(NOT uri_error AND blob_uri MATCHES "\\.(o|obj)$")
        list(APPEND objects "${asset_dir}/${blob_uri}")
    endif()
endforeach()
if(NOT objects)
    message(FATAL_ERROR "CPU Program manifest has no relocatable objects: ${MANIFEST}")
endif()

file(REMOVE "${ARCHIVE}")
if(ARCHIVER_FLAVOR STREQUAL "msvc")
    execute_process(COMMAND "${ARCHIVER}" /NOLOGO "/OUT:${ARCHIVE}" ${objects} RESULT_VARIABLE archive_result)
else()
    execute_process(COMMAND "${ARCHIVER}" rcs "${ARCHIVE}" ${objects} RESULT_VARIABLE archive_result)
endif()
if(NOT
   archive_result
   EQUAL
   0)
    message(FATAL_ERROR "Archiving CPU Program objects failed (${archive_result}): ${MANIFEST}")
endif()
