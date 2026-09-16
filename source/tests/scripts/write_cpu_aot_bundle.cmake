include("${CMAKE_CURRENT_LIST_DIR}/../../../cmake/VernonVersions.cmake")

if(NOT DEFINED ARTIFACT
   OR NOT DEFINED OUTPUT
   OR NOT DEFINED OPERATING_SYSTEM
   OR NOT DEFINED ARCHITECTURE)
    message(FATAL_ERROR "CPU AOT bundle generator arguments are incomplete")
endif()

if(OPERATING_SYSTEM STREQUAL "windows")
    set(TARGET_TRIPLE "${ARCHITECTURE}-pc-windows-msvc")
    set(OBJECT_FORMAT "coff")
elseif(OPERATING_SYSTEM STREQUAL "macos" OR OPERATING_SYSTEM STREQUAL "darwin")
    set(TARGET_TRIPLE "${ARCHITECTURE}-apple-darwin")
    set(OBJECT_FORMAT "macho")
else()
    set(TARGET_TRIPLE "${ARCHITECTURE}-unknown-linux-gnu")
    set(OBJECT_FORMAT "elf")
endif()

file(MAKE_DIRECTORY "${OUTPUT}")
get_filename_component(ARTIFACT_NAME "${ARTIFACT}" NAME)
file(
    COPY_FILE
    "${ARTIFACT}"
    "${OUTPUT}/${ARTIFACT_NAME}"
    ONLY_IF_DIFFERENT)
file(SIZE "${ARTIFACT}" ARTIFACT_SIZE)
file(SHA256 "${ARTIFACT}" ARTIFACT_SHA256)

# This fixture intentionally exercises platform mapping in the canonical type:program envelope. CPU object files remain
# external AOT inputs.
set(MANIFEST
    "{\"type\":\"program\",\"program_version\":${VERNON_PROGRAM_VERSION},\
\"target\":{\"kind\":\"cpu\",\"options\":{\"triple\":\"${TARGET_TRIPLE}\"}},\
\"blobs\":{\"${ARTIFACT_SHA256}\":{\"byte_length\":${ARTIFACT_SIZE},\
\"location\":{\"tag\":\"external\",\"uri\":\"${ARTIFACT_NAME}\"},\"sha256\":\"${ARTIFACT_SHA256}\"}},\
\"variants\":[{\"key\":[],\"program\":{},\"artifact_system\":{\
\"object_format\":\"${OBJECT_FORMAT}\",\"artifacts\":{}}}]}")
file(WRITE "${OUTPUT}/cpu_fill.program.json" "${MANIFEST}\n")
