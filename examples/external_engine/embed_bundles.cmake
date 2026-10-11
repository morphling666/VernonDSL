# Inputs: TEMPLATE, OUTPUT, CPU_MANIFEST, CPU_RUNTIME_DIR, GRAPHICS_MANIFEST, GRAPHICS_RUNTIME_DIR.

# MSVC rejects string literals longer than 16380 bytes, so manifests become adjacent raw-string chunks.
function(
    vernon_chunk_raw_string
    content
    delimiter
    output_variable)
    set(literal)
    set(first_chunk TRUE)
    string(LENGTH "${content}" remaining)
    while(remaining GREATER 0)
        if(remaining GREATER 12000)
            set(chunk_size 12000)
        else()
            set(chunk_size "${remaining}")
        endif()
        string(
            SUBSTRING "${content}"
                      0
                      "${chunk_size}"
                      chunk)
        if(NOT first_chunk)
            string(APPEND literal ")${delimiter}\"\n    R\"${delimiter}(")
        endif()
        string(APPEND literal "${chunk}")
        set(first_chunk FALSE)
        string(
            SUBSTRING "${content}"
                      "${chunk_size}"
                      -1
                      content)
        string(LENGTH "${content}" remaining)
    endwhile()
    set(${output_variable}
        "${literal}"
        PARENT_SCOPE)
endfunction()

file(READ "${CPU_MANIFEST}" _vernon_cpu_manifest)
file(READ "${GRAPHICS_MANIFEST}" _vernon_graphics_manifest)
vernon_chunk_raw_string("${_vernon_cpu_manifest}" vernon_cpu VERNON_EXTERNAL_ENGINE_CPU_MANIFEST_LITERAL)
vernon_chunk_raw_string("${_vernon_graphics_manifest}" vernon_graphics VERNON_EXTERNAL_ENGINE_GRAPHICS_MANIFEST_LITERAL)
set(VERNON_EXTERNAL_ENGINE_CPU_RUNTIME_DIR "${CPU_RUNTIME_DIR}")
set(VERNON_EXTERNAL_ENGINE_GRAPHICS_RUNTIME_DIR "${GRAPHICS_RUNTIME_DIR}")
configure_file("${TEMPLATE}" "${OUTPUT}" @ONLY)
