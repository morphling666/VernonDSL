if(NOT EXISTS "${VERNON_OPT}")
    message(FATAL_ERROR "VERNON_OPT must name the vernon-opt executable")
endif()
if(NOT EXISTS "${INPUT}")
    message(FATAL_ERROR "accumulation lowering fixture is missing")
endif()

execute_process(
    COMMAND "${VERNON_OPT}" "--vernon-lower-accumulation=supports-atomic-f32=true" "${INPUT}"
    RESULT_VARIABLE result
    OUTPUT_VARIABLE output
    ERROR_VARIABLE error)
if(NOT
   result
   EQUAL
   0)
    message(FATAL_ERROR "accumulation lowering failed:\n${error}")
endif()

foreach(
    required IN
    ITEMS "vernon.physical_load"
          "vernon.physical_store"
          "vernon.physical_atomic"
          "arith.addf"
          "vernon.serial_dispatch"
          "vernon.workgroup_size = array<i32: 1, 1, 1>"
          "scf.for")
    string(FIND "${output}" "${required}" position)
    if(position EQUAL -1)
        message(FATAL_ERROR "accumulation lowering did not produce '${required}':\n${output}")
    endif()
endforeach()
foreach(forbidden IN ITEMS "vernon.reduce_sum" "vernon.scatter_add")
    string(FIND "${output}" "${forbidden}" position)
    if(NOT
       position
       EQUAL
       -1)
        message(FATAL_ERROR "accumulation lowering retained illegal semantic op '${forbidden}':\n${output}")
    endif()
endforeach()

string(
    REGEX MATCHALL
          "vernon.physical_load"
          serial_loads
          "${output}")
list(LENGTH serial_loads serial_load_count)
if(serial_load_count LESS 2)
    message(FATAL_ERROR "deterministic and unsupported-atomic paths did not both lower to load/add/store:\n${output}")
endif()
