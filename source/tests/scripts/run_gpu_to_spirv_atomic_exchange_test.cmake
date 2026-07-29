if(NOT EXISTS "${VERNON_OPT}")
    message(FATAL_ERROR "VERNON_OPT must name the vernon-opt executable")
endif()
if(NOT EXISTS "${INPUT}")
    message(FATAL_ERROR "INPUT must name the pass-level MLIR fixture")
endif()

execute_process(
    COMMAND "${VERNON_OPT}" --vernon-convert-gpu-to-spirv "${INPUT}"
    RESULT_VARIABLE result
    OUTPUT_VARIABLE output
    ERROR_VARIABLE error)
if(NOT
   result
   EQUAL
   0)
    message(FATAL_ERROR "standalone GPU-to-SPIR-V pass failed:\n${error}")
endif()

string(FIND "${output}" "spirv.AtomicExchange <Workgroup> <AcquireRelease>" exchange_position)
if(exchange_position EQUAL -1)
    message(FATAL_ERROR "standalone pass did not produce the expected SPIR-V atomic exchange:\n${output}")
endif()
