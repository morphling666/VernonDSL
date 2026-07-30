if(NOT EXISTS "${VERNON_OPT}")
    message(FATAL_ERROR "VERNON_OPT must name the vernon-opt executable")
endif()
if(NOT EXISTS "${INPUT}")
    message(FATAL_ERROR "INPUT must name the pass-level MLIR fixture")
endif()

execute_process(
    COMMAND "${VERNON_OPT}" --vernon-materialize-storage-projections --vernon-to-gpu "${INPUT}"
    RESULT_VARIABLE result
    OUTPUT_VARIABLE output
    ERROR_VARIABLE error)
if(NOT
   result
   EQUAL
   0)
    message(FATAL_ERROR "standalone TensorView-to-GPU conversion failed:\n${error}")
endif()

string(FIND "${output}" "gpu.module @vernon_kernels" gpu_module_position)
if(gpu_module_position EQUAL -1)
    message(FATAL_ERROR "TensorView-to-GPU conversion did not outline a GPU module:\n${output}")
endif()
string(
    SUBSTRING "${output}"
              ${gpu_module_position}
              -1
              gpu_output)

foreach(forbidden IN ITEMS "!vernon.tensor_view" "vernon.physical_" "unrealized_conversion_cast")
    string(FIND "${gpu_output}" "${forbidden}" position)
    if(NOT
       position
       EQUAL
       -1)
        message(FATAL_ERROR "outlined GPU module retained '${forbidden}':\n${gpu_output}")
    endif()
endforeach()

string(
    REGEX MATCH
          "gpu.func @storage_lowering\\([^\n]*\\) kernel"
          gpu_signature
          "${gpu_output}")
string(
    REGEX MATCHALL
          "memref<[^>]*>"
          storage_arguments
          "${gpu_signature}")
list(LENGTH storage_arguments storage_argument_count)
if(NOT
   storage_argument_count
   EQUAL
   3)
    message(FATAL_ERROR "aggregate TensorView argument was not expanded to three storage leaves:\n${output}")
endif()
foreach(required IN ITEMS "memref.load" "memref.store" "memref.atomic_rmw addi")
    string(FIND "${gpu_output}" "${required}" position)
    if(position EQUAL -1)
        message(FATAL_ERROR "TensorView-to-GPU conversion did not produce '${required}':\n${output}")
    endif()
endforeach()
