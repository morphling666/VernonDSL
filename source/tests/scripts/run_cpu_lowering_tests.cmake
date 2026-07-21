cmake_minimum_required(VERSION 3.20)

if(NOT DEFINED VERNON_OPT OR VERNON_OPT STREQUAL "")
  message(FATAL_ERROR "VERNON_OPT must name the vernon-opt executable")
endif()
if(NOT EXISTS "${VERNON_OPT}")
  message(FATAL_ERROR "vernon-opt does not exist: ${VERNON_OPT}")
endif()

if(NOT DEFINED TEST_ROOT OR TEST_ROOT STREQUAL "")
  set(TEST_ROOT "${CMAKE_CURRENT_LIST_DIR}/integration")
endif()

function(run_case name fixture pass_name expect_failure)
  set(input "${TEST_ROOT}/${fixture}")
  if(NOT EXISTS "${input}")
    message(FATAL_ERROR "${name}: fixture does not exist: ${input}")
  endif()

  execute_process(
    COMMAND "${VERNON_OPT}" "${input}" "${pass_name}"
    RESULT_VARIABLE result
    OUTPUT_VARIABLE stdout
    ERROR_VARIABLE stderr
  )
  set(output "${stdout}\n${stderr}")

  if(expect_failure)
    if(result EQUAL 0)
      message(FATAL_ERROR
        "${name}: expected vernon-opt to fail\nOutput:\n${output}")
    endif()
  elseif(NOT result EQUAL 0)
    message(FATAL_ERROR
      "${name}: vernon-opt failed with ${result}\nOutput:\n${output}")
  endif()

  set("${name}_OUTPUT" "${output}" PARENT_SCOPE)
endfunction()

function(require_text case_name text needle)
  string(FIND "${text}" "${needle}" position)
  if(position EQUAL -1)
    message(FATAL_ERROR
      "${case_name}: missing expected text '${needle}'\nOutput:\n${text}")
  endif()
endfunction()

function(reject_text case_name text needle)
  string(FIND "${text}" "${needle}" position)
  if(NOT position EQUAL -1)
    message(FATAL_ERROR
      "${case_name}: found forbidden text '${needle}'\nOutput:\n${text}")
  endif()
endfunction()

run_case(
  TENSOR
  "cpu-tensor-vector-scf.mlir"
  "--vernon-lower-cpu-tensors"
  FALSE
)
require_text("tensor/vector" "${TENSOR_OUTPUT}" "vector<3xf32>")
require_text("tensor/vector" "${TENSOR_OUTPUT}" "arith.addf")
require_text("SCF structural conversion" "${TENSOR_OUTPUT}" "scf.if")
require_text("SCF structural conversion" "${TENSOR_OUTPUT}" "scf.yield")
reject_text("tensor/vector" "${TENSOR_OUTPUT}" "tensor<3xf32>")
reject_text("struct declaration cleanup" "${TENSOR_OUTPUT}" "vernon.struct")

run_case(
  SWIZZLE
  "cpu-swizzle-aliases.mlir"
  "--vernon-lower-cpu-tensors"
  FALSE
)
require_text("swizzle aliases" "${SWIZZLE_OUTPUT}" "vector.extract")
require_text("alpha swizzle alias" "${SWIZZLE_OUTPUT}" "[3]")
reject_text("swizzle aliases" "${SWIZZLE_OUTPUT}" "vernon.swizzle")

run_case(
  BUFFER
  "cpu-buffer-memref.mlir"
  "--vernon-lower-cpu-resources"
  FALSE
)
require_text("buffer/memref" "${BUFFER_OUTPUT}" "memref<?xf32>")
require_text("buffer/memref" "${BUFFER_OUTPUT}" "memref.load")
require_text("buffer/memref" "${BUFFER_OUTPUT}" "memref.store")
reject_text("buffer/memref" "${BUFFER_OUTPUT}" "vernon.intrinsic")
reject_text("buffer/memref" "${BUFFER_OUTPUT}" "!vernon.buffer")

run_case(
  TEXTURE
  "cpu-texture-helper.mlir"
  "--vernon-lower-cpu-resources"
  FALSE
)
require_text(
  "texture helper"
  "${TEXTURE_OUTPUT}"
  "func.func private @__vernon_cpu_texture_sample"
)
require_text("texture helper" "${TEXTURE_OUTPUT}" "!llvm.ptr")
reject_text(
  "texture helper"
  "${TEXTURE_OUTPUT}"
  "vernon.cpu.requires_texture_callbacks"
)
require_text(
  "texture helper"
  "${TEXTURE_OUTPUT}"
  "call @__vernon_cpu_texture_sample"
)
reject_text("texture helper" "${TEXTURE_OUTPUT}" "!vernon.texture")
reject_text("texture helper" "${TEXTURE_OUTPUT}" "!vernon.sampler")

run_case(
  ILLEGAL
  "cpu-illegal-intrinsic.mlir"
  "--vernon-lower-cpu-tensors"
  TRUE
)
require_text(
  "illegal operation diagnostic"
  "${ILLEGAL_OUTPUT}"
  "unknown Vernon intrinsic 'unimplemented_cpu_operation'"
)
require_text(
  "illegal operation diagnostic"
  "${ILLEGAL_OUTPUT}"
  "in CPU entry 'unsupported_intrinsic'"
)

message(STATUS "All vernon-opt CPU lowering integration tests passed")
