// RUN: %vernon-opt \
// RUN:   --vernon-lower-accumulation="f32-device-atomic=native supports-workgroup-reduction=true" \
// RUN:   --vernon-to-gpu %s | %FileCheck %s --check-prefix=GENERIC
// RUN: %vernon-opt \
// RUN:   --vernon-lower-accumulation="f32-device-atomic=native supports-workgroup-reduction=true" \
// RUN:   --vernon-to-gpu="use-spirv-storage=true use-spirv-workgroup-reduction=true" %s \
// RUN:   | %FileCheck %s --check-prefix=SPIRV
//
// GENERIC-LABEL: gpu.func @reduce(
// GENERIC-COUNT-2: gpu.all_reduce add {{.*}} uniform
// GENERIC: memref.atomic_rmw addf
// GENERIC-NOT: vernon.reduce_sum
//
// SPIRV-COUNT-1: spirv.GlobalVariable @__vernon_workgroup_reduce_reduce_f32
// SPIRV-LABEL: gpu.func @reduce(
// SPIRV: gpu.thread_id x
// SPIRV: gpu.thread_id y
// SPIRV: gpu.thread_id z
// SPIRV: arith.constant 2 : index
// SPIRV: arith.constant 3 : index
// SPIRV: scf.if
// SPIRV: gpu.barrier
// SPIRV-NOT: gpu.all_reduce
// SPIRV-NOT: vernon.reduce_sum

module attributes {vernon.compiler_contract_version = 13 : i64, vernon.pipeline_version = 17 : i64} {
  func.func @reduce(
      %gradient: !vernon.tensor_view<f32, [1], "read_write", "device"> {
        vernon.interface = "resource",
        vernon.set = 0 : i64,
        vernon.binding = 0 : i64
      }) attributes {
        vernon.entry,
        vernon.stage = "compute",
        vernon.workgroup_size = array<i32: 3, 1, 1>
      } {
    %index = arith.constant 0 : index
    %value = arith.constant 1.0 : f32
    "vernon.reduce_sum"(%value, %gradient, %index) {
      deterministic = false
    } : (f32, !vernon.tensor_view<f32, [1], "read_write", "device">, index) -> ()
    "vernon.reduce_sum"(%value, %gradient, %index) {
      deterministic = false
    } : (f32, !vernon.tensor_view<f32, [1], "read_write", "device">, index) -> ()
    return
  }
}
