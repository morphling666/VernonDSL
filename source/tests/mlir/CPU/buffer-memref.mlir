// RUN: %vernon-opt \
// RUN:   '--pass-pipeline=builtin.module(vernon-materialize-storage-projections,vernon-lower-cpu-resources)' \
// RUN:   %s | %FileCheck %s
//
// CHECK: memref<?xf32>
// CHECK: memref.load
// CHECK: memref.store
// CHECK-NOT: vernon.intrinsic
// CHECK-NOT: !vernon.buffer

module attributes {vernon.compiler_contract_version = 14 : i64, vernon.pipeline_version = 18 : i64} {
  func.func @tensor_view_round_trip(
      %view: !vernon.tensor_view<f32, [-1], "read_write", "device">,
      %index: index, %value: f32) {
    %loaded = "vernon.load"(%view, %index) :
      (!vernon.tensor_view<f32, [-1], "read_write", "device">, index) -> f32
    "vernon.store"(%value, %view, %index) :
      (f32, !vernon.tensor_view<f32, [-1], "read_write", "device">, index) -> ()
    return
  }
}
