// RUN: %vernon-opt \
// RUN:   '--pass-pipeline=builtin.module(vernon-materialize-storage-projections,vernon-lower-cpu-abi,vernon-lower-cpu-tensors,vernon-lower-cpu-resources)' \
// RUN:   %s | %FileCheck %s
//
// CHECK-LABEL: func.func private @tensor_view_round_trip(
// CHECK-SAME: memref<?xf32>
// CHECK-SAME: %{{.+}}: i64, %{{.+}}: i64, %{{.+}}: i64
// CHECK: memref.load
// CHECK: memref.store
// CHECK-NOT: vernon.intrinsic
// CHECK-NOT: vernon.tuple_get
// CHECK-NOT: !vernon.tensor_view

module attributes {llvm.data_layout = "e", vernon.compiler_contract_version = 1 : i64, vernon.program_version = 2 : i64} {
  func.func @tensor_view_round_trip(
      %view: !vernon.tensor_view<f32, [-1], "read_write", "device"> {
        vernon.interface = "resource",
        vernon.set = 0 : i64,
        vernon.binding = 0 : i64
      },
      %index: index {vernon.dtype = "index"}, %value: f32 {vernon.dtype = "f32"}) attributes {
        vernon.entry,
        vernon.stage = "compute",
        vernon.workgroup_size = array<i32: 1, 1, 1>
      } {
    %loaded = "vernon.load"(%view, %index) :
      (!vernon.tensor_view<f32, [-1], "read_write", "device">, index) -> f32
    "vernon.store"(%value, %view, %index) :
      (f32, !vernon.tensor_view<f32, [-1], "read_write", "device">, index) -> ()
    return
  }
}
