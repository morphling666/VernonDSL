// RUN: %vernon-opt --vernon-lower-accumulation=supports-atomic-f32=true %s -o %t
// RUN: %FileCheck %s --check-prefix=CHECK --input-file=%t
// RUN: %FileCheck %s --check-prefix=ABSENT --input-file=%t
//
// CHECK-DAG: vernon.physical_load
// CHECK-DAG: vernon.physical_load
// CHECK-DAG: vernon.physical_atomic
// CHECK-DAG: arith.addf
// CHECK-DAG: vernon.serial_dispatch
// CHECK-DAG: vernon.workgroup_size = array<i32: 1, 1, 1>
// CHECK-DAG: scf.for
// CHECK-DAG: vernon.physical_store
//
// ABSENT: module
// ABSENT-NOT: vernon.reduce_sum
// ABSENT-NOT: vernon.scatter_add

module {
  func.func @disjoint(%gradient: !vernon.tensor_view<f32, [-1], "read_write", "device">,
                      %value: f32, %index: index) {
    "vernon.scatter_add"(%value, %gradient, %index) {
      deterministic = true,
      disjoint
    } : (f32, !vernon.tensor_view<f32, [-1], "read_write", "device">, index) -> ()
    return
  }

  func.func @atomic(%gradient: !vernon.tensor_view<f32, [-1], "read_write", "device">,
                    %value: f32, %index: index) {
    "vernon.reduce_sum"(%value, %gradient, %index) {
      deterministic = false
    } : (f32, !vernon.tensor_view<f32, [-1], "read_write", "device">, index) -> ()
    return
  }

  func.func @deterministic(%gradient: !vernon.tensor_view<f32, [-1], "read_write", "device">,
                           %value: f32, %index: index) {
    "vernon.scatter_add"(%value, %gradient, %index) {
      deterministic = true
    } : (f32, !vernon.tensor_view<f32, [-1], "read_write", "device">, index) -> ()
    return
  }

  func.func @unsupported_atomic_falls_back(
      %gradient: !vernon.tensor_view<f64, [-1], "read_write", "device">,
      %value: f64, %index: index) {
    "vernon.reduce_sum"(%value, %gradient, %index) {
      deterministic = false
    } : (f64, !vernon.tensor_view<f64, [-1], "read_write", "device">, index) -> ()
    return
  }

  func.func @serial_entry(
      %launch: !vernon.tensor_view<tensor<3xi32>, [1], "read", "device">
          {vernon.source_name = "__vernon_launch"},
      %gradient: !vernon.tensor_view<f64, [1], "read_write", "device">,
      %global_id: tensor<3xi32> {vernon.builtin = "global_invocation_id"})
      attributes {vernon.entry, vernon.workgroup_size = array<i32: 4, 1, 1>} {
    %index = arith.constant 0 : index
    %value = arith.constant 1.0 : f64
    "vernon.reduce_sum"(%value, %gradient, %index) {
      deterministic = false
    } : (f64, !vernon.tensor_view<f64, [1], "read_write", "device">, index) -> ()
    return
  }
}
