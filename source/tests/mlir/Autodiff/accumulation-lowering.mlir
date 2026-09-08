// RUN: %vernon-opt \
// RUN:   --vernon-lower-accumulation="f32-device-atomic=native" \
// RUN:   %s -o %t
// RUN: %FileCheck %s --check-prefix=CHECK --input-file=%t
// RUN: %FileCheck %s --check-prefix=ABSENT --input-file=%t
//
// CHECK: vernon.physical_store
// CHECK: "vernon.physical_atomic"{{.*}}vernon.atomic_implementation = "native"
// CHECK: "vernon.physical_atomic"{{.*}}vernon.atomic_implementation = "native"
//
// ABSENT: module
// ABSENT-NOT: vernon.reduce_sum
// ABSENT-NOT: vernon.scatter_add

module attributes {vernon.ad_profile = "backward"} {
  func.func @disjoint(%gradient: !vernon.tensor_view<f32, [-1, -1, -1], "read_write", "device">,
                      %value: f32,
                      %gid: tensor<3xi32> {vernon.builtin = "global_invocation_id"}) {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %two = arith.constant 2 : index
    %gx_i32 = tensor.extract %gid[%zero] : tensor<3xi32>
    %gy_i32 = tensor.extract %gid[%one] : tensor<3xi32>
    %gz_i32 = tensor.extract %gid[%two] : tensor<3xi32>
    %gx = arith.index_castui %gx_i32 : i32 to index
    %gy = arith.index_castui %gy_i32 : i32 to index
    %gz = arith.index_castui %gz_i32 : i32 to index
    "vernon.scatter_add"(%value, %gradient, %gx, %gy, %gz) {
      deterministic = true,
      disjoint
    } : (f32, !vernon.tensor_view<f32, [-1, -1, -1], "read_write", "device">,
         index, index, index) -> ()
    return
  }

  func.func @atomic(%gradient: !vernon.tensor_view<f32, [-1], "read_write", "device">,
                    %value: f32, %index: index) {
    "vernon.reduce_sum"(%value, %gradient, %index) {
      deterministic = false
    } : (f32, !vernon.tensor_view<f32, [-1], "read_write", "device">, index) -> ()
    return
  }

  func.func @explicit_atomic(%gradient: !vernon.tensor_view<f32, [-1], "read_write", "device">,
                             %value: f32, %index: index) {
    %old = "vernon.atomic"(%gradient, %index, %value) {
      atomic_kind = "add",
      ordering = "relaxed"
    } : (!vernon.tensor_view<f32, [-1], "read_write", "device">, index, f32) -> f32
    return
  }
}
