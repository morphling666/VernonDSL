// RUN: %vernon-opt \
// RUN:   --vernon-lower-accumulation="aggregate-gradient-storage=invocation-private-staging" \
// RUN:   %s | %FileCheck %s
//
// CHECK-LABEL: func.func @private_shaped_gradient
// CHECK: vernon.physical_load
// CHECK: arith.addf
// CHECK: vernon.physical_store
// CHECK-NOT: vernon.physical_atomic
// CHECK-NOT: vernon.scatter_add

module {
  func.func @private_shaped_gradient(
      %gradient: !vernon.tensor_view<tensor<2xf32>, [1], "read_write", "device">,
      %value: tensor<2xf32>, %index: index) {
    "vernon.scatter_add"(%value, %gradient, %index) {
      deterministic = false,
      vernon.accumulation_ownership = "invocation_private"
    } : (tensor<2xf32>,
         !vernon.tensor_view<tensor<2xf32>, [1], "read_write", "device">,
         index) -> ()
    return
  }
}
