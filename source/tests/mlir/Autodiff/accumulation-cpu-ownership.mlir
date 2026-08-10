// RUN: %vernon-opt --vernon-lower-cpu-autodiff %s | %FileCheck %s
//
// CHECK-LABEL: func.func @cpu_backward
// CHECK: "vernon.scatter_add"
// CHECK-NOT: vernon.accumulation_ownership
// CHECK: "vernon.scatter_add"{{.*}}vernon.accumulation_ownership = "invocation_private"

module {
  func.func @cpu_backward(
      %root: index {vernon.builtin = "ad_tape_root_region"},
      %aggregate_gradient:
          !vernon.tensor_view<tensor<2xf32>, [1], "read_write", "device">,
      %single_gradient:
          !vernon.tensor_view<tensor<1xf32>, [1], "read_write", "device">,
      %aggregate: tensor<2xf32>, %single: tensor<1xf32>, %index: index) {
    "vernon.scatter_add"(%aggregate, %aggregate_gradient, %index) {
      deterministic = false
    } : (tensor<2xf32>,
         !vernon.tensor_view<tensor<2xf32>, [1], "read_write", "device">,
         index) -> ()
    "vernon.scatter_add"(%single, %single_gradient, %index) {
      deterministic = false,
      vernon.accumulation_ownership = "invocation_private"
    } : (tensor<1xf32>,
         !vernon.tensor_view<tensor<1xf32>, [1], "read_write", "device">,
         index) -> ()
    return
  }
}
