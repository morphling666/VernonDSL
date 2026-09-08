// RUN: %vernon-opt \
// RUN:   --vernon-lower-accumulation="f32-device-atomic=native" \
// RUN:   %s | %FileCheck %s
//
// Shared Vector TensorView dests keep the constructor. This pass lowers
// scatter_add of a cell to one PhysicalAtomic of that cell; GPU/CPU conversion
// unpacks ABI leaves the same way PhysicalStore does.
//
// CHECK-LABEL: func.func @shaped_atomic
// CHECK-SAME: !vernon.tensor_view<tensor<2xf32>, [1], "write", "device">
// CHECK: "vernon.physical_atomic"{{.*}}vernon.atomic_implementation = "native"
// CHECK-NOT: vernon.scatter_add

module {
  func.func @shaped_atomic(
      %gradient: !vernon.tensor_view<tensor<2xf32>, [1], "write", "device">,
      %value: tensor<2xf32>, %index: index) {
    "vernon.scatter_add"(%value, %gradient, %index) {
      deterministic = false
    } : (tensor<2xf32>,
         !vernon.tensor_view<tensor<2xf32>, [1], "write", "device">,
         index) -> ()
    return
  }
}
