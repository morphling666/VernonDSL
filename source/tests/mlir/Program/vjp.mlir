// RUN: %vernon-opt %s \
// RUN:   --vernon-program-vjp="wrt=left,right forward=forward backward=backward" \
// RUN:   --vernon-program-select-implementations \
// RUN:   --vernon-program-build-executable \
// RUN:   | %FileCheck %s

module {
  func.func @primal(
      %left: tensor<2x3xf32> {vernon.source_name = "left"},
      %right: tensor<3x2xf32> {vernon.source_name = "right"})
      -> tensor<2x2xf32> attributes {vernon_program.graph = "primal"} {
    %normalized = "vernon_program.compute"(%left) {
      callee = "normalize",
      features = [],
      grid = array<i64: 1, 1, 1>,
      operand_names = ["input"],
      result_names = ["result"]
    } : (tensor<2x3xf32>) -> tensor<2x3xf32>
    %main = "vernon.intrinsic"(%normalized, %right) {name = "matmul"}
        : (tensor<2x3xf32>, tensor<3x2xf32>) -> tensor<2x2xf32>
    %skip = "vernon.intrinsic"(%left, %right) {name = "matmul"}
        : (tensor<2x3xf32>, tensor<3x2xf32>) -> tensor<2x2xf32>
    %result = arith.addf %main, %skip : tensor<2x2xf32>
    func.return %result : tensor<2x2xf32>
  }
}

// CHECK-LABEL: func.func @forward
// CHECK-SAME: vernon_program.graph = "forward"
// CHECK-NOT: vernon.entry
// CHECK-NOT: vernon.stage
// CHECK: "vernon_program.compute"
// CHECK-SAME: callee = "normalize.forward_with_tape"
// CHECK-SAME: vernon_program.dependencies = array<i64>
// CHECK: "vernon_program.compute"
// CHECK-SAME: callee = "vernon.builtin.matmul"
// CHECK-SAME: vernon_program.dependencies = array<i64: 0>
// CHECK: "vernon_program.compute"
// CHECK-SAME: callee = "vernon.builtin.matmul"
// CHECK-SAME: vernon_program.dependencies = array<i64>
// CHECK: "vernon_program.compute"
// CHECK-SAME: callee = "vernon.builtin.add"
// CHECK-SAME: vernon_program.dependencies = array<i64: 1, 2>
// CHECK-LABEL: func.func @backward
// CHECK-SAME: vernon_program.graph = "backward"
// CHECK-NOT: vernon.entry
// CHECK-NOT: vernon.stage
// CHECK: "vernon_program.compute"
// CHECK: callee = "normalize.vjp"
// CHECK-SAME: operand_names = ["tape",
// Gradients from normalize.vjp and the skip matmul must fan in to %left.
// CHECK: "vernon_program.compute"
// CHECK-SAME: callee = "vernon.builtin.add"
// CHECK: return %
