// RUN: %vernon-opt %s \
// RUN:   --vernon-program-vjp="wrt=source forward=forward backward=backward" \
// RUN:   --vernon-program-select-implementations \
// RUN:   --vernon-program-build-executable \
// RUN:   | %FileCheck %s

module {
  func.func @primal(
      %source: !vernon.tensor_view<f32, [1], "read_write", "device">
          {vernon.source_name = "source"})
      -> !vernon.tensor_view<f32, [1], "read_write", "device">
      attributes {vernon_program.graph = "primal"} {
    %buffer = "vernon.intrinsic"(%source) {name = "empty_like"}
        : (!vernon.tensor_view<f32, [1], "read_write", "device">) ->
          !vernon.tensor_view<f32, [1], "read_write", "device">
    %result = "vernon_program.compute"(%source, %buffer) {
      callee = "square",
      features = [],
      grid = array<i64: 1, 1, 1>,
      operand_names = ["source", "output"],
      result_names = ["output"],
      vernon_program.operand_accesses = ["read", "write"],
      vernon_program.result_resource_sources = array<i64: 1>
    } : (!vernon.tensor_view<f32, [1], "read_write", "device">,
         !vernon.tensor_view<f32, [1], "read_write", "device">) ->
        !vernon.tensor_view<f32, [1], "read_write", "device">
    func.return %result : !vernon.tensor_view<f32, [1], "read_write", "device">
  }
}

// CHECK-LABEL: func.func @forward
// CHECK: "vernon.intrinsic"
// CHECK-SAME: name = "empty_like"
// CHECK: "vernon_program.compute"
// CHECK-SAME: callee = "square.forward_with_tape"
// CHECK-LABEL: func.func @backward
// CHECK: callee = "square.vjp"
// CHECK-SAME: operand_names = ["tape", "primal.source", "primal.output", "result.output", "cotangent.output", "gradient.source"]
// CHECK-SAME: result_names = ["source"]}>
// CHECK-NOT: name = "empty_like"
