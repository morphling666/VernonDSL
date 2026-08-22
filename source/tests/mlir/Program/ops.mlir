// RUN: %vernon-opt %s | %FileCheck %s

module {
  func.func @forward(%input: tensor<4xf32>) -> tensor<4xf32> attributes {
    vernon_program.graph = "forward",
    vernon_program.argument_names = ["input.value"],
    vernon_program.result_names = ["output.value"]
  } {
    %result = "vernon_program.compute"(%input) {
      callee = "normalize",
      features = [],
      grid = array<i64: 1, 1, 1>,
      operand_names = ["input"],
      result_names = ["result"]
    } : (tensor<4xf32>) -> tensor<4xf32>
    func.return %result : tensor<4xf32>
  }

  func.func @render(%target: tensor<4xi32>, %vertices: tensor<3x4xf32>) -> tensor<4xi32> attributes {
    vernon_program.graph = "forward",
    vernon_program.argument_names = ["render_target", "input.vertices"],
    vernon_program.result_names = ["output.target"]
  } {
    %updated = "vernon_program.graphics"(%target, %vertices) {
      callee = "draw_mesh",
      topology = "triangle_list",
      features = [],
      operand_names = ["vertices"],
      result_names = ["target"]
    } : (tensor<4xi32>, tensor<3x4xf32>) -> tensor<4xi32>
    func.return %updated : tensor<4xi32>
  }
}

// CHECK: "vernon_program.compute"
// CHECK: "vernon_program.graphics"
