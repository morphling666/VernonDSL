// RUN: %not %vernon-opt --vernon-validate %s 2>&1 | %FileCheck %s
//
// CHECK-DAG: result #1 integer varyings must be flat
// CHECK-DAG: result #2 'vernon.interpolation' must be "smooth", "noperspective", or "flat"
// CHECK-DAG: argument #0 'vernon.interpolation' is only valid on located vertex outputs and fragment inputs
// CHECK-DAG: fragment input #0 at location 3: varying interpolation differs between stages

module attributes {vernon.compiler_contract_version = 1 : i64, vernon.program_version = 2 : i64} {
  func.func @vertex_main(
      %position: tensor<4xf32> {
        vernon.interface = "input", vernon.location = 0 : i64,
        vernon.interpolation = "flat"
      }) -> (
      tensor<4xf32> {
        vernon.interface = "output", vernon.builtin = "position"
      },
      tensor<2xi32> {
        vernon.interface = "output", vernon.location = 1 : i64,
        vernon.interpolation = "smooth"
      },
      f32 {
        vernon.interface = "output", vernon.location = 2 : i64,
        vernon.interpolation = "centroid"
      },
      f32 {
        vernon.interface = "output", vernon.location = 3 : i64,
        vernon.interpolation = "noperspective"
      }) attributes {
        vernon.entry, vernon.stage = "vertex"
      } {
    %identifier = arith.constant dense<7> : tensor<2xi32>
    %value = arith.constant 0.5 : f32
    return %position, %identifier, %value, %value : tensor<4xf32>, tensor<2xi32>, f32, f32
  }

  func.func @fragment_main(
      %value: f32 {
        vernon.interface = "input", vernon.location = 3 : i64
      }) -> (
      f32 {
        vernon.interface = "output", vernon.location = 0 : i64
      }) attributes {
        vernon.entry, vernon.stage = "fragment"
      } {
    return %value : f32
  }
}
