// RUN: %vernon-opt --vernon-validate %s -o %t

module attributes {vernon.compiler_contract_version = 1 : i64, vernon.program_version = 1 : i64} {
  func.func @vertex_main(
      %position: tensor<4xf32> {
        vernon.interface = "input", vernon.location = 0 : i64
      },
      %color: tensor<3xf32> {
        vernon.interface = "input", vernon.location = 1 : i64,
        vernon.instance_divisor = 1 : i64
      }) -> (
      tensor<4xf32> {
        vernon.interface = "output", vernon.builtin = "position"
      },
      tensor<3xf32> {
        vernon.interface = "output", vernon.location = 0 : i64
      }) attributes {
        vernon.entry, vernon.stage = "vertex"
      } {
    return %position, %color : tensor<4xf32>, tensor<3xf32>
  }

  func.func @fragment_main(
      %color: tensor<3xf32> {
        vernon.interface = "input", vernon.location = 0 : i64
      },
      %material: tensor<4xf32> {
        vernon.interface = "uniform",
        vernon.set = 0 : i64, vernon.binding = 0 : i64
      }) -> (
      tensor<3xf32> {
        vernon.interface = "output", vernon.location = 0 : i64
      }) attributes {
        vernon.entry, vernon.stage = "fragment"
      } {
    return %color : tensor<3xf32>
  }

  func.func @compute_main(
      %buffer: tensor<16xf32> {
        vernon.interface = "resource",
        vernon.set = 0 : i64, vernon.binding = 1 : i64
      }) attributes {
        vernon.entry, vernon.stage = "compute",
        vernon.workgroup_size = array<i32: 8, 4, 1>
      } {
    return
  }
}
