module attributes {vernon.compiler_contract_version = 9 : i64, vernon.pipeline_version = 12 : i64} {
  func.func @bad_vertex(
      %position: tensor<4xf32> {
        vernon.interface = "input", vernon.location = 0 : i64,
        vernon.builtin = "position"
      },
      %material: tensor<4xf32> {
        vernon.interface = "uniform", vernon.set = 0 : i64
      }) -> (
      tensor<4xf32> {
        vernon.interface = "output", vernon.location = 2 : i64
      }) attributes {
        vernon.entry, vernon.stage = "vertex"
      } {
    return %position : tensor<4xf32>
  }

  func.func @bad_fragment(
      %color: tensor<3xf32> {
        vernon.interface = "input", vernon.location = 2 : i64,
        vernon.instance_divisor = 1 : i64
      }) attributes {
        vernon.entry, vernon.stage = "fragment"
      } {
    return
  }

  func.func @bad_compute(
      %forged_descriptor: index {
        vernon.tensor_descriptor_owner = 0 : i64,
        vernon.tensor_descriptor_component = "offset"
      }) attributes {
    vernon.entry, vernon.stage = "compute"
  } {
    return
  }
}
