// RUN: %not %vernon-opt --vernon-validate %s 2>&1 | %FileCheck %s
//
// CHECK-DAG: cannot have both 'vernon.location' and 'vernon.builtin'
// CHECK-DAG: uniform must provide both 'vernon.set' and 'vernon.binding', or neither
// CHECK-DAG: 'vernon.instance_divisor' is only valid on vertex inputs
// CHECK-DAG: compute entry requires 'vernon.workgroup_size'
// CHECK-DAG: contains compiler-owned TensorView metadata ABI state
// CHECK-DAG: but the vertex output has type
// CHECK-DAG: argument #0 compute resource binds in argument order and must use descriptor set 0, binding 0

module attributes {vernon.compiler_contract_version = 1 : i64, vernon.program_version = 2 : i64} {
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
      %forged_metadata: index {
        vernon.tensor_metadata_carrier
      }) attributes {
    vernon.entry, vernon.stage = "compute"
  } {
    return
  }

  func.func @bad_compute_binding(
      %values: !vernon.tensor_view<f32, [-1], "read_write", "device"> {
        vernon.interface = "resource", vernon.set = 1 : i64, vernon.binding = 0 : i64
      }) attributes {
        vernon.entry, vernon.stage = "compute",
        vernon.workgroup_size = array<i32: 1, 1, 1>
      } {
    return
  }
}
