// RUN: %vernon-opt --vernon-to-gpu %s | %FileCheck %s
//
// CHECK: gpu.module @vernon_kernels
// CHECK-LABEL: gpu.func @storage_main(
// CHECK-SAME: !vernon.texture<"3d", f32, "rgba32_float", "read_write">
// CHECK-SAME: spirv.interface_var_abi = #spirv.interface_var_abi<(0, 0)>
// CHECK: name = "texture_load"
// CHECK: name = "texture_store"

module attributes {vernon.compiler_contract_version = 12 : i64, vernon.pipeline_version = 16 : i64} {
  func.func @storage_main(
      %image: !vernon.texture<"3d", f32, "rgba32_float", "read_write"> {
        vernon.interface = "resource",
        vernon.set = 0 : i64,
        vernon.binding = 0 : i64
      },
      %coordinate: tensor<3xi32> {
        vernon.interface = "uniform"
      }) attributes {
        vernon.entry,
        vernon.stage = "compute",
        vernon.workgroup_size = array<i32: 1, 1, 1>
      } {
    %loaded = "vernon.intrinsic"(%image, %coordinate) {
      name = "texture_load"
    } : (!vernon.texture<"3d", f32, "rgba32_float", "read_write">, tensor<3xi32>) -> tensor<4xf32>
    "vernon.intrinsic"(%image, %coordinate, %loaded) {
      name = "texture_store"
    } : (!vernon.texture<"3d", f32, "rgba32_float", "read_write">, tensor<3xi32>, tensor<4xf32>) -> ()
    return
  }
}
