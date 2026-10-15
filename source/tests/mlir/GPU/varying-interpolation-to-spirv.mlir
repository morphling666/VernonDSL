// RUN: %vernon-opt --vernon-to-spirv %s | %FileCheck %s
//
// CHECK: spirv.GlobalVariable @shade {location = 0 : i32} : !spirv.ptr<f32, Output>
// CHECK: spirv.GlobalVariable @depth {location = 1 : i32, no_perspective} : !spirv.ptr<f32, Output>
// CHECK: spirv.GlobalVariable @identifier {flat, location = 2 : i32} : !spirv.ptr<ui32, Output>
// CHECK: %[[ID:.*]] = spirv.Bitcast %{{.*}} : i32 to ui32
// CHECK: spirv.Store "Output" %identifier_addr, %[[ID]] : ui32

module attributes {vernon.compiler_contract_version = 1 : i64, vernon.program_version = 2 : i64} {
  func.func @vertex_main(
      %position: tensor<4xf32> {
        vernon.interface = "input", vernon.location = 0 : i64
      }) -> (
      tensor<4xf32> {
        vernon.interface = "output", vernon.builtin = "position"
      },
      f32 {
        vernon.interface = "output", vernon.location = 0 : i64, vernon.source_name = "shade"
      },
      f32 {
        vernon.interface = "output", vernon.location = 1 : i64, vernon.source_name = "depth",
        vernon.interpolation = "noperspective"
      },
      i32 {
        vernon.interface = "output", vernon.location = 2 : i64, vernon.source_name = "identifier",
        vernon.abi_leaf_dtypes = ["u32"]
      }) attributes {
        vernon.entry, vernon.stage = "vertex"
      } {
    %value = arith.constant 0.5 : f32
    %identifier = arith.constant 7 : i32
    return %position, %value, %value, %identifier : tensor<4xf32>, f32, f32, i32
  }
}
