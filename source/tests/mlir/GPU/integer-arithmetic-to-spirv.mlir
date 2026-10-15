// RUN: %vernon-opt --vernon-to-spirv %s | %FileCheck %s
//
// CHECK: spirv.ConvertFToS %{{.*}} : vector<2xf32> to vector<2xi32>
// CHECK: spirv.SDiv %{{.*}}, %{{.*}} : vector<2xi32>
// CHECK: spirv.SRem %{{.*}}, %{{.*}} : vector<2xi32>
// CHECK: spirv.ConvertFToU %{{.*}} : f32 to i32
// CHECK: spirv.UDiv %{{.*}}, %{{.*}} : i32
// CHECK: spirv.UMod %{{.*}}, %{{.*}} : i32
// CHECK: spirv.FRem %{{.*}}, %{{.*}} : f32

module attributes {vernon.compiler_contract_version = 1 : i64, vernon.program_version = 2 : i64} {
  func.func @fragment_main(
      %pixel: tensor<2xf32> {
        vernon.interface = "input", vernon.location = 0 : i64
      },
      %shade: f32 {
        vernon.interface = "input", vernon.location = 1 : i64
      }) -> (tensor<2xi32> {
        vernon.interface = "output", vernon.location = 0 : i64
      }, i32 {
        vernon.interface = "output", vernon.location = 1 : i64, vernon.abi_leaf_dtypes = ["u32"]
      }, f32 {
        vernon.interface = "output", vernon.location = 2 : i64
      }) attributes {
        vernon.entry, vernon.stage = "fragment"
      } {
    %texel = arith.fptosi %pixel : tensor<2xf32> to tensor<2xi32>
    %tile = arith.constant dense<3> : tensor<2xi32>
    %row = arith.divsi %texel, %tile : tensor<2xi32>
    %column = arith.remsi %texel, %row : tensor<2xi32>
    %level = arith.fptoui %shade : f32 to i32
    %bucket = arith.constant 7 : i32
    %quotient = arith.divui %level, %bucket : i32
    %identifier = arith.remui %quotient, %bucket : i32
    %period = arith.constant 0.25 : f32
    %phase = arith.remf %shade, %period : f32
    return %column, %identifier, %phase : tensor<2xi32>, i32, f32
  }
}
