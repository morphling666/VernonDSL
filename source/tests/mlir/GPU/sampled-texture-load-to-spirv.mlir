// RUN: %vernon-opt --vernon-to-spirv %s | %FileCheck %s
//
// CHECK: %[[IMAGE:.*]] = spirv.Image %{{.*}} : !spirv.sampled_image<!spirv.image<i32, Dim2D, NoDepth, NonArrayed, SingleSampled, NeedSampler, Unknown>>
// CHECK: spirv.ImageFetch %[[IMAGE]], %{{.*}} : {{.*}} -> vector<4xi32>
// CHECK-NOT: spirv.ImageRead

module attributes {vernon.compiler_contract_version = 1 : i64, vernon.program_version = 2 : i64} {
  func.func @fragment_main(
      %identifiers: !vernon.texture<"2d", i32, "unknown", "sampled"> {
        vernon.interface = "resource", vernon.set = 0 : i64, vernon.binding = 0 : i64
      },
      %coordinates: tensor<2xi32> {
        vernon.interface = "input", vernon.location = 0 : i64, vernon.interpolation = "flat"
      }) -> (i32 {
        vernon.interface = "output", vernon.location = 0 : i64, vernon.abi_leaf_dtypes = ["u32"]
      }) attributes {
        vernon.entry, vernon.stage = "fragment"
      } {
    %texel = "vernon.intrinsic"(%identifiers, %coordinates) <{
      name = "texture_load"
    }> : (!vernon.texture<"2d", i32, "unknown", "sampled">, tensor<2xi32>) -> tensor<4xi32>
    %index = arith.constant 0 : index
    %identifier = tensor.extract %texel[%index] : tensor<4xi32>
    return %identifier : i32
  }
}
