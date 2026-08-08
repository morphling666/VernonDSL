// RUN: %vernon-opt --vernon-lower-cpu-resources %s | %FileCheck %s
//
// CHECK: func.func private @__vernon_cpu_texture_sample
// CHECK: !llvm.ptr
// CHECK: call @__vernon_cpu_texture_sample
// CHECK-NOT: vernon.cpu.requires_texture_callbacks
// CHECK-NOT: !vernon.texture
// CHECK-NOT: !vernon.sampler

module attributes {vernon.compiler_contract_version = 10 : i64, vernon.pipeline_version = 14 : i64} {
  func.func @sample_texture(
      %texture: !vernon.texture<"2d", f32>,
      %sampler: !vernon.sampler,
      %coordinates: vector<2xf32>) -> vector<4xf32>
      attributes {vernon.entry, vernon.stage = "fragment"} {
    %sample = "vernon.intrinsic"(%texture, %sampler, %coordinates) <{
      name = "texture_sample"
    }> : (!vernon.texture<"2d", f32>, !vernon.sampler, vector<2xf32>)
        -> vector<4xf32>
    return %sample : vector<4xf32>
  }
}
