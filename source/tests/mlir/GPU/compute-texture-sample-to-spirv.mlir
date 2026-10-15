// RUN: %vernon-opt \
// RUN:   --pass-pipeline='builtin.module(vernon-to-gpu{use-spirv-storage=true},gpu.module(vernon-lower-gpu-tensors{use-spirv-tuple-abi=true}),vernon-convert-gpu-to-spirv)' \
// RUN:   %s | %FileCheck %s
//
// The sampler binds through the texture's combined image-sampler, so it leaves no SPIR-V argument while the output
// keeps the binding index that counts it.
// CHECK-LABEL: spirv.func @sample_main(
// CHECK-SAME: %[[IMAGE:[a-z0-9]+]]: !spirv.ptr<!spirv.sampled_image<!spirv.image<f32, Dim2D, NoDepth, NonArrayed, SingleSampled, NeedSampler, Unknown>>, UniformConstant> {spirv.interface_var_abi = #spirv.interface_var_abi<(0, 0)>, vernon.texture_access = "sampled"}, %{{[a-z0-9]+}}: !spirv.ptr<!spirv.struct<{{[^{]*}}>, StorageBuffer> {spirv.interface_var_abi = #spirv.interface_var_abi<(0, 2)>})
// CHECK: %[[SAMPLED:.*]] = spirv.Load "UniformConstant" %[[IMAGE]]
// CHECK: spirv.ImageSampleExplicitLod %[[SAMPLED]], %{{.*}} ["Lod"], %{{.*}} : {{.*}} -> vector<4xf32>

module attributes {vernon.compiler_contract_version = 1 : i64, vernon.program_version = 2 : i64} {
  func.func @sample_main(
      %image: !vernon.texture<"2d", f32, "unknown", "sampled"> {
        vernon.interface = "resource",
        vernon.set = 0 : i64,
        vernon.binding = 0 : i64
      },
      %sampler: !vernon.sampler {
        vernon.interface = "resource",
        vernon.set = 0 : i64,
        vernon.binding = 1 : i64
      },
      %output: !vernon.tensor_view<f32, [-1], "write", "device"> {
        vernon.element_abi_leaf_dtypes = ["f32"],
        vernon.interface = "resource",
        vernon.set = 0 : i64,
        vernon.binding = 2 : i64
      }) attributes {
        vernon.entry,
        vernon.stage = "compute",
        vernon.workgroup_size = array<i32: 1, 1, 1>
      } {
    %u = arith.constant 0.25 : f32
    %v = arith.constant 0.5 : f32
    %coordinate = "vernon.intrinsic"(%u, %v) {name = "construct"} : (f32, f32) -> tensor<2xf32>
    %lod = arith.constant 0.0 : f32
    %color = "vernon.intrinsic"(%image, %sampler, %coordinate, %lod) {
      name = "texture_sample"
    } : (!vernon.texture<"2d", f32, "unknown", "sampled">, !vernon.sampler, tensor<2xf32>, f32) -> tensor<4xf32>
    %zero = arith.constant 0 : index
    %red = tensor.extract %color[%zero] : tensor<4xf32>
    "vernon.store"(%red, %output, %zero) : (f32, !vernon.tensor_view<f32, [-1], "write", "device">, index) -> ()
    return
  }
}
