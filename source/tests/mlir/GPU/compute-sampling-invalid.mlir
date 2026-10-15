// RUN: %vernon-opt --split-input-file --verify-diagnostics \
// RUN:   --pass-pipeline='builtin.module(vernon-to-gpu{use-spirv-storage=true},gpu.module(vernon-lower-gpu-tensors{use-spirv-tuple-abi=true}),vernon-convert-gpu-to-spirv)' \
// RUN:   %s

module attributes {vernon.compiler_contract_version = 1 : i64, vernon.program_version = 2 : i64} {
  func.func @implicit_lod(
      %image: !vernon.texture<"2d", f32, "unknown", "sampled"> {vernon.interface = "resource", vernon.set = 0 : i64, vernon.binding = 0 : i64},
      %sampler: !vernon.sampler {vernon.interface = "resource", vernon.set = 0 : i64, vernon.binding = 1 : i64},
      %output: !vernon.tensor_view<f32, [-1], "write", "device"> {
        vernon.element_abi_leaf_dtypes = ["f32"], vernon.interface = "resource", vernon.set = 0 : i64, vernon.binding = 2 : i64
      }) attributes {vernon.entry, vernon.stage = "compute", vernon.workgroup_size = array<i32: 1, 1, 1>} {
    %u = arith.constant 0.25 : f32
    %coordinate = "vernon.intrinsic"(%u, %u) {name = "construct"} : (f32, f32) -> tensor<2xf32>
    // expected-error @+1 {{texture_sample without lod is supported only in graphics stages}}
    %color = "vernon.intrinsic"(%image, %sampler, %coordinate) {
      name = "texture_sample"
    } : (!vernon.texture<"2d", f32, "unknown", "sampled">, !vernon.sampler, tensor<2xf32>) -> tensor<4xf32>
    %zero = arith.constant 0 : index
    %red = tensor.extract %color[%zero] : tensor<4xf32>
    "vernon.store"(%red, %output, %zero) : (f32, !vernon.tensor_view<f32, [-1], "write", "device">, index) -> ()
    return
  }
}

// -----

module attributes {vernon.compiler_contract_version = 1 : i64, vernon.program_version = 2 : i64} {
  func.func @sampled_fetch(
      %image: !vernon.texture<"2d", f32, "unknown", "sampled"> {vernon.interface = "resource", vernon.set = 0 : i64, vernon.binding = 0 : i64},
      %output: !vernon.tensor_view<f32, [-1], "write", "device"> {
        vernon.element_abi_leaf_dtypes = ["f32"], vernon.interface = "resource", vernon.set = 0 : i64, vernon.binding = 1 : i64
      }) attributes {vernon.entry, vernon.stage = "compute", vernon.workgroup_size = array<i32: 1, 1, 1>} {
    %i = arith.constant 0 : i32
    %coordinate = "vernon.intrinsic"(%i, %i) {name = "construct"} : (i32, i32) -> tensor<2xi32>
    // expected-error @+1 {{texture_load from a sampled texture is supported only in graphics stages}}
    %texel = "vernon.intrinsic"(%image, %coordinate) {
      name = "texture_load"
    } : (!vernon.texture<"2d", f32, "unknown", "sampled">, tensor<2xi32>) -> tensor<4xf32>
    %zero = arith.constant 0 : index
    %red = tensor.extract %texel[%zero] : tensor<4xf32>
    "vernon.store"(%red, %output, %zero) : (f32, !vernon.tensor_view<f32, [-1], "write", "device">, index) -> ()
    return
  }
}

// -----

module attributes {vernon.compiler_contract_version = 1 : i64, vernon.program_version = 2 : i64} {
  func.func @forwarded_sampler(
      %image: !vernon.texture<"2d", f32, "unknown", "sampled"> {vernon.interface = "resource", vernon.set = 0 : i64, vernon.binding = 0 : i64},
      %sampler: !vernon.sampler {vernon.interface = "resource", vernon.set = 0 : i64, vernon.binding = 1 : i64},
      %output: !vernon.tensor_view<f32, [-1], "write", "device"> {
        vernon.element_abi_leaf_dtypes = ["f32"], vernon.interface = "resource", vernon.set = 0 : i64, vernon.binding = 2 : i64
      }) attributes {vernon.entry, vernon.stage = "compute", vernon.workgroup_size = array<i32: 1, 1, 1>} {
    %u = arith.constant 0.25 : f32
    %coordinate = "vernon.intrinsic"(%u, %u) {name = "construct"} : (f32, f32) -> tensor<2xf32>
    %true = arith.constant true
    // expected-error @+1 {{compute sampler argument #1 may be used only as the sampler of texture_sample}}
    %chosen = arith.select %true, %sampler, %sampler : !vernon.sampler
    %color = "vernon.intrinsic"(%image, %chosen, %coordinate, %u) {
      name = "texture_sample"
    } : (!vernon.texture<"2d", f32, "unknown", "sampled">, !vernon.sampler, tensor<2xf32>, f32) -> tensor<4xf32>
    %zero = arith.constant 0 : index
    %red = tensor.extract %color[%zero] : tensor<4xf32>
    "vernon.store"(%red, %output, %zero) : (f32, !vernon.tensor_view<f32, [-1], "write", "device">, index) -> ()
    return
  }
}
