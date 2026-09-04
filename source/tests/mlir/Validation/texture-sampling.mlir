// RUN: %vernon-opt --vernon-validate %s -o %t

module attributes {vernon.compiler_contract_version = 14 : i64, vernon.pipeline_version = 18 : i64} {
  func.func @sample_2d(
      %texture: !vernon.texture<"2d", f32, "unknown", "sampled"> {
        vernon.interface = "resource",
        vernon.set = 0 : i64,
        vernon.binding = 0 : i64
      },
      %sampler: !vernon.sampler {
        vernon.interface = "resource",
        vernon.set = 0 : i64,
        vernon.binding = 1 : i64
      },
      %coordinates: tensor<2xf32> {
        vernon.interface = "input",
        vernon.location = 0 : i64
      }) -> (tensor<4xf32> {
        vernon.interface = "output",
        vernon.location = 0 : i64
      }) attributes {vernon.entry, vernon.stage = "fragment"} {
    %sample = "vernon.intrinsic"(%texture, %sampler, %coordinates) <{
      name = "texture_sample"
    }> : (!vernon.texture<"2d", f32, "unknown", "sampled">, !vernon.sampler, tensor<2xf32>)
        -> tensor<4xf32>
    return %sample : tensor<4xf32>
  }

  func.func @sample_cube(
      %texture: !vernon.texture<"cube", f32, "unknown", "sampled"> {
        vernon.interface = "resource",
        vernon.set = 0 : i64,
        vernon.binding = 2 : i64
      },
      %sampler: !vernon.sampler {
        vernon.interface = "resource",
        vernon.set = 0 : i64,
        vernon.binding = 3 : i64
      },
      %coordinates: tensor<3xf32> {
        vernon.interface = "input",
        vernon.location = 1 : i64
      }) -> (tensor<4xf32> {
        vernon.interface = "output",
        vernon.location = 1 : i64
      }) attributes {vernon.entry, vernon.stage = "fragment"} {
    %sample = "vernon.intrinsic"(%texture, %sampler, %coordinates) <{
      name = "texture_sample"
    }> : (!vernon.texture<"cube", f32, "unknown", "sampled">, !vernon.sampler, tensor<3xf32>)
        -> tensor<4xf32>
    return %sample : tensor<4xf32>
  }
}
