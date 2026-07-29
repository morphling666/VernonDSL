module attributes {vernon.compiler_contract_version = 5 : i64, vernon.pipeline_version = 7 : i64} {
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
