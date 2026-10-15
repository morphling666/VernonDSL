// RUN: %vernon-opt --split-input-file --verify-diagnostics %s

func.func @load_cube(%texture: !vernon.texture<"cube", f32, "unknown", "sampled">, %coordinates: tensor<3xi32>)
    -> tensor<4xf32> {
  // expected-error @+1 {{texture_load cannot address texels of a cube texture}}
  %texel = "vernon.intrinsic"(%texture, %coordinates) <{
    name = "texture_load"
  }> : (!vernon.texture<"cube", f32, "unknown", "sampled">, tensor<3xi32>) -> tensor<4xf32>
  return %texel : tensor<4xf32>
}

// -----

func.func @store_sampled(%texture: !vernon.texture<"2d", f32, "unknown", "sampled">, %coordinates: tensor<2xi32>,
                         %texel: tensor<4xf32>) {
  // expected-error @+1 {{texture_store operand #0 must be a Vernon storage Texture}}
  "vernon.intrinsic"(%texture, %coordinates, %texel) <{
    name = "texture_store"
  }> : (!vernon.texture<"2d", f32, "unknown", "sampled">, tensor<2xi32>, tensor<4xf32>) -> ()
  return
}
