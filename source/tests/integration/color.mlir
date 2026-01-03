module {
  //=== Vertex Shader ===//

  func @vertex_main(%pos: tensor<3xf32> {vernon.semantic = "input"},
            %model: tensor<4x4xf32> {vernon.semantic = "uniform"},
            %view: tensor<4x4xf32> {vernon.semantic = "uniform"},
            %proj: tensor<4x4xf32> {vernon.semantic = "uniform"})
        -> (tensor<4xf32> {vernon.semantic="output"},
            tensor<3xf32> {vernon.semantic="varying"}) { // VS out -> FS in
    // Extend pos to vec4(pos,1.0)
    %pos4 : tensor<4xf32> = tensor.insert_slice %pos[0] 1 : tensor<3xf32> -> tensor<4xf32>
    %pos4_h : tensor<4xf32> = tensor.insert %pos4, 1.0[3] : tensor<4xf32>

    // Compute MVP * pos4
    %model_pos : tensor<4xf32> = linalg.matmul %model, %pos4_h : tensor<4x4xf32>, tensor<4xf32> -> tensor<4xf32>
    %view_model_pos : tensor<4xf32> = linalg.matmul %view, %model_pos : tensor<4x4xf32>, tensor<4xf32> -> tensor<4xf32>
    %gl_Position : tensor<4xf32> = linalg.matmul %proj, %view_model_pos : tensor<4x4xf32>, tensor<4xf32> -> tensor<4xf32>

    // Pass through varying
    %frag_pos : tensor<3xf32> = %pos

    return %gl_Position, %frag_pos
  }

  //=== Fragment Shader ===//
  func @frag_main(%color: tensor<3xf32> {vernon.semantic = "uniform"},
                %frag_pos: tensor<3xf32> {vernon.semantic = "varying"})
            ->tensor<4xf32> {vernon.semantic="output"}  {
    // Extend color to vec4(color,1.0)
    %color4 : tensor<4xf32> = tensor.insert %color, 1.0[3] : tensor<3xf32> -> tensor<4xf32>

    // Output FragColor
    return %color4
  }
}