module {
  //=== Vertex Shader ===//

  func.func @vertex_main(%pos: tensor<3xf32> {vernon.semantic = "input"},
                        %model: tensor<4x4xf32> {vernon.semantic = "uniform"},
                        %view: tensor<4x4xf32> {vernon.semantic = "uniform"},
                        %proj: tensor<4x4xf32> {vernon.semantic = "uniform"})
                    -> (tensor<4xf32> {vernon.semantic="output"},
                        tensor<3xf32> {vernon.semantic="varying"}) { // VS out -> FS in
    // Extend pos to vec4(pos,1.0)
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index

    %p0 = tensor.extract %pos[%c0] : tensor<3xf32>
    %p1 = tensor.extract %pos[%c1] : tensor<3xf32>
    %p2 = tensor.extract %pos[%c2] : tensor<3xf32>
    %one = arith.constant 1.0 : f32

    %pos4 = tensor.from_elements %p0, %p1, %p2, %one: tensor<4xf32>

    // Compute MVP * pos4
    %zero = arith.constant 0.0 : f32
    %model_pos = tensor.from_elements %zero, %zero, %zero, %zero: tensor<4xf32>
    %view_model_pos = tensor.from_elements %zero, %zero, %zero, %zero: tensor<4xf32>
    %gl_Position = tensor.from_elements %zero, %zero, %zero, %zero: tensor<4xf32>

    //linalg.matvec ins(%model, %pos4 : tensor<4x4xf32>, tensor<4xf32>)
    //              outs(%model_pos : tensor<4xf32>)

    //linalg.matvec ins(%view, %model_pos : tensor<4x4xf32>, tensor<4xf32>)
    //              outs(%view_model_pos : tensor<4xf32>)

    //linalg.matvec ins(%proj, %view_model_pos : tensor<4x4xf32>, tensor<4xf32>)
    //              outs(%gl_Position : tensor<4xf32>)

    // Pass through varying
    return %gl_Position, %pos: tensor<4xf32>, tensor<3xf32>
  }

  //=== Fragment Shader ===//
  func.func @frag_main(%color: tensor<3xf32> {vernon.semantic = "uniform"},
                      %frag_pos: tensor<3xf32> {vernon.semantic = "varying"})
                    -> (tensor<4xf32> {vernon.semantic="output"})  {
    // Extend color to vec4(color,1.0)
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index

    %p0 = tensor.extract %color[%c0] : tensor<3xf32>
    %p1 = tensor.extract %color[%c1] : tensor<3xf32>
    %p2 = tensor.extract %color[%c2] : tensor<3xf32>
    %one = arith.constant 1.0 : f32

    %color4 = tensor.from_elements %p0, %p1, %p2, %one: tensor<4xf32>

    // Output FragColor
    return %color4: tensor<4xf32>
  }
}