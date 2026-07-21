module {
  func.func @unsupported_intrinsic(%input: tensor<4xf32>)
      -> tensor<4xf32> {
    %result = "vernon.intrinsic"(%input) <{
      name = "unimplemented_cpu_operation"
    }> : (tensor<4xf32>) -> tensor<4xf32>
    return %result : tensor<4xf32>
  }
}
