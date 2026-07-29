module attributes {vernon.frontend_version = 4 : i64, vernon.value_abi_version = 1 : i64} {
  func.func @unsupported_intrinsic(%input: tensor<4xf32>)
      -> tensor<4xf32> {
    %result = "vernon.intrinsic"(%input) <{
      name = "unimplemented_cpu_operation"
    }> : (tensor<4xf32>) -> tensor<4xf32>
    return %result : tensor<4xf32>
  }
}
