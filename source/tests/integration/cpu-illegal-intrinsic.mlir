module attributes {vernon.compiler_contract_version = 9 : i64, vernon.pipeline_version = 12 : i64} {
  func.func @unsupported_intrinsic(%input: tensor<4xf32>)
      -> tensor<4xf32> {
    %result = "vernon.intrinsic"(%input) <{
      name = "unimplemented_cpu_operation"
    }> : (tensor<4xf32>) -> tensor<4xf32>
    return %result : tensor<4xf32>
  }
}
