// RUN: %not %vernon-opt --vernon-lower-cpu-tensors %s 2>&1 | %FileCheck %s
//
// CHECK: unknown Vernon intrinsic 'unimplemented_cpu_operation'
// CHECK: in CPU entry 'unsupported_intrinsic'

module attributes {vernon.compiler_contract_version = 10 : i64, vernon.pipeline_version = 14 : i64} {
  func.func @unsupported_intrinsic(%input: tensor<4xf32>)
      -> tensor<4xf32> {
    %result = "vernon.intrinsic"(%input) <{
      name = "unimplemented_cpu_operation"
    }> : (tensor<4xf32>) -> tensor<4xf32>
    return %result : tensor<4xf32>
  }
}
