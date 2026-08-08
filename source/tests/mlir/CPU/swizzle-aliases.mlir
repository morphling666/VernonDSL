// RUN: %vernon-opt --vernon-lower-cpu-tensors %s | %FileCheck %s
//
// CHECK: vector.extract
// CHECK: [3]
// CHECK-NOT: vernon.swizzle

module attributes {vernon.compiler_contract_version = 10 : i64, vernon.pipeline_version = 14 : i64} {
  func.func @swizzle_aliases(%input: tensor<4xf32>)
      -> (tensor<4xf32>, tensor<3xf32>, f32) {
    %rgba = "vernon.swizzle"(%input) {
      mask = "rgba"
    } : (tensor<4xf32>) -> tensor<4xf32>
    %rgb = "vernon.swizzle"(%input) {
      mask = "rgb"
    } : (tensor<4xf32>) -> tensor<3xf32>
    %alpha = "vernon.swizzle"(%input) {
      mask = "a"
    } : (tensor<4xf32>) -> f32
    return %rgba, %rgb, %alpha : tensor<4xf32>, tensor<3xf32>, f32
  }
}
