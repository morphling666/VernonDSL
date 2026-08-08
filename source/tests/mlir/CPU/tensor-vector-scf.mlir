// RUN: %vernon-opt --vernon-lower-cpu-tensors %s | %FileCheck %s
//
// CHECK: vector<3xf32>
// CHECK: arith.addf
// CHECK: scf.if
// CHECK: scf.yield
// CHECK: arith.extf {{.*}} : vector<2xf16> to vector<2xf32>
// CHECK: arith.truncf {{.*}} : vector<2xf32> to vector<2xf16>
// CHECK-NOT: tensor<3xf32>
// CHECK-NOT: vernon.struct

module attributes {vernon.compiler_contract_version = 11 : i64, vernon.pipeline_version = 15 : i64} {
  "vernon.struct"() <{
    fields = ["value"],
    sym_name = "LoweringOnlyMetadata"
  }> : () -> ()

  func.func @select_and_add(
      %input: tensor<3xf32>, %condition: i1) -> tensor<3xf32> {
    %offset = arith.constant dense<[1.0, 2.0, 3.0]> : tensor<3xf32>
    %sum = arith.addf %input, %offset : tensor<3xf32>
    %selected = scf.if %condition -> tensor<3xf32> {
      scf.yield %sum : tensor<3xf32>
    } else {
      scf.yield %offset : tensor<3xf32>
    }
    return %selected : tensor<3xf32>
  }

  func.func @extend_f16_tensor(%input: tensor<2xf16>) -> tensor<2xf32> {
    %extended = arith.extf %input : tensor<2xf16> to tensor<2xf32>
    return %extended : tensor<2xf32>
  }

  func.func @truncate_f32_tensor(%input: tensor<2xf32>) -> tensor<2xf16> {
    %truncated = arith.truncf %input : tensor<2xf32> to tensor<2xf16>
    return %truncated : tensor<2xf16>
  }
}
