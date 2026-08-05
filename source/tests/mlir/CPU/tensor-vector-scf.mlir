// RUN: %vernon-opt --vernon-lower-cpu-tensors %s | %FileCheck %s
//
// CHECK: vector<3xf32>
// CHECK: arith.addf
// CHECK: scf.if
// CHECK: scf.yield
// CHECK-NOT: tensor<3xf32>
// CHECK-NOT: vernon.struct

module attributes {vernon.compiler_contract_version = 10 : i64, vernon.pipeline_version = 13 : i64} {
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
}
