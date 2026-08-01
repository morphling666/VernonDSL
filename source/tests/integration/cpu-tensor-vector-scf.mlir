module attributes {vernon.compiler_contract_version = 8 : i64, vernon.pipeline_version = 10 : i64} {
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
