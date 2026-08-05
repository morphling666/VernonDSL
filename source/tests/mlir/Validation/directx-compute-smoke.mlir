// RUN: %vernon-opt --vernon-validate %s -o %t

module attributes {vernon.compiler_contract_version = 10 : i64, vernon.pipeline_version = 13 : i64} {
  func.func @increment(
      %values: !vernon.tensor_view<f32, [1], "read_write", "device"> {
        vernon.interface = "resource",
        vernon.set = 0 : i64,
        vernon.binding = 0 : i64
      },
      %id: index {
        vernon.interface = "input",
        vernon.builtin = "global_invocation_id"
      }) attributes {
        vernon.entry,
        vernon.stage = "compute",
        vernon.workgroup_size = array<i32: 8, 1, 1>
      } {
    %value = "vernon.load"(%values, %id) :
      (!vernon.tensor_view<f32, [1], "read_write", "device">, index) -> f32
    %one = arith.constant 1.0 : f32
    %sum = arith.addf %value, %one : f32
    "vernon.store"(%sum, %values, %id) :
      (f32, !vernon.tensor_view<f32, [1], "read_write", "device">, index) -> ()
    return
  }
}
