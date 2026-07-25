module {
  func.func @increment(
      %values: !vernon.tensor_view<f32, 1, "read_write"> {
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
    %value = "vernon.intrinsic"(%values, %id) {
      name = "tensor_view_load"
    } : (!vernon.tensor_view<f32, 1, "read_write">, index) -> f32
    %one = arith.constant 1.0 : f32
    %sum = arith.addf %value, %one : f32
    "vernon.intrinsic"(%values, %id, %sum) {
      name = "tensor_view_store"
    } : (!vernon.tensor_view<f32, 1, "read_write">, index, f32) -> ()
    return
  }
}
