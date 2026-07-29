module attributes {vernon.frontend_version = 4 : i64, vernon.value_abi_version = 1 : i64} {
  func.func @tensor_view_round_trip(
      %view: !vernon.tensor_view<f32, [-1], "read_write", "device"> {
        vernon.tensor_strides = array<i64: 1>,
        vernon.tensor_offset = 0 : i64
      },
      %index: index, %value: f32) {
    %loaded = "vernon.load"(%view, %index) :
      (!vernon.tensor_view<f32, [-1], "read_write", "device">, index) -> f32
    "vernon.store"(%value, %view, %index) :
      (f32, !vernon.tensor_view<f32, [-1], "read_write", "device">, index) -> ()
    return
  }
}
