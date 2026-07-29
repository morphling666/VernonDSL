module attributes {vernon.compiler_contract_version = 5 : i64, vernon.pipeline_version = 7 : i64} {
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
