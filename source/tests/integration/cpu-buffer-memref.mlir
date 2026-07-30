module attributes {vernon.compiler_contract_version = 7 : i64, vernon.pipeline_version = 10 : i64} {
  func.func @tensor_view_round_trip(
      %view: !vernon.tensor_view<f32, [-1], "read_write", "device">,
      %index: index, %value: f32) {
    %loaded = "vernon.load"(%view, %index) :
      (!vernon.tensor_view<f32, [-1], "read_write", "device">, index) -> f32
    "vernon.store"(%value, %view, %index) :
      (f32, !vernon.tensor_view<f32, [-1], "read_write", "device">, index) -> ()
    return
  }
}
