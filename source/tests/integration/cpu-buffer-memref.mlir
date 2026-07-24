module {
  func.func @tensor_view_round_trip(
      %view: !vernon.tensor_view<f32, 1, "read_write">,
      %index: index, %value: f32) {
    %loaded = "vernon.intrinsic"(%view, %index) <{
      name = "tensor_view_load"
    }> : (!vernon.tensor_view<f32, 1, "read_write">, index) -> f32
    "vernon.intrinsic"(%view, %index, %value) <{
      name = "tensor_view_store"
    }> : (!vernon.tensor_view<f32, 1, "read_write">, index, f32) -> ()
    return
  }
}
