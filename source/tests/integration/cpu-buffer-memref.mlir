module {
  func.func @buffer_round_trip(
      %buffer: !vernon.buffer<f32, "read_write">,
      %index: index, %value: f32) {
    %loaded = "vernon.intrinsic"(%buffer, %index) <{
      name = "buffer_load"
    }> : (!vernon.buffer<f32, "read_write">, index) -> f32
    "vernon.intrinsic"(%buffer, %index, %value) <{
      name = "buffer_store"
    }> : (!vernon.buffer<f32, "read_write">, index, f32) -> ()
    return
  }
}
