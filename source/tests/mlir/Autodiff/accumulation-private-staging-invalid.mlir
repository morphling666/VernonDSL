// RUN: %vernon-opt --verify-diagnostics \
// RUN:   --vernon-lower-accumulation="aggregate-gradient-storage=invocation-private-staging" \
// RUN:   %s

module {
  func.func @target_model_is_not_ownership_proof(
      %gradient: !vernon.tensor_view<tensor<2xf32>, [1], "read_write", "device">,
      %value: tensor<2xf32>, %index: index) {
    // expected-error @+1 {{shared shaped accumulation requires proven invocation-private gradient staging}}
    "vernon.scatter_add"(%value, %gradient, %index) {
      deterministic = false
    } : (tensor<2xf32>,
         !vernon.tensor_view<tensor<2xf32>, [1], "read_write", "device">,
         index) -> ()
    return
  }
}
