// RUN: %vernon-opt --split-input-file --verify-diagnostics --vernon-lower-accumulation %s

module {
  func.func @deterministic_needs_reduction_kernel(
      %gradient: !vernon.tensor_view<f32, [1], "read_write", "device">,
      %value: f32, %index: index) {
    // expected-error @+1 {{deterministic shared accumulation requires a dedicated reduction kernel}}
    "vernon.reduce_sum"(%value, %gradient, %index) {
      deterministic = true
    } : (f32, !vernon.tensor_view<f32, [1], "read_write", "device">,
         index) -> ()
    return
  }
}

// -----

module {
  func.func @scalar_needs_atomic(
      %gradient: !vernon.tensor_view<f32, [1], "read_write", "device">,
      %value: f32, %index: index) {
    // expected-error @+1 {{shared scalar accumulation requires a supported atomic add}}
    "vernon.reduce_sum"(%value, %gradient, %index) {
      deterministic = false
    } : (f32, !vernon.tensor_view<f32, [1], "read_write", "device">,
         index) -> ()
    return
  }
}

// -----

module {
  func.func @f64_device_needs_atomic(
      %gradient: !vernon.tensor_view<f64, [1], "read_write", "device">,
      %value: f64, %index: index) {
    // expected-error @+1 {{shared scalar accumulation requires a supported atomic add}}
    "vernon.reduce_sum"(%value, %gradient, %index) {deterministic = false} :
      (f64, !vernon.tensor_view<f64, [1], "read_write", "device">, index) -> ()
    return
  }
}

// -----

module {
  func.func @shaped_needs_proven_private_staging(
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

// -----

module {
  func.func @private_metadata_needs_target_capability(
      %gradient: !vernon.tensor_view<tensor<2xf32>, [1], "read_write", "device">,
      %value: tensor<2xf32>, %index: index) {
    // expected-error @+1 {{invocation-private accumulation requires target support for invocation-private gradient staging}}
    "vernon.scatter_add"(%value, %gradient, %index) {
      deterministic = false,
      vernon.accumulation_ownership = "invocation_private"
    } : (tensor<2xf32>,
         !vernon.tensor_view<tensor<2xf32>, [1], "read_write", "device">,
         index) -> ()
    return
  }
}

// -----

module {
  func.func @unproven_disjoint(
      %gradient: !vernon.tensor_view<f32, [1], "read_write", "device">,
      %value: f32, %unknown: index) {
    // expected-error @+1 {{scatter_add 'disjoint' hint requires a lane-exclusive global invocation index proof}}
    "vernon.scatter_add"(%value, %gradient, %unknown) {
      deterministic = false,
      disjoint
    } : (f32, !vernon.tensor_view<f32, [1], "read_write", "device">,
         index) -> ()
    return
  }
}

// -----

module {
  func.func @private_metadata_requires_private_staging(
      %gradient: !vernon.tensor_view<f32, [1], "read_write", "device">,
      %value: f32, %index: index) {
    // expected-error @+1 {{invocation-private accumulation requires target support for invocation-private gradient staging}}
    "vernon.scatter_add"(%value, %gradient, %index) {
      deterministic = false,
      vernon.accumulation_ownership = "invocation_private"
    } : (f32, !vernon.tensor_view<f32, [1], "read_write", "device">,
         index) -> ()
    return
  }
}
