// RUN: %vernon-opt --split-input-file --verify-diagnostics \
// RUN:   --vernon-verify-generated-accumulation %s

module {
  func.func @missing_legalization(
      %gradient: !vernon.tensor_view<f32, [1], "read_write", "device">,
      %value: f32, %index: index) {
    // expected-error @+1 {{generated floating atomic has no selected legalization}}
    %old = "vernon.atomic"(%gradient, %index, %value) {
      atomic_kind = "add",
      ordering = "relaxed"
    } : (!vernon.tensor_view<f32, [1], "read_write", "device">, index, f32) -> f32
    return
  }
}

// -----

module {
  func.func @conflicting_legalization(
      %gradient: !vernon.tensor_view<f32, [1], "read_write", "workgroup">,
      %value: f32, %index: index) {
    // expected-error @+1 {{generated floating atomic requires 'native' but the target profile provides 'unsupported' for workgroup scope}}
    %old = "vernon.atomic"(%gradient, %index, %value) {
      atomic_kind = "add",
      ordering = "relaxed",
      vernon.atomic_implementation = "native"
    } : (!vernon.tensor_view<f32, [1], "read_write", "workgroup">, index, f32) -> f32
    return
  }
}

// -----

module {
  func.func @unsupported_f64_workgroup(
      %gradient: !vernon.tensor_view<f64, [1], "read_write", "workgroup">,
      %value: f64, %index: index) {
    // expected-error @+1 {{generated floating atomic requires 'native' but the target profile provides 'unsupported' for workgroup scope}}
    %old = "vernon.atomic"(%gradient, %index, %value) {
      atomic_kind = "add",
      ordering = "relaxed",
      vernon.atomic_implementation = "native"
    } : (!vernon.tensor_view<f64, [1], "read_write", "workgroup">, index, f64) -> f64
    return
  }
}
