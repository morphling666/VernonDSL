// RUN: %vernon-opt \
// RUN:   --vernon-lower-accumulation="f32-device-atomic=native f64-device-atomic=native f32-workgroup-atomic=native f64-workgroup-atomic=native" \
// RUN:   %s | %FileCheck %s --check-prefix=NATIVE
// RUN: %vernon-opt \
// RUN:   --vernon-lower-accumulation="f32-device-atomic=integer_cas f64-device-atomic=integer_cas f32-workgroup-atomic=integer_cas f64-workgroup-atomic=integer_cas" \
// RUN:   %s | %FileCheck %s --check-prefix=CAS
//
// NATIVE-COUNT-4: "vernon.physical_atomic"{{.*}}vernon.atomic_implementation = "native"
// CAS-COUNT-4: "vernon.physical_atomic"{{.*}}vernon.atomic_implementation = "integer_cas"
// NATIVE-NOT: vernon.reduce_sum
// CAS-NOT: vernon.reduce_sum

module {
  func.func @f32_device(
      %gradient: !vernon.tensor_view<f32, [1], "read_write", "device">,
      %value: f32, %index: index) {
    "vernon.reduce_sum"(%value, %gradient, %index) {deterministic = false} :
      (f32, !vernon.tensor_view<f32, [1], "read_write", "device">, index) -> ()
    return
  }

  func.func @f64_device(
      %gradient: !vernon.tensor_view<f64, [1], "read_write", "device">,
      %value: f64, %index: index) {
    "vernon.reduce_sum"(%value, %gradient, %index) {deterministic = false} :
      (f64, !vernon.tensor_view<f64, [1], "read_write", "device">, index) -> ()
    return
  }

  func.func @f32_workgroup(
      %gradient: !vernon.tensor_view<f32, [1], "read_write", "workgroup">,
      %value: f32, %index: index) {
    %old = "vernon.atomic"(%gradient, %index, %value) {
      atomic_kind = "add", ordering = "relaxed"
    } : (!vernon.tensor_view<f32, [1], "read_write", "workgroup">, index, f32) -> f32
    return
  }

  func.func @f64_workgroup(
      %gradient: !vernon.tensor_view<f64, [1], "read_write", "workgroup">,
      %value: f64, %index: index) {
    %old = "vernon.atomic"(%gradient, %index, %value) {
      atomic_kind = "add", ordering = "relaxed"
    } : (!vernon.tensor_view<f64, [1], "read_write", "workgroup">, index, f64) -> f64
    return
  }
}
