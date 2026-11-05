// RUN: %vernon-opt --pass-pipeline='builtin.module(vernon-to-gpu)' %s | %FileCheck %s
// RUN: %vernon-opt --pass-pipeline='builtin.module(vernon-to-gpu{unit-stride-versions=false})' %s \
// RUN:   | %FileCheck %s --check-prefix=SINGLE
//
// On CUDA the body runs in two copies, chosen by whether every innermost TensorView stride is 1; the first copy reads
// those strides as the constant 1 and the second reads them from the metadata carrier. Members 2 and 5 of the carrier
// are the two views' strides.
// CHECK-LABEL: gpu.func @copy(
// CHECK: %[[SOURCE_STRIDE:.*]] = "vernon.tuple_get"(%arg2) <{index = 2 : i64}>
// CHECK: %[[SOURCE_UNIT:.*]] = arith.cmpi eq, %[[SOURCE_STRIDE]]
// CHECK: %[[TARGET_STRIDE:.*]] = "vernon.tuple_get"(%arg2) <{index = 5 : i64}>
// CHECK: %[[TARGET_UNIT:.*]] = arith.cmpi eq, %[[TARGET_STRIDE]]
// CHECK: %[[UNIT:.*]] = arith.andi %[[SOURCE_UNIT]], %[[TARGET_UNIT]]
// CHECK: scf.if %[[UNIT]] {
// CHECK-NOT: <{index = 2 : i64}>
// CHECK-NOT: <{index = 5 : i64}>
// CHECK: memref.load %arg0
// CHECK-NOT: <{index = 5 : i64}>
// CHECK: memref.store
// CHECK: } else {
// CHECK: <{index = 2 : i64}>
// CHECK: memref.load %arg0
// CHECK: <{index = 5 : i64}>
// CHECK: memref.store
// CHECK: }
// CHECK-NEXT: gpu.return
// SINGLE-LABEL: gpu.func @copy(
// SINGLE-NOT: scf.if
// SINGLE: gpu.return

module attributes {vernon.compiler_contract_version = 1 : i64, vernon.program_version = 2 : i64} {
  func.func @copy(
      %source: !vernon.tensor_view<f32, [-1], "read", "device"> {
        vernon.element_abi_leaf_dtypes = ["f32"],
        vernon.interface = "resource",
        vernon.set = 0 : i64,
        vernon.binding = 0 : i64
      },
      %target: !vernon.tensor_view<f32, [-1], "write", "device"> {
        vernon.element_abi_leaf_dtypes = ["f32"],
        vernon.interface = "resource",
        vernon.set = 0 : i64,
        vernon.binding = 1 : i64
      }) attributes {
        vernon.entry,
        vernon.stage = "compute",
        vernon.workgroup_size = array<i32: 1, 1, 1>
      } {
    %zero = arith.constant 0 : index
    %value = "vernon.load"(%source, %zero) : (!vernon.tensor_view<f32, [-1], "read", "device">, index) -> f32
    "vernon.store"(%value, %target, %zero) : (f32, !vernon.tensor_view<f32, [-1], "write", "device">, index) -> ()
    return
  }
}
