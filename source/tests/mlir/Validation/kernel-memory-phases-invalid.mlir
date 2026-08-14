// RUN: %not %vernon-opt --vernon-validate %s 2>&1 | %FileCheck %s
//
// CHECK-COUNT-2: device TensorView write followed by a workgroup barrier and cross-lane or unproven read

module attributes {vernon.compiler_contract_version = 12 : i64, vernon.pipeline_version = 16 : i64} {
  func.func @cross_lane_device_epoch(
      %scratch: !vernon.tensor_view<f32, [-1, -1, -1], "read_write", "device"> {
        vernon.interface = "resource", vernon.set = 0 : i64, vernon.binding = 0 : i64
      },
      %gid: tensor<3xi32> {
        vernon.interface = "input", vernon.builtin = "global_invocation_id"
      }) attributes {
        vernon.entry, vernon.stage = "compute",
        vernon.workgroup_size = array<i32: 8, 1, 1>
      } {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %two = arith.constant 2 : index
    %eight = arith.constant 8 : index
    %gx_i32 = tensor.extract %gid[%zero] : tensor<3xi32>
    %gy_i32 = tensor.extract %gid[%one] : tensor<3xi32>
    %gz_i32 = tensor.extract %gid[%two] : tensor<3xi32>
    %gx = arith.index_cast %gx_i32 : i32 to index
    %gy = arith.index_cast %gy_i32 : i32 to index
    %gz = arith.index_cast %gz_i32 : i32 to index
    %value = arith.constant 1.0 : f32
    "vernon.store"(%value, %scratch, %gx, %gy, %gz)
        : (f32, !vernon.tensor_view<f32, [-1, -1, -1], "read_write", "device">,
           index, index, index) -> ()
    "vernon.barrier"() {ordering = "acquire_release", scope = "workgroup"} : () -> ()
    %next = arith.addi %gx, %one : index
    %neighbor = arith.remui %next, %eight : index
    %loaded = "vernon.load"(%scratch, %neighbor, %gy, %gz)
        : (!vernon.tensor_view<f32, [-1, -1, -1], "read_write", "device">,
           index, index, index) -> f32
    return
  }

  func.func @nested_barrier_device_epoch(
      %scratch: !vernon.tensor_view<f32, [-1, -1, -1], "read_write", "device"> {
        vernon.interface = "resource", vernon.set = 0 : i64, vernon.binding = 0 : i64
      },
      %gid: tensor<3xi32> {
        vernon.interface = "input", vernon.builtin = "global_invocation_id"
      }) attributes {
        vernon.entry, vernon.stage = "compute",
        vernon.workgroup_size = array<i32: 8, 1, 1>
      } {
    %true = arith.constant true
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %two = arith.constant 2 : index
    %eight = arith.constant 8 : index
    %gx_i32 = tensor.extract %gid[%zero] : tensor<3xi32>
    %gy_i32 = tensor.extract %gid[%one] : tensor<3xi32>
    %gz_i32 = tensor.extract %gid[%two] : tensor<3xi32>
    %gx = arith.index_cast %gx_i32 : i32 to index
    %gy = arith.index_cast %gy_i32 : i32 to index
    %gz = arith.index_cast %gz_i32 : i32 to index
    %value = arith.constant 1.0 : f32
    "vernon.store"(%value, %scratch, %gx, %gy, %gz)
        : (f32, !vernon.tensor_view<f32, [-1, -1, -1], "read_write", "device">,
           index, index, index) -> ()
    scf.if %true {
      "vernon.barrier"() {ordering = "acquire_release", scope = "workgroup"} : () -> ()
    }
    %next = arith.addi %gx, %one : index
    %neighbor = arith.remui %next, %eight : index
    %loaded = "vernon.load"(%scratch, %neighbor, %gy, %gz)
        : (!vernon.tensor_view<f32, [-1, -1, -1], "read_write", "device">,
           index, index, index) -> f32
    return
  }
}
