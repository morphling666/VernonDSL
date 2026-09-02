// RUN: %vernon-opt --vernon-validate %s -o %t

module attributes {vernon.compiler_contract_version = 13 : i64, vernon.pipeline_version = 17 : i64} {
  func.func @same_lane_device_epoch(
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
    %gx_i32 = tensor.extract %gid[%zero] : tensor<3xi32>
    %gy_i32 = tensor.extract %gid[%one] : tensor<3xi32>
    %gz_i32 = tensor.extract %gid[%two] : tensor<3xi32>
    %gx = arith.index_cast %gx_i32 : i32 to index
    %gy = arith.index_cast %gy_i32 : i32 to index
    %gz = arith.index_cast %gz_i32 : i32 to index
    %offset = arith.addi %gx, %one : index
    %value = arith.constant 1.0 : f32
    "vernon.store"(%value, %scratch, %offset, %gy, %gz)
        : (f32, !vernon.tensor_view<f32, [-1, -1, -1], "read_write", "device">,
           index, index, index) -> ()
    "vernon.barrier"() {ordering = "acquire_release", scope = "workgroup"} : () -> ()
    %loaded = "vernon.load"(%scratch, %offset, %gy, %gz)
        : (!vernon.tensor_view<f32, [-1, -1, -1], "read_write", "device">,
           index, index, index) -> f32
    return
  }

  func.func @same_lane_device_epoch_2d(
      %scratch: !vernon.tensor_view<f32, [-1, -1, -1], "read_write", "device"> {
        vernon.interface = "resource", vernon.set = 0 : i64, vernon.binding = 0 : i64
      },
      %gid: tensor<3xi32> {
        vernon.interface = "input", vernon.builtin = "global_invocation_id"
      }) attributes {
        vernon.entry, vernon.stage = "compute",
        vernon.workgroup_size = array<i32: 8, 4, 1>
      } {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %two = arith.constant 2 : index
    %gx_i32 = tensor.extract %gid[%zero] : tensor<3xi32>
    %gy_i32 = tensor.extract %gid[%one] : tensor<3xi32>
    %gz_i32 = tensor.extract %gid[%two] : tensor<3xi32>
    %gx = arith.index_cast %gx_i32 : i32 to index
    %gy = arith.index_cast %gy_i32 : i32 to index
    %gz = arith.index_cast %gz_i32 : i32 to index
    %value = arith.constant 1.0 : f32
    "vernon.store"(%value, %scratch, %gy, %gx, %gz)
        : (f32, !vernon.tensor_view<f32, [-1, -1, -1], "read_write", "device">,
           index, index, index) -> ()
    "vernon.barrier"() {ordering = "acquire_release", scope = "workgroup"} : () -> ()
    %loaded = "vernon.load"(%scratch, %gy, %gx, %gz)
        : (!vernon.tensor_view<f32, [-1, -1, -1], "read_write", "device">,
           index, index, index) -> f32
    return
  }

  func.func @same_lane_device_epoch_3d(
      %scratch: !vernon.tensor_view<f32, [-1, -1, -1], "read_write", "device"> {
        vernon.interface = "resource", vernon.set = 0 : i64, vernon.binding = 0 : i64
      },
      %gid: tensor<3xi32> {
        vernon.interface = "input", vernon.builtin = "global_invocation_id"
      }) attributes {
        vernon.entry, vernon.stage = "compute",
        vernon.workgroup_size = array<i32: 4, 2, 2>
      } {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %two = arith.constant 2 : index
    %gx_i32 = tensor.extract %gid[%zero] : tensor<3xi32>
    %gy_i32 = tensor.extract %gid[%one] : tensor<3xi32>
    %gz_i32 = tensor.extract %gid[%two] : tensor<3xi32>
    %gx = arith.index_cast %gx_i32 : i32 to index
    %gy = arith.index_cast %gy_i32 : i32 to index
    %gz = arith.index_cast %gz_i32 : i32 to index
    %value = arith.constant 1.0 : f32
    "vernon.store"(%value, %scratch, %gz, %gy, %gx)
        : (f32, !vernon.tensor_view<f32, [-1, -1, -1], "read_write", "device">, index, index, index) -> ()
    "vernon.barrier"() {ordering = "acquire_release", scope = "workgroup"} : () -> ()
    %loaded = "vernon.load"(%scratch, %gz, %gy, %gx)
        : (!vernon.tensor_view<f32, [-1, -1, -1], "read_write", "device">, index, index, index) -> f32
    return
  }
}
