// RUN: %vernon-opt --vernon-validate %s -o %t

module attributes {vernon.compiler_contract_version = 11 : i64, vernon.pipeline_version = 15 : i64} {
  func.func @increment(
      %values: !vernon.tensor_view<f32, [1, 1, 1], "read_write", "device"> {
        vernon.interface = "resource",
        vernon.set = 0 : i64,
        vernon.binding = 0 : i64
      },
      %id: tensor<3xi32> {
        vernon.interface = "input",
        vernon.builtin = "global_invocation_id"
      }) attributes {
        vernon.entry,
        vernon.stage = "compute",
        vernon.workgroup_size = array<i32: 8, 1, 1>
      } {
    %zero = arith.constant 0 : index
    %one_index = arith.constant 1 : index
    %two = arith.constant 2 : index
    %gx_i32 = tensor.extract %id[%zero] : tensor<3xi32>
    %gy_i32 = tensor.extract %id[%one_index] : tensor<3xi32>
    %gz_i32 = tensor.extract %id[%two] : tensor<3xi32>
    %gx = arith.index_castui %gx_i32 : i32 to index
    %gy = arith.index_castui %gy_i32 : i32 to index
    %gz = arith.index_castui %gz_i32 : i32 to index
    %value = "vernon.load"(%values, %gx, %gy, %gz) :
      (!vernon.tensor_view<f32, [1, 1, 1], "read_write", "device">,
       index, index, index) -> f32
    %one = arith.constant 1.0 : f32
    %sum = arith.addf %value, %one : f32
    "vernon.store"(%sum, %values, %gx, %gy, %gz) :
      (f32, !vernon.tensor_view<f32, [1, 1, 1], "read_write", "device">,
       index, index, index) -> ()
    return
  }
}
