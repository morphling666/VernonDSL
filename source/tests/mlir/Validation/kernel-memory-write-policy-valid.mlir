// RUN: %vernon-opt --vernon-validate %s | %FileCheck %s
//
// CHECK-LABEL: func.func @rank_one_global_x
// CHECK-SAME: vernon.dispatch_contract = {requires_unit_workgroup = false, unit_grid_axes = array<i32: 1, 2>}
// CHECK-LABEL: func.func @valid_write_policies
// CHECK-SAME: vernon.dispatch_contract = {requires_unit_workgroup = true, unit_grid_axes = array<i32: 0, 1, 2>}
// CHECK-LABEL: func.func @partial_leader_guarded_workgroup_store
// CHECK-SAME: vernon.dispatch_contract = {requires_unit_workgroup = false, unit_grid_axes = array<i32: 1, 2>}

module attributes {vernon.compiler_contract_version = 11 : i64, vernon.pipeline_version = 15 : i64} {
  func.func @rank_one_global_x(
      %output: !vernon.tensor_view<f32, [-1], "write", "device"> {
        vernon.interface = "resource", vernon.set = 0 : i64, vernon.binding = 0 : i64
      },
      %gid: tensor<3xi32> {
        vernon.interface = "input", vernon.builtin = "global_invocation_id"
      }) attributes {
        vernon.entry, vernon.stage = "compute",
        vernon.workgroup_size = array<i32: 8, 1, 1>
      } {
    %zero = arith.constant 0 : index
    %gx_i32 = tensor.extract %gid[%zero] : tensor<3xi32>
    %gx = arith.index_castui %gx_i32 : i32 to index
    %value = arith.constant 1.0 : f32
    "vernon.store"(%value, %output, %gx)
        : (f32, !vernon.tensor_view<f32, [-1], "write", "device">, index) -> ()
    return
  }

  func.func @valid_write_policies(
      %output: !vernon.tensor_view<f32, [-1, -1, -1], "read_write", "device"> {
        vernon.interface = "resource", vernon.set = 0 : i64, vernon.binding = 0 : i64
      },
      %gid: tensor<3xi32> {
        vernon.interface = "input", vernon.builtin = "global_invocation_id"
      },
      %value: f32 {
        vernon.interface = "input", vernon.location = 0 : i64
      }) attributes {
        vernon.entry, vernon.stage = "compute",
        vernon.workgroup_size = array<i32: 1, 1, 1>,
        vernon.storage_effects = [
          {kind = "write", owner = "output", region = "element", indices = array<i64: 0>}
        ]
      } {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %two = arith.constant 2 : index
    %gx_i32 = tensor.extract %gid[%zero] : tensor<3xi32>
    %gy_i32 = tensor.extract %gid[%one] : tensor<3xi32>
    %gz_i32 = tensor.extract %gid[%two] : tensor<3xi32>
    %gx = arith.index_castui %gx_i32 : i32 to index
    %gy = arith.index_castui %gy_i32 : i32 to index
    %gz = arith.index_castui %gz_i32 : i32 to index
    "vernon.store"(%value, %output, %gx, %gy, %gz)
        : (f32, !vernon.tensor_view<f32, [-1, -1, -1], "read_write", "device">,
           index, index, index) -> ()
    "vernon.store"(%value, %output, %zero, %zero, %zero) {
      vernon.accumulation_ownership = "invocation_private"
    } : (f32, !vernon.tensor_view<f32, [-1, -1, -1], "read_write", "device">,
         index, index, index) -> ()
    %previous = "vernon.atomic"(%output, %zero, %zero, %zero, %value) {
      atomic_kind = "add",
      ordering = "relaxed"
    } : (!vernon.tensor_view<f32, [-1, -1, -1], "read_write", "device">,
         index, index, index, f32) -> f32
    "vernon.reduce_sum"(%value, %output, %zero, %zero, %zero) {
      deterministic = false
    } : (f32, !vernon.tensor_view<f32, [-1, -1, -1], "read_write", "device">,
         index, index, index) -> ()
    "vernon.scatter_add"(%value, %output, %gx, %gy, %gz) {
      deterministic = false,
      disjoint
    } : (f32, !vernon.tensor_view<f32, [-1, -1, -1], "read_write", "device">,
         index, index, index) -> ()
    return
  }

  func.func @distinct_parameter_owners(
      %output: !vernon.tensor_view<f32, [-1, -1, -1], "write", "device"> {
        vernon.interface = "resource", vernon.set = 0 : i64, vernon.binding = 0 : i64
      },
      %input: !vernon.tensor_view<f32, [-1, -1, -1], "read", "device"> {
        vernon.interface = "resource", vernon.set = 0 : i64, vernon.binding = 1 : i64
      },
      %gid: tensor<3xi32> {
        vernon.interface = "input", vernon.builtin = "global_invocation_id"
      }) attributes {
        vernon.entry, vernon.stage = "compute",
        vernon.workgroup_size = array<i32: 1, 1, 1>
      } {
    %value = arith.constant 1.0 : f32
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %two = arith.constant 2 : index
    %gx_i32 = tensor.extract %gid[%zero] : tensor<3xi32>
    %gy_i32 = tensor.extract %gid[%one] : tensor<3xi32>
    %gz_i32 = tensor.extract %gid[%two] : tensor<3xi32>
    %gx = arith.index_castui %gx_i32 : i32 to index
    %gy = arith.index_castui %gy_i32 : i32 to index
    %gz = arith.index_castui %gz_i32 : i32 to index
    "vernon.store"(%value, %output, %gx, %gy, %gz)
        : (f32, !vernon.tensor_view<f32, [-1, -1, -1], "write", "device">,
           index, index, index) -> ()
    "vernon.barrier"() {ordering = "acquire_release", scope = "workgroup"} : () -> ()
    %loaded = "vernon.load"(%input, %gx, %gy, %gz)
        : (!vernon.tensor_view<f32, [-1, -1, -1], "read", "device">,
           index, index, index) -> f32
    return
  }

  func.func @same_width_bitcast_global_id(
      %output: !vernon.tensor_view<f32, [-1, -1, -1], "write", "device"> {
        vernon.interface = "resource", vernon.set = 0 : i64, vernon.binding = 0 : i64
      },
      %gid: tensor<3xi32> {
        vernon.interface = "input", vernon.builtin = "global_invocation_id"
      }) attributes {
        vernon.entry, vernon.stage = "compute",
        vernon.workgroup_size = array<i32: 4, 1, 1>
      } {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %two = arith.constant 2 : index
    %gx_i32 = tensor.extract %gid[%zero] : tensor<3xi32>
    %gy_i32 = tensor.extract %gid[%one] : tensor<3xi32>
    %gz_i32 = tensor.extract %gid[%two] : tensor<3xi32>
    %gx_bits = arith.bitcast %gx_i32 : i32 to i32
    %gy_bits = arith.bitcast %gy_i32 : i32 to i32
    %gz_bits = arith.bitcast %gz_i32 : i32 to i32
    %gx = arith.index_castui %gx_bits : i32 to index
    %gy = arith.index_castui %gy_bits : i32 to index
    %gz = arith.index_castui %gz_bits : i32 to index
    %value = arith.constant 1.0 : f32
    "vernon.store"(%value, %output, %gx, %gy, %gz)
        : (f32, !vernon.tensor_view<f32, [-1, -1, -1], "write", "device">,
           index, index, index) -> ()
    return
  }

  func.func @reconstructed_global_id(
      %output: !vernon.tensor_view<f32, [-1, -1, -1], "write", "device"> {
        vernon.interface = "resource", vernon.set = 0 : i64, vernon.binding = 0 : i64
      },
      %group: tensor<3xi32> {
        vernon.interface = "input", vernon.builtin = "workgroup_id"
      },
      %local: tensor<3xi32> {
        vernon.interface = "input", vernon.builtin = "local_invocation_id"
      }) attributes {
        vernon.entry, vernon.stage = "compute",
        vernon.workgroup_size = array<i32: 8, 1, 1>
      } {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %two = arith.constant 2 : index
    %eight = arith.constant 8 : i32
    %unit = arith.constant 1 : i32
    %group_x = tensor.extract %group[%zero] : tensor<3xi32>
    %group_y = tensor.extract %group[%one] : tensor<3xi32>
    %group_z = tensor.extract %group[%two] : tensor<3xi32>
    %local_x = tensor.extract %local[%zero] : tensor<3xi32>
    %local_y = tensor.extract %local[%one] : tensor<3xi32>
    %local_z = tensor.extract %local[%two] : tensor<3xi32>
    %base_x = arith.muli %group_x, %eight : i32
    %base_y = arith.muli %group_y, %unit : i32
    %base_z = arith.muli %group_z, %unit : i32
    %lane_x = arith.addi %base_x, %local_x : i32
    %lane_y = arith.addi %base_y, %local_y : i32
    %lane_z = arith.addi %base_z, %local_z : i32
    %index_x = arith.index_castui %lane_x : i32 to index
    %index_y = arith.index_castui %lane_y : i32 to index
    %index_z = arith.index_castui %lane_z : i32 to index
    %value = arith.constant 1.0 : f32
    "vernon.store"(%value, %output, %index_x, %index_y, %index_z)
        : (f32, !vernon.tensor_view<f32, [-1, -1, -1], "write", "device">,
           index, index, index) -> ()
    return
  }

  func.func @leader_guarded_workgroup_store(
      %output: !vernon.tensor_view<f32, [-1, -1, -1], "write", "device"> {
        vernon.interface = "resource", vernon.set = 0 : i64, vernon.binding = 0 : i64
      },
      %group: tensor<3xi32> {
        vernon.interface = "input", vernon.builtin = "workgroup_id"
      },
      %local: tensor<3xi32> {
        vernon.interface = "input", vernon.builtin = "local_invocation_id"
      }) attributes {
        vernon.entry, vernon.stage = "compute",
        vernon.workgroup_size = array<i32: 8, 1, 1>
      } {
    %zero_index = arith.constant 0 : index
    %one_index = arith.constant 1 : index
    %two_index = arith.constant 2 : index
    %zero = arith.constant 0 : i32
    %local_x = tensor.extract %local[%zero_index] : tensor<3xi32>
    %leader = arith.cmpi eq, %local_x, %zero : i32
    scf.if %leader {
      %group_x = tensor.extract %group[%zero_index] : tensor<3xi32>
      %group_y = tensor.extract %group[%one_index] : tensor<3xi32>
      %group_z = tensor.extract %group[%two_index] : tensor<3xi32>
      %index_x = arith.index_castui %group_x : i32 to index
      %index_y = arith.index_castui %group_y : i32 to index
      %index_z = arith.index_castui %group_z : i32 to index
      %value = arith.constant 1.0 : f32
      "vernon.store"(%value, %output, %index_x, %index_y, %index_z)
          : (f32, !vernon.tensor_view<f32, [-1, -1, -1], "write", "device">,
             index, index, index) -> ()
    }
    return
  }

  func.func @partial_leader_guarded_workgroup_store(
      %output: !vernon.tensor_view<f32, [-1], "write", "device"> {
        vernon.interface = "resource", vernon.set = 0 : i64, vernon.binding = 0 : i64
      },
      %group: tensor<3xi32> {
        vernon.interface = "input", vernon.builtin = "workgroup_id"
      },
      %local: tensor<3xi32> {
        vernon.interface = "input", vernon.builtin = "local_invocation_id"
      }) attributes {
        vernon.entry, vernon.stage = "compute",
        vernon.workgroup_size = array<i32: 8, 1, 1>
      } {
    %zero_index = arith.constant 0 : index
    %zero = arith.constant 0 : i32
    %local_x = tensor.extract %local[%zero_index] : tensor<3xi32>
    %leader = arith.cmpi eq, %local_x, %zero : i32
    scf.if %leader {
      %group_x = tensor.extract %group[%zero_index] : tensor<3xi32>
      %index_x = arith.index_castui %group_x : i32 to index
      %value = arith.constant 1.0 : f32
      "vernon.store"(%value, %output, %index_x)
          : (f32, !vernon.tensor_view<f32, [-1], "write", "device">, index) -> ()
    }
    return
  }
}
