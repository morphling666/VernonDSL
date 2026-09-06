// RUN: %not %vernon-opt --vernon-validate %s 2>&1 | %FileCheck %s
//
// CHECK-COUNT-12: ordinary device TensorView store is not proven lane-exclusive
// CHECK: scatter_add 'disjoint' hint requires a lane-exclusive global invocation index proof

module attributes {vernon.compiler_contract_version = 14 : i64, vernon.program_version = 19 : i64} {
  func.func @unproven_store(
      %output: !vernon.tensor_view<f32, [-1], "write", "device"> {
        vernon.interface = "resource", vernon.set = 0 : i64, vernon.binding = 0 : i64
      },
      %unknown: index,
      %value: f32) attributes {
        vernon.entry, vernon.stage = "compute",
        vernon.workgroup_size = array<i32: 2, 1, 1>
      } {
    "vernon.store"(%value, %output, %unknown)
        : (f32, !vernon.tensor_view<f32, [-1], "write", "device">, index) -> ()
    return
  }

  func.func @constant_store_with_multiple_workgroup_lanes(
      %output: !vernon.tensor_view<f32, [-1], "write", "device"> {
        vernon.interface = "resource", vernon.set = 0 : i64, vernon.binding = 0 : i64
      },
      %value: f32) attributes {
        vernon.entry, vernon.stage = "compute",
        vernon.workgroup_size = array<i32: 4, 1, 1>
      } {
    %zero = arith.constant 0 : index
    "vernon.store"(%value, %output, %zero)
        : (f32, !vernon.tensor_view<f32, [-1], "write", "device">, index) -> ()
    return
  }

  func.func @local_id_only_across_workgroups(
      %output: !vernon.tensor_view<f32, [-1], "write", "device">,
      %local: i32 {vernon.builtin = "local_invocation_id"}) attributes {
        vernon.entry, vernon.stage = "compute",
        vernon.workgroup_size = array<i32: 4, 1, 1>
      } {
    %index = arith.index_castui %local : i32 to index
    %value = arith.constant 1.0 : f32
    "vernon.store"(%value, %output, %index)
        : (f32, !vernon.tensor_view<f32, [-1], "write", "device">, index) -> ()
    return
  }

  func.func @group_id_without_leader_guard(
      %output: !vernon.tensor_view<f32, [-1], "write", "device">,
      %group: i32 {vernon.builtin = "workgroup_id"}) attributes {
        vernon.entry, vernon.stage = "compute",
        vernon.workgroup_size = array<i32: 4, 1, 1>
      } {
    %index = arith.index_castui %group : i32 to index
    %value = arith.constant 1.0 : f32
    "vernon.store"(%value, %output, %index)
        : (f32, !vernon.tensor_view<f32, [-1], "write", "device">, index) -> ()
    return
  }

  func.func @group_id_under_unknown_predicate(
      %output: !vernon.tensor_view<f32, [-1], "write", "device">,
      %group: i32 {vernon.builtin = "workgroup_id"},
      %predicate: i1) attributes {
        vernon.entry, vernon.stage = "compute",
        vernon.workgroup_size = array<i32: 4, 1, 1>
      } {
    scf.if %predicate {
      %index = arith.index_castui %group : i32 to index
      %value = arith.constant 1.0 : f32
      "vernon.store"(%value, %output, %index)
          : (f32, !vernon.tensor_view<f32, [-1], "write", "device">, index) -> ()
    }
    return
  }

  func.func @reconstructed_id_with_wrong_static_size(
      %output: !vernon.tensor_view<f32, [-1], "write", "device">,
      %group: i32 {vernon.builtin = "workgroup_id"},
      %local: i32 {vernon.builtin = "local_invocation_id"}) attributes {
        vernon.entry, vernon.stage = "compute",
        vernon.workgroup_size = array<i32: 8, 1, 1>
      } {
    %four = arith.constant 4 : i32
    %base = arith.muli %group, %four : i32
    %lane = arith.addi %base, %local : i32
    %index = arith.index_castui %lane : i32 to index
    %value = arith.constant 1.0 : f32
    "vernon.store"(%value, %output, %index)
        : (f32, !vernon.tensor_view<f32, [-1], "write", "device">, index) -> ()
    return
  }

  func.func @global_id_missing_active_workgroup_axis(
      %output: !vernon.tensor_view<f32, [-1], "write", "device">,
      %gid: tensor<3xi32> {vernon.builtin = "global_invocation_id"}) attributes {
        vernon.entry, vernon.stage = "compute",
        vernon.workgroup_size = array<i32: 4, 1, 1>
      } {
    %one = arith.constant 1 : index
    %gy_i32 = tensor.extract %gid[%one] : tensor<3xi32>
    %gy = arith.index_castui %gy_i32 : i32 to index
    %value = arith.constant 1.0 : f32
    "vernon.store"(%value, %output, %gy)
        : (f32, !vernon.tensor_view<f32, [-1], "write", "device">, index) -> ()
    return
  }

  func.func @remainder_global_id(
      %output: !vernon.tensor_view<f32, [-1], "write", "device">,
      %gid: i32 {vernon.builtin = "global_invocation_id"}) attributes {
        vernon.entry, vernon.stage = "compute",
        vernon.workgroup_size = array<i32: 4, 1, 1>
      } {
    %four = arith.constant 4 : i32
    %wrapped = arith.remui %gid, %four : i32
    %index = arith.index_castui %wrapped : i32 to index
    %value = arith.constant 1.0 : f32
    "vernon.store"(%value, %output, %index)
        : (f32, !vernon.tensor_view<f32, [-1], "write", "device">, index) -> ()
    return
  }

  func.func @divided_global_id(
      %output: !vernon.tensor_view<f32, [-1], "write", "device">,
      %gid: i32 {vernon.builtin = "global_invocation_id"}) attributes {
        vernon.entry, vernon.stage = "compute",
        vernon.workgroup_size = array<i32: 4, 1, 1>
      } {
    %four = arith.constant 4 : i32
    %quotient = arith.divui %gid, %four : i32
    %index = arith.index_castui %quotient : i32 to index
    %value = arith.constant 1.0 : f32
    "vernon.store"(%value, %output, %index)
        : (f32, !vernon.tensor_view<f32, [-1], "write", "device">, index) -> ()
    return
  }

  func.func @truncated_global_id(
      %output: !vernon.tensor_view<f32, [-1], "write", "device">,
      %gid: i64 {vernon.builtin = "global_invocation_id"}) attributes {
        vernon.entry, vernon.stage = "compute",
        vernon.workgroup_size = array<i32: 4, 1, 1>
      } {
    %short = arith.trunci %gid : i64 to i32
    %index = arith.index_castui %short : i32 to index
    %value = arith.constant 1.0 : f32
    "vernon.store"(%value, %output, %index)
        : (f32, !vernon.tensor_view<f32, [-1], "write", "device">, index) -> ()
    return
  }

  func.func @scalar_global_id_is_x_only(
      %output: !vernon.tensor_view<f32, [-1], "write", "device">,
      %gid: index {vernon.builtin = "global_invocation_id"}) attributes {
        vernon.entry, vernon.stage = "compute",
        vernon.workgroup_size = array<i32: 1, 1, 1>
      } {
    %value = arith.constant 1.0 : f32
    "vernon.store"(%value, %output, %gid)
        : (f32, !vernon.tensor_view<f32, [-1], "write", "device">, index) -> ()
    return
  }

  func.func @full_workgroup_tuple_without_leader_guard(
      %output: !vernon.tensor_view<f32, [-1, -1, -1], "write", "device">,
      %group: tensor<3xi32> {vernon.builtin = "workgroup_id"}) attributes {
        vernon.entry, vernon.stage = "compute",
        vernon.workgroup_size = array<i32: 4, 1, 1>
      } {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %two = arith.constant 2 : index
    %gx_i32 = tensor.extract %group[%zero] : tensor<3xi32>
    %gy_i32 = tensor.extract %group[%one] : tensor<3xi32>
    %gz_i32 = tensor.extract %group[%two] : tensor<3xi32>
    %gx = arith.index_castui %gx_i32 : i32 to index
    %gy = arith.index_castui %gy_i32 : i32 to index
    %gz = arith.index_castui %gz_i32 : i32 to index
    %value = arith.constant 1.0 : f32
    "vernon.store"(%value, %output, %gx, %gy, %gz)
        : (f32, !vernon.tensor_view<f32, [-1, -1, -1], "write", "device">,
           index, index, index) -> ()
    return
  }

  func.func @leader_guard_with_partial_workgroup_tuple(
      %output: !vernon.tensor_view<f32, [-1], "write", "device">,
      %group: tensor<3xi32> {vernon.builtin = "workgroup_id"},
      %local: tensor<3xi32> {vernon.builtin = "local_invocation_id"}) attributes {
        vernon.entry, vernon.stage = "compute",
        vernon.workgroup_size = array<i32: 4, 1, 1>
      } {
    %zero_index = arith.constant 0 : index
    %zero = arith.constant 0 : i32
    %local_x = tensor.extract %local[%zero_index] : tensor<3xi32>
    %leader = arith.cmpi eq, %local_x, %zero : i32
    scf.if %leader {
      %group_x_i32 = tensor.extract %group[%zero_index] : tensor<3xi32>
      %group_x = arith.index_castui %group_x_i32 : i32 to index
      %value = arith.constant 1.0 : f32
      "vernon.store"(%value, %output, %group_x)
          : (f32, !vernon.tensor_view<f32, [-1], "write", "device">, index) -> ()
    }
    return
  }

  func.func @invalid_disjoint_scatter(
      %output: !vernon.tensor_view<f32, [-1], "read_write", "device"> {
        vernon.interface = "resource", vernon.set = 0 : i64, vernon.binding = 0 : i64
      },
      %unknown: index,
      %value: f32) attributes {
        vernon.entry, vernon.stage = "compute",
        vernon.workgroup_size = array<i32: 1, 1, 1>
      } {
    "vernon.scatter_add"(%value, %output, %unknown) {
      deterministic = false,
      disjoint
    } : (f32, !vernon.tensor_view<f32, [-1], "read_write", "device">, index) -> ()
    return
  }
}
