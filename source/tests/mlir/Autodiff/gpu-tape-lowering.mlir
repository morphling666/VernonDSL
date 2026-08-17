// RUN: %vernon-opt --vernon-lower-gpu-autodiff %s | %FileCheck %s

// CHECK-LABEL: func.func @static_forward(
// CHECK-SAME: !vernon.tensor_view<i32, [-1], "read_write", "device">
// CHECK-SAME: vernon.autodiff_role = "tape"
// CHECK-SAME: vernon.source_name = "__vernon_ad_tape"
// CHECK-SAME: !vernon.tensor_view<i32, [-1], "read", "device">
// CHECK-SAME: vernon.source_name = "__vernon_ad_segment"
// CHECK-SAME: !vernon.tensor_view<i32, [-1], "read_write", "device">
// CHECK-SAME: vernon.source_name = "__vernon_ad_status"
// CHECK-SAME: tensor<3xi32> {vernon.builtin = "local_invocation_id"
// CHECK-NOT: !vernon.ad_tape
// CHECK-NOT: !vernon.ad_region_header
// CHECK-NOT: arith.constant {{.*}} : i64
// CHECK: "vernon.physical_load"
// CHECK: "vernon.physical_store"
// CHECK-COUNT-2: "vernon.physical_atomic"
// CHECK-SAME: atomic_kind = "umax"
// CHECK-SAME: ordering = "relaxed"
// CHECK: return

// CHECK-LABEL: func.func @static_backward(
// CHECK-SAME: !vernon.tensor_view<i32, [-1], "read_write", "device">
// CHECK-SAME: !vernon.tensor_view<i32, [-1], "read", "device">
// CHECK-NOT: vernon.source_name = "__vernon_ad_status"
// CHECK-NOT: !vernon.ad_tape
// CHECK-NOT: !vernon.ad_region_header
// CHECK-NOT: arith.constant {{.*}} : i64
// CHECK: "vernon.physical_load"
// CHECK: "vernon.physical_load"
// CHECK: vector.bitcast
// CHECK: return

module attributes {vernon.ad_profile = "static"} {
  func.func @static_forward(
      %gid: tensor<3xi32> {vernon.builtin = "global_invocation_id"},
      %lid: tensor<3xi32> {vernon.builtin = "local_invocation_id"})
      -> tuple<!vernon.ad_tape, !vernon.ad_region_header>
      attributes {
        vernon.ad.residual_storage = "static",
        vernon.entry,
        vernon.stage = "compute",
        vernon.workgroup_size = array<i32: 8, 1, 1>
      } {
    %state:5 = "vernon.ad.capture"() ({
      %tape = "vernon.ad.begin_invocation"() : () -> !vernon.ad_tape
      %root = "vernon.ad.begin_region"(%tape)
          : (!vernon.ad_tape) -> !vernon.ad_region_header
      %record = "vernon.ad.reserve_record"(%root)
          {record_alignment = 8 : i64, record_size = 8 : i64}
          : (!vernon.ad_region_header) -> index
      %saved = arith.constant 3.0 : f64
      "vernon.ad.write_leaf"(%root, %record, %saved)
          {leaf_offset = 0 : i64}
          : (!vernon.ad_region_header, index, f64) -> ()
      %one = arith.constant 1 : index
      %normal = arith.constant 0 : i32
      "vernon.ad.end_region"(%root, %one, %normal)
          : (!vernon.ad_region_header, index, i32) -> ()
      "vernon.ad.capture_yield"(%tape, %root)
          : (!vernon.ad_tape, !vernon.ad_region_header) -> ()
    }) : () -> (!vernon.ad_tape, !vernon.ad_region_header, i1, index, i1)
    %result = "vernon.tuple_create"(%state#0, %state#1)
        : (!vernon.ad_tape, !vernon.ad_region_header)
            -> tuple<!vernon.ad_tape, !vernon.ad_region_header>
    func.return %result : tuple<!vernon.ad_tape, !vernon.ad_region_header>
  }

  func.func @static_backward(
      %tape: !vernon.ad_tape,
      %root: !vernon.ad_region_header,
      %gid: tensor<3xi32> {vernon.builtin = "global_invocation_id"},
      %lid: tensor<3xi32> {vernon.builtin = "local_invocation_id"})
      attributes {
        vernon.ad.residual_storage = "static",
        vernon.entry,
        vernon.stage = "compute",
        vernon.workgroup_size = array<i32: 8, 1, 1>
      } {
    %zero = arith.constant 0 : index
    %saved = "vernon.ad.read_leaf"(%root, %zero)
        {leaf_offset = 0 : i64, record_alignment = 8 : i64,
         record_size = 8 : i64}
        : (!vernon.ad_region_header, index) -> f64
    %unused = arith.addf %saved, %saved : f64
    func.return
  }
}
