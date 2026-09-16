// RUN: %vernon-opt %s -o %t
// RUN: %vernon-opt %t -o %t.roundtrip
// RUN: %FileCheck %s --check-prefix=CHECK --input-file=%t.roundtrip
// RUN: %FileCheck %s --check-prefix=NEUTRAL --input-file=%t.roundtrip
//
// CHECK-DAG: !vernon.ad_tape
// CHECK-DAG: !vernon.ad_region_header
// CHECK-DAG: "vernon.ad.capture"
// CHECK-DAG: "vernon.ad.begin_invocation"
// CHECK-DAG: "vernon.ad.begin_region"
// CHECK-DAG: "vernon.ad.reserve_record"
// CHECK-DAG: "vernon.ad.checked_increment"
// CHECK-DAG: "vernon.ad.write_leaf"
// CHECK-DAG: "vernon.ad.end_region"
// CHECK-DAG: "vernon.ad.capture_yield"
// CHECK-DAG: "vernon.ad.commit"
// CHECK-DAG: "vernon.ad.commit_yield"
// CHECK-DAG: "vernon.ad.read_record_offset"
// CHECK-DAG: "vernon.ad.read_executed_count"
// CHECK-DAG: "vernon.ad.read_exit_kind"
// CHECK-DAG: "vernon.ad.read_nested_region"
// CHECK-DAG: "vernon.ad.read_leaf"
//
// NEUTRAL: module
// NEUTRAL-NOT: cpu
// NEUTRAL-NOT: metal
// NEUTRAL-NOT: vulkan
// NEUTRAL-NOT: binding
// NEUTRAL-NOT: status_word
// NEUTRAL-NOT: status-word

module {
  func.func @logical_ad_roundtrip(
      %output: !vernon.tensor_view<f32, [1], "write", "device">)
      -> (index, index, i32, !vernon.ad_region_header, f64) {
    %tape, %root, %success, %required, %overflow =
        "vernon.ad.capture"() ({
          %owned_tape = "vernon.ad.begin_invocation"()
              : () -> !vernon.ad_tape
          %root_region = "vernon.ad.begin_region"(%owned_tape)
              : (!vernon.ad_tape) -> !vernon.ad_region_header

          %one = arith.constant 1 : index
          %two = "vernon.ad.checked_increment"(%one) : (index) -> index
          %normal_exit = arith.constant 0 : i32
          %root_record = "vernon.ad.reserve_record"(%root_region)
              {record_size = 16 : i64, record_alignment = 8 : i64}
              : (!vernon.ad_region_header) -> index
          %saved_f64 = arith.constant 2.000000e+00 : f64
          "vernon.ad.write_leaf"(%root_region, %root_record, %saved_f64)
              {leaf_offset = 0 : i64}
              : (!vernon.ad_region_header, index, f64) -> ()

          %nested_region = "vernon.ad.begin_region"(
              %owned_tape, %root_region, %root_record)
              {child_ordinal = 0 : i64}
              : (!vernon.ad_tape, !vernon.ad_region_header, index)
                  -> !vernon.ad_region_header
          %nested_record = "vernon.ad.reserve_record"(%nested_region)
              {record_size = 4 : i64, record_alignment = 4 : i64}
              : (!vernon.ad_region_header) -> index
          %saved_i32 = arith.constant 7 : i32
          "vernon.ad.write_leaf"(%nested_region, %nested_record, %saved_i32)
              {leaf_offset = 0 : i64}
              : (!vernon.ad_region_header, index, i32) -> ()
          "vernon.ad.end_region"(%nested_region, %one, %normal_exit)
              : (!vernon.ad_region_header, index, i32) -> ()
          "vernon.ad.end_region"(%root_region, %one, %normal_exit)
              : (!vernon.ad_region_header, index, i32) -> ()
          "vernon.ad.capture_yield"(%owned_tape, %root_region)
              : (!vernon.ad_tape, !vernon.ad_region_header) -> ()
        }) : () -> (!vernon.ad_tape, !vernon.ad_region_header, i1, index, i1)

    "vernon.ad.commit"(%tape, %success) ({
      %zero = arith.constant 0 : index
      %committed = arith.constant 1.000000e+00 : f32
      "vernon.store"(%committed, %output, %zero)
          : (f32, !vernon.tensor_view<f32, [1], "write", "device">, index) -> ()
      "vernon.ad.commit_yield"() : () -> ()
    }) : (!vernon.ad_tape, i1) -> ()

    %record_offset = "vernon.ad.read_record_offset"(%root)
        : (!vernon.ad_region_header) -> index
    %executed_count = "vernon.ad.read_executed_count"(%root)
        : (!vernon.ad_region_header) -> index
    %exit_kind = "vernon.ad.read_exit_kind"(%root)
        : (!vernon.ad_region_header) -> i32
    %zero_record = arith.constant 0 : index
    %nested = "vernon.ad.read_nested_region"(%root, %zero_record)
        {child_ordinal = 0 : i64}
        : (!vernon.ad_region_header, index) -> !vernon.ad_region_header
    %restored = "vernon.ad.read_leaf"(%root, %zero_record)
        {record_size = 16 : i64, record_alignment = 8 : i64,
         leaf_offset = 0 : i64}
        : (!vernon.ad_region_header, index) -> f64
    return %record_offset, %executed_count, %exit_kind, %nested, %restored
        : index, index, i32, !vernon.ad_region_header, f64
  }
}
