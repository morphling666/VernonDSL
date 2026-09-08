// RUN: %vernon-opt --split-input-file --verify-diagnostics %s

// expected-error @+1 {{is only legal inside vernon.ad.capture}}
%tape = "vernon.ad.begin_invocation"() : () -> !vernon.ad_tape

// -----

func.func @capture_cannot_write_visible_storage(
    %storage: !vernon.tensor_view<f32, [1], "write", "device">) {
  %tape, %root, %ok, %bytes, %overflow = "vernon.ad.capture"() ({
    %owned = "vernon.ad.begin_invocation"() : () -> !vernon.ad_tape
    %region = "vernon.ad.begin_region"(%owned)
        : (!vernon.ad_tape) -> !vernon.ad_region_header
    %zero = arith.constant 0 : index
    %exit = arith.constant 0 : i32
    %value = arith.constant 1.000000e+00 : f32
    // expected-error @+1 {{has unclassified or externally visible effects and is illegal during autodiff capture}}
    "vernon.store"(%value, %storage, %zero)
        : (f32, !vernon.tensor_view<f32, [1], "write", "device">, index) -> ()
    "vernon.ad.end_region"(%region, %zero, %exit)
        : (!vernon.ad_region_header, index, i32) -> ()
    "vernon.ad.capture_yield"(%owned, %region)
        : (!vernon.ad_tape, !vernon.ad_region_header) -> ()
  }) : () -> (!vernon.ad_tape, !vernon.ad_region_header, i1, index, i1)
  return
}

// -----

func.func @read_is_not_legal_during_capture() {
  %tape, %root, %ok, %bytes, %overflow = "vernon.ad.capture"() ({
    %owned = "vernon.ad.begin_invocation"() : () -> !vernon.ad_tape
    %region = "vernon.ad.begin_region"(%owned)
        : (!vernon.ad_tape) -> !vernon.ad_region_header
    // expected-error @+1 {{is only legal in the reverse read phase outside capture and commit}}
    %count = "vernon.ad.read_executed_count"(%region)
        : (!vernon.ad_region_header) -> index
    %zero = arith.constant 0 : index
    %exit = arith.constant 0 : i32
    "vernon.ad.end_region"(%region, %zero, %exit)
        : (!vernon.ad_region_header, index, i32) -> ()
    "vernon.ad.capture_yield"(%owned, %region)
        : (!vernon.ad_tape, !vernon.ad_region_header) -> ()
  }) : () -> (!vernon.ad_tape, !vernon.ad_region_header, i1, index, i1)
  return
}

// -----

func.func @foreign_invocation_owner(%foreign: !vernon.ad_tape) {
  %tape, %root, %ok, %bytes, %overflow = "vernon.ad.capture"() ({
    %owned = "vernon.ad.begin_invocation"() : () -> !vernon.ad_tape
    // expected-error @+1 {{tape must come from the enclosing capture's begin_invocation}}
    %region = "vernon.ad.begin_region"(%foreign)
        : (!vernon.ad_tape) -> !vernon.ad_region_header
    %zero = arith.constant 0 : index
    %exit = arith.constant 0 : i32
    "vernon.ad.end_region"(%region, %zero, %exit)
        : (!vernon.ad_region_header, index, i32) -> ()
    "vernon.ad.capture_yield"(%owned, %region)
        : (!vernon.ad_tape, !vernon.ad_region_header) -> ()
  }) : () -> (!vernon.ad_tape, !vernon.ad_region_header, i1, index, i1)
  return
}

// -----

func.func @region_requires_exactly_one_end() {
  %tape, %root, %ok, %bytes, %overflow = "vernon.ad.capture"() ({
    %owned = "vernon.ad.begin_invocation"() : () -> !vernon.ad_tape
    // expected-error @+1 {{region handle must be finalized exactly once; found 0 vernon.ad.end_region users}}
    %region = "vernon.ad.begin_region"(%owned)
        : (!vernon.ad_tape) -> !vernon.ad_region_header
    "vernon.ad.capture_yield"(%owned, %region)
        : (!vernon.ad_tape, !vernon.ad_region_header) -> ()
  }) : () -> (!vernon.ad_tape, !vernon.ad_region_header, i1, index, i1)
  return
}

// -----

func.func @record_layout_is_checked() {
  %tape, %root, %ok, %bytes, %overflow = "vernon.ad.capture"() ({
    %owned = "vernon.ad.begin_invocation"() : () -> !vernon.ad_tape
    %region = "vernon.ad.begin_region"(%owned)
        : (!vernon.ad_tape) -> !vernon.ad_region_header
    // expected-error @+1 {{requires record_alignment to be a positive power of two}}
    %record = "vernon.ad.reserve_record"(%region)
        {record_size = 12 : i64, record_alignment = 3 : i64}
        : (!vernon.ad_region_header) -> index
    %zero = arith.constant 0 : index
    %exit = arith.constant 0 : i32
    "vernon.ad.end_region"(%region, %zero, %exit)
        : (!vernon.ad_region_header, index, i32) -> ()
    "vernon.ad.capture_yield"(%owned, %region)
        : (!vernon.ad_tape, !vernon.ad_region_header) -> ()
  }) : () -> (!vernon.ad_tape, !vernon.ad_region_header, i1, index, i1)
  return
}

// -----

func.func @iteration_counter_overflow_is_checked() {
  %tape, %root, %ok, %bytes, %overflow = "vernon.ad.capture"() ({
    %owned = "vernon.ad.begin_invocation"() : () -> !vernon.ad_tape
    %region = "vernon.ad.begin_region"(%owned)
        : (!vernon.ad_tape) -> !vernon.ad_region_header
    %maximum = arith.constant 9223372036854775807 : index
    // expected-error @+1 {{constant counter increment overflows index representation}}
    %wrapped = "vernon.ad.checked_increment"(%maximum) : (index) -> index
    %zero = arith.constant 0 : index
    %exit = arith.constant 0 : i32
    "vernon.ad.end_region"(%region, %zero, %exit)
        : (!vernon.ad_region_header, index, i32) -> ()
    "vernon.ad.capture_yield"(%owned, %region)
        : (!vernon.ad_tape, !vernon.ad_region_header) -> ()
  }) : () -> (!vernon.ad_tape, !vernon.ad_region_header, i1, index, i1)
  return
}

// -----

func.func @leaf_write_requires_reservation() {
  %tape, %root, %ok, %bytes, %overflow = "vernon.ad.capture"() ({
    %owned = "vernon.ad.begin_invocation"() : () -> !vernon.ad_tape
    %region = "vernon.ad.begin_region"(%owned)
        : (!vernon.ad_tape) -> !vernon.ad_region_header
    %unchecked = arith.constant 0 : index
    %value = arith.constant 1.000000e+00 : f32
    // expected-error @+1 {{record_offset must come directly from checked ad.reserve_record for this region}}
    "vernon.ad.write_leaf"(%region, %unchecked, %value)
        {leaf_offset = 0 : i64}
        : (!vernon.ad_region_header, index, f32) -> ()
    %exit = arith.constant 0 : i32
    "vernon.ad.end_region"(%region, %unchecked, %exit)
        : (!vernon.ad_region_header, index, i32) -> ()
    "vernon.ad.capture_yield"(%owned, %region)
        : (!vernon.ad_tape, !vernon.ad_region_header) -> ()
  }) : () -> (!vernon.ad_tape, !vernon.ad_region_header, i1, index, i1)
  return
}

// -----

func.func @nested_region_requires_parent_reservation() {
  %tape, %root, %ok, %bytes, %overflow = "vernon.ad.capture"() ({
    %owned = "vernon.ad.begin_invocation"() : () -> !vernon.ad_tape
    %parent = "vernon.ad.begin_region"(%owned)
        : (!vernon.ad_tape) -> !vernon.ad_region_header
    %unchecked = arith.constant 0 : index
    // expected-error @+1 {{nested region requires a checked parent record from its direct parent region}}
    %child = "vernon.ad.begin_region"(%owned, %parent, %unchecked)
        {child_ordinal = 0 : i64}
        : (!vernon.ad_tape, !vernon.ad_region_header, index)
            -> !vernon.ad_region_header
    %exit = arith.constant 0 : i32
    "vernon.ad.end_region"(%child, %unchecked, %exit)
        : (!vernon.ad_region_header, index, i32) -> ()
    "vernon.ad.end_region"(%parent, %unchecked, %exit)
        : (!vernon.ad_region_header, index, i32) -> ()
    "vernon.ad.capture_yield"(%owned, %parent)
        : (!vernon.ad_tape, !vernon.ad_region_header) -> ()
  }) : () -> (!vernon.ad_tape, !vernon.ad_region_header, i1, index, i1)
  return
}

// -----

func.func @leaf_layout_must_be_canonical(%region: !vernon.ad_region_header) {
  %record = arith.constant 0 : index
  // expected-error @+1 {{record_alignment is smaller than the canonical leaf alignment}}
  %value = "vernon.ad.read_leaf"(%region, %record)
      {record_size = 4 : i64, record_alignment = 4 : i64,
       leaf_offset = 0 : i64}
      : (!vernon.ad_region_header, index) -> f64
  return
}

// -----

func.func @region_cannot_be_used_after_end() {
  %tape, %root, %ok, %bytes, %overflow = "vernon.ad.capture"() ({
    %owned = "vernon.ad.begin_invocation"() : () -> !vernon.ad_tape
    %region = "vernon.ad.begin_region"(%owned)
        : (!vernon.ad_tape) -> !vernon.ad_region_header
    %zero = arith.constant 0 : index
    %exit = arith.constant 0 : i32
    "vernon.ad.end_region"(%region, %zero, %exit)
        : (!vernon.ad_region_header, index, i32) -> ()
    // expected-error @+1 {{uses a region handle after vernon.ad.end_region}}
    %late = "vernon.ad.reserve_record"(%region)
        {record_size = 4 : i64, record_alignment = 4 : i64}
        : (!vernon.ad_region_header) -> index
    "vernon.ad.capture_yield"(%owned, %region)
        : (!vernon.ad_tape, !vernon.ad_region_header) -> ()
  }) : () -> (!vernon.ad_tape, !vernon.ad_region_header, i1, index, i1)
  return
}

// -----

func.func @region_cannot_be_used_in_nested_control_flow_after_end() {
  %tape, %root, %ok, %bytes, %overflow = "vernon.ad.capture"() ({
    %owned = "vernon.ad.begin_invocation"() : () -> !vernon.ad_tape
    %region = "vernon.ad.begin_region"(%owned)
        : (!vernon.ad_tape) -> !vernon.ad_region_header
    %zero = arith.constant 0 : index
    %exit = arith.constant 0 : i32
    %condition = arith.constant true
    "vernon.ad.end_region"(%region, %zero, %exit)
        : (!vernon.ad_region_header, index, i32) -> ()
    scf.if %condition {
      // expected-error @+1 {{uses a region handle after vernon.ad.end_region}}
      %late = "vernon.ad.reserve_record"(%region)
          {record_size = 4 : i64, record_alignment = 4 : i64}
          : (!vernon.ad_region_header) -> index
    }
    "vernon.ad.capture_yield"(%owned, %region)
        : (!vernon.ad_tape, !vernon.ad_region_header) -> ()
  }) : () -> (!vernon.ad_tape, !vernon.ad_region_header, i1, index, i1)
  return
}

// -----

func.func @recursive_effect_op_cannot_hide_visible_write(
    %storage: memref<1xf32>) {
  %tape, %root, %ok, %bytes, %overflow = "vernon.ad.capture"() ({
    %owned = "vernon.ad.begin_invocation"() : () -> !vernon.ad_tape
    %region = "vernon.ad.begin_region"(%owned)
        : (!vernon.ad_tape) -> !vernon.ad_region_header
    %zero = arith.constant 0 : index
    // expected-error @+1 {{has unclassified or externally visible effects and is illegal during autodiff capture}}
    %value = memref.generic_atomic_rmw %storage[%zero] : memref<1xf32> {
      ^bb0(%current: f32):
        memref.atomic_yield %current : f32
    }
    %exit = arith.constant 0 : i32
    "vernon.ad.end_region"(%region, %zero, %exit)
        : (!vernon.ad_region_header, index, i32) -> ()
    "vernon.ad.capture_yield"(%owned, %region)
        : (!vernon.ad_tape, !vernon.ad_region_header) -> ()
  }) : () -> (!vernon.ad_tape, !vernon.ad_region_header, i1, index, i1)
  return
}

// -----

func.func @commit_requires_matching_capture_success() {
  %tape, %root, %ok, %bytes, %overflow = "vernon.ad.capture"() ({
    %owned = "vernon.ad.begin_invocation"() : () -> !vernon.ad_tape
    %region = "vernon.ad.begin_region"(%owned)
        : (!vernon.ad_tape) -> !vernon.ad_region_header
    %zero = arith.constant 0 : index
    %exit = arith.constant 0 : i32
    "vernon.ad.end_region"(%region, %zero, %exit)
        : (!vernon.ad_region_header, index, i32) -> ()
    "vernon.ad.capture_yield"(%owned, %region)
        : (!vernon.ad_tape, !vernon.ad_region_header) -> ()
  }) : () -> (!vernon.ad_tape, !vernon.ad_region_header, i1, index, i1)
  %unrelated = arith.constant false
  // expected-error @+1 {{tape and capture_success must be paired results of the same capture}}
  "vernon.ad.commit"(%tape, %unrelated) ({
    "vernon.ad.commit_yield"() : () -> ()
  }) : (!vernon.ad_tape, i1) -> ()
  return
}

// -----

func.func @commit_rejects_unproven_block_arguments(
    %tape: !vernon.ad_tape, %success: i1) {
  // expected-error @+1 {{tape and capture_success must be paired results of the same capture}}
  "vernon.ad.commit"(%tape, %success) ({
    "vernon.ad.commit_yield"() : () -> ()
  }) : (!vernon.ad_tape, i1) -> ()
  return
}
