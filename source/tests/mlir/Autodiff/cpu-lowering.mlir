// RUN: %vernon-opt --vernon-prepare-cpu-autodiff-signatures %s -o %t.prepare
// RUN: %FileCheck %s --check-prefix=PREPARE --input-file=%t.prepare
// RUN: %vernon-opt --vernon-prepare-cpu-autodiff-signatures --vernon-lower-cpu-autodiff %s -o %t.lower
// RUN: %FileCheck %s --check-prefix=LOWER --input-file=%t.lower
// RUN: %vernon-opt --vernon-prepare-cpu-autodiff-signatures --vernon-lower-cpu-autodiff \
// RUN:   --vernon-cpu-autodiff-to-llvm %s -o %t.llvm
// RUN: %FileCheck %s --check-prefix=LLVM --input-file=%t.llvm
//
// PREPARE-LABEL: func.func @forward(
// PREPARE-SAME: index {vernon.builtin = "ad_tape_allocator", vernon.interface = "input"}
// PREPARE-LABEL: func.func @backward(
// PREPARE-SAME: !vernon.ad_tape {vernon.builtin = "ad_tape_allocator", vernon.interface = "input"}
// PREPARE-SAME: !vernon.ad_region_header {vernon.builtin = "ad_tape_root_region", vernon.interface = "input"}
//
// LOWER-LABEL: func.func @forward(
// LOWER-SAME: index {vernon.builtin = "ad_tape_allocator", vernon.interface = "input"}
// LOWER: "vernon.cpu_ad.callback"
// LOWER-LABEL: func.func @backward(
// LOWER-SAME: index {vernon.builtin = "ad_tape_allocator", vernon.interface = "input"}
// LOWER-SAME: index {vernon.builtin = "ad_tape_root_region", vernon.interface = "input"}
// LOWER: "vernon.cpu_ad.callback"
// LOWER: llvm.load
// LOWER-NOT: "vernon.ad.adjoint_buffer.peek"
// LOWER-NOT: !vernon.ad_tape
// LOWER-NOT: !vernon.ad_region_header
//
// LLVM-LABEL: func.func @forward(
// LLVM: llvm.call
// LLVM-LABEL: func.func @backward(
// LLVM: llvm.call
// LLVM-NOT: "vernon.cpu_ad.callback"
// LLVM-NOT: !vernon.ad_tape
// LLVM-NOT: !vernon.ad_region_header

module {
  func.func @forward() -> tuple<!vernon.ad_tape, !vernon.ad_region_header>
      attributes {vernon.entry, vernon.stage = "compute"} {
    %tape, %root, %success, %required, %overflow =
        "vernon.ad.capture"() ({
          %owned = "vernon.ad.begin_invocation"() : () -> !vernon.ad_tape
          %region = "vernon.ad.begin_region"(%owned)
              : (!vernon.ad_tape) -> !vernon.ad_region_header
          %record = "vernon.ad.reserve_record"(%region)
              {record_size = 8 : i64, record_alignment = 8 : i64}
              : (!vernon.ad_region_header) -> index
          %saved = arith.constant 2.000000e+00 : f64
          "vernon.ad.write_leaf"(%region, %record, %saved)
              {leaf_offset = 0 : i64}
              : (!vernon.ad_region_header, index, f64) -> ()
          %one = arith.constant 1 : index
          %normal = arith.constant 0 : i32
          "vernon.ad.end_region"(%region, %one, %normal)
              : (!vernon.ad_region_header, index, i32) -> ()
          "vernon.ad.capture_yield"(%owned, %region)
              : (!vernon.ad_tape, !vernon.ad_region_header) -> ()
        }) : () -> (!vernon.ad_tape, !vernon.ad_region_header, i1, index, i1)
    %result = "vernon.tuple_create"(%tape, %root)
        : (!vernon.ad_tape, !vernon.ad_region_header)
            -> tuple<!vernon.ad_tape, !vernon.ad_region_header>
    return %result : tuple<!vernon.ad_tape, !vernon.ad_region_header>
  }

  func.func @backward(
      %tape: !vernon.ad_tape,
      %root: !vernon.ad_region_header,
      %seed: f64) -> f64
      attributes {vernon.entry, vernon.stage = "compute"} {
    %zero = arith.constant 0 : index
    %saved = "vernon.ad.read_leaf"(%root, %zero)
        {record_size = 8 : i64, record_alignment = 8 : i64,
         leaf_offset = 0 : i64}
        : (!vernon.ad_region_header, index) -> f64
    %buffer = "vernon.ad.adjoint_buffer.create"() {ownership = "lane_private"}
        : () -> !vernon.ad_adjoint_buffer<f64, [1], 1>
    %peek = "vernon.ad.adjoint_buffer.peek"(%buffer, %zero)
        : (!vernon.ad_adjoint_buffer<f64, [1], 1>, index) -> f64
    %product = arith.mulf %saved, %seed : f64
    %gradient = arith.addf %product, %peek : f64
    return %gradient : f64
  }
}
