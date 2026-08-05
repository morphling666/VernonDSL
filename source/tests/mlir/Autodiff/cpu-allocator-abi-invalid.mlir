// RUN: %not %vernon-opt --vernon-verify-cpu-autodiff-abi %s 2>&1 | %FileCheck %s
//
// CHECK-DAG: argument #0 ad_tape_allocator must have index type
// CHECK-DAG: argument #0 ad_tape_allocator is only valid on a compute entry
// CHECK-DAG: argument #0 ad_tape_allocator must use the input interface
// CHECK-DAG: contains more than one ad_tape_allocator argument
// CHECK-DAG: argument #0 ad_tape_allocator cannot carry a user interface location or name

module {
  func.func @wrong_type(
      %allocator: f32 {
        vernon.interface = "input",
        vernon.builtin = "ad_tape_allocator"
      }) attributes {
        vernon.entry,
        vernon.stage = "compute"
      } {
    return
  }

  func.func @duplicate(
      %first: index {
        vernon.interface = "input",
        vernon.builtin = "ad_tape_allocator"
      },
      %second: index {
        vernon.interface = "input",
        vernon.builtin = "ad_tape_allocator"
      }) attributes {
        vernon.entry,
        vernon.stage = "compute"
      } {
    return
  }

  func.func @wrong_stage(
      %allocator: index {
        vernon.interface = "input",
        vernon.builtin = "ad_tape_allocator"
      }) attributes {
        vernon.entry,
        vernon.stage = "fragment"
      } {
    return
  }

  func.func @wrong_interface(
      %allocator: index {
        vernon.interface = "resource",
        vernon.builtin = "ad_tape_allocator"
      }) attributes {
        vernon.entry,
        vernon.stage = "compute"
      } {
    return
  }

  func.func @user_named(
      %allocator: index {
        vernon.interface = "input",
        vernon.builtin = "ad_tape_allocator",
        vernon.source_name = "allocator"
      }) attributes {
        vernon.entry,
        vernon.stage = "compute"
      } {
    return
  }
}
