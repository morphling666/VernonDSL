module {
  func.func @noop(
      %value: f32 {
        vernon.interface = "input",
        vernon.location = 0 : i64
      }) attributes {
    vernon.entry,
    vernon.stage = "compute",
    vernon.workgroup_size = array<i32: 1, 1, 1>
  } {
    return
  }
}
