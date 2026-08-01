module attributes {vernon.compiler_contract_version = 9 : i64, vernon.pipeline_version = 12 : i64} {
  "vernon.struct"() {
    sym_name = "Pair",
    fields = ["left:i32", "right:f32"],
    abi_leaf_dtypes = ["i32", "f32"]
  } : () -> ()
  func.func @storage_lowering(
      %pairs: !vernon.tensor_view<!vernon.struct<"Pair">, [-1], "read_write", "device"> {
        vernon.interface = "resource",
        vernon.set = 0 : i64,
        vernon.binding = 0 : i64
      },
      %counters: !vernon.tensor_view<i32, [-1], "read_write", "device"> {
        vernon.interface = "resource",
        vernon.set = 0 : i64,
        vernon.binding = 1 : i64
      }) attributes {
        vernon.entry,
        vernon.stage = "compute",
        vernon.workgroup_size = array<i32: 1, 1, 1>
      } {
    %index = arith.constant 0 : index
    %pair = "vernon.load"(%pairs, %index) :
      (!vernon.tensor_view<!vernon.struct<"Pair">, [-1], "read_write", "device">, index) ->
      !vernon.struct<"Pair">
    "vernon.store"(%pair, %pairs, %index) :
      (!vernon.struct<"Pair">,
       !vernon.tensor_view<!vernon.struct<"Pair">, [-1], "read_write", "device">,
       index) -> ()
    %one = arith.constant 1 : i32
    %previous = "vernon.atomic"(%counters, %index, %one) {
      atomic_kind = "add", ordering = "relaxed"
    } : (!vernon.tensor_view<i32, [-1], "read_write", "device">, index, i32) -> i32
    return
  }
}
