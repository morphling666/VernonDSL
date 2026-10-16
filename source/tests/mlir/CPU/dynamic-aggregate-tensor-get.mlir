// RUN: %vernon-opt --vernon-lower-cpu-tensors %s | %FileCheck %s
//
// A runtime index reads an aggregate Tensor through a function-local copy allocated in the entry block, and a
// constant index extracts the element directly.
// CHECK-LABEL: func.func @dynamic_pair(
// CHECK-NEXT: llvm.mlir.constant(1 : i64)
// CHECK-NEXT: llvm.alloca %{{.*}} x !llvm.array<3 x struct<packed (i32, f32)>>
// CHECK-NOT: llvm.alloca
// CHECK: scf.for
// CHECK: llvm.store %{{.*}}, %{{.*}} : !llvm.array<3 x struct<packed (i32, f32)>>, !llvm.ptr
// CHECK: llvm.getelementptr %{{.*}}[0, %{{.*}}]
// CHECK: llvm.load %{{.*}} : !llvm.ptr -> !llvm.struct<packed (i32, f32)>
// CHECK-NOT: arith.select
// CHECK-NOT: arith.cmpi
// CHECK: return
//
// CHECK-LABEL: func.func @constant_pair(
// CHECK-NOT: llvm.alloca
// CHECK: llvm.extractvalue %{{.*}}[2] : !llvm.array<3 x struct<packed (i32, f32)>>
// CHECK-NOT: arith.select

module attributes {vernon.compiler_contract_version = 1 : i64, vernon.program_version = 2 : i64} {
  "vernon.struct"() {
    sym_name = "Pair",
    fields = ["left:i32", "right:f32"],
    abi_leaf_dtypes = ["i32", "f32"]
  } : () -> ()

  func.func @dynamic_pair(%pairs: !vernon.tensor<!vernon.struct<"Pair">, [3]>, %count: index) -> f32 {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %initial = arith.constant 0.0 : f32
    %sum = scf.for %index = %zero to %count step %one iter_args(%accumulated = %initial) -> (f32) {
      %pair = "vernon.tensor_get"(%pairs, %index) :
        (!vernon.tensor<!vernon.struct<"Pair">, [3]>, index) -> !vernon.struct<"Pair">
      %right = "vernon.struct_get"(%pair) {field = "right", index = 1 : i64} : (!vernon.struct<"Pair">) -> f32
      %next = arith.addf %accumulated, %right : f32
      scf.yield %next : f32
    }
    return %sum : f32
  }

  func.func @constant_pair(%pairs: !vernon.tensor<!vernon.struct<"Pair">, [3]>) -> f32 {
    %two = arith.constant 2 : index
    %pair = "vernon.tensor_get"(%pairs, %two) :
      (!vernon.tensor<!vernon.struct<"Pair">, [3]>, index) -> !vernon.struct<"Pair">
    %right = "vernon.struct_get"(%pair) {field = "right", index = 1 : i64} : (!vernon.struct<"Pair">) -> f32
    return %right : f32
  }
}
