// RUN: %vernon-opt \
// RUN:   --pass-pipeline='builtin.module(vernon-to-gpu{use-spirv-storage=true},gpu.module(vernon-lower-gpu-tensors{use-spirv-tuple-abi=true}),vernon-convert-gpu-to-spirv)' \
// RUN:   %s | %FileCheck %s --check-prefix=SPIRV
// RUN: %vernon-opt --pass-pipeline='builtin.module(vernon-to-gpu,gpu.module(vernon-lower-gpu-tensors))' %s \
// RUN:   | %FileCheck %s --check-prefix=CUDA
//
// A runtime index reads a register vector with a dynamic extract and a register array through one function-local copy
// per read, allocated in the entry block, instead of selecting over every element.
// The single select before the dynamic vector extract is the upstream clamp of the poison index -1.
// SPIRV-LABEL: spirv.func @dynamic_reads(
// SPIRV-COUNT-3: spirv.Variable : !spirv.ptr<!spirv.array<{{8 x f32|20 x f32|3 x !spirv.struct<\(i32, f32\)>}}{{.*}}>, Function>
// SPIRV-NOT: spirv.Variable
// SPIRV-NOT: spirv.Select
// SPIRV: spirv.Store "Function" %{{.*}}, %{{.*}} : !spirv.array<8 x f32
// SPIRV: spirv.AccessChain
// SPIRV: spirv.Load "Function"
// SPIRV-NOT: spirv.Select
// SPIRV: spirv.Store "Function" %{{.*}}, %{{.*}} : !spirv.array<20 x f32
// SPIRV: spirv.AccessChain
// SPIRV: spirv.Load "Function"
// SPIRV: spirv.Select
// SPIRV-NOT: spirv.Select
// SPIRV: spirv.VectorExtractDynamic {{.*}} : vector<3xf32>
// SPIRV-NOT: spirv.Select
// SPIRV: spirv.Store "Function" %{{.*}}, %{{.*}} : !spirv.array<3 x !spirv.struct
// SPIRV: spirv.AccessChain
// SPIRV: spirv.Load "Function"
// SPIRV-NOT: spirv.Select
// SPIRV: spirv.Return
//
// CUDA-LABEL: gpu.func @dynamic_reads(
// CUDA-COUNT-2: llvm.alloca %{{.*}} x !llvm.array<{{20 x f32|3 x struct<\(i32, f32\)>}}>
// CUDA-NOT: llvm.alloca
// CUDA: vector.extract %{{.*}}[%{{.*}}] : f32 from vector<8xf32>
// CUDA: llvm.store %{{.*}}, %{{.*}} : !llvm.array<20 x f32>
// CUDA: llvm.getelementptr %{{.*}}[0, %{{.*}}]
// CUDA: llvm.load %{{.*}} -> f32
// CUDA: vector.extract %{{.*}}[%{{.*}}] : f32 from vector<3xf32>
// CUDA: llvm.store %{{.*}}, %{{.*}} : !llvm.array<3 x struct<(i32, f32)>>
// CUDA: llvm.getelementptr %{{.*}}[0, %{{.*}}]
// CUDA: llvm.load %{{.*}} -> !llvm.struct<(i32, f32)>
// CUDA-NOT: arith.select
// CUDA-NOT: arith.cmpi
// CUDA: gpu.return

module attributes {vernon.compiler_contract_version = 1 : i64, vernon.program_version = 2 : i64} {
  "vernon.struct"() {
    sym_name = "Pair",
    fields = ["left:i32", "right:f32"],
    abi_leaf_dtypes = ["i32", "f32"]
  } : () -> ()
  func.func @dynamic_reads(
      %pairs: !vernon.tensor<!vernon.struct<"Pair">, [3]> {
        vernon.interface = "input",
        vernon.location = 0 : i64,
        vernon.abi_leaf_dtypes = ["i32", "f32", "i32", "f32", "i32", "f32"],
        vernon.element_abi_leaf_dtypes = ["i32", "f32"]
      },
      %selectors: !vernon.tensor_view<i32, [-1], "read", "device"> {
        vernon.element_abi_leaf_dtypes = ["i32"],
        vernon.interface = "resource",
        vernon.set = 0 : i64,
        vernon.binding = 0 : i64
      },
      %output: !vernon.tensor_view<f32, [-1], "write", "device"> {
        vernon.element_abi_leaf_dtypes = ["f32"],
        vernon.interface = "resource",
        vernon.set = 0 : i64,
        vernon.binding = 1 : i64
      }) attributes {
        vernon.entry,
        vernon.stage = "compute",
        vernon.workgroup_size = array<i32: 1, 1, 1>
      } {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %two = arith.constant 2 : index
    %three = arith.constant 3 : index
    %selector = "vernon.load"(%selectors, %zero) : (!vernon.tensor_view<i32, [-1], "read", "device">, index) -> i32
    %index = arith.index_cast %selector : i32 to index
    %f0 = arith.constant 0.0 : f32
    %f1 = arith.constant 1.0 : f32
    %f2 = arith.constant 2.0 : f32
    %f3 = arith.constant 3.0 : f32
    %eight = "vernon.intrinsic"(%f0, %f1, %f2, %f3, %f3, %f2, %f1, %f0) {name = "construct"} :
      (f32, f32, f32, f32, f32, f32, f32, f32) -> tensor<8xf32>
    %twenty = "vernon.intrinsic"(%f0, %f1, %f2, %f3, %f0, %f1, %f2, %f3, %f0, %f1, %f2, %f3, %f0, %f1, %f2, %f3,
                                 %f0, %f1, %f2, %f3) {name = "construct"} :
      (f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)
      -> tensor<20xf32>
    %three_lanes = "vernon.intrinsic"(%f1, %f2, %f3) {name = "construct"} : (f32, f32, f32) -> tensor<3xf32>
    %a = tensor.extract %eight[%index] : tensor<8xf32>
    "vernon.store"(%a, %output, %zero) : (f32, !vernon.tensor_view<f32, [-1], "write", "device">, index) -> ()
    %b = tensor.extract %twenty[%index] : tensor<20xf32>
    "vernon.store"(%b, %output, %one) : (f32, !vernon.tensor_view<f32, [-1], "write", "device">, index) -> ()
    %c = tensor.extract %three_lanes[%index] : tensor<3xf32>
    "vernon.store"(%c, %output, %two) : (f32, !vernon.tensor_view<f32, [-1], "write", "device">, index) -> ()
    %pair = "vernon.tensor_get"(%pairs, %index) : (!vernon.tensor<!vernon.struct<"Pair">, [3]>, index) -> !vernon.struct<"Pair">
    %d = "vernon.struct_get"(%pair) {field = "right", index = 1 : i64} : (!vernon.struct<"Pair">) -> f32
    "vernon.store"(%d, %output, %three) : (f32, !vernon.tensor_view<f32, [-1], "write", "device">, index) -> ()
    return
  }
}
