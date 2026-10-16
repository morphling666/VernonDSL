// RUN: %vernon-opt --vernon-to-spirv %s | %FileCheck %s
//
// Aligned Tensor leaves of a uniform block are 4-vector members, so a runtime index reads one element in place.
// CHECK: spirv.GlobalVariable @fragment_main_arg_0 bind(0, 0) : !spirv.ptr<!spirv.struct<fragment_main_arg_0_block, (!spirv.array<8 x vector<4xf32>, stride=16> [0], vector<4xf32> [128], f32 [144]), Block>, Uniform>
// CHECK-LABEL: spirv.func @fragment_main()
// CHECK: %[[ARRAY:.*]] = spirv.Variable : !spirv.ptr<!spirv.array<5 x f32, stride=16>, Function>
// CHECK: %[[BLOCK:.*]] = spirv.mlir.addressof @fragment_main_arg_0
// CHECK-NOT: spirv.Load "Uniform"
// CHECK: %[[SELECTED:.*]] = spirv.ConvertFToS
// CHECK: %[[VECTOR:.*]] = spirv.SDiv %[[LINEAR:.*]], %{{.*}} : i32
// CHECK: %[[LANE:.*]] = spirv.SRem %[[LINEAR]], %{{.*}} : i32
// CHECK: %[[MATRIX_ELEMENT:.*]] = spirv.AccessChain %[[BLOCK]][%{{.*}}, %[[VECTOR]], %[[LANE]]]
// CHECK: spirv.Load "Uniform" %[[MATRIX_ELEMENT]] : f32
// CHECK: %[[SPLIT:.*]] = spirv.AccessChain %[[BLOCK]][%{{.*}}, %[[SELECTED]]]
// CHECK: spirv.Load "Uniform" %[[SPLIT]] : f32
// CHECK: %[[BLEND:.*]] = spirv.AccessChain %[[BLOCK]][%{{.*}}] : {{.*}}, i32 -> !spirv.ptr<f32, Uniform>
// CHECK: spirv.Load "Uniform" %[[BLEND]] : f32
// CHECK-NOT: spirv.Load "Uniform"
// CHECK-NOT: spirv.Select
// CHECK: spirv.VectorExtractDynamic %{{.*}}[%[[SELECTED]]] : vector<3xf32>, i32
// CHECK: spirv.Store "Function" %[[ARRAY]], %{{.*}} : !spirv.array<5 x f32, stride=16>
// CHECK: %[[ELEMENT:.*]] = spirv.AccessChain %[[ARRAY]][%[[SELECTED]]]
// CHECK: spirv.Load "Function" %[[ELEMENT]] : f32
// CHECK-NOT: spirv.Select
// CHECK: spirv.Return
//
// A Tensor leaf off a 16-byte boundary stays scalar members; its field read loads only its own leaves.
// CHECK: spirv.GlobalVariable @vertex_main_arg_1 bind(0, 1) : !spirv.ptr<!spirv.struct<vertex_main_arg_1_block, (f32 [0], f32 [4], f32 [8], f32 [12], f32 [16]), Block>, Uniform>
// CHECK-LABEL: spirv.func @vertex_main()
// CHECK: %[[UNALIGNED:.*]] = spirv.mlir.addressof @vertex_main_arg_1
// CHECK: %[[INDEX:.*]] = spirv.ConvertFToS
// CHECK: spirv.AccessChain %[[UNALIGNED]][%cst1_i32]
// CHECK: spirv.AccessChain %[[UNALIGNED]][%cst2_i32]
// CHECK: spirv.AccessChain %[[UNALIGNED]][%cst3_i32]
// CHECK: spirv.AccessChain %[[UNALIGNED]][%cst4_i32]
// CHECK-NOT: spirv.Select
// CHECK: spirv.VectorExtractDynamic %{{.*}}[%[[INDEX]]] : vector<4xf32>, i32
// CHECK-NOT: spirv.Select
// CHECK: spirv.Return

module attributes {vernon.compiler_contract_version = 1 : i64, vernon.program_version = 2 : i64} {
  "vernon.struct"() {
    sym_name = "Shadows",
    fields = ["matrices:tensor<2x4x4xf32>", "splits:tensor<4xf32>", "blend:f32"],
    abi_leaf_dtypes = ["f32", "f32", "f32"]
  } : () -> ()
  "vernon.struct"() {
    sym_name = "Unaligned",
    fields = ["bias:f32", "weights:tensor<4xf32>"],
    abi_leaf_dtypes = ["f32", "f32"]
  } : () -> ()
  func.func @fragment_main(
      %shadows: !vernon.struct<"Shadows"> {
        vernon.interface = "uniform", vernon.set = 0 : i64, vernon.binding = 0 : i64
      },
      %selector: f32 {
        vernon.interface = "input", vernon.location = 0 : i64
      }) -> (
      f32 {
        vernon.interface = "output", vernon.location = 0 : i64, vernon.source_name = "shade"
      }) attributes {
        vernon.entry, vernon.stage = "fragment"
      } {
    %selected = arith.fptosi %selector : f32 to i32
    %index = arith.index_cast %selected : i32 to index
    %one = arith.constant 1 : index
    %matrices = "vernon.struct_get"(%shadows) {field = "matrices", index = 0 : i64} :
      (!vernon.struct<"Shadows">) -> tensor<2x4x4xf32>
    %matrix = tensor.extract %matrices[%index, %one, %one] : tensor<2x4x4xf32>
    %splits = "vernon.struct_get"(%shadows) {field = "splits", index = 1 : i64} :
      (!vernon.struct<"Shadows">) -> tensor<4xf32>
    %split = tensor.extract %splits[%index] : tensor<4xf32>
    %blend = "vernon.struct_get"(%shadows) {field = "blend", index = 2 : i64} : (!vernon.struct<"Shadows">) -> f32
    %vector = tensor.from_elements %matrix, %split, %blend : tensor<3xf32>
    %lane = tensor.extract %vector[%index] : tensor<3xf32>
    %array = tensor.from_elements %matrix, %split, %blend, %lane, %selector : tensor<5xf32>
    %element = tensor.extract %array[%index] : tensor<5xf32>
    return %element : f32
  }
  func.func @vertex_main(
      %scale: f32 {
        vernon.interface = "uniform"
      },
      %unaligned: !vernon.struct<"Unaligned"> {
        vernon.interface = "uniform"
      },
      %selector: f32 {
        vernon.interface = "input", vernon.location = 0 : i64
      }) -> (
      tensor<4xf32> {
        vernon.interface = "output", vernon.builtin = "position"
      }) attributes {
        vernon.entry, vernon.stage = "vertex"
      } {
    %selected = arith.fptosi %selector : f32 to i32
    %index = arith.index_cast %selected : i32 to index
    %weights = "vernon.struct_get"(%unaligned) {field = "weights", index = 1 : i64} :
      (!vernon.struct<"Unaligned">) -> tensor<4xf32>
    %split = tensor.extract %weights[%index] : tensor<4xf32>
    %position = tensor.from_elements %split, %split, %scale, %scale : tensor<4xf32>
    return %position : tensor<4xf32>
  }
}
