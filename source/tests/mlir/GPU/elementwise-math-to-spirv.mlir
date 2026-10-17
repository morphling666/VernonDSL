// RUN: %vernon-opt \
// RUN:   --pass-pipeline='builtin.module(vernon-to-gpu{use-spirv-storage=true},gpu.module(vernon-lower-gpu-tensors{use-spirv-tuple-abi=true}),vernon-convert-gpu-to-spirv)' \
// RUN:   %s | %FileCheck %s --check-prefix=SPIRV
// RUN: %vernon-opt --pass-pipeline='builtin.module(vernon-to-gpu,gpu.module(vernon-lower-gpu-tensors))' %s \
// RUN:   | %FileCheck %s --check-prefix=CUDA
//
// Math on a register vector stays one vector operation; math on a register array applies per element.
// SPIRV-LABEL: spirv.func @elementwise_math(
// SPIRV: spirv.GL.Exp %{{.*}} : vector<3xf32>
// SPIRV: spirv.GL.Sqrt %{{.*}} : vector<3xf32>
// SPIRV: spirv.GL.FAbs %{{.*}} : vector<3xf32>
// SPIRV-COUNT-8: spirv.GL.Exp %{{.*}} : f32
// SPIRV-NOT: spirv.GL.Exp
// SPIRV: spirv.Return
//
// CUDA-LABEL: gpu.func @elementwise_math(
// CUDA: math.exp %{{.*}} : vector<3xf32>
// CUDA: math.sqrt %{{.*}} : vector<3xf32>
// CUDA: math.absf %{{.*}} : vector<3xf32>
// CUDA: math.exp %{{.*}} : vector<8xf32>
// CUDA: gpu.return

module attributes {vernon.compiler_contract_version = 1 : i64, vernon.program_version = 2 : i64} {
  func.func @elementwise_math(
      %output: !vernon.tensor_view<f32, [-1], "write", "device"> {
        vernon.element_abi_leaf_dtypes = ["f32"],
        vernon.interface = "resource",
        vernon.set = 0 : i64,
        vernon.binding = 0 : i64
      }) attributes {
        vernon.entry,
        vernon.stage = "compute",
        vernon.workgroup_size = array<i32: 1, 1, 1>
      } {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %f1 = arith.constant 1.0 : f32
    %f2 = arith.constant 2.0 : f32
    %f3 = arith.constant 3.0 : f32
    %lanes = "vernon.intrinsic"(%f1, %f2, %f3) {name = "construct"} : (f32, f32, f32) -> tensor<3xf32>
    %exp = math.exp %lanes : tensor<3xf32>
    %sqrt = math.sqrt %exp : tensor<3xf32>
    %abs = math.absf %sqrt : tensor<3xf32>
    %a = tensor.extract %abs[%zero] : tensor<3xf32>
    "vernon.store"(%a, %output, %zero) : (f32, !vernon.tensor_view<f32, [-1], "write", "device">, index) -> ()
    %eight = "vernon.intrinsic"(%f1, %f2, %f3, %f1, %f2, %f3, %f1, %f2) {name = "construct"} :
      (f32, f32, f32, f32, f32, f32, f32, f32) -> tensor<8xf32>
    %wide = math.exp %eight : tensor<8xf32>
    %b = tensor.extract %wide[%one] : tensor<8xf32>
    "vernon.store"(%b, %output, %one) : (f32, !vernon.tensor_view<f32, [-1], "write", "device">, index) -> ()
    return
  }
}
