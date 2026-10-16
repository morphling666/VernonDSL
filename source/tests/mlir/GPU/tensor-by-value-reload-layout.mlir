// RUN: %vernon-opt --vernon-to-gpu %s | %FileCheck %s --check-prefix=CUDA
// RUN: %vernon-opt --vernon-to-gpu="use-spirv-storage=true" %s | %FileCheck %s --check-prefix=SHADER
//
// The prologue reloads a Tensor-by-value argument with the strides Runtime packs it with: row-major kernel-parameter
// bytes for CUDA and the std430 column-major native matrix for SPIR-V targets.
// CUDA-LABEL: gpu.func @reload(
// CUDA: memref.load %arg0[%c0]
// CUDA-NEXT: %c1 = arith.constant 1 : index
// CUDA-NEXT: memref.load %arg0[%c1]
// CUDA-NEXT: %c2 = arith.constant 2 : index
// CUDA-NEXT: memref.load %arg0[%c2]
// CUDA: tensor.from_elements
//
// SHADER-LABEL: gpu.func @reload(
// SHADER: memref.load %arg0[%c0]
// SHADER-NEXT: %c2 = arith.constant 2 : index
// SHADER-NEXT: memref.load %arg0[%c2]
// SHADER-NEXT: %c4 = arith.constant 4 : index
// SHADER-NEXT: memref.load %arg0[%c4]
// SHADER: tensor.from_elements

module attributes {vernon.compiler_contract_version = 1 : i64, vernon.program_version = 2 : i64} {
  func.func @reload(
      %table: tensor<2x4xf32> {
        vernon.interface = "input",
        vernon.location = 0 : i64,
        vernon.dtype = "f32",
        vernon.abi_leaf_dtypes = ["f32"],
        vernon.element_abi_leaf_dtypes = ["f32"]
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
    %value = tensor.extract %table[%zero, %one] : tensor<2x4xf32>
    "vernon.store"(%value, %output, %zero) : (f32, !vernon.tensor_view<f32, [-1], "write", "device">, index) -> ()
    return
  }
}
