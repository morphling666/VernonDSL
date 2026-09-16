// RUN: %vernon-opt --vernon-materialize-storage-projections %s | %FileCheck %s --check-prefix=SEMANTIC
// RUN: %vernon-opt --vernon-to-gpu %s | %FileCheck %s --check-prefix=CUDA
// RUN: %vernon-opt --vernon-to-gpu="use-spirv-storage=true" %s | %FileCheck %s --check-prefix=SHADER
// RUN: %vernon-opt --pass-pipeline='builtin.module(vernon-to-gpu,gpu.module(vernon-lower-gpu-tensors))' %s \
// RUN:   | %FileCheck %s --check-prefix=CUDA-ABI
// RUN: %vernon-opt \
// RUN:   --pass-pipeline='builtin.module(vernon-to-gpu{use-spirv-storage=true},gpu.module(vernon-lower-gpu-tensors{use-spirv-tuple-abi=true}))' \
// RUN:   %s | %FileCheck %s --check-prefix=SHADER-ABI
// RUN: %vernon-opt \
// RUN:   --pass-pipeline='builtin.module(vernon-to-gpu{use-spirv-storage=true},gpu.module(vernon-lower-gpu-tensors{use-spirv-tuple-abi=true}),vernon-convert-gpu-to-spirv)' \
// RUN:   %s | %FileCheck %s --check-prefix=SPIRV
//
// SEMANTIC-LABEL: func.func @metadata(
// SEMANTIC-SAME: !vernon.tensor_view<f32, [-1, -1], "read", "device">
// SEMANTIC-SAME: tuple<index, index, index, index, index>
// SEMANTIC-SAME: vernon.tensor_metadata_carrier
// SEMANTIC-SAME: vernon.tensor_metadata_field_count = 5
// SEMANTIC-NOT: vernon.tensor_descriptor_
// SEMANTIC: "vernon.tuple_get"
// SEMANTIC-SAME: index = 0
// SEMANTIC: "vernon.tuple_get"
// SEMANTIC-SAME: index = 3
// SEMANTIC: "vernon.tuple_get"
// SEMANTIC-SAME: index = 4
//
// CUDA-LABEL: gpu.func @metadata(
// CUDA-SAME: memref<?xf32>
// CUDA-SAME: tuple<i64, i64, i64, i64, i64>
// CUDA-SAME: vernon.tensor_metadata_profile = "cuda_kernel_metadata_i64"
// CUDA-NOT: vernon.tensor_descriptor_
//
// SHADER-LABEL: gpu.func @metadata(
// SHADER-SAME: memref<?xf32, #spirv.storage_class<StorageBuffer>>
// SHADER-SAME: tuple<i32, i32, i32, i32, i32>
// SHADER-SAME: spirv.interface_var_abi
// SHADER-SAME: vernon.tensor_metadata_profile = "portable_shader_metadata_i32"
// SHADER-NOT: vernon.tensor_descriptor_
//
// CUDA-ABI-LABEL: gpu.func @metadata(
// CUDA-ABI-SAME: !llvm.struct<(i64, i64, i64, i64, i64)>
// CUDA-ABI-COUNT-3: llvm.extractvalue
// CUDA-ABI-NOT: vernon.tuple_get
//
// SHADER-ABI-LABEL: gpu.func @metadata(
// SHADER-ABI-SAME: !spirv.struct<(i32 [0], i32 [4], i32 [8], i32 [12], i32 [16])>
// SHADER-ABI-COUNT-3: spirv.CompositeExtract
// SHADER-ABI-NOT: vernon.tuple_get
//
// SPIRV: spirv.GlobalVariable @metadata_metadata bind(0, 1)
// SPIRV-SAME: !spirv.ptr<!spirv.struct<(i32 [0], i32 [4], i32 [8], i32 [12], i32 [16]), Block>, Uniform>
// SPIRV-LABEL: spirv.func @metadata(
// SPIRV: spirv.mlir.addressof @metadata_metadata
// SPIRV-COUNT-3: spirv.Load "Uniform"
// SPIRV: spirv.Load "StorageBuffer"

module attributes {vernon.compiler_contract_version = 1 : i64, vernon.program_version = 1 : i64} {
  func.func @metadata(
      %view: !vernon.tensor_view<f32, [-1, -1], "read", "device"> {
        vernon.interface = "resource",
        vernon.set = 0 : i64,
        vernon.binding = 0 : i64
      }) attributes {
        vernon.entry,
        vernon.stage = "compute",
        vernon.workgroup_size = array<i32: 1, 1, 1>
      } {
    %i = arith.constant 2 : index
    %j = arith.constant 3 : index
    %value = "vernon.load"(%view, %i, %j) :
      (!vernon.tensor_view<f32, [-1, -1], "read", "device">, index, index) -> f32
    return
  }
}
