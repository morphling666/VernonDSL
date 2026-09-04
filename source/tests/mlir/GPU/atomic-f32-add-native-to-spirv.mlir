// RUN: %vernon-opt --vernon-convert-gpu-to-spirv %s | %FileCheck %s
//
// CHECK: spirv.EXT.AtomicFAdd <Workgroup> <None>
// CHECK-NOT: spirv.AtomicCompareExchange
// CHECK-NOT: spirv.mlir.loop

module attributes {vernon.compiler_contract_version = 14 : i64, vernon.pipeline_version = 18 : i64} {
  gpu.module @kernels attributes {
    spirv.target_env = #spirv.target_env<
      #spirv.vce<v1.3, [Shader, AtomicFloat32AddEXT],
        [SPV_EXT_shader_atomic_float_add]>,
      api=Vulkan, #spirv.resource_limits<>>
  } {
    gpu.func @add(%storage: memref<1xf32, #spirv.storage_class<Workgroup>>) kernel
        attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [1, 1, 1]>} {
      %index = arith.constant 0 : index
      %value = arith.constant 1.0 : f32
      %previous = memref.atomic_rmw addf %value, %storage[%index]
          {vernon.atomic_implementation = "native"} :
        (f32, memref<1xf32, #spirv.storage_class<Workgroup>>) -> f32
      gpu.return
    }
  }
}
