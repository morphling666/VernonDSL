// RUN: %vernon-opt --vernon-convert-gpu-to-spirv %s | %FileCheck %s
//
// CHECK: spirv.AtomicCompareExchange <Workgroup> <None> <None>
// CHECK: spirv.mlir.loop
// CHECK: spirv.FAdd
// CHECK: spirv.AtomicCompareExchange <Workgroup> <None> <None>
// CHECK-NOT: spirv.EXTAtomicFAdd
// CHECK-NOT: AtomicFloat32AddEXT
// CHECK-NOT: !spirv.ptr<f32, Workgroup>

module attributes {vernon.compiler_contract_version = 14 : i64, vernon.pipeline_version = 18 : i64} {
  gpu.module @kernels {
    gpu.func @add(%storage: memref<1xf32, #spirv.storage_class<Workgroup>>) kernel
        attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [1, 1, 1]>} {
      %index = arith.constant 0 : index
      %value = arith.constant 1.0 : f32
      %loaded = memref.load %storage[%index] : memref<1xf32, #spirv.storage_class<Workgroup>>
      memref.store %loaded, %storage[%index] : memref<1xf32, #spirv.storage_class<Workgroup>>
      %previous = memref.atomic_rmw addf %value, %storage[%index]
          {vernon.atomic_implementation = "integer_cas"} :
        (f32, memref<1xf32, #spirv.storage_class<Workgroup>>) -> f32
      gpu.return
    }
  }
}
