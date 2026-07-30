module attributes {vernon.compiler_contract_version = 6 : i64, vernon.pipeline_version = 9 : i64} {
  gpu.module @kernels {
    gpu.func @exchange(%storage: memref<1xi32, #spirv.storage_class<Workgroup>>) kernel
        attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [1, 1, 1]>} {
      %index = arith.constant 0 : index
      %value = arith.constant 7 : i32
      %previous = memref.atomic_rmw assign %value, %storage[%index] :
        (i32, memref<1xi32, #spirv.storage_class<Workgroup>>) -> i32
      gpu.return
    }
  }
}
