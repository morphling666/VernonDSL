# Compiler design

## Tensor-first type system

The Python language has one compound numeric type,
`Tensor[element_type, shape]`. Vector and matrix names are aliases that add
constructors and operations such as swizzle, dot, cross, transpose, and matrix
multiplication; they are not separate types.

The MLIR representation is selected by use rather than source spelling:

- Small fixed-size value tensors lower to `vector` and target-native
  vector/matrix types.
- General value computation uses `tensor` and `linalg`.
- Addressable resources use `memref` or target buffer types.

## Backend split

SPIR-V is not the universal backend IR. Target lowering branches from typed
Vernon and standard MLIR:

- Graphics stages lower through Vernon interface semantics to SPIR-V.
- Compute kernels lower to `gpu.module`/`gpu.func`, then to SPIR-V, NVVM, or
  ROCDL according to the target.
- CPU reference execution lowers directly to LLVM.

CUDA therefore uses GPU to NVVM to NVPTX and never depends on SPIR-V.

## Python and native boundary

The restricted Python frontend is an AOT tool. It parses source without
executing user code and emits textual MLIR. The C API accepts that MLIR,
validates it, emits reflection, and invokes an available target pipeline. It
does not embed CPython.

A target is reported as available only after its complete lowering and
artifact generation pipeline is registered. An IR-only prototype must return
`VERNON_STATUS_UNSUPPORTED_TARGET`.

## CPU resource ABI

CPU shader execution will receive plain ABI input/output structures plus an
explicit callback table for texture sampling and queries. It executes shader
entry functions for reference testing; it is not a software rasterizer.

## Target routing invariant

The common typed MLIR is the last shared representation. Backend routing is:

```text
Graphics DSL -> Vernon graphics IR -> SPIR-V
                                  |-> Vulkan consumes SPIR-V directly
                                  |-> SPIRV-Cross -> GLSL for OpenGL/OpenGL ES
                                  |-> SPIRV-Cross -> MSL for Metal
                                  `-> SPIRV-Cross -> HLSL -> DXC -> DXIL

Compute DSL -> MLIR GPU dialect
                            |-> SPIR-V for Vulkan compute
                            |-> NVVM -> NVPTX/PTX for CUDA
                            |-> ROCDL for AMD GPU
                            `-> standard MLIR -> LLVM for CPU reference
```

SPIR-V is canonical for graphics and Vulkan compute, but it is not the universal compute backend. CUDA must not be routed through SPIR-V. The project must not maintain independent GLSL, MSL, or HLSL emitters; those languages are produced through SPIRV-Cross, with target capability validation before translation.

## Next implementation session

The next priority is reusable, multi-file DSL functions. The current frontend
ignores Python import statements, and although calls to functions in the same
source file can be emitted as `func.call`, those calls are not yet supported by
all backend lowerings. Therefore shared functions such as shadow evaluation
cannot yet be used end to end.

Implement this without executing imported Python:

1. Resolve project-local absolute and relative imports from source files.
2. Collect transitive structs and function signatures into a namespaced module
   graph; diagnose missing symbols, duplicate symbols, and import cycles.
3. Build and validate the function call graph. Recursion is forbidden.
4. Inline non-entry helper functions in common typed MLIR before the graphics,
   GPU, and CPU backend split. This keeps backend-specific call lowering out of
   each target and allows one helper implementation to serve all targets.
5. Include transitive source hashes in compiler cache keys and reflection
   dependencies.
6. Add a shared shadow helper in a separate module and test it through Vulkan
   SPIR-V, CUDA PTX where applicable, and CPU reference execution.

After module support, the remaining integration work is:

- connect compiler artifacts and reflection to Vernon's shader/material asset
  service;
- add shader variant keys, dependency invalidation, and persistent artifact
  caching;
- bind reflected resources and CPU entry ABI from C++;
- extend graphics/CPU control flow and resource coverage as real shaders
  require it;
- complete the DirectX route with DXC-to-DXIL when DXC is available.
