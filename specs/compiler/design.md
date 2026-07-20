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

GLSL language versions are compile options, not backend constants. A zero
version selects the target default; OpenGL and OpenGL ES callers may request a
specific version through the stable C API or CLI. Other targets reject this
option rather than silently ignoring it.

## Completed module and artifact work

1. Project-local absolute and relative imports are resolved from source without
   executing imported Python.
2. Transitive symbols are namespaced, with diagnostics for missing or duplicate
   symbols and import cycles.
3. The function call graph is validated and recursion is rejected.
4. Non-entry helpers are deterministically inlined before Vulkan, CUDA, and CPU
   backend routing.
5. Transitive SHA-256 source dependencies are emitted into MLIR and reflection,
   so the canonical module hash changes with imported source.
6. A shared branchless shadow helper is compiled through Vulkan SPIR-V, CPU
   LLVM, and OpenGL GLSL integration tests. CUDA helper coverage remains pending
   until a compute shader uses the shared module.
7. OpenGL and OpenGL ES GLSL versions are selectable through the stable C API
   and `--glsl-version`; zero retains the target default.
8. Vernon can compile paired generated GLSL source and publish it through
   `ShaderProvider`. This is source loading only; reflected resources are not
   yet bound.
9. Compiler reflection schema 2 includes target options and an explicit
   entry-point/stage/format/filename artifact table. The CLI cooks a normally
   named directory containing `shader.json` and separate readable OpenGL
   artifacts; Vernon validates, registers, mounts, and lazily links these
   bundles through `ShaderProvider`. This first runtime asset slice requires one
   vertex and one fragment artifact.
10. The Python frontend specializes `feature`, `When`, and compile-time feature
    branches, prunes compilation to a selected stage entry, and infers stable
    interface locations from the unspecialized signature. Shader-module and
    shader-pipeline manifests cook explicit variant sets into a schema-2
    `shader.json`; unchanged stages are content-deduplicated. Vernon parses this
    variant map and resolves exact canonical feature sets without fallback.

## Next implementation session

Make compiler code and reflection a single runtime-consumable shader bundle:

Shader variant authoring, stage composition, cooking, and asset integration are
specified in [shader_variant_asset_plan.md](shader_variant_asset_plan.md).

1. Extend reflection with an artifact table mapping entry point, stage, target,
   format, and artifact filename. Do not infer stage pairing from filenames.
2. Record each cross-compiled resource's exact generated block/uniform/member
   name and layout alongside its DSL argument index, kind, set, and binding.
3. Add a Vernon `CompiledShaderBundle` loader that validates reflection schema,
   target, GLSL version, module hash, dependencies, and required vertex/fragment
   artifacts before publishing a program.
4. Implement OpenGL reflection binding. Inputs/outputs continue to use explicit
   locations; uniforms use reflected UBO layouts and textures/samplers use a
   deterministic `(set, binding)` to OpenGL binding-point mapping.
5. Make materials provide typed values by `(set, binding)` instead of generated
   GLSL names. Keep the existing named-uniform path for legacy shaders.
6. Add shader variant keys and persistent cache invalidation from target,
   requested GLSL version, feature set, module hash, and dependency hashes.
7. Test bundle parsing and binding without a window where possible. GPU/GUI
   shader tests must be run explicitly by the developer.

After bundle integration, remaining backend work includes CUDA shared-helper
coverage, graphics/CPU control flow and texture coverage required by real
materials, and DXC-to-DXIL artifact generation when DXC is available.
