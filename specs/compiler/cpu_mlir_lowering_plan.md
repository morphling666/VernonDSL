# CPU MLIR lowering migration

Status: implemented. This document is retained as the migration and design
record; its phased instructions describe the work that produced the current
`vernon-cpu-pipeline`.

## Status and scope

This document records the replacement of the hand-written `CpuLLVMEmitter` in
`source/lib/VernonCompiler.cpp` with an extensible MLIR conversion pipeline.

This migration changes compiler internals only. It must not change the public
compiler/runtime C ABI, reflection schema, CPU bundle schema, Python `@kernel`
behavior, or the CUDA and Vulkan artifact pipelines.

The Python frontend remains responsible for parsing the restricted Python AST,
specializing compile-time values and Tensor shapes, and emitting typed textual
Vernon MLIR. Python must not emit LLVM IR and must not interpret kernel bodies.

## Motivation

Before this migration, the CPU backend validated and inlined the Vernon MLIR
module, then `CpuLLVMEmitter` recursively walked MLIR operations and
constructed LLVM IR with `llvm::IRBuilder`. Every newly supported arithmetic
operation, intrinsic, or structured-control-flow form therefore required
another custom case. The emitter duplicated behavior already represented by
standard MLIR dialects and their maintained LLVM conversions.

CUDA and Vulkan do not have this problem: both run in-process MLIR pass
pipelines and use standard NVVM or SPIR-V lowering. CPU should follow the same
architecture while retaining its target-specific host invocation wrapper.

## Target architecture

```text
restricted Python AST
        |
        v
typed Vernon MLIR
        |
        +-- vernon-validate
        +-- inline shared helpers
        |
        v
CPU-normalized standard MLIR
  - func, arith, math
  - vector/tensor value operations
  - memref resource operations
  - scf, then cf
        |
        v
LLVM dialect
        |
        v
translateModuleToLLVMIR
        |
        +-- generate __vernon_cpu_<entry> wrappers
        |
        v
LLVM module.ll
        |
        +-- LLJIT reference entry
        `-- clang AOT shared library and validated compute bundle
```

Production compilation runs these passes in process through
`mlir::PassManager`. `vernon-opt` exposes the same named passes and pipeline for
debugging and FileCheck tests. Production code must not spawn `mlir-opt`.

## Stable contracts

The following are migration invariants:

1. The exported entry symbol remains exactly `__vernon_cpu_<entry>`.
2. `VernonCpuInvocation`, `VernonCpuTextureCallbacks`, and
   `VERNON_CPU_INVOCATION_ABI_VERSION` remain unchanged.
3. Argument packing continues to use reflection `cpu_offset`, `cpu_size`, and
   `cpu_arguments_size`, including the existing alignment rules.
4. The wrapper continues to validate null pointers and argument/result sizes
   and returns the existing `VernonStatus` values.
5. Texture operations continue to use the callback table supplied through the
   invocation. A required but missing callback table is an invalid invocation.
6. CPU reflection continues to name the wrapper symbol, not the internal
   lowered function.
7. The CPU compiler artifact remains `module.ll`; `vernon-compile` continues to
   use clang to produce `compute.dll`, `compute.so`, or `compute.dylib`.
8. CPU compute and pipeline bundle manifests and their OS, architecture, size,
   path, and SHA-256 validation remain unchanged.
9. Helper inlining and reflection generation happen while the canonical typed
   Vernon module and its source-facing interface metadata are still available.
10. Unknown or unlowered Vernon operations fail compilation with a diagnostic
    that includes the operation and entry name.
11. Windows wrappers retain `dllexport`; other platforms retain default symbol
    visibility.
12. Row-major Tensor element ordering and buffer index semantics do not change.

## Lowering decisions

### Standard dialects before LLVM

Kernel body semantics lower through standard MLIR rather than directly
creating LLVM dialect operations:

- small static value Tensors become `vector`;
- general value computation remains `tensor`/`linalg` until existing standard
  transformations lower it;
- `scf.if` and `scf.while` use MLIR's SCF-to-CF conversion;
- arithmetic and math use the upstream arith/math conversions;
- addressable buffers use memref load/store semantics;
- only target-specific operations that have no suitable standard form receive
  a late custom LLVM lowering.

The initial CPU implementation should retain the current value-Tensor
capabilities. It must not accidentally adopt the GPU 16-element limit for
values the old CPU emitter accepted. If a common pattern requires a limit, make
the limit a pass option and test the CPU policy explicitly.

### Buffer boundary

The runtime ABI supplies a raw data pointer for each `!vernon.buffer`; it does
not supply a public memref descriptor. The CPU resource pass lowers
`buffer_load` and `buffer_store` to rank-one, row-major memref operations.

The internal lowered kernel may receive a memref descriptor, but the external
wrapper must continue accepting the raw pointer packed in
`VernonCpuInvocation`. The wrapper constructs the internal descriptor with:

- allocated pointer and aligned pointer equal to the runtime pointer;
- offset zero;
- stride one;
- static extent from specialized `vernon.tensor_shape` metadata when present.

Legacy C API modules may omit `vernon.tensor_shape`. Their load/store semantics
must continue working because generated address calculation requires only the
pointer, offset, index, and stride. The descriptor extent must not introduce a
new runtime bounds check or become a new ABI requirement.

Do not change the runtime launch argument format to expose MLIR memref
descriptors.

### Texture callbacks

`texture_sample` remains a CPU-specific late-lowered operation. Introduce a
dedicated intermediate operation or conversion pattern that:

- records that an entry requires texture callbacks;
- loads `sample_2d` and `user_data` from the callback table;
- preserves texture and sampler handles as their current integer ABI values;
- writes the callback result into the expected fixed vector value.

The generic value/intrinsic pattern library must not contain CPU callback-table
layout knowledge.

### ABI wrapper placement

Generate wrappers after `translateModuleToLLVMIR` with a dedicated LLVM module
component, provisionally `VernonCpuAbiWrapper`. This keeps host structure
offsets, argument blob unpacking, return-value storage, platform export flags,
and status returns out of shader semantic conversion patterns.

Extract the existing wrapper logic before deleting `CpuLLVMEmitter`. Wrapper
generation consumes explicit per-entry metadata captured before lowering:

- internal function symbol;
- source argument/result types and packing offsets;
- whether texture callbacks are required;
- exported wrapper symbol.

## Pass structure

Add a central `buildVernonCpuPassPipeline()` used by both `compileCpu()` and
`vernon-opt`. The expected order is:

1. `vernon-validate`;
2. inline shared helpers;
3. `vernon-lower-cpu-tensors`;
4. `vernon-lower-cpu-resources`;
5. generic Vernon intrinsic and swizzle normalization;
6. CPU texture marker/call preparation;
7. any required tensor/linalg bufferization;
8. `convert-scf-to-cf`;
9. vector, math, index, arith, memref, control-flow, and func conversion to the
   LLVM dialect in the order required by the installed MLIR version;
10. a final conversion check that permits only the module and LLVM dialect.

Use explicit conversion targets and legality. Do not use
`allowUnknownOps = true` in the CPU pipeline. A partially converted module must
fail before LLVM translation.

After the pass manager succeeds:

1. call `translateModuleToLLVMIR`;
2. generate CPU ABI wrappers;
3. run `llvm::verifyModule`;
4. print `module.ll`;
5. add the same module to LLJIT for the compiler reference API;
6. set reflection wrapper symbols with the existing naming rule.

## Shared pattern extraction

Extract backend-independent patterns currently owned by
`VernonLowerGPUTensorsPass`:

- static Tensor to vector type conversion;
- constant, splat, from-elements, and extract conversion;
- swizzle;
- construct, dot, normalize, cross, reflect, and matrix multiplication;
- min, max, clamp, and pow normalization;
- elementwise arithmetic and SCF structural type conversion.

The shared library provides patterns and type-conversion helpers. Backend
passes remain responsible for:

- operation legality;
- entry container (`func.func` or `gpu.func`);
- resource representation;
- target-specific math and texture operations;
- target limits and capabilities.

Refactor the existing GPU pass to consume the extracted patterns before using
them for CPU. This first step must be behavior-neutral and must keep all
CUDA/Vulkan tests green.

## Implementation phases

### Phase 0: establish parity fixtures

Before replacing behavior:

1. capture the existing C API CPU modules for value vectors, intrinsics,
   buffers, texture callbacks, invalid invocation packets, and results;
2. add CPU SCF `if`/`while` coverage with loop-carried scalar and vector values;
3. add an unknown-intrinsic negative test;
4. add a temporary test-only switch that can compile the same module through
   the old and new paths for numerical parity.

The switch is a migration tool, not a supported build option, and is removed
with the old emitter.

### Phase 1: extract shared value patterns

1. Move reusable conversion patterns out of
   `VernonLowerGPUTensors.cpp`.
2. Keep `VernonLowerGPUTensorsPass` behavior and legality unchanged.
3. Add focused FileCheck tests for the shared pattern outputs.
4. Run CUDA and Vulkan compiler/runtime tests before beginning CPU conversion.

### Phase 2: lower CPU values and resources

1. Implement `VernonLowerCPUTensors`.
2. Implement `VernonLowerCPUResources`.
3. Lower generic intrinsics and swizzles to standard dialect operations.
4. Preserve a dedicated CPU texture operation until its late lowering.
5. Register each pass and expose it through `vernon-opt`.

Exit criterion: the CPU-normalized module contains no generic Vernon
operations; only an explicitly legal CPU texture operation may remain.

### Phase 3: convert to LLVM dialect

1. Run upstream SCF-to-CF conversion.
2. Convert vector, math, index, arith, memref, CF, and func operations.
3. Configure host data layout from the same target-machine information used by
   LLJIT.
4. Add a full-conversion legality check.
5. Translate the resulting module to `llvm::Module`.

Exit criterion: the translated internal functions pass LLVM verification
without any wrapper generation.

### Phase 4: extract and attach ABI wrappers

1. Move `emitWrapper` behavior into `VernonCpuAbiWrapper`.
2. Feed it captured entry metadata rather than inspecting already-lowered
   source types.
3. Verify invalid packet behavior, result storage, texture callbacks, symbol
   visibility, and reflection symbol names.
4. Emit `module.ll` and load the wrapper symbols through LLJIT.

Exit criterion: all old/new parity cases produce the same status and numerical
results.

### Phase 5: switch production and remove legacy code

1. Make `compileCpu()` use only `buildVernonCpuPassPipeline()`.
2. Delete `CpuLLVMEmitter` and the test-only dual-path switch.
3. Remove now-unused direct IRBuilder helpers and includes.
4. Keep the wrapper generator as the only intentional LLVM IRBuilder layer.
5. Run the complete compiler, runtime, Python, CUDA, and Vulkan test suites.

No permanent fallback to `CpuLLVMEmitter` is allowed.

## Expected file changes

Names may be adjusted to existing project conventions, but responsibilities
must remain separate.

New headers and implementations:

```text
source/include/mlir/Dialect/Vernon/Transforms/VernonSharedValuePatterns.h
source/lib/Dialect/Vernon/Transforms/VernonSharedValuePatterns.cpp
source/include/mlir/Dialect/Vernon/Transforms/VernonLowerCPUTensors.h
source/lib/Dialect/Vernon/Transforms/VernonLowerCPUTensors.cpp
source/include/mlir/Dialect/Vernon/Transforms/VernonLowerCPUResources.h
source/lib/Dialect/Vernon/Transforms/VernonLowerCPUResources.cpp
source/include/mlir/Dialect/Vernon/Transforms/VernonCpuPipeline.h
source/lib/Dialect/Vernon/Transforms/VernonCpuPipeline.cpp
source/lib/VernonCpuAbiWrapper.h
source/lib/VernonCpuAbiWrapper.cpp
```

Existing files:

- `source/lib/Dialect/Vernon/Transforms/VernonLowerGPUTensors.cpp`: consume
  shared value patterns without changing GPU semantics.
- `source/lib/Dialect/Vernon/Transforms/CMakeLists.txt`: compile and link new
  passes and required conversion libraries.
- `source/CMakeLists.txt`: link the LLVM dialect/export and standard conversion
  libraries required by `VernonDSLCompiler`.
- `source/lib/VernonCompiler.cpp`: retain orchestration, reflection, target
  machine setup, translation, JIT, and artifact publication; remove direct
  kernel body emission.
- `source/tools/vernon_opt/vernon_opt.cpp`: register CPU passes and the complete
  CPU debug pipeline.
- `source/tests/`: add pass-level and end-to-end parity fixtures.
- `specs/compiler/design.md`: replace the direct-SCF-emission description with
  the accepted standard MLIR conversion architecture after implementation.

Expected upstream CMake dependencies include the installed-version equivalents
of:

```text
MLIRArithToLLVM
MLIRControlFlowToLLVM
MLIRFuncToLLVM
MLIRIndexToLLVM
MLIRMathToLLVM
MLIRMemRefToLLVM
MLIRVectorToLLVM
MLIRLLVMCommonConversion
MLIRLLVMDialect
MLIRSCFToControlFlow
MLIRTargetLLVMIRExport
```

Use actual exported target names from the checked-in LLVM/MLIR version rather
than assuming names from another release.

## Verification gates

### Pass-level tests

Add FileCheck inputs covering:

- static Tensor/vector construction and extraction;
- row-major multi-dimensional extraction;
- every generic Vernon intrinsic supported by CPU;
- resource load/store;
- `scf.if` results;
- `scf.while` loop-carried scalar and vector values;
- CPU texture callback lowering;
- complete removal of illegal Vernon operations;
- explicit failure for an unknown intrinsic.

Each test should be runnable through `vernon-opt` at the relevant pipeline
stage.

### Compiler API tests

The existing `vernon-compiler-c-api-test` and
`vernon-compiler-cpp-api-test` remain mandatory. Extend the C test to verify:

- old/new parity during migration;
- wrapper null and undersized argument/result rejection;
- buffer mutation through a packed raw pointer;
- texture callback success and missing-callback rejection;
- exact `__vernon_cpu_<entry>` lookup;
- SCF and math numerical behavior.

### Runtime and AOT tests

Required tests:

- `vernon-runtime-cpu-aot-test`;
- `vernon-runtime-cpu-pipeline-test`;
- CPU `vernon-compile --compute-bundle` output loaded from a fresh directory;
- artifact size/hash and host OS/architecture validation;
- Windows DLL export or Linux/macOS symbol visibility.

### Python and cross-backend tests

Run:

- all `python/tests`;
- CPU `fill`, shared helper, vector while, fractal, Tensor residency, and cache
  tests;
- `shared_struct_methods.py --arch cpu`;
- conditional CUDA and Vulkan numerical parity;
- CUDA/Vulkan compiler and pipeline regression tests after shared-pattern
  extraction.

The migration is complete only when no test or production path references
`CpuLLVMEmitter`.

## Risks and mitigations

- **Raw pointer versus memref ABI:** construct descriptors only inside the
  wrapper and test pointer arithmetic directly; never expose descriptors in
  the public runtime ABI.
- **Data-layout mismatch:** use the host target machine data layout for MLIR
  LLVM conversion, wrapper packing, translation, and LLJIT.
- **SCF semantic drift:** compare status and numerical results against the old
  emitter before deleting it.
- **Vector layout drift:** test row-major extraction and matrix operations at
  the IR and runtime levels.
- **Texture callback drift:** isolate callback lowering and preserve all
  invalid-invocation tests.
- **Incomplete conversion hidden by permissive legality:** use explicit
  conversion targets and a final full-conversion check.
- **Windows-only export failure:** inspect the generated shared library symbol
  in addition to checking JIT lookup.
- **GPU regression from shared patterns:** make extraction behavior-neutral and
  run CUDA/Vulkan tests before using the shared library from CPU.
- **MLIR API/version differences:** verify pass and CMake target names against
  this repository's LLVM checkout before coding each phase.

## Completion criteria

The migration is complete when:

1. `CpuLLVMEmitter` and its body-emission helpers are deleted.
2. CPU kernel bodies lower through registered MLIR conversion passes.
3. Only the dedicated CPU ABI wrapper component constructs LLVM IR directly.
4. `compileCpu()` and `vernon-opt` use the same CPU pipeline builder.
5. Existing C ABI, reflection, runtime, and bundle contracts remain unchanged.
6. Pass-level, compiler, runtime, AOT, Python, CUDA, and Vulkan gates pass.
7. `specs/compiler/design.md` describes the implemented pipeline rather than
   the removed direct emitter.

## Next steps after CPU migration

These improvements are intentionally non-blocking and should be planned
separately after the CPU conversion pipeline is stable:

1. Extract the pass assembly currently inside `compileCuda()` and
   `compileVulkan()` into reusable backend pipeline builders callable by
   `vernon-opt`.
2. Make CPU, CUDA, and Vulkan consume the shared static Tensor, vector, and
   generic intrinsic pattern library; backend passes retain only
   target-specific legality and resource rules.
3. Narrow CUDA's `allowUnknownOps = true` bufferization configuration by
   explicitly legalizing expected intermediate operations, so unknown Vernon
   operations fail near their source with operation/entry diagnostics.
4. Keep `VernonLowerCUDAMathPass` and Vernon-to-SPIR-V conversion as normal
   target-specific lowering, while aligning their naming, registration, and
   staged IR tests.
5. Add `vernon-opt` FileCheck coverage for CUDA and Vulkan pipeline builders.
   Production compilation continues to run pass managers in process and does
   not invoke `mlir-opt`.
