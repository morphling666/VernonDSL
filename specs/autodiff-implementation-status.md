# Autodiff implementation status

Last updated: 2026-08-04

This document records implemented behavior. `autodiff.md` remains the
normative design and contract document.

## Implemented

### Program transform and reverse planning

- Declarative compute VJP through `ProgramExpression` and
  `ProgramTransformSpec`.
- Typed semantic graph, reverse plan, tape layout, and launch/resource plan
  construction with deterministic identities and derivative-rule validation.
- Scalar, static Tensor, Tuple, Struct, and static-shape TensorView gradients.
- Static and dynamic Tensor indexing with gather/scatter adjoints.
- Tensor scalar broadcasting and reduction back to source shape.
- Pure helper inlining, bounded branches, pure-Value early returns,
  literal-range loops, and literal-bounded loops with a dynamic leading break.
- Reusable CPU reference pullbacks with an explicit batched launch executor.

### Stateful Storage AD

- Functionalized TensorView loads and stores in the ProgramGraph.
- Explicit `storage` resource role in compiler reflection and Runtime binding.
- Fresh TensorView gradients for supported overwrite semantics.
- Static and dynamically indexed Storage effects.
- Runtime rejection of overlapping writable Storage inputs.
- Grid-independent accumulation analysis:
  - `ReduceSum` represents shared Value contributions;
  - `ScatterAdd` represents indexed TensorView contributions;
  - complete `(x,y,z)` global-invocation indexing carries a
    `DisjointScatter` evidence hint;
  - invocation-conflicting static stores are rejected;
  - non-injective or shared-Value accumulation remains semantic until
    target lowering selects an available physical strategy.

### GPU lowering and Runtime

- Independent forward-with-tape and backward modules for CUDA, Vulkan, Metal,
  DirectX, OpenGL, and OpenGL ES lowering paths.
- Explicit GPU resources for inputs, Storage, outputs, tape, cotangents, and
  gradients.
- Typed native resource binding plans shared by the split common, CPU, and GPU
  emitters.
- Direct TensorView resource SSA lowering: static and dynamic reads/writes use
  one `vernon.load`/`vernon.store` at the access site. Reverse emits semantic
  `vernon.reduce_sum` and `vernon.scatter_add`; target lowering selects direct,
  atomic, or canonical deterministic serial reduction.
- Tensor Value gradient accumulation uses rank-proportional `scf.for` loops
  around semantic accumulation operations instead of emitting one operation
  per static element.
- TensorView state is excluded from tape; dynamic indices, branch predicates,
  and reverse-required scalar/Tensor Values remain explicit tape leaves.
- Immutable `ValueAbi` and `ResourceAbi` descriptors retain logical shape,
  alignment, dynamic physical shape, byte size, access, role, and the
  runtime-carrier marker.
- External-buffer and external-command-encoder entry points through the
  internal `GpuGraphExecutable` interface.
- Immediate `vernonAdPipelineForward` is a one-node adapter over the same
  compiled ExecutionGraph forward/pullback implementation used by composed
  graphs; the duplicate immediate GPU buffer execution path has been removed.
- Per-invocation positive three-dimensional VJP grid supplied to each Runtime
  forward call and retained by its pullback.
- Dynamic `(z,y,x)` output, cotangent, and tape carriers with X-fastest layout
  and tail-invocation guards.
- Freshly zeroed gradient buffers before accumulation.
- f32 atomic-add lowering where target capability permits it.
- Bitwise-deterministic canonical X-fastest serial reduction fallback where
  atomics are unavailable or deterministic execution is required.

### ExecutionGraph integration

- Public C++ composition API in `VernonAutodiffGraph.h`.
- Multiple cooked GPU VJP nodes in a validated DAG.
- External input binding, node-output to node-input connections, and one
  selected sink output.
- Builder topology compiles to a reusable `CompiledAutodiffGraph`; each forward
  invocation supplies a fresh external value set and creates an independent
  pullback.
- Forward profiles encoded as ExecutionGraph compute passes.
- Buffer hazards and forward barriers derived by ExecutionGraph.
- Compute passes resolve logical graph buffers through `ExecutionResources`
  instead of reading embedded native handles.
- Graph-owned output and tape resources retained by the pullback.
- Reverse-topological backward planning and cotangent routing.
- One device-resident reverse ExecutionGraph per pullback application, with
  connected cotangents and repeated external gradients accumulated directly in
  shared GPU buffers.
- ExecutionGraph-derived reverse hazards and barriers; acceptance checks one
  forward submission and one backward submission for a multi-node graph.
- Reusable pullbacks with fresh backward gradient resources.
- Pipeline and bundle objects may be destroyed after graph compilation; the
  compiled plan retains executable ownership and each pullback retains its
  forward graph and tape.
- Shared RAII context leases guard unexecuted graphs and immediate/graph
  pullbacks without reusing the pullback handle counter.
- GPU profiles cache ordered resource descriptors and materialize dynamic
  carrier shape/stride views from each invocation grid.
- Compiled plans are lock-free immutable objects. Pullbacks retain one
  object-local lock for reusable tape/stats, while same-device command encoder
  serialization belongs to ExecutionGraph rather than an AD context lock.
- AD invocation diagnostics are thread-local per Runtime context, avoiding
  concurrent writes to the shared context diagnostic string. Ordinary Runtime
  operation boundaries clear stale AD diagnostics before publishing their own
  errors.
- Failed graph forwards clear transient resource handles before retry, and
  graph bindings reject overlapping writable Storage host ranges.
- Compiler target lowering owns device-storage and float atomic capability
  selection; those capabilities do not alter the frontend derivative graph.

### Contracts and acceptance

- Compiler contract version: 10.
- Pipeline contract version: 13.
- One grid-independent manifest launch/resource plan containing workgroup,
  semantic accumulation operations, invocation-axis evidence, and dynamic
  carrier ABI.
- Cooked pure-Value and Stateful Storage VJP acceptance on Vulkan and Metal.
- Cooked parallel disjoint gather/scatter acceptance on Vulkan and Metal.
- Multi-node ExecutionGraph VJP acceptance on Vulkan.
- CUDA acceptance is enabled when CUDA hardware is available.
- All current Python autodiff tests pass: 49 tests and 12 subtests.
- All current dual-backend CTest tests pass: 201 tests, with unavailable
  hardware/backend cases skipped.

## Implemented capability boundaries

- Every forward invocation accepts a positive runtime `(x, y, z)` grid; the
  grid is absent from assets, manifests, profile identity, and artifacts.
- CUDA uses the f32 atomic fast path for conflicting accumulation. Conflicting
  f64 gradients use deterministic serial reduction because CUDA f64 atomics are
  not enabled.
- Portable SPIR-V targets use direct stores for proven disjoint scatter and
  deterministic serial reduction for conflicting accumulation.
- ExecutionGraph AD currently exposes a C++ GPU composition API with one
  selected sink output.
- Native CPU execution supports the reflected single-Value-leaf input ABI.

## Not implemented / remaining work

### Accumulation backend architecture

- Portable deterministic scheduling is currently a canonical single-dispatch
  X-fastest loop selected while emitting the target profile. The scheduling
  decision must move entirely into compiler backend lowering so the frontend
  emitter remains target independent.
- Contribution buffers, deterministic sorting, and segmented reduction are not
  implemented. The current serial fallback is correct and deterministic but
  does not scale to large invocation domains.

### Runtime and ABI completion

- ExecutionGraph AD does not yet support cross-node Storage mutation edges,
  multiple selected sink outputs, or C/Python composition APIs.
- Native CPU profile loading does not yet support aggregate Struct/Tuple inputs
  with multiple reflected Value leaves, although ProgramGraph and the CPU
  reference support those value types.
- `VernonAdValue` carries dtype and byte size but not a complete logical shape,
  so Runtime boundary validation cannot distinguish equal-size values with
  different logical shapes.
- GPU gradient initialization uses bounded host-to-device zero uploads because
  the RHI has no device buffer-fill operation.
- Forward and backward backend pipelines are resolved eagerly rather than
  lazily.

### Unified Runtime diagnostics

The current implementation keeps AD invocation diagnostics in a per-thread,
per-context channel and clears stale AD diagnostics at ordinary Runtime API
boundaries. This is the safe incremental behavior, but the long-term design is
one unified diagnostic system:

- all Runtime errors are written to a per-thread, per-context diagnostic;
- `context->error` is used only as temporary storage while a backend operation
  is being constructed;
- every public Runtime API commits its final diagnostic through one common exit
  path;
- `vernonRuntimeGetLastError` reads only the unified diagnostic channel.

### Language and differentiation coverage

- Stateful control flow does not support general dynamic loops, `continue`, or
  arbitrary in-loop `return`.
- Graphics-stage backward lowering and differentiable rendering rules are not
  implemented.
- Forward-mode JVP, materialized Jacobians, Hessians, and higher-order AD are
  not implemented.

### Platform acceptance

- CUDA execution still requires hardware CI coverage.
- DirectX requires DXC-backed Windows validation.
- OpenGL and OpenGL ES need broader cooked AD acceptance coverage.

## Source map

- Frontend graph and analysis:
  `python/vernon_dsl/frontend/autodiff.py`
- CPU reference:
  `python/vernon_dsl/frontend/autodiff_cpu.py`
- Stable native lowering façade:
  `python/vernon_dsl/frontend/autodiff_native.py`
- Typed native ABI and split emitters:
  `python/vernon_dsl/frontend/autodiff_native_{abi,common,cpu,gpu}.py`
- Profile ABI:
  `python/vernon_dsl/frontend/autodiff_profiles.py`
- GPU Runtime:
  `source/lib/runtime/runtime_autodiff_gpu.cpp`
- ExecutionGraph AD Runtime:
  `source/lib/runtime/runtime_autodiff_graph.cpp`
- Public graph API:
  `source/include/VernonAutodiffGraph.h`
- Manifest contract:
  `source/lib/runtime/pipeline_manifest.{h,cpp}`
- Normative design:
  `specs/autodiff.md`
