# Autodiff implementation status

Last updated: 2026-08-05

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
- Stateful TensorView conditionals, literal-range loops, dynamic leading-break
  loops, and branch-local early returns.
- Reusable CPU reference pullbacks with an explicit batched launch executor.

### Stateful Storage AD

- Functionalized TensorView loads and stores in the ProgramGraph.
- Explicit `storage` resource role in compiler reflection and Runtime binding.
- Fresh TensorView gradients for supported overwrite semantics.
- Static and dynamically indexed Storage effects.
- Independent lowering of multiple writable Storage parameters, with alias
  validation deferred to Runtime binding.
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
- GPU forward emission uses independent tape-capture, Storage-commit, and
  output-observation phases. Tape capture interprets functional Storage SSA
  without physical writes, and each Storage effect is committed exactly once.
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
- Native CPU/GPU forward and reverse lowering for every frontend derivative
  rule, including `acos`, `atan2`, `abs`, `pow`, `dot`, `matmul`, `norm`,
  `normalize`, `cross`, and `reflect`.
- Portable SPIR-V `atan2` lowering is shared by compute and graphics paths and
  preserves signed-zero quadrants, zero-origin cases, and infinite operands.
- f16 primals with f32 cotangents and gradients on native CPU/GPU, including
  the public Python `pipeline.vjp()` path.
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
- Diagnostics for context-bound Runtime APIs are thread-local per context.
  Nested operations commit once through a common outer scope, and
  `vernonRuntimeGetLastError` reads only the published channel.
- Failed graph forwards clear transient resource handles before retry, and
  graph bindings reject overlapping writable Storage host ranges.
- Compiler target lowering owns device-storage and float atomic capability
  selection; those capabilities do not alter the frontend derivative graph.

### Contracts and acceptance

- Compiler contract version: 10.
- Pipeline contract version: 13.
- Versioned `VernonAdTapeAllocator` CPU ABI with a hidden
  `ad_tape_allocator` packed argument, checked region handles, exact
  post-capacity required-byte accounting, latched failure states, per-thread
  invocation ownership, reset semantics, and immutable pullback-owned tape
  snapshots. The descriptor is an internal Compiler/Runtime header, and a
  dedicated CPU AD ABI pass verifies the hidden argument before CPU lowering.
- One grid-independent manifest launch/resource plan containing workgroup,
  semantic accumulation operations, invocation-axis evidence, and dynamic
  carrier ABI.
- Cooked pure-Value and Stateful Storage VJP acceptance on Vulkan and Metal.
- One cooked numerical VJP fixture covers all frontend math rules, dynamic
  leading-break loops, branch-local early return, both branches of Stateful
  Storage mutation, multiple writable resources, and overlap rejection.
- Fixture cooking covers CPU, CUDA, Vulkan, Metal, DirectX, OpenGL, and OpenGL
  ES. Runtime numerical acceptance has been executed on CPU, Metal, and Vulkan
  on current hardware. CUDA, OpenGL, and OpenGL ES are currently
  cooking-validated and execute the same numerical test when capability
  discovery reports an available device; DirectX cooking requires DXC and its
  numerical test requires an available DirectX runtime.
- Cooked parallel disjoint gather/scatter acceptance on Vulkan and Metal.
- Multi-node ExecutionGraph VJP acceptance on Vulkan.
- CUDA acceptance is enabled when CUDA hardware is available.
- The current Python autodiff suite passes 54 tests; native numerical CTest
  acceptance passes on CPU, Metal, and Vulkan on the current host.

## Implemented capability boundaries

- Every forward invocation accepts a positive runtime `(x, y, z)` grid; the
  grid is absent from assets, manifests, profile identity, and artifacts.
- CUDA uses the f32 atomic fast path for conflicting accumulation. Conflicting
  f64 gradients use deterministic serial reduction because CUDA f64 atomics are
  not enabled.
- Portable SPIR-V targets use direct stores for proven disjoint scatter and
  deterministic serial reduction for conflicting accumulation.
- GPU backward emission is target independent. Compiler backend accumulation
  lowering selects direct, atomic, or canonical serial scheduling and reflects
  an explicit serial-dispatch contract to Runtime.
- ExecutionGraph AD currently exposes a C++ GPU composition API with one
  selected sink output.
- Native CPU execution accepts flattened reflected Value leaves for Scalar,
  Tensor, Tuple, and Struct inputs.
- Native CPU backward lowering and Python bindings preserve Struct/Tuple leaf
  paths for leaf-level `wrt` gradients.
- `VernonAdValue` carries complete logical shape and Runtime rejects equal-size
  values whose shapes differ from reflection.

## Implementation roadmap

This document is an implemented-behavior snapshot and does not maintain
roadmap checkboxes. The ordered, authoritative task list and acceptance gates
are in [`autodiff-task-list.md`](autodiff-task-list.md).

The current highest-priority boundary remains general runtime-bounded control
flow with checked dynamic tape, followed by structured aggregate parity and
the GPU physical-ABI/capture-retry-commit migration. Later language, graph,
graphics, higher-order AD, efficiency, and platform work is ordered in the
same task list.

## Source map

- Frontend graph and analysis:
  `python/vernon_dsl/frontend/autodiff.py`
- CPU reference:
  `python/vernon_dsl/frontend/autodiff_cpu.py`
- Stable native lowering façade:
  `python/vernon_dsl/frontend/autodiff_native.py`
- Typed native ABI and split emitters:
  `python/vernon_dsl/frontend/autodiff_native_{abi,common,cpu,gpu,math}.py`
- Profile ABI:
  `python/vernon_dsl/frontend/autodiff_profiles.py`
- GPU Runtime:
  `source/lib/runtime/autodiff/runtime_autodiff_gpu.cpp`
- CPU tape allocator:
  `source/lib/runtime/autodiff/host_tape_allocator.{h,cpp}`
- ExecutionGraph AD Runtime:
  `source/lib/runtime/autodiff/runtime_autodiff_graph.cpp`
- Public graph API:
  `source/include/VernonAutodiffGraph.h`
- Manifest contract:
  `source/lib/runtime/pipeline_manifest.{h,cpp}`
- Normative design:
  `specs/autodiff.md`
