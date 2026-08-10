# Multi-Kernel TensorView and Autodiff Architecture

## Status

This is the final architecture for mutable device `TensorView` state,
cross-workgroup communication, execution graphs, and portable autodiff.

It supersedes:

- `specs/cpu-cooperative-workgroup-autodiff-hardening.md`;
- `specs/cpu-cooperative-workgroup-next-session-plan.md`.

The superseded documents combined valid CPU scheduler hardening with an invalid
long-term premise: using a workgroup barrier to divide mutable device
`TensorView` communication into globally ordered phases inside one kernel.
The valid CPU, allocation, AOT, and workgroup-local autodiff requirements are
retained below.

Do not update `VERNON_COMPILER_CONTRACT_VERSION` or
`VERNON_PIPELINE_VERSION` until the release that publishes the graph schema.
Do not add compatibility readers or fallback spellings for the new schema.

## Fixed decisions

### One Storage abstraction

`TensorView` remains the only device-language Storage abstraction.

- A kernel parameter `TensorView` uses device address space.
- `workgroup_storage()` returns a `TensorView` in workgroup address space.
- Runtime-owned intermediate allocations are ordinary `TensorStorage` owners
  exposed to kernels through ordinary `TensorView` bindings.

Physical placement does not change the logical Storage type. A backend may
cache a device `TensorView` region in native shared memory only as a proven
optimization.

### Runtime-dynamic grid

`grid=(x,y,z)` is the workgroup count on every backend:

```text
extent = grid * workgroup_size
global_id = workgroup_id * workgroup_size + local_id
```

Grid remains runtime data. It is not part of artifact specialization or
frontend cache identity. Kernel correctness must hold for every legal runtime
grid and explicitly masked logical extent.

### Workgroup barriers are local

A workgroup barrier synchronizes only lanes in one workgroup. It never creates
a happens-before edge between workgroups.

Device-address-space `TensorView` state must not use:

```text
lane/workgroup A writes
workgroup_barrier
lane/workgroup B reads the written value
```

as a global phase boundary. If the producer and consumer may belong to
different workgroups, the program is invalid even when one test invocation
happens to use `grid=(1,1,1)`.

The portable global synchronization boundary is:

```text
dispatch A completes
device/resource fence
dispatch B begins
```

Cross-workgroup communication therefore uses multiple kernels.

An optional future grid-wide barrier may exist behind a backend capability
such as cooperative launch, but it is not part of the portable core language
and cannot be required by a portable PipelineAsset.

## Single-kernel memory rules

Each kernel is one globally unordered parallel phase.

Legal device `TensorView` patterns include:

- read-only gather and stencil inputs;
- pointwise or otherwise proven lane-exclusive writes;
- explicit atomic updates;
- explicit scatter/reduction operations with defined collision semantics;
- lane-local read/write versions whose ownership is proven.

Illegal patterns include:

- a device write followed by barrier-mediated cross-lane consumption;
- relying on a workgroup barrier for device-wide visibility or completion;
- non-atomic overlapping writes with unspecified winner;
- software global barriers based only on an atomic arrival counter.

An atomic counter does not provide a portable global barrier. Waiting
workgroups can occupy all resident hardware slots and prevent unscheduled
workgroups from reaching the counter. Memory ordering also does not imply an
execution-progress guarantee.

`VernonValidation` must implement a `KernelMemoryPhaseAnalysis` that:

1. treats the complete kernel as one device-memory epoch;
2. does not split the device epoch at workgroup barriers;
3. builds alias sets for TensorView owners and subviews;
4. classifies reads, writes, atomics, scatters, and reductions;
5. proves exclusive writes where required;
6. rejects barrier-separated mutable device communication with a diagnostic
   that recommends a multi-kernel graph;
7. preserves normal barrier analysis for workgroup-address-space TensorViews.

Compiler analysis is conservative. Runtime binding must additionally validate
dynamic shape, signed stride, offset, byte range, and overlapping aliases.

## Execution graph is the global memory model

Cross-kernel data flow is represented by a typed execution graph.

Required semantic nodes are:

- `Dispatch`: one compute kernel with runtime grid and TensorView bindings;
- `Storage`: one runtime-owned TensorStorage allocation;
- `For`: structured dispatch iteration with explicit loop-carried versions;
- `Reduce`: a reduction with a defined identity and merge operation;
- `Accumulate`: deterministic gradient or result accumulation;
- graph inputs and outputs.

Every writable Storage edge has an SSA-like version:

```text
state.v0 -> force -> forced.v0
forced.v0 -> advect -> advected.v0
advected.v0 -> divergence -> divergence.v0
pressure_a.vN, pressure_b.vN -> jacobi -> pressure_a.vN+1, pressure_b.vN+1
```

Edges define producer/consumer dependencies and global happens-before.
Multiple readers can share a version. Mutation creates a new version instead
of hiding ordering in side effects.

Graph construction may use Python tracing or a declarative builder, but direct
kernel invocation remains available. The resulting internal graph must be
identical for direct and cooked execution.

The graph belongs in the execution-graph/compiler layer. `RuntimeCore` keeps
its current responsibility boundary: it parses reflection and binding plans,
while the provider owns native resources, barriers, command recording,
submission, and completion.

## Graph-level autodiff

Autodiff operates at two levels.

### Kernel-local autodiff

Structured VJP differentiates one kernel phase.

- Lane-private values retain lane-private adjoints.
- True workgroup-address-space TensorViews use workgroup-shared adjoints.
- Reverse workgroup barriers execute in reverse order.
- Device TensorViews are phase inputs, phase outputs, or explicit
  accumulation destinations; they are not reclassified as hidden
  workgroup-coupled scratch.
- Every lane contribution is preserved. A generic leader-only gradient store
  is invalid.

The syntactic three-axis mutable-device-slice ownership matcher is removed.
Workgroup-coupled internal adjoints are selected from actual workgroup address
space, not inferred from global index spelling.

### Graph VJP

Graph VJP:

1. versions every forward Storage edge;
2. traverses dispatch nodes in reverse topological order;
3. invokes each kernel's structured VJP;
4. reverses structured `For` nodes;
5. applies explicit transpose rules for gather, scatter, reduce, and
   accumulate nodes;
6. merges cotangents from multiple consumers deterministically;
7. retains or recomputes forward versions according to a checkpoint policy;
8. publishes gradients only after the complete graph pullback succeeds.

Graph-owned intermediate TensorStorage is not a new language type. It is a
runtime owner whose views satisfy the same TensorView descriptor ABI.

## Smoke-fluid decomposition

The smoke-fluid example is a graph, not one giant cooperative kernel:

1. apply forces;
2. advect velocity;
3. compute divergence;
4. execute Jacobi pressure iterations with ping-pong TensorView versions;
5. project velocity and transport density;
6. reduce loss.

Every phase accepts dynamic TensorViews and a runtime grid. Intermediate
owners are allocated by the graph storage planner. The pressure iteration is a
structured graph loop. The loss is an explicit graph/kernel reduction.

`examples/autodiff_smoke_mpc.py` differentiates the graph program. Tests must
compare graph VJP against finite differences at multiple dynamic shapes and
grids.

## Runtime graph execution

The graph planner must complete before the first dispatch:

- resolve dynamic TensorView shapes and layouts;
- validate aliases and access modes;
- calculate all intermediate byte sizes with checked arithmetic;
- allocate required intermediate owners transactionally;
- establish Storage liveness and safe reuse;
- plan forward checkpoints and pullback lifetimes;
- build provider resource hazards and dispatch dependencies.

Failure before or during execution must not publish partial external outputs.
Observable writes use the existing transaction/commit model or an equivalent
graph-scoped transaction.

Backend synchronization maps the same graph edge to native mechanisms:

- CPU: complete the producer dispatch before starting the consumer;
- CUDA: ordered launches and events in the selected stream;
- Vulkan: compute dispatch dependencies and pipeline/resource barriers;
- Metal: command-encoder ordering and resource barriers/fences;
- DirectX 12: transitions and UAV barriers;
- OpenGL/OpenGL ES: dispatch ordering and the required memory-barrier bits.

The provider adapter remains responsible for native barrier details.

## CPU cooperative execution retained requirements

Multi-kernel global semantics do not remove true workgroup-local cooperation.
The CPU implementation must retain:

- a validated maximum workgroup volume;
- persistent reusable lane workers, never one OS thread per lane per dispatch;
- bounded parallel cooperative teams;
- generation-based workgroup barriers with static site IDs;
- all-lanes-started gating;
- first-error preservation and peer cancellation;
- deterministic destruction after jobs release their contexts;
- diagnostics containing workgroup ID, local ID, barrier site, and original
  cause when available.

Workgroup allocation must retain:

- the portable 16 KiB primal workgroup arena;
- separate lane-private and workgroup-shared adjoint storage;
- checked size/alignment arithmetic;
- per-entry allocation-site namespaces;
- complete preflight before lane execution;
- no poison pointer, alias fallback, or post-preflight backing-store growth.

Storage-gradient staging remains O(workgroups). Lane tapes and lane-private
state may remain O(invocations).

## CPU helper and AOT boundary

Until a versioned release explicitly changes it:

- keep the existing `VernonCpuRuntimeHelpersV1` registration boundary;
- executable JIT obtains helpers from its owning compiler context;
- artifact-only compilation does not require helper addresses;
- do not use process-global `dlsym` fallback for required helpers;
- `.o/.obj` artifacts are linked by the host and registered explicitly;
- `VernonRuntime` does not own LLVM, LLJIT, or a relocatable-object linker.

Do not extend the V1 helper struct layout. Add a separate V2 registration only
if a future released capability cannot be expressed internally.

## Optimization model

The unfused graph is the correctness reference.

Allowed optimizations include:

- liveness-based intermediate buffer reuse;
- safe fusion of adjacent pointwise phases;
- Taichi-style block-local caching of proven TensorView regions;
- halo prefetch into native workgroup storage;
- checkpoint/recompute selection;
- command-buffer batching.

Fusion must prove that removing a dispatch boundary does not remove a required
global happens-before edge. Block-local caching must preserve the original
global TensorView semantics. Optimization failure falls back to the correct
multi-dispatch graph.

## Contract migration

### Frozen-contract phase

With compiler contract 11 and pipeline 15:

- restore the smoke-fluid source from experimental partial rewrites;
- implement the kernel memory validator;
- remove mutable-device workgroup-coupling heuristics;
- fix remaining true workgroup-storage cooperative AD defects;
- build internal graph IR, graph VJP, storage planning, and direct-path tests;
- add no required reflection field and change no V1 helper ABI.

### Release phase

At the next deliberate contract release:

- serialize graph nodes, structured loops, Storage version edges, dynamic grid
  bindings, and checkpoint policy;
- use one schema for direct, cooked, and AOT execution;
- update compiler and pipeline versions together where required;
- remove old schema readers instead of adding compatibility translation.

## Verification

### Kernel validation

- reject device write/barrier/cross-lane-read communication;
- accept read-only stencils and proven-exclusive writes;
- validate atomic, scatter, and reduction collision semantics;
- cover 1D, 2D, and 3D dynamic shapes, masks, subviews, strides, and offsets.

### Graph execution

- dynamic runtime grids and non-divisible logical extents;
- dependency and hazard correctness on every available backend;
- structured pressure loops and ping-pong Storage versions;
- allocation failure before externally visible writes;
- buffer reuse only after the last reader;
- direct, cooked, and AOT parity.

### Graph autodiff

- multiple nonzero cotangent carriers;
- non-leader lane contributions;
- gather/scatter/reduction transpose rules;
- overwrite and alias cases;
- reusable and concurrent pullback application;
- scaled cotangents;
- finite-difference comparison for smoke-fluid at multiple shapes and grids.

### True workgroup-local autodiff

- multiple lanes and multiple independent workgroups;
- repeated barrier generations;
- shared-storage isolation;
- divergent barriers, early completion, lane error, and exception;
- dynamic shared-adjoint preflight;
- deterministic shutdown and cancellation.

Run focused tests first, then full CTest and Python suites, sanitizer-capable
tests, and available Metal/Vulkan/CUDA/DirectX/OpenGL parity suites. Do not
widen numerical tolerances solely to make a parallel implementation pass.

## Implementation order

1. Restore the pre-experiment smoke-fluid mathematics and keep its giant
   single-kernel form as a failing validation fixture.
2. Add `KernelMemoryPhaseAnalysis` and remove device workgroup-coupling
   inference.
3. Define internal graph nodes and TensorView Storage-version edges.
4. Split smoke-fluid into graph phases and establish forward parity.
5. Implement graph Storage allocation, hazards, loops, and transactions.
6. Implement graph VJP and explicit reduction/scatter transpose rules.
7. Finish true workgroup-storage cooperative AD and allocation hardening.
8. Add backend barrier mappings and optimizer-off parity.
9. Add buffer reuse, block-local caching, and proven-safe fusion.
10. Publish the graph schema only at the planned contract release.

## Acceptance criteria

The architecture is complete only when:

- public device-language Storage remains `TensorView`;
- grid remains runtime data and kernels are grid-generic;
- no portable kernel relies on cross-workgroup barriers;
- mutable global phase communication is represented by graph edges;
- smoke-fluid uses multiple kernels and matches finite differences;
- graph VJP is numerically consistent across available backends;
- true workgroup-local AD preserves every lane contribution;
- allocation and execution failure cannot publish partial output;
- optimization on/off produces the same defined result;
- current contract versions remain unchanged until the release phase.

## Next-session handoff

### Current workspace state

The workspace is not a Git repository and cannot be restored with `git reset`.
The following files contain unfinished experiments from the abandoned
single-kernel mutable-device approach:

- `examples/autodiff_smoke_fluid_kernels.py`
  - six external mutable scratch TensorView parameters were added;
  - workgroup/local builtins and tile-neighbor expressions were added;
  - the original bilinear advection and loss reduction were partially replaced;
  - this file is not an authoritative implementation and must be reconstructed
    as separate kernels.
- `examples/autodiff_smoke_mpc.py`
  - six scratch TensorStorage owners and bindings were added only to satisfy the
    experimental giant-kernel signature;
  - replace them with graph-owned intermediate Storage.
- `python/tests/test_kernel_runtime.py` and
  `source/tests/python/autodiff_dynamic_smoke_fluid_runtime_test.py`
  - temporary scratch bindings and positional argument changes follow the
    experimental signature;
  - replace them with graph forward/VJP tests.
- `source/lib/Dialect/Vernon/Transforms/VernonAutodiffAnalysis.cpp`
  - contains prototype mutable-device ownership matching, structured-barrier
    ancestry changes, signed-remainder matching, and temporary diagnostics;
  - remove the mutable-device workgroup-coupling path;
  - retain storage activity/version analysis, external-gradient
    classification, and true workgroup-address-space ownership.

Do not blindly revert the CPU scheduler, helper registration, allocation
preflight, invocation-local pullback state, or true workgroup-storage adjoint
changes. Audit them against this specification; those changes address
independent valid requirements.

The intended smoke-fluid mathematics remains:

1. force density and velocity from controls;
2. bilinearly advect velocity;
3. compute central-difference divergence;
4. run ping-pong Jacobi pressure iterations;
5. project velocity;
6. bilinearly transport density and apply the Laplacian term;
7. reduce density error and control regularization into loss.

Preserve this mathematics while changing only the phase structure.

### Multi-Kernel Forward first milestone

This milestone is forward-only. It does not claim graph autodiff, graph Storage
SSA, structured graph `For`, checkpointing, or recomputation. Kernel-local
structured VJP remains the only implemented autodiff boundary until the
forward graph is stable.

All backends execute one backend-independent `ExecutionGraph` description and
one compiled dependency schedule. CPU maps graph edges to synchronous
producer-completes-before-consumer dispatch. GPU providers map the same edges
to native command ordering and resource barriers. Examples must not maintain
separate CPU and GPU phase lists.

The smoke graph owns ordinary `TensorStorage` intermediates and exposes them
to dispatches as ordinary `TensorView` bindings. Jacobi iterations are expanded
as runtime ping-pong dispatch nodes for this milestone. The loss phase is a
separate deterministic kernel with `workgroup_size=(1,1,1)` and runtime
`grid=(1,1,1)`; its single invocation serially traverses the dynamic views.
Parallel graph reduction is deferred.

Internal kernel-adjoint ownership is independent from external gradient
publication:

- read-only Storage and Storage requiring no internal adjoint use `None`;
- true workgroup-address-space Storage requiring an internal adjoint uses
  `WorkgroupCoupled`;
- device Storage requiring an internal adjoint uses `LanePrivate` only when
  lane ownership is proven injective;
- an unproved device mapping fails closed instead of allocating one complete
  dynamic adjoint buffer per lane;
- `externalGradientDestination` remains a separate classification.

The first injectivity proof accepts direct global-ID tuples and explicit
injective affine maps with unique axes. Remainder, division, truncation,
nonlinear terms, repeated axes, and unknown expressions are not proof.
Barrier ancestry belongs to `KernelMemoryPhaseAnalysis`, which treats the
whole kernel as one device epoch. The production giant kernel is removed;
minimal fixtures test that device write, workgroup barrier, then cross-lane
read is rejected.

The next session should not attempt the complete graph VJP at once. The first
milestone is complete when:

1. experimental giant-kernel edits are removed or replaced;
2. `KernelMemoryPhaseAnalysis` rejects device
   write→workgroup-barrier→cross-lane-read communication;
3. smoke-fluid forward execution is split into the seven mathematical phases;
4. the phases execute through the existing execution-graph/provider command
   path with runtime grid and ordinary TensorViews;
5. CPU and every available GPU backend produce matching forward outputs;
6. compiler contract 11 and pipeline 15 remain unchanged;
7. focused tests, full CTest, and full Python tests pass.

Only after this baseline is stable should implementation proceed to graph
Storage SSA, structured graph loops, graph VJP, checkpointing, buffer reuse,
block-local caching, and fusion.

### Required starting checks

Before editing:

- read this specification completely;
- inspect the current smoke-fluid and ownership-analysis files listed above;
- inspect existing `VernonExecutionGraph` and provider hazard tracking before
  creating new graph infrastructure;
- preserve RuntimeCore/provider ownership boundaries from
  `specs/runtime/design.md`;
- use one `*build/` directory at a time;
- invoke formatting tools through `.venv`.
