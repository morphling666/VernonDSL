# VernonDSL completion roadmap

## Current position

VernonDSL 0.1.1a1 is a Windows-first alpha developer preview. The supported
near-term product range is:

- CPU compute reference execution;
- CUDA compute;
- Vulkan compute and offscreen graphics;
- OpenGL/OpenGL ES compute and graphics with a supplied context;
- DirectX 12 compute and offscreen graphics on Windows;
- experimental Metal compute and offscreen graphics Runtime in macOS source
  builds and CI, consuming cooked MSL bundles on Apple but not offered as a
  stable wheel or GA capability.

The released frontend remains version 3. Language v4 is the normative target,
not a released compatibility claim.

## Priority 1: compiler architecture debt

1. Implement the general runtime TensorView descriptor ABI and storage
   lowering contract defined in
   [`stable_release_plan.md`](stable_release_plan.md). Runtime shape, signed
   strides, and offset must reach one precompiled artifact through the ABI
   without layout specialization. Replace pass-state `physical_index`
   attributes with internal physical load/store/atomic operations, and replace
   the ToGPU unrealized-cast/manual clone bridge with MLIR one-to-many
   conversion. No backend-local projection or layout-specialization fallback
   may remain.
2. Remove cross-language Value ABI algorithm duplication. One declarative
   source or native planner must define sizes, alignment, offsets, leaves, and
   hashes consumed by Python and host packing.
3. Add executable aggregate workgroup coverage on available GPU runtimes,
   including nested values, non-zero indices, padding, branches, loops,
   independent workgroups, and barrier-visible writes.
4. Minimize derivable interface ABI metadata in one deliberate
   compiler-contract bump. Retired fields must be rejected rather than
   accepted through compatibility parsing.

Acceptance requires one production path per invariant, deterministic artifacts
and diagnostics, complete Python/native Release suites, relevant available GPU
execution, and no compatibility fallback for retired internal forms.

## Priority 2: post-0.1.1 language-v4 acceptance

The detailed gates live in
[`language/future_language_roadmap.md`](language/future_language_roadmap.md).
Autodiff is not a `0.1.1` feature or stable-release gate. The following work
applies to the later frontend-v4 declaration.
Before declaring frontend version 4:

- complete first-order JVP/VJP for specialized pure `@func` code and compare
  derivatives with finite differences;
- prove workgroup storage, barriers, and relaxed atomics on every advertised
  runtime where hardware is available;
- close dynamic TensorView layout and synchronization guarantees required by
  the language contract;
- retain deterministic cache identity and explicit diagnostics for unsupported
  operations;
- decide whether remaining autodiff gates are required for the v4 declaration.

Stateful kernel differentiation, nested transforms, Hessians, and general
differentiable rendering remain outside the initial v4 target unless the
language contract is amended.

## Priority 3: GPU lifetime and throughput

Phase 1 stable logical resource records are implemented. Continue with
[`deferred_gpu_resource_lifetime_plan.md`](deferred_gpu_resource_lifetime_plan.md)
only when asynchronous submission or multiple frames in flight become product
requirements:

- track per-device submission and completion serials;
- reclaim resources only after logical and execution lifetimes end;
- remove unconditional queue waits at explicit synchronization boundaries;
- preserve external ownership contracts for borrowed resources;
- measure pending bytes, queue wait, reclamation latency, and CPU encode time.

Performance work must follow measurements. Current targets are:

- warm frontend compilation below 5% of cold frontend time;
- cooking cost proportional to unique specialized stages, not
  `variants * stages`;
- reused pipelines and unchanged bindings allocate nothing on warm dispatch;
- transfers, queue waits, execution, and readback reported separately.

## Priority 4: production engineering

Required before beta:

1. Add Linux CI and verify compiler, Runtime, tests, install, and source builds
   outside Windows.
2. Define the supported public API and deprecation policy.
3. Publish compatibility guarantees for release, compiler-contract, pipeline,
   cache, and native ABI axes.
4. Add scheduled or self-hosted GPU jobs with validation/debug layers.
5. Add fuzzing for source, manifest, reflection, TensorView, and
   ExecutionGraph inputs.
6. Add benchmark reporting and stable regression thresholds.
7. Verify every published wheel in a fresh environment through a real compile,
   dispatch, readback, CLI, and bundled-source check.

Optional product tracks require independent justification and acceptance:

- promotion of the experimental Metal Runtime beyond macOS source builds and
  CI into a stable wheel or GA capability;
- AMD/ROCDL compiler and Runtime support;
- explicit constrained and const generics;
- enums, `Option[T]`, exhaustive `match`, and compile-time data structures.

## Stable / GA gates

Stable release requires:

- satisfaction of [`stable_release_plan.md`](stable_release_plan.md);
- repeatable release CI on every supported platform;
- production-ready installation and binary distribution;
- proven resource lifetime and synchronization across supported runtimes;
- published ABI, cache, security, support, and release policies;
- removal or explicit de-scoping of major deferred compiler/runtime work;
- release automation that tags and publishes only an exact green commit.

## Change policy

For architecture changes:

1. document the invariant and add characterization tests;
2. introduce the smallest stable boundary;
3. migrate every enabled backend or frontend path;
4. delete the superseded implementation in the same change;
5. measure correctness and performance;
6. update the applicable canonical specification.

Git history is the record of completed milestones. This roadmap contains only
active future work.
