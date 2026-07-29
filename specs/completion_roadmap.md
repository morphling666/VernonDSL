# VernonDSL completion and refactoring roadmap

## Status and scope

This document prioritizes the work required to move VernonDSL from a
demonstration-capable multi-backend compiler to a production-ready graphics and
compute language. It covers language completion, compiler and Runtime
refactoring, backend consistency, performance, and engineering gates.

The near-term supported product matrix is CPU compute, Vulkan, OpenGL,
OpenGL ES, CUDA compute, and DirectX 12. Metal Runtime and AMD/ROCDL are
separate product decisions rather than blockers for this matrix.

Completion estimates are directional planning inputs, not coverage metrics:

- the current graphics/compute language subset is approximately 70–80% complete;
- the full language-v4 contract is approximately 55–65% complete;
- the desktop compiler/artifact/Runtime path is approximately 65–75% complete;
- the independent optimization layer is approximately 25% complete.

## Baseline completed

Priority 0 was completed on 2026-07-28:

- the complete Python suite passes with 308 tests and 297 subtests;
- a clean Windows Release build passes all 103 native and integration CTest
  entries;
- that baseline executes CPU, CUDA, Vulkan compute/graphics, and D3D12 WARP
  paths; OpenGL is covered through the external-provider contract tests;
- README, the language contract, and the language roadmap consistently state
  that checked phases are implemented under the current
  `COMPILER_CONTRACT_VERSION` while open
  correctness gates remain tracked in `specs/compiler/root_cause_audit.md`;
- the compiler/Runtime support matrix is recorded in README;
- stale completion-plan and archived-proposal references were removed;
- `scripts/benchmark_baseline.py` provides reproducible frontend, cooking,
  kernel/transfer, and multi-pass showcase scenarios documented in README.

Further implementation starts at Priority 1.

## Priority 1: architecture and refactoring foundation

Refactoring belongs before feature expansion when the feature would otherwise
add another parallel implementation. It is not a repository-wide rewrite.
Each refactor must retain behavior through tests, remove duplication, and
produce a boundary used immediately by the next feature or optimization.

### Compiler pipeline

Parse MLIR once per compile request. Validation, helper inlining, reflection,
and target lowering must consume a shared owned intermediate representation
instead of reparsing source and rerunning equivalent passes in each backend.

Introduce one explicit prepared-module boundary with:

- validated and deterministically inlined IR;
- reflection inputs and source dependency identity;
- target-independent semantic metadata;
- clone/ownership rules for target-specific mutation.

Acceptance:

- compilation performs one source parse and one common validation/inlining
  sequence;
- reflection and target artifacts remain byte deterministic;
- existing C API ownership and diagnostic behavior remain unchanged;
- medium shader cold compilation improves by at least 40%, or profiling
  explains why the target was not reached.

Progress (2026-07-28):

- `PreparedModule` now owns the validated, deterministically inlined common IR;
- CPU, CUDA, and SPIR-V target pipelines clone that IR and no longer reparse
  source or repeat common validation/inlining;
- repeated Vulkan compilation now checks byte-identical artifacts and
  reflection, and all 103 native and integration CTest entries pass;
- on the fixed Blinn-Phong Vulkan fixture, 15 independent cold-process samples
  improved from a 5.2142 ms median to 4.8276 ms (7.4%); 100 same-context
  samples improved from 1.2519 ms to 1.0785 ms (13.9%);
- differential profiling shows duplicate preparation was only 0.3866 ms of
  the old cold median. The remaining 92.6% is context/dialect initialization,
  canonical reflection, target lowering, and artifact serialization, so this
  refactor cannot reach 40% without optimizing those shared costs.

### Python frontend

Status: **In progress (2026-07-28 audit reopened).**

Split the large frontend implementation by semantic responsibility rather than
by syntax convenience:

- project/module loading and dependency identity;
- type inference and helper specialization;
- typed control-flow and effect analysis;
- Value, Storage, Resource, and interface lowering;
- MLIR text emission.

Typed semantic records must be the source of lowering decisions. Python AST
objects remain source syntax and location payload and must not become an
unversioned second semantic model.

Add a dependency-aware frontend cache keyed by source digests and semantic
inputs. Graphics stages from one project load must share module parsing and
dependency discovery.

Acceptance:

- warm compilation does not repeat project parsing, inference, or lowering
  when semantic inputs are unchanged;
- changing a transitive dependency invalidates every affected entry;
- successful cache hits preserve generated symbols and MLIR bytes; failed
  requests are not cached and reproduce diagnostics from current sources;
- frontend and language coverage gates do not regress.

Progress (2026-07-28):

- immutable frontend results are cached by complete
  `FrontendCompileRequest` semantic inputs;
- cache hits rehash all discovered source dependencies and transitive changes
  invalidate the affected specialization;
- graphics stages rooted at the same source now share parsed modules, resolved
  imports, and dependency discovery while retaining independent specialization
  and typed semantic records;
- runtime shape and captured-constant source specialization is isolated from
  semantic analysis and MLIR emission in `frontend.specialization`;
- module attributes, Struct ABI declarations, deterministic function assembly,
  and module-level syntax checks are isolated in `frontend.emission`;
- lowering-only Value/context records, TensorView physical storage lowering,
  and texture Resource lowering are separated into `lowering_types`,
  `storage_lowering`, and `resource_lowering` without introducing a second
  semantic model;
- literal and numeric operation lowering, aggregate construction, and
  expression-level structured control-flow emission are separated into
  `numeric_lowering`, `aggregate_lowering`, and `control_flow_lowering`;
- range/while serialization and break, continue, and nested-return state
  propagation are delegated to `loop_lowering` and `control_flow_lowering`
  using typed branch merges and loop-depth records;
- dead pre-extraction loop implementations are removed, active guarded-block
  state emission is centralized in `control_flow_lowering`, and compiler
  orchestration lives in `frontend.compiler`;
- runtime workgroup size now affects emitted compute-entry MLIR as well as cache
  identity, while raw-source and whole-module validation APIs are explicitly
  uncached;
- Resource reads have structured `ResourceEffect` records instead of the
  legacy expression-name fallback, and source digest/type helpers have one
  shared implementation;
- the fixed frontend benchmark improved from a 2.4447 ms warm median to
  0.1232 ms, or 4.3% of its 2.8895 ms cold compile.

### Runtime, RHI, and backend adapters

Status: **In progress — ABI-profile redesign is incomplete (2026-07-28).**

Continue enforcing the dependency direction:

```text
RuntimeCore -> provider SPI <- VernonRuntimeRHIAdapter -> VernonRHI
```

Split backend-neutral planning and lifecycle policy from backend command
encoding. Large translation units must be divided by resource records,
submission, pipeline preparation, and command encoding when those parts have
independent invariants.

Use one stable logical resource-record contract across enabled GPU backends.
Do not retain raw recyclable slot addresses or add backend-specific lifetime
fallbacks.

Acceptance:

- prepared bindings survive public owner destruction according to the documented
  logical-record lifetime;
- generation and slot-reuse tests cover buffers, images, and samplers;
- warm invocation performs no allocation solely because bindings are unchanged;
- RuntimeCore remains independent of VernonRHI and native graphics SDK types.

Progress (2026-07-28):

- `LogicalResourceRecord` is the single generation, public-owner, prepared-
  binding retention, and slot-reuse contract used by CUDA, Vulkan, DirectX 12,
  OpenGL, and OpenGL ES resources;
- the RHI entry layer now dispatches through a backend-neutral function table;
  OpenGL, CUDA, Vulkan, and DirectX 12 each own their device registry, resource
  records, command encoding, submission, barriers, and native interop;
- public owner destruction invalidates the public handle immediately while the
  logical record keeps the native object alive until its final prepared binding
  is released;
- backend-parameterized tests cover buffer reuse on CUDA, Vulkan, and DirectX
  12, and image/sampler reuse on the backends that expose them; external-context
  tests cover the same three resource kinds on OpenGL;
- unchanged Vulkan graphics invocation reuses binding snapshots, command
  buffers, descriptor pools, staging allocations, layouts, pipelines, and
  implicit samplers;
- RuntimeCore links only the provider SPI and JSON support; configure-time and
  C-header checks reject a VernonRHI dependency, while native SDK command
  encoding remains in the RHI adapter target.

### Canonical ABI and reflection

Status: **Complete (2026-07-28).**

Use one canonical planner for Tensor, Tuple, Struct, TensorView leaf expansion,
vertex attributes, and inline values. CPU, CUDA, SPIR-V, and Runtime packing
must consume that plan rather than reimplementing layout decisions.

Acceptance:

- the same source Value has one reflected logical ABI identity across targets;
- backend-specific physical mappings are explicit and tested;
- cross-backend numeric and byte-layout parity tests cover nested aggregates.

Completion verification (2026-07-28):

- Physical planning is keyed by semantic profiles:
  `HostValue`, `CudaKernelParameter`, `VulkanStd140UniformBuffer`,
  `VulkanStd430StorageBuffer`, `VulkanPushConstant`,
  `OpenGLNativeUniform`, `DirectXConstantBuffer`, and
  `MetalConstantBuffer`. Backend and transport jointly select the profile.
- Reflection publishes only `physical_layouts` keyed by profile and records
  each packed argument's `value_transport`. Manifests carry one selected
  `physical_value_layout` with explicit `profile` and `transport`.
- Packed compute values use the same path as graphics values. Cooked Vulkan
  and DirectX compute bundles now include the scalar `factor` layout and pass
  byte packing through the Runtime core provider.
- The mixed backend/profile route table and the obsolete generic graphics and
  uniform-only layout fields were removed. Compiler reflection, cooking,
  manifest validation, direct artifact loading, and Runtime dispatch consume
  the same schema without compatibility parsing.
- Typed TensorView resource plans, consolidated Struct field resolution,
  reflection-local caches, explicit std140/std430/push-constant planning, and
  non-square/nested aggregate checks are active.
- A clean Release build passed. The native suite completed 109 tests with 108
  passes and one unsupported CUDA image/sampler parameterization skipped; the Python
  suite passed 313 tests plus 308 subtests across CPU, CUDA, OpenGL, DirectX,
  Vulkan, Metal source cooking, non-square matrices, and nested aggregate
  parity.

## Priority 2: complete the core product path

Implement these items after, or together with, the relevant Priority-1
boundary. Do not wait for unrelated refactoring to finish.

1. Complete multidimensional `TensorView` lowering for runtime-specialized
   shapes, signed element strides, and offsets.
2. Complete reflected Resource/material `(set, binding)` planning, cooking,
   manifest serialization, Runtime validation, and backend binding.
3. Specify and implement CUDA static Tensor-by-value launch ABI or reject it at
   a target-independent capability boundary with a precise diagnostic.
4. Connect reflected resources to the OpenGL `ShaderProvider` path.
5. Add workgroup storage, barriers, and atomics with typed effects, address
   spaces, ordering, scope, and backend capability validation.
6. Extend graphics stage topology only through `COMPILER_CONTRACT_VERSION`
   and target capability checks.

Acceptance:

- representative programs have CPU, CUDA, and Vulkan parity where each target
  advertises support;
- a cooked material pipeline can be loaded and bound without external
  target-specific reflection logic;
- unsupported combinations fail before artifact or Runtime state mutation;
- race and synchronization semantics have positive and negative tests.

### Priority 2 implementation handoff (2026-07-29)

Do not weaken backend coverage or remove a backend from a parity test merely to
make the suite pass. In particular, restore CUDA in
`KernelTensorRuntimeTests._aggregate_compute_backends`; its temporary removal
was an invalid test workaround.

Priority 2 is now implemented:

- CUDA static aggregate Tensor-by-value dispatch uses a reflected physical
  kernel-parameter ABI with matching host packing; CUDA remains in aggregate
  backend parity coverage.
- CPU/CUDA storage uses no longer require descriptor bindings, ordinary
  non-negative JSON integers parse as sampled texture bindings, and stable
  builtin/topology diagnostics are restored.
- `atomic_*` accepts writable scalar i32/u32 `TensorView` owners with device
  scope and lowers to CPU atomics, CUDA global atomics, and SPIR-V
  `StorageBuffer` atomics. Runtime dirty tracking treats atomic-only kernels as
  writers. OpenGL and DirectX reject this capability before artifact emission.
- The unfinished Geometry Shader feature has been removed from the frontend,
  Vernon/MLIR and SPIR-V lowering, `PIPELINE_VERSION` program handling, provider/RHI
  stage bits, Vulkan/OpenGL runtime paths, tests, and product documentation.
  The compiler contract now exposes only `vertex -> fragment`.
- Mesh/Task shaders remain out of scope until a concrete GPU-driven rendering
  requirement exists.

Final validation completed all 122 CTest entries successfully; the unavailable
CUDA image/sampler lifetime and OpenGL synchronization cases were explicitly
skipped. The Python suite passed all 322 tests, including CUDA aggregate
dispatch and CPU/CUDA/Vulkan global atomic race coverage.

#### Post-implementation audit follow-up

The synchronization audit is complete. Runtime writability and dispatch borrows
now use the frontend's typed storage effects and declared `TensorView` access
contract, atomic scope classification fails on missing or unexpected owner
types, and the IR verifier rejects every atomic ordering except the only
currently implemented ordering, `relaxed`. The obsolete CUDA
`static_tensor_by_value` diagnostic has also been removed.

本轮同步收口：

1. `StorageOwner` 显式区分 parameter 与 workgroup-local；参数 owner 缺失、
   local owner 越界传播以及非 compute local effect 均 fail-closed。
2. `AtomicOp::verify()` 从 GPU address space 或 SPIR-V storage class 推导
   workgroup/device scope，并拒绝未知 memory space。
3. 多 workgroup 运行测试覆盖独立共享存储、barrier 可见性和多 lane atomic
   结果；后端可用时执行，不可用时明确 skip。
4. 语言契约明确允许后端在保持语义的前提下加强 atomic ordering。
5. atomic exchange 的 GPU→SPIR-V conversion 已移入独立 Vernon transform，
   compiler pipeline 只组合该 pass，并由 pass-level 测试覆盖。

## Priority 3: structural performance work

Optimize measured structural costs in this order:

1. Remove duplicate native MLIR parsing, validation, helper inlining, and
   reflection work.
2. Add the dependency-aware Python frontend cache.
3. Make cooking scale with unique specialized stages rather than
   `variants * stages`.
4. Combine SPIRV-Cross interface analysis and final source generation to avoid
   repeated compiler construction.
5. Replace per-stage DXC subprocess and temporary-file overhead when an
   in-process or safely batched implementation is available.
6. Replace Vulkan submit-and-immediate-wait with a fence ring and synchronize at
   explicit readback or compatibility boundaries.
7. Batch host/device transfers, reuse pinned staging, and avoid unconditional
   full-buffer copies.
8. Replace per-dispatch source-file hashing with a metadata fast path and
   content verification on invalidation.

Required benchmark scenarios:

- B1: one compute kernel, cold call versus the 100th unchanged call;
- B2: one vertex/fragment pipeline with frontend, native compiler,
  SPIRV-Cross, and load timings;
- B3: a four-variant PipelineAsset with frontend call and cache-hit counts;
- B4: an eight-pass Vulkan ExecutionGraph over 180 frames;
- B5: repeated 8M-element `f32` upload, kernel, and download;
- B6: four concurrent CPU compilations, including LLD lock time;
- B7: a DirectX two-stage pipeline, including DXC process and I/O counts.

Performance acceptance:

- warm frontend time is below 5% of cold frontend time;
- cooking complexity approaches the number of unique specialized stages;
- at least two owned GPU submissions may remain in flight;
- performance gates report compile, CPU encode, queue wait, transfer, and
  readback time separately.

## Priority 4: production engineering

1. Add Linux CI for standalone Runtime and compiler CTest.
2. Replace Windows-only validation/frontend integration harnesses with
   cross-platform CMake or Python runners.
3. Add scheduled or self-hosted Vulkan, CUDA, and DirectX hardware jobs with
   validation/debug layers enabled.
4. Run one compute and one graphics headless showcase as end-to-end smoke tests.
5. Add ASan CI for native ownership and resource-lifetime tests.
6. Add fuzz targets for source/manifest parsing, reflection normalization,
   TensorView validation, and ExecutionGraph mutation.
7. Add benchmark reporting and regression thresholds without placing unstable
   GPU wall-clock limits on ordinary pull-request CI.
8. Validate built wheels in fresh environments beyond import-only smoke tests.

Acceptance:

- a green CI result identifies which real backends executed and which skipped;
- Linux and Windows build, test, install, and package paths are continuously
  exercised;
- sanitizer, validation-layer, and fuzz regressions retain minimal reproducers.

## Priority 5: optional product expansion

These tracks must have independent product justification and acceptance gates:

- Metal Runtime and macOS execution;
- AMD/ROCDL compiler and Runtime support;
- first-order pure-function autodiff;
- explicit constrained and const generics;
- enums, `Option[T]`, exhaustive `match`, and compile-time data structures.

For autodiff, complete specialized pure `@func` JVP/VJP and finite-difference
validation before stateful kernel differentiation. Do not couple stateful
autodiff to initial v4 publication unless the language contract is explicitly
changed.

## Refactoring policy

Allocate approximately 25–35% of near-term work to the Priority-1 foundation,
then continue with feature-coupled refactoring:

1. identify the invariant and add characterization tests;
2. create the smallest stable boundary;
3. migrate all existing implementations to it;
4. delete the superseded path in the same change;
5. implement the feature or optimization that required the boundary;
6. measure behavior, compile time, and runtime performance.

Avoid both extremes:

- do not pause delivery for a multi-month whole-repository rewrite;
- do not add new targets or language features on top of known duplicated
  parsing, ABI, reflection, or resource-lifetime logic.

Refactoring is complete only when duplication is removed, dependencies are
one-directional, tests express the invariant, and the next change uses the new
boundary.
