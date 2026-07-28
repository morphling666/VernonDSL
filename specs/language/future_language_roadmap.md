# Language v4 implementation roadmap

This roadmap tracks the remaining gates for the normative target in
`contract.md`. Checked phases are implemented while the current compiler and
frontend continue to report language version 3. A phase may not silently
broaden Python semantics or publish frontend version 4 before all contract
acceptance gates pass.

## Design position

Vernon is a statically typed GPU and graphics DSL, not unrestricted Python.
The v4 semantic categories are Value, Storage, and Resource. `Tensor` is an
immutable logical Value; `TensorStorage` owns dense memory; `TensorView`
borrows shaped or strided memory; runtime-only `vd.interop.RawBuffer` is
low-level host-byte interop; `Texture` and `Sampler` are Resources.

Core v4 remains dense. Sparse formats are explicit library structures composed
from dense storage. `Array` is not introduced: `Tensor[T, (N,)]` is the fixed
homogeneous sequence Value. Arbitrary Python objects are never Tensor elements.
First-order autodiff operates on specialized typed IR and does not require an
object or dual-number dtype.

## Phase 1: Value aggregates and canonical types

- [x] Add recursively ABI-stable Tensor elements: Scalar, Tensor, Tuple, and
      Struct Values; reject Storage and Resource elements.
- [x] Canonicalize nested Tensor Values by logical shape composition while
      retaining nominal Struct boundaries.
- [x] Make `Tensor([...])` the canonical constructor and Vector/Matrix rank
      aliases; remove the v3 `vec`, `mat`, `vec*`, and `mat*` spellings.
- [x] Add heterogeneous structural Tuple construction, constant indexing, and
      destructuring.
- [x] Use rank-1 Tensor everywhere a fixed `Array[T, N]` would have appeared.
- [x] Define deterministic Tensor/Tuple/Struct ABI layout for CPU, CUDA,
      Vulkan, OpenGL, and host/device shared functions.

## Phase 2: Storage, layouts, and ownership

- [x] Introduce host-owning dense `TensorStorage[T]`.
- [x] Introduce borrowed `TensorView[T, rank, access]` with runtime shape,
      signed element strides, offset, owner, access mode, and lifetime.
- [x] Define byte-stride/offset reflection after leaf and Struct layout is
      resolved; reject ambiguous external layout units.
- [x] Implement TensorView load/store projection for Scalar and recursively
      ABI-stable aggregate Value elements through runtime-specialized shape,
      signed element strides, and offset for direct compute dispatch.
- [x] Implement deterministic AoS `TensorStorage[Struct]` and Struct-field
      TensorView projections, including interleaved vertex attributes.
- [x] Validate bounds, internal injectivity, shared-owner aliasing, writable
      overlap, asynchronous dispatch borrows, and owner lifetime.
- [x] Remove public `Buffer`; provide host-byte typed-view interop as the
      runtime-only `vd.interop.RawBuffer`, outside the frontend parser/model.

## Audited implementation order after Phase 2

The original numeric order mixed semantic prerequisites, user ergonomics,
pure-function autodiff, and multi-dispatch runtime work. Continue in the
dependency order below. Program-internal graph work does not block pure
functions or local structured control flow.

## Phase 3 audit: existing execution boundaries

- [x] Enforce the host/device parsing boundary: source modules are parsed
      without executing Python; TensorStorage and RawBuffer remain host-only;
      device intrinsics and device-only types are rejected from shared host
      functions.
- [x] Keep Texture and Sampler Resource operations separate from TensorView
      load/store across types, intrinsics, lowering, reflection, and runtime
      binding.
- [x] Bind each cooked pipeline variant directly to either one compute program
      or one graphics program and provide invocation planning for that program.

## Phase 3A: owner/region effect foundation

- [x] Replace statement-only PURE/READ/WRITE classification with structured
      `read(owner, region)` and `write(owner, region)` effects in typed semantic
      nodes; retain coarse summaries only as derived diagnostics.
- [x] Propagate effects through specialized helper calls and validate them
      against stage, parameter access, alias, and purity contracts.
- [x] Reserve atomic and barrier effect records with explicit ordering and
      scope while rejecting unsupported language operations. Actual GPU
      atomics, workgroup storage, and barriers remain Phase 6.
- [x] Reflect effect summaries deterministically where entry/runtime validation
      needs them; do not encode host allocation or transfer as device effects.

## Phase 5A: expression-level structured control flow

- [x] Add lazy short-circuit `and` and `or` with value-yielding structured
      control flow.
- [x] Add conditional expressions with branch type unification and explicit
      effect merging.

## Phase 4A: complete first-order pure autodiff

- [ ] Add typed transform APIs for `jvp`, `vjp`, `grad`, and
      `value_and_grad`, plus `stop_gradient`.
- [ ] Transform only specialized, validated pure non-recursive `@func` IR;
      include derivative transform and rule versions in deterministic cache
      identity.
- [ ] Derive tangent/adjoint Values recursively for floating Scalar, Tensor,
      Tuple, and Struct leaves.
- [ ] Reject integer, Boolean, Storage, Resource, sampler, and opaque
      differentiation unless a custom operation rule explicitly handles it.
- [ ] Implement forward JVP and reverse VJP for the accepted pure `@func`
      subset, including accepted Phase 5A expressions.
- [ ] Add versioned custom JVP/VJP registration and validate all primal,
      tangent, and adjoint signatures.
- [ ] Specify derivatives and non-differentiable points for arithmetic, casts,
      Tensor construction, `matmul`, and supported math intrinsics.
- [ ] Allocate `TensorStorage.grad` as separate companion storage; never embed
      autodiff state into Tensor element types.
- [ ] Explicitly diagnose nested transforms, Hessians, and HVPs as unsupported
      in v4.
- [ ] Compare analytical derivatives with finite differences on CPU and with
      CUDA/Vulkan where supported.

## Phase 5B: statement and loop control flow

- [x] Add `break` and `continue` with explicit loop-carried termination.
- [x] Support early returns with explicit region/CFG termination.
- [x] Define dynamic range bounds and steps, including backend legality and
      termination diagnostics.

These ergonomics are accepted as part of the stable v4 core. Autodiff support
for Phase 5B requires separate derivative and tape policies.

## Low priority: program-internal autodiff graph

The graph used by autodiff is compiler IR inside one specialized program. It is
not a host graph of PipelineAssets, dispatches, render passes, or backend
transitions.

- [ ] Build a typed `ProgramGraph` from one specialized, validated Kernel or
      graphics Pipeline program.
- [ ] Represent value flow, control flow, Storage effects, alias constraints,
      and differentiability boundaries without changing source semantics.
- [ ] Use graph transforms for JVP/VJP, mutation functionalization, reverse
      traversal, tape planning, and checkpointing.
- [ ] Lower transformed graphs through the existing target pipelines; do not
      serialize ProgramGraph as a deployment or orchestration asset.

This work is post-v4 and lower priority than the accepted language,
PipelineAsset target coverage, and first-order pure-function autodiff.
Runtime host orchestration is unrelated and is specified independently in
`specs/runtime/design.md`.

A public graph-level execution model may become useful for neural-network
workloads that compose many kernels, parameters, and differentiable operators.
It is not part of the current roadmap. `ProgramGraph` remains private compiler
IR and must not implicitly become that public execution API.

## Phase 6: GPU memory and synchronization

- [ ] Model private, function-local, workgroup/shared, storage, uniform, and
      host-visible address spaces.
- [ ] Add workgroup storage with backend-validated alignment and layout.
- [ ] Add atomic operations for explicitly supported leaf types.
- [ ] Add barriers and shared CUDA/SPIR-V/CPU ordering and scope semantics.
- [ ] Extend effect analysis to atomics, barriers, races, and
      differentiability-relevant reads and overwrites.

## Phase 7: stateful kernel autodiff

- [ ] Functionalize local mutation and TensorView writes before reverse-mode
      transformation; reject unresolved aliasing and races.
- [ ] Define gather/scatter adjoints, atomic accumulation, and deterministic
      reduction alternatives.
- [ ] Differentiate accepted branches and loops with specified tape layout,
      bounds, checkpointing, and recomputation.
- [ ] Generate explicit primal and companion-gradient storage bindings without
      changing primal ABI identity.
- [ ] Reverse the program-internal graph while preserving validated Storage
      effects and alias constraints.
- [ ] Bound tape memory for multi-step fluid workloads and test gradients
      across checkpoint intervals.
- [ ] Define texture-sampling custom gradients separately for coordinates and
      texels; sampler state remains non-differentiable.
- [ ] Keep rasterization, visibility, depth/blend decisions, and discontinuous
      material branches non-differentiable until custom primitives are chosen.
- [ ] Validate stencil-fluid and fixed-visibility PBR derivatives against
      finite differences without claiming general differentiable rendering.

This phase remains post-v4 research unless the language contract is amended.

## Phase 8: explicit generics

- [ ] Add `Numeric`, `Integer`, `Float`, and recursively `Differentiable`
      constraints.
- [ ] Add const generics for dimensions and capacities.
- [ ] Give explicit and inferred specializations one deterministic key.
- [ ] Add type aliases and compile-time assertions.
- [ ] Retain call-site inference as convenience rather than the only
      polymorphism mechanism.

## Phase 9: algebraic and compile-time data

- [ ] Add enums with deterministic underlying representation.
- [ ] Add tagged unions and `Option[T]`.
- [ ] Add exhaustive `match`.
- [ ] Define copy, equality, and ABI rules for nested product/algebraic Values.
- [ ] Add deterministic constant-expression evaluation and compile-time
      conditionals.
- [ ] Add fixed compile-time lookup tables and `ConstMap[K, V, N]`.

## Non-core library work

- [ ] Implement fixed-capacity vector/list containers over TensorStorage.
- [ ] Implement sparse and hash structures over TensorStorage/RawBuffer only
      after atomic and memory semantics are stable.
- [ ] Keep logical length, capacity, overflow, collision, allocation, and
      concurrency policies explicit in each library type.

## Acceptance criteria for every phase

- The language contract is updated before implementation.
- Typed semantic nodes fully represent the feature before lowering changes.
- CPU reference behavior has numeric and diagnostic tests.
- CUDA and Vulkan behavior is compared where supported.
- Unsupported target combinations fail explicitly.
- Cache identity, reflection, and generated symbols remain deterministic.
- Public terminology maps one-to-one to Value, Storage, or Resource semantics.
- Autodiff work compares analytical derivatives with finite differences and
  documents non-differentiable operations and memory bounds.
