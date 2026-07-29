# Language v4 implementation roadmap

This roadmap tracks the remaining gates for the normative target in
`contract.md`. The Python frontend reports language version 4
(`FRONTEND_VERSION`). Checked phases below are implemented in source and
covered by frontend or runtime tests where noted. Unchecked gates, compile-only
coverage, and open correctness findings in `specs/compiler/root_cause_audit.md`
are not treated as complete end-to-end acceptance.

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

## Showcase porting findings (2026-07-29)

Porting the Aurora and Mandelbulb fragment shaders exposed the following
language and backend gaps. These are not Python-style requests; each item is a
failure inside syntax that the frontend currently accepts.

Resolved correctness gaps:

- [x] Lower floating-point casts of `range` induction values through a legal
      intermediate integer cast.
- [x] Allow explicitly sampled Texture reads in ordinary `@func` helpers,
      propagate their Resource effects, and validate them at the entry stage.
- [x] Resolve nominal Struct layouts from the source module when SPIR-V
      structured control flow converts carried values.
- [x] Record and type-check unary Boolean `not` operands in statement control
      flow.

Resolved during the port:

- [x] Add public `acos`, `atan2`, and `floor` intrinsics with frontend type
      inference and MLIR Math lowering.
- [x] Teach the custom SPIR-V structured-control-flow translator to lower
      `math.acos` and `math.floor`, and lower `math.atan2` through a
      quadrant-aware `acos` construction. Vulkan, DirectX, and OpenGL artifact
      tests cover the resulting shader math.

Resolved runtime acceptance gaps:

- [x] Keep the generated OpenGL `resolution()` uniform name consistent between
      SPIR-V/GLSL emission and PipelineAsset reflection. Aurora and Mandelbulb
      now use `resolution()` without an explicit `viewport_size` uniform.
- [x] Preserve logical `i32`/`u32` signedness on SPIR-V interface variables so
      generated GLSL uniform declarations match the Provider ABI's
      `glUniform*i` upload path. Mandelbulb now uses native `i32` loop budgets.

A generated OpenGL PipelineAsset acceptance test covers `resolution()` and a
signed integer uniform through compilation, reflection, loading, and provider
upload. The generated resolution value now works in Vulkan, DirectX 12, and
OpenGL showcases.

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

Source syntax, typed IR, and compile-time effect metadata for device/workgroup
address spaces, `workgroup_storage`, atomics, and compute barriers are
implemented and covered by frontend tests. End-to-end runtime correctness for
aggregate workgroup storage, the runtime TensorView layout ABI, and cross-backend
parity remain open; see `specs/compiler/root_cause_audit.md`.

- [x] Model device and workgroup address spaces on unified TensorView Storage.
- [ ] Model private, function-local, storage, uniform, and host-visible address
      spaces beyond the current TensorView surface.
- [x] Add workgroup storage source syntax, IR allocation, and compile-time size
      validation (`workgroup_storage`).
- [ ] Validate workgroup physical allocation, nested control flow, and backend
      limits at runtime on every supported GPU backend.
- [x] Add atomic operations for explicitly supported leaf types in source and IR.
- [ ] Prove atomic semantics end-to-end on every available GPU runtime.
- [x] Add barriers with typed effect records in compute kernels.
- [ ] Prove barrier and shared-memory ordering semantics end-to-end on every
      available GPU runtime.
- [x] Extend effect analysis to atomics and barriers at compile time.
- [ ] Extend runtime validation for atomics, barriers, races, and
      differentiability-relevant reads and overwrites.

### Phase 6 unified TensorView Storage

The accepted source, IR, binding, ABI, and migration contract is maintained in
[`tensor_view.md`](tensor_view.md). This section records the motivation and
roadmap boundary; the dedicated contract is authoritative where details
overlap.

`workgroup_storage(element, shape=(...))` replaces the removed
`workgroup_array(T, N)` spelling. Workgroup memory is modeled as TensorView
Storage with `address_space = workgroup` rather than a separate workgroup type:

```text
TensorView[
    element = i32,
    shape = (8, 8),
    strides = (8, 1),
    offset = 0,
    address_space = workgroup,
    access = read_write,
]
```

This is Storage, not a `Tensor` Value. Shape and strides describe indexing;
`address_space` determines ownership, visibility, lifetime, legal operations,
and synchronization scope. In particular:

- `private` storage belongs to one invocation;
- `workgroup` storage is shared only by invocations in one workgroup and each
  workgroup receives an independent allocation;
- `device` storage is visible through an externally owned allocation such as a
  kernel `TensorView`;
- uniform and host-visible spaces require separate mutability and interface
  rules rather than aliases for device storage.

Address space is part of the compiler's concrete semantic Storage type, but is
not written in an ordinary source annotation. It is inferred from origin:

- a kernel `TensorView` parameter is device storage;
- `workgroup_storage` produces workgroup storage;
- future private-storage constructors produce invocation-private storage.

The source API constructs a legal storage class rather than accepting an
arbitrary address-space string:

```python
input: vd.TensorView[vd.f32, (vd.dyn, vd.dyn), vd.read]
output: vd.TensorView[vd.f32, (vd.dyn, vd.dyn), vd.write]
shared = vd.workgroup_storage(vd.i32, shape=(8, 8))
shared[y, x] = value
value = shared[y, x]
previous = vd.atomic_add(shared, (y, x), 1)
```

`vd.dyn` is an immutable type-level marker for one runtime-resolved dimension,
so annotation rank remains explicit. Static and dynamic dimensions may be
mixed. Shape is part of the source Storage contract; strides and offset are
layout metadata rather than core type identity.

At an external entry boundary, a host `TensorStorage` owner is accepted as its
canonical full `TensorView`; an explicitly constructed subview is accepted
without losing its layout. For example, one row-major `TensorStorage` with
shape `(4, 4)` may provide two disjoint `(2, 2)` inputs:

```python
upper_left = storage.view(
    shape=(2, 2),
    strides=(4, 1),
    offset=0,
    access="read",
)
lower_right = storage.view(
    shape=(2, 2),
    strides=(4, 1),
    offset=10,
    access="read",
)
kernel(upper_left, lower_right)
```

The concrete descriptors have the same owner and device address space but
different offsets. Their logical indices are projected through
`offset + y * 4 + x`. Region analysis therefore proves these two views
disjoint even though they borrow one owner. Passing the owner itself instead
means `shape=(4, 4)`, canonical `strides=(4, 1)`, and `offset=0`.

Dynamic dimensions are valid for externally owned device `TensorView`
parameters. Every `workgroup_storage` extent must be a positive compile-time
integer after specialization. Literal constants and captured host constants
may determine those extents, with captured constants participating in cache
identity and causing recompilation when changed. Kernel arguments and other
device-runtime values may not determine workgroup allocation size. The core
language does not expose CUDA-only dynamic shared memory.

Owned `workgroup_storage` uses contiguous row-major layout. Its strides are
derived canonically from shape and are not exposed as constructor arguments.
Strided workgroup projections are deferred until a concrete algorithm
requires them. The multi-dimensional atomic index is lowered through the same
checked linearization as an ordinary load or store:

```text
linear_index = offset + sum(index[d] * stride[d])
```

Atomic scope is inferred from the storage address space and is not an
independent source argument:

```text
workgroup storage -> workgroup scope
device storage    -> device scope
```

An explicit scope that contradicts the address space must fail verification.
Barrier scope remains explicit because a barrier is not tied to one storage
operand. Atomic ordering remains explicit in the semantic effect even when the
initial public operation supports only `relaxed`.

The design must not introduce a fourth semantic category. A rank-one
workgroup allocation and a rank-two workgroup allocation are both
`TensorView`; `workgroup_array` must not remain as a parallel semantic model.
Whether it is removed directly or replaced by the canonical constructor is a
language-version migration decision, not a backend compatibility path.

The accepted source-level direction is therefore:

- `Tensor` is an immutable Value;
- `TensorView` is the only non-owning shaped Storage type;
- `TensorStorage` remains the host owner and is not a device-language type;
- address space is inferred and cannot be forged in ordinary source;
- `workgroup_storage` replaces `workgroup_array` directly, without a
  compatibility alias;
- shape uses static integer extents and `vd.dyn` dimensions, while
  strides and offset remain layout metadata;
- workgroup allocation shape is fully static after specialization, while
  device `TensorView` parameters may use dynamic dimensions;
- owned workgroup layout is contiguous row-major and does not expose stride or
  offset controls.

The remaining design choices are resolved as follows:

1. **Atomic indexing:** rank one accepts a scalar index; higher ranks require a
   tuple with exactly one index per dimension.
2. **Atomic element types:** the initial set remains i32/u32. Wider integers
   and floating-point atomics require explicit capability contracts and are not
   emulated implicitly.
3. **IR representation:** Vernon IR uses one address-space-parameterized
   Storage type with unified ranked load, store, and atomic operations.
4. **Workgroup elements:** every recursively ABI-stable Value element is legal
   and lowers through canonical ABI leaves.
5. **Portable allocation:** workgroup storage is fully static after
   specialization and initially limited to 16 KiB after ABI layout.

Acceptance requires frontend inference, effect ownership, MLIR verification,
CPU reference behavior, backend lowering, reflection where externally
visible, and multi-workgroup runtime tests for every accepted shape/index
form. Tests must demonstrate independent allocations between workgroups and
barrier-visible writes within one workgroup.

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
