# Language feature test inventory

Status: Phase 0 coverage inventory for the active cross-backend language
testing plan.

This inventory assigns stable test IDs to the supported regions of
[`language/contract.md`](../language/contract.md),
[`language/tensor_view.md`](../language/tensor_view.md), the implemented
foundations distinguished from future gates in
[`language/future_language_roadmap.md`](../language/future_language_roadmap.md),
and the closed shader API registries in
[`python/vernon_dsl/shader_contracts.py`](../../python/vernon_dsl/shader_contracts.py).
Deferred roadmap items are boundaries of a supported region, not supported
features.

## 1. Record format

IDs are permanent once used by a test. Splitting a record creates new IDs and
retires the old ID; IDs are never reassigned to different semantics.

Required test layers are:

- `F`: frontend parsing, type/effect analysis, and diagnostic;
- `I`: typed Vernon IR and MLIR verification;
- `C`: every applicable target compiler;
- `A`: cooked Program artifact, reflection, and deterministic identity;
- `R`: execution with a numeric, publication, ordering, or effect oracle.

Capability requirements use the Phase 1 matrix vocabulary:

- compile: `compute`, `graphics`, `storage_buffers`,
  `device_storage_atomics`, `f32_atomic_add`, `f64_atomic_add`,
  `texture_sampler`, `storage_texture`, `workgroup_memory`, and `program_vjp`;
- runtime: the matching capability plus an available device/context; OpenGL
  compute additionally requires 4.3+, and OpenGL ES compute requires 3.1+.

`none` means that the layer is backend-independent. `N/A` means that runtime
execution is not part of the language region. A capability rejection is a
compile failure with the stated stable reason, never a runtime skip.

## 2. Semantic categories and Values

### `LANG-SEM-001` — closed semantic categories

- Valid: every language type is exactly one of Value, Storage, or Resource;
  loads produce Values and stores consume Values.
- Invalid: using Storage or Resource as a Value, Tensor element, copied handle,
  or ordinary arithmetic operand.
- Capabilities: compile `none`; runtime `N/A`.
- Layers: `F`, `I`.
- Diagnostic/oracle: identify the offending category and required Value
  context; typed nodes retain exactly one category.

### `LANG-SCALAR-001` — scalar type set and aliases

- Valid: `bool`, `i32`, `u32`, `f16`, `f32`, `f64`; annotation/cast aliases
  `int -> i32` and `float -> f32`.
- Invalid: unknown scalar names and use of `bool` in numeric arithmetic.
- Capabilities: compile `none`; runtime `compute`.
- Layers: `F`, `I`, `C`, `A`, `R`.
- Diagnostic/oracle: unknown type or numeric-operand reason; round-trip values,
  dtypes, and canonical ABI widths.

### `LANG-SCALAR-002` — contextual literals and conversions

- Valid: contextual integer/floating literals, defaults `i32`/`f32`, safe
  integer-to-floating and `f16 -> f32 -> f64`, and explicit casts for all
  narrowing or signedness changes.
- Invalid: implicit floating narrowing, floating-to-integer conversion, and
  dynamic `i32`/`u32` mixing.
- Capabilities: compile `none`; runtime `compute`.
- Layers: `F`, `I`, `C`, `R`.
- Diagnostic/oracle: incompatible conversion reason; exact result type and
  representative boundary values.

### `LANG-SCALAR-003` — arithmetic, comparisons, and true division

- Valid: typed unary/binary arithmetic, one comparison, and `/` producing at
  least `f32` for integer operands.
- Invalid: incompatible operand types and chained comparisons.
- Capabilities: compile `none`; runtime `compute`.
- Layers: `F`, `I`, `C`, `R`.
- Diagnostic/oracle: operand or chained-comparison reason; numeric comparison
  with a host reference.

### `LANG-TENSOR-001` — immutable Tensor type

- Valid: positive static shapes, including rank zero, with recursively
  ABI-stable Value elements.
- Invalid: dynamic Tensor Value extents, non-positive extents, Storage,
  Resource, or arbitrary Python-object elements.
- Capabilities: compile `none`; runtime `compute`.
- Layers: `F`, `I`, `C`, `A`, `R`.
- Diagnostic/oracle: static-shape or ABI-stable-Value reason; reflected logical
  element type, rank, and shape.

### `LANG-TENSOR-002` — nested Tensor normalization

- Valid: recursively concatenate outer and inner Tensor shapes while retaining
  Struct boundaries.
- Invalid: treating nested and canonical shapes as different types or
  flattening through a nominal Struct.
- Capabilities: compile `none`; runtime `compute`.
- Layers: `F`, `I`, `C`, `A`, `R`.
- Diagnostic/oracle: one canonical type/cache identity and equal numeric
  indexing behavior.

### `LANG-TENSOR-003` — Tensor, Vector, and Matrix constructors

- Valid: rectangular common-element `Tensor` literals; rank-one `Vector`
  literals with scalar/rank-one concatenation; rank-two `Matrix` literals.
- Invalid: empty or ragged literals, incompatible elements, and constructor
  rank mismatches.
- Capabilities: compile `none`; runtime `compute`.
- Layers: `F`, `I`, `C`, `R`.
- Diagnostic/oracle: rectangularity, common-type, or rank reason; exact shape,
  element order, and numeric contents.

### `LANG-TENSOR-004` — Vector and Matrix rank aliases

- Valid: `Vector[T, N] == Tensor[T, (N,)]` and
  `Matrix[T, R, C] == Tensor[T, (R, C)]`.
- Invalid: assigning an alias to a different rank or extent.
- Capabilities: compile `none`; runtime `N/A`.
- Layers: `F`, `I`.
- Diagnostic/oracle: canonical Tensor type equality or shape-mismatch reason.

### `LANG-TENSOR-005` — Tensor indexing, shape, and swizzles

- Valid: one integer index per Tensor dimension; `.shape` on rank-one-or-greater
  Tensor/TensorView as `Tensor[u32, (rank,)]`; rank-one Tensor swizzles over
  `xyzw`/`rgba`, with color masks canonicalized to positional masks.
- Invalid: wrong index count/type, `.shape` on rank zero or another category,
  non-rank-one swizzle, unknown component, or out-of-bounds component.
- Capabilities: compile `none`; runtime `compute`.
- Layers: `F`, `I`, `C`, `R`.
- Diagnostic/oracle: index/shape/swizzle rank or bounds reason; extracted
  values, dynamic extents, and masks match the host reference.

### `LANG-TUPLE-001` — structural Tuple Values

- Valid: non-empty heterogeneous construction, constant indexing,
  destructuring, and structural type equality.
- Invalid: dynamic or out-of-bounds indexing, incompatible destructuring, and
  non-Value elements.
- Capabilities: compile `none`; runtime `compute`.
- Layers: `F`, `I`, `C`, `A`, `R`.
- Diagnostic/oracle: literal-index/bounds/value-element reason; field order and
  values match the host reference.

### `LANG-STRUCT-001` — nominal immutable Struct Values

- Valid: `@struct` declarations, declared-field construction/access, methods
  normalized to helpers, and recursively ABI-stable Value fields.
- Invalid: recursive or Storage/Resource fields, mutation, wrong constructor
  arity/type, and interchange of distinct nominal declarations.
- Capabilities: compile `none`; runtime `compute`.
- Layers: `F`, `I`, `C`, `A`, `R`.
- Diagnostic/oracle: nominal/field/ABI-stability reason; reflected field order,
  offsets, identity, and values.

### `LANG-ABI-001` — portable Value ABI

- Valid: specified scalar size/alignment, row-major Tensor repetition,
  declaration-order products, padding, empty-product layout, and explicit
  boundary conversions from backend carriers.
- Invalid: backend-dependent public layout or omitted/misaligned aggregate
  fields.
- Capabilities: compile `none`; runtime `compute`.
- Layers: `I`, `C`, `A`, `R`.
- Diagnostic/oracle: reflection equals canonical size, alignment, offsets, and
  element stride; byte round trips preserve every field.

## 3. Storage, views, and interop

### `LANG-STORAGE-001` — TensorStorage ownership boundary

- Valid: host-owned dense typed allocation and canonical full-view creation.
- Invalid: TensorStorage in device-language annotations or Value operations.
- Capabilities: compile `none`; runtime `compute`.
- Layers: `F`, `A`, `R`.
- Diagnostic/oracle: device annotation identifies a host-only owner; runtime
  preserves shape, layout, ownership, and synchronization state.

### `LANG-VIEW-001` — TensorView source type

- Valid: ABI-stable Value elements; `read`, `write`, or `read_write`; positive
  static extents, `vd.dyn`, and rank-zero `()`.
- Invalid: unknown access, non-positive static extents, callable `vd.dyn`,
  dynamic rank, or non-Value elements.
- Capabilities: compile `storage_buffers`; runtime `storage_buffers`.
- Layers: `F`, `I`, `C`, `A`, `R`.
- Diagnostic/oracle: access/shape/element reason; reflected constraints and
  bound rank/shape match.

### `LANG-VIEW-002` — inferred address space

- Valid: entry parameters infer device storage and `workgroup_storage` infers
  workgroup storage.
- Invalid: source-authored address-space strings or treating uniform/attribute
  transport as Storage.
- Capabilities: compile `storage_buffers` or `workgroup_memory`; runtime
  matching capability.
- Layers: `F`, `I`, `C`, `A`.
- Diagnostic/oracle: forged-address-space reason; IR/reflection carries the
  origin-derived address space.

### `LANG-VIEW-003` — strict direct binding

- Valid: Tensor Values snapshot compatible immutable host/NumPy values;
  TensorView accepts TensorStorage, its full view, or an explicit subview.
- Invalid: Tensor accepting Storage/View or TensorView implicitly allocating
  from a raw NumPy array.
- Capabilities: compile `storage_buffers`; runtime `storage_buffers`.
- Layers: `F`, `A`, `R`.
- Diagnostic/oracle: category-specific bind error; asynchronous execution
  retains owners and observes the bound descriptor.

### `LANG-VIEW-004` — invocation-time layout descriptor

- Valid: runtime shape, signed element strides, and element offset vary between
  invocations without recompilation; static extents are checked.
- Invalid: baking a concrete dynamic shape/layout into capture, artifact,
  manifest, reflection, or cache identity.
- Capabilities: compile `storage_buffers`; runtime `storage_buffers`.
- Layers: `I`, `C`, `A`, `R`.
- Diagnostic/oracle: one artifact identity serves contiguous, non-contiguous,
  offset, and negative-stride bindings with correct projected results.

### `LANG-VIEW-005` — bounds, injectivity, alias, and lifetime

- Valid: in-bounds views, internally injective writable layouts, compatible
  readers, and proven-disjoint projections of a shared owner.
- Invalid: out-of-bounds mapping, non-injective writable views, unproven
  read/write overlap, incompatible concurrent borrows, or expired owners.
- Capabilities: compile `storage_buffers`; runtime `storage_buffers`.
- Layers: `F`, `A`, `R`.
- Diagnostic/oracle: stable bounds/injectivity/alias/lifetime reason before
  mutation; accepted disjoint projections update only their regions.

### `LANG-VIEW-006` — ranked load and store

- Valid: exactly one integer index per rank, including `view[()]` for rank
  zero; load requires readable access and store requires writable access.
- Invalid: wrong index count/type or access-mode violation.
- Capabilities: compile `storage_buffers`; runtime `storage_buffers`.
- Layers: `F`, `I`, `C`, `R`.
- Diagnostic/oracle: rank/index/access reason; scalar and aggregate loads and
  stores match physical projection.

### `LANG-VIEW-007` — aggregate elements and AoS projections

- Valid: Tensor/Tuple/Struct elements, canonical leaf expansion, and
  internally injective Struct-field projections sharing one owner.
- Invalid: observable backend leaf expansion or projection with wrong record
  stride/field offset.
- Capabilities: compile `storage_buffers`; runtime `storage_buffers`.
- Layers: `F`, `I`, `C`, `A`, `R`.
- Diagnostic/oracle: reflected leaf bindings reconstruct canonical aggregate
  values and interleaved fields exactly.

### `LANG-INTEROP-001` — RawBuffer boundary

- Valid: runtime-only `vd.interop.RawBuffer` with explicit external byte ABI
  and typed-view validation.
- Invalid: RawBuffer in kernel, shader, or shared-function annotations, or
  implicit typed shape/alias guarantees.
- Capabilities: compile `none`; runtime `storage_buffers`.
- Layers: `F`, `A`, `R`.
- Diagnostic/oracle: frontend identifies RawBuffer as non-language interop;
  malformed alignment/layout fails before execution.

## 4. Workgroup storage and effects

### `LANG-WORKGROUP-001` — static workgroup TensorView allocation

- Valid: `workgroup_storage(T, shape=...)` in compute, including rank zero;
  ABI-stable elements, compile-time extents, row-major layout, and at most
  16 KiB total canonical footprint per kernel.
- Invalid: use outside compute, runtime-dependent/non-positive dimensions,
  explicit stride/offset, dynamic shared memory, or footprint over 16 KiB.
- Capabilities: compile `compute`, `workgroup_memory`; runtime `compute`,
  `workgroup_memory`.
- Layers: `F`, `I`, `C`, `A`, `R`.
- Diagnostic/oracle: compute-stage/dimension/portable-limit reason; independent
  allocation per workgroup and correct aggregate/non-zero-index values.

### `LANG-EFFECT-001` — typed Storage and Resource effects

- Valid: read, write, atomic, barrier, query, and sample effects retain owner,
  region, ordering, and scope and propagate through reachable helpers.
- Invalid: effectful shared helpers, missing propagated effects, or treating
  host runtime operations as device expressions.
- Capabilities: compile according to the represented operation; runtime `N/A`.
- Layers: `F`, `I`.
- Diagnostic/oracle: typed effect model is complete, deterministic, and rejects
  forbidden host/device-pure operations.

### `LANG-ATOMIC-001` — integer TensorView atomics

- Valid: relaxed `atomic_add`, `atomic_min`, `atomic_max`, and
  `atomic_exchange` on writable `i32`/`u32` TensorViews; rank one uses one
  scalar index and higher rank uses an exact-length integer tuple.
- Invalid: read-only/non-View owner, unnamed owner expression, wrong element or
  index/value type, and `atomic_add(view[index], value)`.
- Capabilities: compile `device_storage_atomics` for device scope or
  `workgroup_memory` for workgroup scope; runtime matching capability.
- Layers: `F`, `I`, `C`, `A`, `R`.
- Diagnostic/oracle: writable-owner/element/index reason; returned old values
  and final contended min/max/add/exchange results are legal serializations.

### `LANG-ATOMIC-002` — floating atomic add

- Valid: relaxed `atomic_add` on writable `f32` and `f64` TensorViews when the
  selected scope/target profile has a legal native or compare-exchange
  implementation.
- Invalid: floating min/max/exchange or target profile without the required
  floating atomic-add implementation.
- Capabilities: compile `f32_atomic_add` or `f64_atomic_add` for the selected
  type and scope; runtime matching capability.
- Layers: `F`, `I`, `C`, `A`, `R`.
- Diagnostic/oracle: floating-atomic capability reason; NaN, signed zero, and
  contended sums follow the declared atomic-add semantics.

### `LANG-BARRIER-001` — workgroup and storage barriers

- Valid: no-argument typed barriers in compute; acquire-release ordering and
  explicit workgroup/device scope.
- Invalid: arguments, non-compute use, or a target without a legal requested
  scope.
- Capabilities: compile/runtime `workgroup_memory` for workgroup scope and a
  target-specific device-barrier capability for storage scope.
- Layers: `F`, `I`, `C`, `A`, `R`.
- Diagnostic/oracle: argument/stage/scope capability reason; writes become
  visible in the ordering region.

### `LANG-DISPATCH-001` — ordinary-write dispatch contract

- Valid: launch geometry satisfying compiler-derived unit-grid and
  unit-workgroup residuals.
- Invalid: non-injective ordinary writes under the requested launch; runtime
  serialization as a fallback.
- Capabilities: compile `compute`; runtime `compute`.
- Layers: `F`, `I`, `A`, `R`.
- Diagnostic/oracle: dispatch-contract reason occurs before allocation,
  staging, mutation, or submission; legal launches publish exact results.

## 5. Entries, interfaces, and Resources

### `LANG-ENTRY-001` — registered entry stages

- Valid: externally visible `@kernel`, `@vertex`, and `@fragment`; graphics
  topology is exactly `vertex -> fragment`.
- Invalid: unknown entry decorators, CUDA graphics entries, or invalid/missing
  graphics topology.
- Capabilities: compile/runtime `compute` for kernel and `graphics` for
  vertex/fragment.
- Layers: `F`, `I`, `C`, `A`, `R`.
- Diagnostic/oracle: stage/topology/target reason; compute dispatch or graphics
  draw produces the expected output.

### `LANG-ENTRY-002` — entry signatures and calls

- Valid: annotated parameters and non-None results; required positional device
  calls.
- Invalid: unresolved entry types, defaults, variadics, keyword device calls,
  or wrong call arity/type.
- Capabilities: compile according to stage; runtime `N/A`.
- Layers: `F`, `I`.
- Diagnostic/oracle: annotation/positional/arity/type reason.

### `LANG-HELPER-001` — specialized non-recursive helpers

- Valid: reachable `@func` specializations with partial annotations resolved by
  call sites; unreachable helpers are pruned.
- Invalid: recursion or a reachable unresolved/incompatible specialization.
- Capabilities: compile according to reachable entry; runtime matching stage.
- Layers: `F`, `I`, `C`, `R`.
- Diagnostic/oracle: recursion/specialization reason; deterministic symbols
  and helper result parity.

### `LANG-HELPER-002` — host/device-pure shared helpers

- Valid: `@func(shared=True)` over host/device-common Values and pure
  operations.
- Invalid: Storage, atomic, barrier, generated interface, Texture, or Sampler
  operations.
- Capabilities: compile `none`; runtime `N/A`.
- Layers: `F`, `I`.
- Diagnostic/oracle: identify the device-only type or operation.

### `LANG-SPECIALIZE-001` — deterministic source specialization

- Valid: source imports without execution, deterministic captured constants,
  `vd.feature`/`When`, and declared ProgramAsset variants.
- Invalid: arbitrary Python capture/execution, undeclared feature combination,
  or runtime feature conditions.
- Capabilities: compile according to entry; runtime matching stage.
- Layers: `F`, `I`, `C`, `A`, `R`.
- Diagnostic/oracle: capture/variant reason; dependencies, constants, features,
  interfaces, workgroup size, and transform identity affect deterministic
  cache identity.

### `LANG-RESOURCE-001` — Texture and Sampler categories

- Valid: opaque Texture and Sampler Resources with texture-specific operations
  and separately bound sampler state.
- Invalid: TensorView-style indexing/stores, Value arithmetic, or
  differentiation of handles/state.
- Capabilities: compile/runtime `texture_sampler` or `storage_texture`.
- Layers: `F`, `I`, `C`, `A`, `R`.
- Diagnostic/oracle: Resource-operation/category reason; reflected bindings and
  sampled/loaded values match.

### `LANG-RESOURCE-002` — Texture type regions

- Valid: sampled `Texture[2d|3d|cube, f32|i32|u32]`; storage
  `Texture[2d|3d, format, read|write|read_write]` with `r8_unorm`,
  `r16_float`, `r32_float`, `rg8_unorm`, `rgba8_unorm`, `rgba16_float`, or
  `rgba32_float`.
- Invalid: another dimension/sample type/format/access, cube storage Texture,
  or malformed type arity.
- Capabilities: compile/runtime `texture_sampler` for sampled and
  `storage_texture` for storage.
- Layers: `F`, `I`, `C`, `A`, `R`.
- Diagnostic/oracle: dimension/sample/format/access reason; reflection exactly
  preserves the accepted Resource type.

### `LANG-INTERFACE-001` — graphics interface metadata

- Valid: `attribute()` with inferred/explicit non-negative location and
  non-negative divisor, `uniform()` with optional set/binding, `varying()`,
  `resource(set, binding)`, and `builtin(name)` in their legal stage/direction
  regions.
- Invalid: unknown or duplicate/conflicting metadata, wrong literal arity/type,
  negative location/divisor, binding collision, or metadata on an incompatible
  semantic category/stage.
- Capabilities: compile/runtime `graphics` and Resource capability where used.
- Layers: `F`, `I`, `C`, `A`, `R`.
- Diagnostic/oracle: metadata/location/binding/stage reason; deterministic
  locations, sets, bindings, divisors, varying links, and reflected interfaces.

The low-level builtin registry gives each legal stage/direction/type region a
stable ID:

- `LANG-BUILTIN-POSITION`: vertex output, `Tensor[f32, (4,)]`;
- `LANG-BUILTIN-VERTEX-INDEX`: vertex input, `u32`;
- `LANG-BUILTIN-INSTANCE-INDEX`: vertex input, `u32`;
- `LANG-BUILTIN-FRAG-COORD`: fragment input, `Tensor[f32, (4,)]`;
- `LANG-BUILTIN-FRONT-FACING`: fragment input, `bool`;
- `LANG-BUILTIN-GLOBAL-ID`: compute input, `Tensor[u32, (3,)]`;
- `LANG-BUILTIN-LOCAL-ID`: compute input, `Tensor[u32, (3,)]`;
- `LANG-BUILTIN-WORKGROUP-ID`: compute input, `Tensor[u32, (3,)]`.

For every builtin ID:

- Valid: exactly the listed stage, direction, and type through `builtin(...)`.
- Invalid: unknown name or any other stage, direction, or type.
- Capabilities: compile/runtime capability of the listed stage.
- Layers: `F`, `I`, `C`, `A`, `R`.
- Diagnostic/oracle: stable unknown-builtin or allowed-use/type reason;
  reflection and invocation values match the backend builtin.

Generated interface APIs also have one ID per closed region:

- `LANG-GENERATED-RESOLUTION`: fragment `resolution()`, uniform
  `Tensor[f32, (2,)]`;
- `LANG-GENERATED-FRAGMENT-COORD`: fragment `fragment_coord()`, frag-coord
  input `Tensor[f32, (4,)]`;
- `LANG-GENERATED-FRONT-FACING`: fragment `front_facing()`, front-facing input
  `bool`;
- `LANG-GENERATED-VERTEX-ID`: vertex `vertex_id()`, vertex-index input `u32`;
- `LANG-GENERATED-INSTANCE-ID`: vertex `instance_id()`, instance-index input
  `u32`.

For every generated-interface ID:

- Valid: no arguments in exactly the listed stage.
- Invalid: arguments or use in any other stage/helper context.
- Capabilities: compile/runtime `graphics`.
- Layers: `F`, `I`, `C`, `A`, `R`.
- Diagnostic/oracle: no-argument or stage-specific reason; generated interface
  and runtime value match the corresponding builtin/uniform.

### `LANG-TEXTURE-SAMPLE-IMPLICIT` — implicit sampler, implicit LOD

- Valid: `texture_sample(texture, coordinates)` only in fragment.
- Invalid: vertex/compute use, non-sampled Texture, or wrong coordinate rank or
  floating element type.
- Capabilities: compile/runtime `graphics`, `texture_sampler`.
- Layers: `F`, `I`, `C`, `A`, `R`.
- Diagnostic/oracle: fragment-only/coordinate reason; filtered texel oracle.

### `LANG-TEXTURE-SAMPLE-EXPLICIT` — explicit sampler, implicit LOD

- Valid: `texture_sample(texture, sampler, coordinates)` only in fragment.
- Invalid: vertex/compute use, wrong Resource kinds, or mixing implicit and
  explicit sampler modes for one entry Texture parameter.
- Capabilities: compile/runtime `graphics`, `texture_sampler`.
- Layers: `F`, `I`, `C`, `A`, `R`.
- Diagnostic/oracle: fragment-only/sampler-mode reason; sampler state affects
  the filtered texel as expected.

### `LANG-TEXTURE-SAMPLE-LOD` — implicit sampler, explicit LOD

- Valid: `texture_sample(texture, coordinates, lod)` in vertex or fragment.
- Invalid: compute use, malformed coordinates, or non-floating LOD.
- Capabilities: compile/runtime `graphics`, `texture_sampler`.
- Layers: `F`, `I`, `C`, `A`, `R`.
- Diagnostic/oracle: graphics-stage/type reason; selected mip value.

### `LANG-TEXTURE-SAMPLE-EXPLICIT-LOD` — explicit sampler and LOD

- Valid: `texture_sample(texture, sampler, coordinates, lod)` in compute,
  vertex, or fragment.
- Invalid: malformed Resource, coordinate, or LOD arguments.
- Capabilities: compile/runtime stage capability plus `texture_sampler`.
- Layers: `F`, `I`, `C`, `A`, `R`.
- Diagnostic/oracle: argument/type reason; selected filtered mip value.

### `LANG-TEXTURE-SIZE` — texture dimensions

- Valid: texture plus optional LOD in graphics stages, producing rank-two
  `u32` for 2D/cube and rank-three `u32` for 3D.
- Invalid: non-Texture, wrong arity/LOD, or unsupported stage.
- Capabilities: compile/runtime `graphics`, `texture_sampler`.
- Layers: `F`, `I`, `C`, `A`, `R`.
- Diagnostic/oracle: texture/LOD/stage reason; exact dimensions.

### `LANG-TEXTURE-STORAGE` — storage texture load/store

- Valid: integer coordinates of the texture rank; load on read/read_write and
  store of a four-component texel on write/read_write.
- Invalid: sampled Texture, wrong coordinate/value type, or access violation.
- Capabilities: compile/runtime stage capability plus `storage_texture`.
- Layers: `F`, `I`, `C`, `A`, `R`.
- Diagnostic/oracle: storage/access/coordinate/value reason; exact loaded or
  stored texel.

## 6. Statements, expressions, and math

### `LANG-CONTROL-001` — core statements

- Valid: `pass`, expression, local/indexed/annotated/augmented assignment,
  `return`, `if`, `range` loop, `while`, `break`, and `continue`.
- Invalid: exceptions, generators, Python object mutation, and unsupported
  statement forms.
- Capabilities: compile `none`; runtime `compute`.
- Layers: `F`, `I`, `C`, `R`.
- Diagnostic/oracle: unsupported-statement reason; structured execution agrees
  with a host reference.

### `LANG-CONTROL-002` — short circuit and conditional expressions

- Valid: typed `and`/`or` short circuit and type-compatible conditional
  expressions.
- Invalid: incompatible branch/result types or evaluation of the skipped side.
- Capabilities: compile `none`; runtime `compute`.
- Layers: `F`, `I`, `C`, `R`.
- Diagnostic/oracle: branch-type reason; side-effect and result oracle proves
  short circuiting.

### `LANG-CONTROL-003` — range semantics

- Valid: literal or dynamic `i32` start/stop/step, positive/negative exclusive
  stop, one-time argument evaluation, and fixed-width overflow termination.
- Invalid: implicit `u32` bounds, literal zero step, or dynamic zero step on a
  target without a legal contract-violation mechanism.
- Capabilities: compile `compute`; runtime `compute`.
- Layers: `F`, `I`, `C`, `A`, `R`.
- Diagnostic/oracle: zero-step/cast/capability reason; iteration sequence
  equals Python range without overflow wrap.

### `LANG-CONTROL-004` — structured exits

- Valid: nested/early return and nearest-loop break/continue, including loop
  `else`.
- Invalid: break/continue outside a loop or incompatible return types.
- Capabilities: compile `none`; runtime `compute`.
- Layers: `F`, `I`, `C`, `R`.
- Diagnostic/oracle: enclosing-loop/return-type reason; path results match a
  host reference.

### `LANG-MATH-001` — portable scalar/Tensor math

- Valid: `sin`, `cos`, `acos`, `atan2`, `exp`, `log`, `sqrt`, `floor`, `abs`,
  `min`, `max`, `pow`, `clamp`, `dot`, `cross`, `norm`, `normalize`, and
  `reflect` on their declared numeric scalar/Tensor regions.
- Invalid: wrong arity, rank, shape, or non-numeric element type.
- Capabilities: compile `none`; runtime `compute`.
- Layers: `F`, `I`, `C`, `R`.
- Diagnostic/oracle: intrinsic-specific type/shape reason; tolerance-based host
  numeric comparison including conventional `atan2(y, x)`.

### `LANG-MATMUL-001` — matrix multiplication

- Valid: compatible numeric vector/matrix/batched Tensor core dimensions.
- Invalid: non-Tensor operands, incompatible core/batch dimensions, or
  incompatible element types.
- Capabilities: compile `none`; runtime `compute`.
- Layers: `F`, `I`, `C`, `R`.
- Diagnostic/oracle: Tensor/dimension/element reason; host matrix-product
  comparison.

## 7. Program boundary and first-order autodiff

### `LANG-PROGRAM-001` — one canonical Program boundary

- Valid: Kernel, graphics pipeline, initialized Module, and supported explicit
  transform as ProgramAsset operands; standalone entries normalize to one
  node.
- Invalid: direct asset execution paths, host pass descriptors, or undeclared
  executable kinds.
- Capabilities: compile/runtime capability of contained stages.
- Layers: `F`, `I`, `C`, `A`, `R`.
- Diagnostic/oracle: unsupported-program reason; load, resolve, instantiate,
  begin, bind, and invoke lifecycle produces the direct semantic result.

### `LANG-AD-001` — public VJP surface and cotangent seeding

- Valid:
  `vd.ad.vjp(program, wrt=..., outputs=..., planning_policy=...)`; explicit
  output cotangents, with omitted seed `1` only for one floating Scalar output;
  reusable pullbacks.
- Invalid: omitted aggregate/Tensor cotangent, invalid `wrt`, implicit backward
  execution, custom derivative-rule arguments, or signature mutation.
- Capabilities: compile/runtime `compute`, `program_vjp`.
- Layers: `F`, `I`, `C`, `A`, `R`.
- Diagnostic/oracle: seed/wrt reason; primal and finite-difference gradient
  agreement across repeated pullback application.

### `LANG-AD-002` — differentiable Value and Storage leaves

- Valid: recursive floating leaves of Scalar/Tensor/Tuple/Struct; fresh owned
  gradient Storage for differentiated TensorView/mutable Storage with dynamic
  extents retained.
- Invalid: requested gradients for integer, Boolean, Resource, sampler, or
  opaque leaves without a consuming custom rule; aliasing primal storage as
  gradient storage.
- Capabilities: compile/runtime `compute`, `program_vjp`, and storage/atomic
  capabilities required by the selected plan.
- Layers: `F`, `I`, `C`, `A`, `R`.
- Diagnostic/oracle: non-differentiable-path reason; tangent layout, ownership,
  and finite-difference values match.

### `LANG-AD-003` — built-in VJP rules and accepted control flow

- Valid: arithmetic, casts, Tensor construction, matmul, supported math, pure
  Value early returns, literal-bounded loops, and one dynamic leading
  `if condition: break` guard.
- Invalid: general dynamic range, while, continue, or loop-local return in
  differentiated code.
- Capabilities: compile/runtime `compute`, `program_vjp`.
- Layers: `F`, `I`, `C`, `A`, `R`.
- Diagnostic/oracle: unsupported-AD-control-flow reason; finite differences
  cover each built-in rule and accepted branch/loop region.

### `LANG-AD-004` — unsupported transform boundary

- Valid: first-order compute Program VJP only.
- Invalid: graphics VJP, public custom compute rules until specified, JVP,
  full Jacobian, convenience grad aliases, nested transforms, Hessian, or HVP.
- Capabilities: compile `none`; runtime `N/A`.
- Layers: `F`, `A`.
- Diagnostic/oracle: deterministic unsupported-capability reason before
  execution or Program mutation.

## 8. Removed and deferred boundaries

### `LANG-LEGACY-001` — removed language spellings

- Valid replacement: TensorStorage/TensorView/RawBuffer, rank-one Tensor,
  Vector/Matrix aliases, and `workgroup_storage`.
- Invalid: public `Buffer`, `Array`, `vec`, `mat`, `vec*`, `mat*`,
  `workgroup_array`, old workgroup IR, compatibility normalization, or legacy
  addressability metadata.
- Capabilities: compile `none`; runtime `N/A`.
- Layers: `F`, `I`, source guards.
- Diagnostic/oracle: unknown/removed-name reason identifies the canonical
  replacement; no legacy symbol or schema appears in generated output.

The following are explicit non-features and therefore negative boundaries of
the records above: dynamic-shape Tensor Values, private-storage constructors,
dynamic shared memory, byte-address device operations, external-memory import,
transparent sparse layouts, recursion, arbitrary Python containers/classes,
exceptions, generators, unrestricted Python control flow, future graphics
stages, graphics derivatives, custom compute VJP declarations, JVP and
higher-order AD, constrained generics, enums, tagged unions, `Option`, and
`match`.

## 9. Required pairwise interactions

Individual IDs provide exhaustive construct-region coverage. The matrix must
also retain these stable pairwise case IDs:

- `LANG-PAIR-001`: nested Tensor element normalization × Struct boundary;
- `LANG-PAIR-002`: aggregate TensorView × dynamic shape/layout descriptor;
- `LANG-PAIR-003`: shared-owner field projections × writable alias proof;
- `LANG-PAIR-004`: helper specialization × captured constant/feature identity;
- `LANG-PAIR-005`: helper effects × concrete entry stage;
- `LANG-PAIR-006`: workgroup aggregate storage × barrier × non-zero indexing;
- `LANG-PAIR-007`: device/workgroup atomic scope × contention;
- `LANG-PAIR-008`: texture sampling overload × stage × sampler mode;
- `LANG-PAIR-009`: structured control flow × Tensor/Struct values;
- `LANG-PAIR-010`: VJP × accepted early-return/bounded-loop control flow;
- `LANG-PAIR-011`: VJP × dynamic/signed-stride TensorView;
- `LANG-PAIR-012`: Module fan-in/fan-out × Value and Storage versions;
- `LANG-PAIR-013`: backend carrier packing × portable aggregate ABI;
- `LANG-PAIR-014`: dispatch injectivity × dynamic launch geometry.

Each pairwise case inherits the union of its component layers and
capabilities. Its diagnostic or runtime oracle must demonstrate the
interaction, not merely execute both constructs independently.
