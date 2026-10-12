# Vernon DSL language contract

> **Status: language-v4 normative target, partially implemented under the
> current `COMPILER_CONTRACT_VERSION` in `versions.toml`. Released builds remain
> frontend version 3 until every required v4 acceptance gate passes. There is
> no independent numeric `FRONTEND_VERSION` axis.**
>
> The checked phases in `future_language_roadmap.md` are implemented in source
> and tested where noted. Unchecked sections and the language-v4 work in
> the [project roadmap](../roadmap.md#language-v4) remain unavailable as end-to-end
> guarantees. Git history is the archive for superseded contract text.

Vernon is a statically typed GPU and graphics DSL embedded in Python syntax.
The frontend parses source without importing or executing the shader module.
Python spelling does not imply Python runtime semantics.

## 1. Semantic categories

Every language type belongs to exactly one category:

- **Value**: immutable data consumed and produced by computation;
- **Storage**: mutable addressable memory from which Values are loaded and to
  which Values are stored;
- **Resource**: opaque device objects whose operations have resource-specific
  semantics.

The central rule is:

> Storage loads produce Values. Computation consumes and produces Values.
> Storage writes consume Values.

Storage and Resource handles are not Values, cannot be Tensor elements, and do
not implicitly copy. Value shape never carries ownership or physical-layout
semantics.

## 2. Value types

### 2.1 Scalars

`bool`, `i32`, `u32`, `f16`, `f32`, and `f64` are scalar Values. Python `int`
and `float` are annotation and explicit-cast aliases for `i32` and `f32`.
Language `u32` is ABI metadata; MLIR storage is signless `i32`. See
[compiler design](../compiler/design.md#mlir-storage-vs-language-abi-dtype).

Integer and floating literals remain contextual until constraints are solved.
Unconstrained literals default to `i32` and `f32`. Safe implicit conversions
are integer-to-floating and `f16 -> f32 -> f64`. Floating narrowing,
floating-to-integer conversion, and dynamic `i32`/`u32` mixing require an
explicit cast. `bool` does not participate in numeric arithmetic. `/` is true
division; integer operands produce at least `f32`.

### 2.2 Tensor Values

`Tensor[T, static_shape]` is an immutable, homogeneous, rectangular Value.
Every extent is a positive compile-time integer after specialization. A Tensor
type exposes only its logical element type, rank, and shape:

```python
Tensor[f32, (4,)]
Tensor[f32, (3, 3)]
Tensor[Particle, (16,)]
```

`T` may be any recursively ABI-stable Value, including a Scalar, Tensor,
Tuple, or Struct. `T` may not be Storage, Resource, or an arbitrary Python
object. Numeric Tensor operations additionally require numeric leaves; element
legality does not imply that every operation is defined for that element.

Stride, offset, order, padding, alignment, address space, and ownership are
not part of a Tensor Value type. The compiler may use registers, vectors,
matrices, aggregate ABI values, or spills for the same source type. This
physical choice does not change type identity.

Nested Tensor Values canonicalize by logical shape composition:

```text
Tensor[Tensor[T, inner_shape], outer_shape]
    == Tensor[T, outer_shape + inner_shape]
```

For example, `Tensor[Tensor[f32, (3,)], (8,)]` normalizes to
`Tensor[f32, (8, 3)]`. Canonicalization is recursive. Struct boundaries remain
nominal and are never flattened by Tensor canonicalization.

`Vector[T, N]` and `Matrix[T, R, C]` are canonical aliases for rank-1 and
rank-2 Tensor Values. `Tensor([...])` is the canonical fixed-value constructor.
It requires a rectangular literal list or tuple with a common element type.
`Vector([...])` and `Matrix([[...]])` are rank-constrained constructors.
`Vector` may concatenate scalar and rank-1 Tensor segments from its single
sequence literal. The v3 `vec`, `mat`, `vec*`, and `mat*` spellings are removed.

`Array` is removed. `Tensor[T, (N,)]` is the fixed-size homogeneous sequence
Value.

### 2.3 Tuple and Struct

`Tuple[T0, ..., Tn]` is a structural anonymous product Value. Tuples support
construction, constant indexing, and destructuring. Two Tuple types are equal
when their ordered element types are equal.

`@struct` declares an immutable nominal product Value. Structs support
construction by declared fields and field access. Two Structs with identical
fields remain different types when their declarations differ.

Tuple and Struct share deterministic product-layout rules at ABI boundaries,
but structural and nominal identity remain distinct. All fields must be
recursively ABI-stable Values.

### 2.4 Portable Value ABI layout

The canonical Value ABI is backend-independent. Scalar size/alignment pairs
in bytes are: `bool` `(1, 1)`, `i32`/`u32`/`f32` `(4, 4)`, `f16` `(2, 2)`,
and `f64` `(8, 8)`. A Tensor is a row-major repetition of its canonical
element layout; its element stride is the element size rounded up to element
alignment. Tuple and Struct fields retain declaration order. Each field starts
at its size-so-far rounded up to that field's alignment, aggregate alignment is
the maximum field alignment, and final aggregate size is rounded up to
aggregate alignment. Empty products have size zero and alignment one.

These are logical boundary rules, not an instruction to use the same physical
representation inside a function. A backend may use vectors, registers, or a
native block layout internally, but must insert an explicit conversion when
that representation differs from the canonical ABI. Reflection records size,
alignment, field offsets, Tensor element stride, and the layout-rule version.

## 3. Storage types and layouts

### 3.1 TensorStorage

`TensorStorage[T]` is a host-runtime owner of a dense typed allocation. Its
runtime descriptor records logical shape, resolved element layout, byte size,
device/backend ownership, and synchronization state. It is not a device
computation Value.

`TensorStorage[Vertex]` is the canonical typed array-of-structures (AoS)
representation for interleaved records. Struct field offsets, alignment, and
record stride are deterministic and reflected for every backend ABI.

At a `Module.forward` boundary, an owner parameter has three canonical
annotation forms:

```python
TensorStorage
TensorStorage[f32]
TensorStorage[f32, (1024,), read]
```

The bare form leaves element type, shape, and access to nested typed Program
calls or an explicit export argument. `TensorStorage[T]` fixes the element type
and infers shape and access. The three-argument form fixes the complete static
contract. Explicit constraints must agree with every nested kernel, Module,
and graphics-pipeline use; read and write uses join to `read_write`, while
`vd.dyn` unifies with a compatible static extent. Cooking rejects any unresolved
element type, extent, or access instead of consulting runtime values.

`TensorStorage[...]` denotes an owner boundary and rejects borrowed
`TensorView` values. A Module parameter intended to accept a full view or
subview uses `TensorView[T, shape, access]`.

### 3.2 TensorView

`TensorView[T, shape, access]` is the only non-owning shaped Storage type used
by kernel parameters, slices, field projections, and local allocations.
`access` is `read`, `write`, or `read_write`. Shape contains positive static
integer extents and `vd.dyn` markers for runtime-resolved extents:

```python
TensorView[f32, (4, 4), read]
TensorView[f32, (vd.dyn, 4), read_write]
```

Address space is inferred from origin and cannot be written in ordinary source:
kernel parameters are device storage and `workgroup_storage` results are
workgroup storage. The concrete semantic descriptor also carries one signed
element stride per dimension, an element offset, owner and lifetime, and the
inferred address space.

A host-side subview is constructed explicitly, for example:

```python
positions = storage.view(
    shape=(vertex_count, 3),
    strides=(8, 1),
    offset=0,
    access="read",
)
```

The corresponding kernel parameter is annotated
`TensorView[f32, (vd.dyn, 3), read]`. Shape constraints are source type
arguments. Concrete strides and offset are layout metadata, not core source
type identity. Runtime binding verifies static extents and resolves every
`vd.dyn` extent from the supplied descriptor.

`T` may be any recursively ABI-stable Value element: Scalar, fixed Tensor,
Tuple, or Struct. Aggregate elements retain their canonical product layout.
Backend lowering may expand them into Scalar leaf descriptors over the same
owner allocation, but that expansion is not observable in source typing,
ownership, access, or dispatch semantics.

Compute artifacts must not specialize for a concrete runtime layout. Shape,
signed element strides, and offset are dispatch descriptor values, and every
logical index is projected to the owner's physical element index before
backend lowering.

Do not feed `TensorStorage`, `TensorView`, or `GraphBuffer` extents into
kernel `_lower`, `specialize`, or compute `finalize` `shape_facts`. Annotation
static extents may be recorded in the Program. `vd.dyn` stays `-1` on the
Value shape; borrowed Storage does not invent a compile-time byte length.
C++ bind/invoke reads shape, strides, offset, and byte length from the bound
buffer. Python must not paper over GPU `finalize` by baking a launch shape
into the native artifact. Graphics `shape_facts` exist only for
image/attachment extents.

Language-level typed shape, strides, and offset use units of the recursively
resolved leaf element. Compiler reflection and `PROGRAM_VERSION` manifests
record static shape constraints and the entry-owned semantic metadata field
sequence, never concrete dispatch values. The selected StageArtifact
implementation records one aggregate physical metadata carrier and its target
location. Runtime descriptors record byte strides and byte offsets after
Tensor and Struct layout is resolved. Runtime validation performs one checked
byte-to-element conversion. External APIs must state whether supplied layout
values are element or byte units; implicit unit conversion is forbidden.

A view is legal when every in-bounds logical index maps within its owner.
Writable views must be internally injective: distinct logical indices cannot
name the same location. Read-only overlapping views may be admitted by an
explicit backend capability, but overlap is never inferred to be safe.
Negative strides are view semantics because they change the base/offset and
address mapping; they are not Tensor Value layout.

Views may share one owner without being internally overlapping. Disjoint
writable projections may be borrowed simultaneously. Potentially overlapping
writable projections, or a writable projection aliasing an active reader,
require rejection unless an explicit synchronization/effect rule proves the
access safe.

Loading through a view materializes a Value independent of physical layout.
Storing performs the inverse projection:

```text
TensorView load  : TensorView[T, shape, read] x indices -> T
TensorView store : TensorView[T, shape, write] x indices x T -> ()
```

The notation describes semantics, not a promise that a whole dynamic view is
copied at once; scalar and sub-Tensor indexing obey the same rule.

`vd.workgroup_storage(T, shape=(...))` creates a workgroup-address-space
`TensorView` with read-write access and contiguous row-major layout. Every
extent must be a positive compile-time integer after specialization. Literal
and captured host constants are legal; device-runtime values are not. The
initial portable limit is 16 KiB after canonical ABI layout. Every recursively
ABI-stable Value element is legal. There is no dynamic shared-memory form,
stride argument, offset argument, or `workgroup_array` compatibility spelling.

### 3.3 Interleaved attributes

Interleaved structured storage is a core dense layout. For a record
`[position(3), normal(3), uv(2)]`, field projections may expose:

- position: shape `(N, 3)`, outer stride `8`, offset `0`;
- normal: shape `(N, 3)`, outer stride `8`, offset `3`;
- UV: shape `(N, 2)`, outer stride `8`, offset `6`.

Each projection is internally injective although all three share one owner.
Field projection from `TensorStorage[Vertex]` is the canonical form. A typed
view over `RawBuffer` is permitted only for an exact byte ABI.

### 3.4 RawBuffer

`vd.interop.RawBuffer` is a low-level host/runtime boundary for copying
externally defined bytes into an owned allocation with explicit alignment and
typed view construction. The source buffer is never a mutable alias of the
runtime allocation; host mutation occurs only through claim-scoped
`TensorView` operations. `RawBuffer` is not a source-language type: the
frontend parser/model must reject it in kernel, shader, and shared-function
annotations. It provides no implicit element type or shape.

The public `Buffer` spelling is removed. A shaped kernel storage parameter uses
`TensorView`; ownership uses `TensorStorage`; untyped host bytes use
`vd.interop.RawBuffer`. Typed backend-facing IR uses
`!vernon.tensor_view<element, shape, access, address_space>`; only opaque runtime
allocation handles retain device-buffer terminology.

The current baseline does not claim byte-address device operations, unsized
trailing-array layouts, or CUDA/Vulkan external-memory import. Those require
separate ownership, synchronization, bounds, and lifetime contracts before
they become supported interop features.

### 3.5 Dense-only core

Core Storage is dense. The current contract has no SNode/layout tree, transparent sparse
Tensor layout, or sparse Value type. CSR, COO, blocked-grid, hash-grid, and
other sparse structures are library data structures composed from dense
`TensorStorage` and, where necessary, `RawBuffer`. Their capacity, indexing,
allocation, failure, and synchronization policies remain explicit.

## 4. Resource types

`Texture` and `Sampler` are opaque Resource handles. A sampled texture is
`Texture[dimension, sample_type]`, where `dimension` is `2d`, `3d`, `cube`,
`2d_array`, or `cube_array` and `sample_type` is `f32`, `i32`, or `u32`. A
storage texture is `Texture[dimension, format, access]`, where `dimension` is
`2d`, `3d`, or `2d_array`,
`access` is `read`, `write`, or `read_write`, and format is one of
`r8_unorm`, `r16_float`, `r32_float`, `rg8_unorm`, `rgba8_unorm`,
`rgba16_float`, or `rgba32_float`.

Texture queries and sampling use resource-specific operations; they are not
ordinary TensorView loads or stores. Sampler state is separate from sampled
storage. Resource format, dimensions, binding, and backend capabilities are
validated at entry interfaces and reflected in pipeline metadata.

The supported texture operation regions are:

- `texture_sample(texture, coordinates)` and
  `texture_sample(texture, sampler, coordinates)` use implicit LOD and are
  fragment-only;
- `texture_sample(texture, coordinates, lod)` uses an implicit sampler and is
  valid in vertex and fragment stages;
- `texture_sample(texture, sampler, coordinates, lod)` is valid in compute,
  vertex, and fragment stages;
- `texture_size(texture[, lod])` is valid in vertex and fragment stages and
  requires an integer scalar LOD when present;
- `texture_load` and `texture_store` operate on storage Textures with integer
  coordinates and enforce the declared read/write access.

Sampling coordinates are floating Tensor Values with rank determined by the
Texture dimension, and sampling LOD is a floating Scalar. Arrayed dimensions
follow SPIR-V: the layer is the last coordinate component and `texture_size`
appends the layer count. A `cube_array` layer selects a whole cube.

| Dimension | Sample coordinates | Storage coordinates | `texture_size` |
|---|---|---|---|
| `2d` | `(u, v)` | `(x, y)` | `(w, h)` |
| `3d` | `(u, v, w)` | `(x, y, z)` | `(w, h, d)` |
| `cube` | `(x, y, z)` direction | — | `(w, h)` |
| `2d_array` | `(u, v, layer)` | `(x, y, layer)` | `(w, h, layers)` |
| `cube_array` | `(x, y, z, cube)` | — | `(w, h, cubes)` |
 One Texture entry
parameter cannot mix implicit and explicit sampler forms in the same stage.
Implicit sampling creates a deterministic sampler interface binding.

## 5. Ownership, borrowing, and effects

A TensorStorage owns its allocation. A TensorView borrows one owner and cannot
outlive it. A dispatch borrows every view for the complete asynchronous device
use, not merely for the duration of the host call. The runtime retains owners
until completion and rejects host mutation, destruction, or incompatible
borrows while device work is outstanding.

Typed semantic nodes record effects independently from types:

- `read(owner, region)`;
- `write(owner, region)`;
- `atomic(owner, region, ordering, scope)`;
- `barrier(ordering, scope)`;
- Resource-specific query/sample effects.

`atomic_add`, `atomic_min`, `atomic_max`, and `atomic_exchange` accept i32/u32
writable `TensorView` elements. `atomic_add` additionally accepts f32/f64 when
the selected target profile has a legal native or integer-compare-exchange
implementation for that type and scope. A rank-one atomic accepts one scalar
index; a higher-rank atomic requires a tuple containing exactly one index per
dimension. Scope is inferred from the Storage address space: workgroup storage
uses workgroup scope and device storage uses device scope. Unsupported
type/scope/target combinations are rejected during capability validation.
The source ordering is `relaxed`: a backend may emit a stronger ordering when
its legal lowering cannot represent relaxed ordering, but must never weaken a
requested ordering.
Read/write effects include projected regions where statically known, and an
atomic read-modify-write also marks its owner writable for runtime
synchronization. Unknown overlap is conservatively aliasing.

Every compute entry carries a compiler-derived `vernon.dispatch_contract`.
Its `unit_grid_axes` and `requires_unit_workgroup` constraints are residual
conditions of the ordinary-write injectivity proof. The lemmas (affine
unique-axis, unsigned mixed-radix, workgroup leader, and the serialized
single-invocation fallback) are
[`compiler/invocation_index_ownership.md`](../compiler/invocation_index_ownership.md).
All public direct, cooked,
AOT, graph, and autodiff dispatch paths validate this contract before
allocation, staging, mutation, or submission. A failed constraint is an error;
the runtime never serializes the dispatch. Constant ordinary writes therefore
require a unit grid and unit workgroup, while unconstrained multi-invocation
accumulation must use a formal accumulation or atomic operation.

TensorView bounds, writable injectivity, physical identity, and overlap use
the shared native validation model. Distinct allocation identities and
non-overlapping byte spans prove disjointness; regular strided views may also
use stride-lattice congruence. An unproven writable overlap is rejected.
Semantic acceptance never depends on enumerating elements or on an
element-count threshold.

Ordinary `@func` code may carry Storage and Resource effects through explicit
parameters and may use atomics or barriers where their concrete entry stage
permits them. Those effects propagate through the call graph and are checked
against parameter access and the concrete entry stage. Shared
`@func(shared=True)` code remains host/device-pure and cannot use Storage or
device-only Resource operations. Kernels and graphics entries may perform
effects allowed by their stage and parameter access modes. Host runtime
allocation, uploads, dispatch, downloads, and resource lifetime are not parsed
device-language expressions.
Multi-node ordering, render state, resource transitions, and cross-backend
synchronization are canonical Program and Runtime Command DAG semantics. They
are not shader-language semantics.

## 6. Functions, interfaces, and specialization

- The currently registered `@kernel`, `@vertex`, and `@fragment` decorators
  declare externally visible entries. Future graphics entry decorators must
  register a distinct stage kind and topology rules. Parameters and non-`None`
  results require annotations.
- `@func` declares a non-recursive, stage-polymorphic helper. Parameter and
  result annotations may be partial; every reachable specialization must
  resolve completely.
- `@func(shared=True)` is limited to host/device common Values and pure
  operations.
- Functions accept required positional parameters only. Device calls do not
  have Python keyword-argument semantics.
- Helpers specialize by qualified declaration, concrete argument types,
  enabled features, and captured constants. Ordering, symbols, and cache keys
  are deterministic.
- Unreachable helpers are pruned. Unreachable generic bodies have no
  diagnostic obligation.

At a direct compute boundary, a `Tensor` parameter is a snapshotted immutable
Value and accepts a compatible NumPy array or immutable host Tensor Value. It
does not accept `TensorStorage` or `TensorView`. A `TensorView` parameter
accepts a compatible `TensorStorage` owner as its canonical full view or an
explicit subview; a raw NumPy array is not implicitly allocated as Storage.
Every asynchronous dispatch retains borrowed owners until completion.

Graphics transport does not change semantic category. A
`Tensor[..., attribute()]` is one immutable per-invocation Value sourced from
a host-bound vertex stream, and a `Tensor[..., uniform()]` is an immutable
Value even if a backend uses a buffer physically. Direct shader addressing,
stores, and atomics require `TensorView[..., resource()]`.

A persistent `ProgramAsset` wraps exactly one executable Program. Its
`program=` is a compute Kernel, a `vd.pipeline(...)` graphics pipeline, an
initialized Module, or a supported explicit Program transform. Standalone
compute and graphics executables normalize to one-node Programs. Modules may
compose compute and graphics Nodes in one static Program DAG.

Each graphics entry carries an explicit stage kind. A target-independent stage
registry validates `vd.pipeline(...)` topology and ordering. The registered
graphics topology is `vertex -> fragment`. Future stage additions can extend
topology validation through a versioned compiler contract.

Generated builtin functions are the preferred authoring API.
`builtin("...")` remains a low-level entry-interface annotation and uses the
same closed stage/direction/type registry.

Imported DSL declarations are loaded through the source module graph without
executing Python. Captured constants must belong to the deterministic
constant-expression subset and are part of specialization identity. A Struct
keeps its declared class name in emitted source, reflection, and host Values
whichever module declares it, so two different reachable Structs with one name
are rejected with `conflicting declarations named`; imported helpers are
module-qualified. The design rationale for sections 6.1 through 6.6 is in
[`static_specialization.md`](static_specialization.md).

### 6.1 Specializations and variants

`vd.specialization(name, kind)` declares a compile-time value whose kind is
`bool`, `i32`, `u32`, `f32`, `f64`, or `vd.type` (section 6.6);
`vd.feature(name)` is the `bool` form. A specialization key binds declared
names to values. Specialization values and `When` branches are compile-time
inputs, not runtime Python conditions.

`ProgramAsset` explicitly enumerates canonical keys through `variants=`, whose
keys may name specializations the asset module declares or imports. Compute
Kernel, Kernel-VJP, and graphics Pipeline assets accept every kind, and both
graphics stages share one key. Module and Module-VJP assets require the empty
key because their child specialization choices are authored explicitly in the
Module graph. There is no independent public shader-variant selector.

The source module graph is the only authority that checks a key against its
declarations: a name no reachable declaration has is rejected with
`requested undeclared specialization`, a kind that differs from the
declaration with `is declared as`, and a missing value where it is used. An
unassigned scalar fails with `missing specialization`, an unassigned Boolean is
false, and a type specialization has no default.

### 6.2 Compile-time expressions and static branches

A compile-time expression contains only literals, captured constants,
specialization names, and these operators:

- `==` and `!=` on every specialization kind;
- `<`, `<=`, `>`, `>=`, `+`, `-`, `*`, `//`, and `%` on numeric kinds;
- `and`, `or`, `not`.

A type specialization compares only with `==` and `!=` against a type
reference: a declared Struct or binding group, a scalar type, or another type
specialization. Any other use in a compile-time expression fails with
`a type specialization compares only with == and != against a type reference`.

When the test of an `if`, `elif`, or conditional expression references a
specialization, the whole test must be a compile-time expression; otherwise
compilation fails with `specialization condition mixes runtime values`. Such a
branch is static:

- it is decided when the key is bound, before method resolution,
  reachability, and typing;
- only the selected branch remains; discarded branches have no diagnostic
  obligation and contribute no callees, resources, or reachability;
- a name bound only in a discarded branch is undefined afterwards.

A module referenced only from a discarded branch is still loaded and recorded
as a dependency. Any other use of a scalar specialization is a typed constant.

### 6.3 Conditional parameters

`When[condition, T]` accepts any compile-time expression as `condition`. A
disabled parameter is absent from the entry interface, host call signature,
reflection, and Program parameters, and using it in a selected branch fails
with `disabled value`. Attribute and varying locations are assigned before any
`When` is evaluated, so disabled parameters keep stable reservations. Resource
bindings are planned after evaluation, so disabled resources take no binding.

The host evaluates annotations with each specialization name bound to its
value in the key (a type specialization to its class or scalar type), and a
`When` annotation resolves to its wrapped type; the host's parameter set comes
from the specialized entry. A `Specialization` object refuses `bool()`,
ordering, and arithmetic, so on Python 3.11 through 3.13 a module whose
annotations contain conditions other than `==` and `!=` needs
`from __future__ import annotations`.

### 6.4 Binding groups

A `@vd.binding_group` class declares a named group of entry interface items.
Its fields are `TensorView`, `Texture`, and `Sampler` resources; Values,
including Structs, which bind as immutable uniforms in graphics stages and as
ordinary Value parameters in Kernels; and nested binding groups. Fields carry
no `attribute`, `varying`, `builtin`, `set`, or `binding` metadata, so one
group is valid in every stage. Graphics TensorView fields follow the read-only
rule of
[`../program/graphics_storage_reads.md`](../program/graphics_storage_reads.md).

A binding group is not a Value. It appears only as an entry parameter, a
helper parameter, a forwarded argument, or the receiver of a field access or
method call. It cannot be constructed in device code, assigned to a local,
returned, compared, used as a Struct field or a Tensor or TensorView element,
or passed to `@func(shared=True)` code (`binding group cannot be`). Fields
cannot declare interface metadata (`binding group field cannot declare`).
Methods are `@vd.func`, cannot reassign fields, and a group has no `shared`
option.

Each leaf is an entry interface item named by its dotted path, for example
`material.noise`, in reflection, Program parameters, host keyword binding, and
derivative paths. On the host the class is a plain container:
`Hologram(params=HologramParams(...), noise=texture)` binds leaf by leaf. One
group occurrence takes one consecutive binding range in field order, including
spilled uniforms, storage leaves, and nested groups, and the same group under
the same name in both stages of a pipeline reuses that range. All ranges are in
descriptor set 0.

Every backend accepts every group leaf kind, including Struct and Tuple Value
leaves in GPU Kernels; leaf scalar types still follow target capabilities.
Kernels, Pipelines, and `Module.forward()` accept groups. `Module.forward()`
group parameters must be annotated, and `forward` may pass a group to the
Kernels and Pipelines it calls or construct one from leaves it holds.

Differentiation treats a group leaf by leaf under
[`../autodiff/contract.md`](../autodiff/contract.md) section 3. A VJP `wrt` or
`outputs` path naming a group root, or a nested group, is shorthand for its
leaves in field order: for `wrt` every floating Value leaf (Struct and Tuple
Values expanded to floating fields) and whole floating TensorView leaf, for
`outputs` the writable floating TensorView leaves. A root with no such leaf
fails with `names a binding group with no ... leaves`. Expansion happens per
specialization key, before the paths become the transform identity.

### 6.5 Typed method resolution

`x.m(...)` resolves on the inferred Struct or binding-group type of `x` during
typed helper specialization, not by syntactic lookup of the receiver. Generic
helpers may call methods on unannotated parameters, and each helper
specialization resolves its calls independently. Methods are reachable only
through resolved calls, so unreachable methods are pruned without diagnostics.
A missing method fails at the call site with `'T' has no method 'm'` and the
helper specialization chain. Methods are not first-class values.

### 6.6 Type-valued specializations

`vd.specialization(name, vd.type)` declares a specialization whose value is a
`@vd.binding_group` class, a `@vd.struct` class, or a scalar type.

- The name is a type wherever a type is allowed: `material: MATERIAL` for a
  binding group, `Annotated[MATERIAL, uniform()]` for a Struct in a graphics
  stage, a `TensorView[ELEMENT, ...]` element, or a constructor call
  `ELEMENT(x)`.
- At key binding, the module that defines the value joins the source module
  graph as if imported; the template's own imports are unchanged.
- `variants=` and Kernel and Pipeline `specializations=` take the class or
  scalar type itself, for example `{MATERIAL: Hologram}`. The frontend loads
  the declaration from its defining source and never executes that module.
- Conformance is checked at each use, not declared.

A declared type's canonical identity is `<module>:<Class>`, where `<module>`
is its defining source path relative to that source's own project root (the
nearest directory containing `pyproject.toml`), with `.` separators and no
`.py` suffix. The type may belong to a different project than the template it
specializes: the graph loads it from its declaring file, and its modules
import absolutely from their own project root before the template's. Two
modules of different projects with the same relative path fail with
`share the project module path`, because the path names their emitted
symbols. Variant keys are scoped to one Program, so identities need only be
unique among one Program's variants: types from two files with one identity
in the same Program fail with `share the identity`. A scalar type's identity is
its scalar name. A class without such a
module-level path fails with `type value has no portable identity`, and a value
that is neither fails with
`type specialization requires a declared type or scalar type`. A missing
assignment fails where it is used with `missing type specialization`.

### 6.7 Processing order and identity

Project processing order is:

1. load and validate source imports;
2. bind captured constants and the canonical typed specialization key:
   substitute specialization values, decide static branches, remove disabled
   `When` parameters, and load type-valued modules;
3. normalize generated builtins;
4. prune to selected entries and validate the call graph;
5. infer, specialize, and validate typed semantic nodes and effects,
   resolving method calls and flattening binding groups;
6. lower typed nodes;
7. validate target capabilities and materialize artifacts.

Semantic cache identity includes `COMPILER_CONTRACT_VERSION`,
`PROGRAM_VERSION`, all source dependency digests (including each type value's
defining module graph), entry, the typed specialization key (type values by
canonical identity), annotation-static shape constraints and interfaces,
captured constants, workgroup size, helper specializations, and
derivative-transform identity. Two keys that differ only in a type value never
share frontend results. Concrete dynamic TensorView descriptors remain
invocation data as defined in [`tensor_view.md`](tensor_view.md). Diagnostics
include source path, one-based line and column, and a stable reason string.

## 7. Stable syntax target and deferred ergonomics

The current core statement subset is `pass`, expression statements, simple local or
indexed assignment, annotated assignment, augmented assignment, `return`,
`if`, `range` loops, `while`, loop `else`, `break`, and `continue`. Return may
terminate a nested structured region; break and continue target the nearest
enclosing loop.

The expression subset includes names, numeric and Boolean literals, arithmetic,
unary operations, one comparison, calls, indexing, Tensor/Tuple/Struct
construction, and constant Tuple indexing. Supported attributes are immutable
Struct fields, `.shape` on rank-one-or-greater Tensor/TensorView values, and
rank-one Tensor swizzles over `xyzw`/`rgba`; color aliases canonicalize to the
corresponding positional mask.

The portable floating-point math surface includes `sin`, `cos`, `acos`,
`atan2`, `exp`, `log`, `sqrt`, `floor`, `abs`, `min`, `max`, `pow`, and
`clamp`, plus vector `dot`, `cross`, `norm`, `normalize`, and `reflect`.
`atan2(y, x)` follows the conventional quadrant-aware argument order.

Tuple destructuring, short-circuit `and`/`or`, conditional expressions,
dynamic `range`, `break`, `continue`, and nested/early return are implemented
current phases under `COMPILER_CONTRACT_VERSION`. Compute autodiff accepts
pure-Value early returns and literal-bounded loops with one dynamic leading
`if condition: break` guard; general dynamic `range`, `while`, `continue`, and
loop-local return remain deferred. Chained comparisons, recursion, dynamic allocation, exceptions,
generators, arbitrary classes, Python list/dict semantics, and Python object
mutation remain deferred.

`range(start, stop, step)` evaluates its arguments once before entering the
loop. Integer literals and dynamic `i32` bounds are accepted; `u32` values
require an explicit cast. Positive and negative steps follow Python's exclusive
stop semantics. A literal zero step is a compile error. A dynamic zero step is
a runtime contract violation and must never silently execute zero iterations.
Fixed-width overflow must terminate the range rather than wrap into an
unbounded loop. A target without a legal contract-violation mechanism must
reject dynamic step values during capability validation. Phase 5B control flow
outside the accepted compute-autodiff subset above remains deferred until
separate derivative and tape policies are specified.

Frontend acceptance guarantees well-typed Vernon IR, not that every target
implements every operation. Unsupported target/type/stage combinations fail
explicitly and never silently narrow or change semantics.

## 8. First-order autodiff

The complete normative design is
[`../autodiff/contract.md`](../autodiff/contract.md).
Autodiff transforms specialized, validated typed IR and never executes Python
to trace a function.

The initial public reverse-mode surface is one transform:
`vd.ad.vjp(program, wrt=..., outputs=..., planning_policy=...)`. It produces a
program whose execution returns primal outputs and a pullback. `outputs` may
select public output paths and `planning_policy` selects a supported
deterministic planning policy; neither is a custom derivative-rule surface.
Applying the pullback to explicit output cotangents returns gradients for the
canonical input paths selected by `wrt`.
One floating Scalar output may omit its cotangent and uses seed `1`.
Tensor/aggregate outputs may not omit cotangents and are never implicitly
reduced.

Authored Kernel and graphics entry signatures do not change. The language has
no `grad_or_not`, `requires_grad`, implicit `.grad`, global gradient clearing,
or context that silently executes backward work. `program_asset()` remains the
only cookable declaration; a VJP is represented by a declarative
`ProgramExpression` in its `program=` operand.

Floating-point Scalar leaves are differentiable. Tensor, Tuple, and Struct
Values derive adjoint structure recursively from floating leaves. Floating
Scalar and immutable Tensor gradients are ordinary Values. A differentiated
TensorView or mutable Storage input produces newly owned gradient Storage.
The backward compiler ABI receives that Storage as a writable TensorView
argument, with dynamic extents preserved in its descriptor; it is not returned
as a fixed-shape Value.
Integer, Boolean, Resource handle, sampler state, and opaque leaves are
non-differentiable unless a custom operation rule consumes them without
requesting a gradient. Gradients never change primal type or identity.

The compiler-internal `ProgramGraph` for one specialized program represents
typed Value flow, structured control flow, Storage effects, aliases, saved
Values, and reverse dependencies. It is not a host orchestration graph.
Stateful differentiation requires legal mutation functionalization, bounded
tape, effect-preserving reverse traversal, alias/race validation, and
deterministic or capability-checked gather/scatter accumulation.

Arithmetic, casts, Tensor construction, `matmul`, and supported math intrinsics
use versioned built-in VJP rules. Graphics Program VJP is not a supported
capability and is rejected during Program capture. There is no public graphics
custom-rule API in this contract. Texture handles and sampler state remain
non-differentiable.

Applying `vd.ad.vjp` to a Module produces explicit forward, backward, and
residual Program topology. Runtime executes that topology without a separate
host pass graph.

JVP, full-Jacobian materialization, convenience `grad` aliases, implicit
gradient accumulation, nested transforms, Hessians, and Hessian-vector
products are outside the initial public surface.

## 9. Legacy migration

- Host runtime `Tensor` ownership becomes `TensorStorage`.
- Kernel storage parameters become `TensorView[T, shape, access]`.
- Pure fixed-shape computation values remain `Tensor[T, static_shape]`.
- Removed `Buffer` becomes a typed `TensorView` or runtime-only
  `vd.interop.RawBuffer`.
- `Array[T, N]` is not introduced; use `Tensor[T, (N,)]`.
- `Vector` and `Matrix` are rank aliases; v3 `vec`, `mat`, `vec*`, and `mat*`
  spellings are rejected.
- V3 addressability metadata becomes explicit Storage typing and access mode.
- Physical shape/stride/offset metadata moves from Tensor Value concepts to
  TensorView descriptors.
- `workgroup_array` becomes `workgroup_storage` directly; no alias remains.

Removed v3 aliases must not create a fourth semantic category or distinct
cache identity.

## 10. Feature status

| Area | Legacy implementation | Current normative target | Deferred |
| --- | --- | --- | --- |
| Tensor | Fixed numeric aggregate; entry addressability metadata | Immutable Value with recursively ABI-stable Value elements and logical shape only | Dynamic-shape Value |
| Ownership | Runtime `Tensor`; removed `Buffer` | `TensorStorage`, borrowed `TensorView`, runtime-only `vd.interop.RawBuffer` escape hatch | General allocator model |
| Layout | Backend/runtime details | Dense strided views, AoS field projections, explicit alias rules | Transparent sparse layouts and SNode trees |
| Aggregates | Nominal immutable Struct; Vector/Matrix constructors | Tensor, structural Tuple, nominal Struct; no Array | Enums and tagged unions |
| Effects | Typed read/write records | Region-aware reads/writes, relaxed i32/u32 atomics, capability-checked f32/f64 atomic add, and typed barriers | Additional atomic operations/orderings and full race model |
| Autodiff | No public transform | First-order compute Program VJP over typed Programs and stateful Kernels; graphics paths fail closed | Graphics VJP/custom rules, JVP, full Jacobians, convenience aliases, and higher-order AD |
| Control flow | Tuple destructuring, short-circuit expressions, early return, dynamic range, break, and continue | Current subset plus explicitly AD-covered flow | Unrestricted recursion and Python-only control flow |
| Rendering | Typed graphics stages, textures, samplers | Same model with versioned rasterization/visibility/depth/blend/texture VJP boundaries | Graphics derivatives without explicit accepted custom rules |

## 11. Workload expressiveness

A dense fluid simulation uses TensorStorage grids, TensorView stencil
projections, pure numerical helpers, and multiple host-ordered compute
dispatches. Sparse acceleration is a library data structure, not a transparent
Tensor layout. Stateful fluid gradients remain gated on mutation, tape,
checkpoint, reverse traversal, and the program-internal autodiff graph.

PBR and SSR use Value Tensors and Structs for local calculations,
TensorView/Texture inputs for scene data, Sampler resources for filtering, and
graphics or compute entries for passes. Their primal execution does not
require arbitrary Python objects. First-order derivatives apply only to the
pure typed arithmetic or explicitly differentiable primitives, not visibility
or raster decisions.

## 12. Current contract acceptance gates

The current `COMPILER_CONTRACT_VERSION` may be declared implemented only when:

- every public term in this document maps to one typed semantic category;
- Tensor element legality, nested normalization, and Struct boundaries have
  parser, inference, ABI, cache-identity, and backend tests;
- TensorStorage ownership, TensorView shape constraints, layout units, lifetime, projection,
  injectivity, and alias diagnostics have CPU reference tests and backend
  parity tests where supported;
- rank-1 Tensor replacement of Array is consistent across constructors,
  interfaces, reflection, and diagnostics;
- Resource operations remain distinct from Storage operations;
- first-order transforms, built-in versioned rules, rejection of public custom
  rules and higher-order transforms, and non-differentiable boundaries have
  deterministic tests;
- v3 migration diagnostics and compatibility aliases are versioned;
- unsupported target combinations fail explicitly;
- cache identity and generated symbols remain deterministic;
- static specialization meets section 6: an integer static branch prunes the
  callees and type errors of discarded branches, and a `When` resource
  disappears from the interface when disabled, for Kernels and graphics
  stages; graphics Pipeline assets cook and render with integer and type
  variants; binding groups with every leaf kind work in Kernels and both
  graphics stages on every backend, reject every misuse with its diagnostic,
  take one consecutive binding range, and yield dotted-leaf VJP cotangents; one
  generic helper calls the same method on two binding groups; a template that
  does not import its material cooks against two binding-group values whose
  frontend results never alias and whose source edits invalidate only their
  own keys; manifest keys carry checkout-independent type identities;
- fluid simulation, PBR/SSR, and first-order autodiff examples compile within
  their stated boundaries.

Until every gate in this document passes, unchecked sections remain unavailable
as end-to-end guarantees even though the frontend reports the current compiler
contract version.
