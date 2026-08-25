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
resolved leaf element. Compiler reflection and `PIPELINE_VERSION` manifests
record static shape constraints and descriptor binding positions, never
concrete dispatch values. Runtime descriptors record byte strides and byte
offsets after Tensor and Struct layout is resolved. Runtime validation performs
one checked byte-to-element conversion. External APIs must state whether
supplied layout values are element or byte units; implicit unit conversion is
forbidden.

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
view over `RawBuffer` is permitted only for an exact external byte ABI.

### 3.4 RawBuffer

`vd.interop.RawBuffer` is a low-level host/runtime interop escape hatch for
externally defined bytes, explicit alignment, and typed view construction. It
is not a source-language type: the frontend parser/model must reject
`RawBuffer` in kernel, shader, and shared-function annotations. It provides no
implicit element type, shape, or safe aliasing guarantee.

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

`Texture` and `Sampler` are opaque Resource handles. Texture queries and
sampling use resource-specific operations; they are not ordinary TensorView
loads or stores. Sampler state is separate from sampled storage. Resource
format, dimensions, binding, and backend capabilities are validated at entry
interfaces and reflected in pipeline metadata.

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

`atomic_add`, `atomic_min`, `atomic_max`, and `atomic_exchange` initially
accept i32/u32 writable `TensorView` elements. A rank-one atomic accepts one
scalar index; a higher-rank atomic requires a tuple containing exactly one
index per dimension. Scope is inferred from the Storage address space:
workgroup storage uses workgroup scope and device storage uses device scope.
Device-scope TensorView atomics are supported by CPU, CUDA, and Vulkan; other
targets reject them during capability validation.
The source ordering is `relaxed`: a backend may emit a stronger ordering when
its legal lowering cannot represent relaxed ordering, but must never weaken a
requested ordering.
Read/write effects include projected regions where statically known, and an
atomic read-modify-write also marks its owner writable for runtime
synchronization. Unknown overlap is conservatively aliasing.

Every compute entry carries a compiler-derived `vernon.dispatch_contract`.
Its `unit_grid_axes` and `requires_unit_workgroup` constraints are residual
conditions of the ordinary-write injectivity proof. All public direct, cooked,
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

Ordinary `@func` code has no Storage, atomic, or barrier effects. It may read
Resources passed explicitly as parameters; those effects propagate through the
call graph and are checked against the concrete entry stage. Shared
`@func(shared=True)` code remains host/device-pure and cannot use device-only
Resource operations. Kernels and graphics entries may perform effects allowed
by their stage and parameter access modes. Host runtime allocation, uploads, dispatch,
downloads, and resource lifetime are not parsed device-language expressions.
Multi-program ordering, render-pass state, resource transitions, and
cross-backend synchronization are Runtime host-orchestration semantics,
specified by `VernonExecutionGraph` in `specs/runtime/design.md`. They are not
current language semantics.

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

A persistent `PipelineAsset` wraps exactly one executable pipeline. Its
`program=` is either one `@kernel` entry or a non-empty tuple containing only
graphics entries. Kernel programs are compute-only; graphics stage tuples form
graphics-only Pipelines. Compute and graphics entries cannot be mixed in one
pipeline.

Each graphics entry carries an explicit stage kind. A target-independent stage
registry validates tuple topology and ordering. The registered graphics
topology is `vertex -> fragment`. Future stage additions can extend topology
validation without changing `PipelineAsset` syntax; such changes are covered by
`COMPILER_CONTRACT_VERSION`.

Generated builtin functions are the preferred authoring API.
`builtin("...")` remains a low-level entry-interface annotation and uses the
same closed stage/direction/type registry. `vd.feature` is the single
compile-time specialization mechanism for program code. `PipelineAsset`
explicitly enumerates accepted canonical feature combinations through
`variants=`; there is no independent public shader-variant selector. Feature
values and `When` branches are compile-time specialization inputs, not runtime
Python conditions. Interface locations are assigned deterministically before
feature pruning so disabled fields retain stable reservations.

Imported DSL declarations are loaded through the source module graph without
executing Python. Captured constants must belong to the deterministic
constant-expression subset and are part of specialization identity. Struct
methods normalize to ordinary typed helper calls before reachability analysis.

Project processing order is:

1. load and validate source imports;
2. bind captured constants and specialize features;
3. normalize Struct methods and generated builtins;
4. prune to selected entries and validate the call graph;
5. infer, specialize, and validate typed semantic nodes and effects;
6. lower typed nodes;
7. validate target capabilities and materialize artifacts.

Semantic cache identity includes `COMPILER_CONTRACT_VERSION`,
`PIPELINE_VERSION`, all source dependency
digests, entry, enabled features, concrete shapes and interfaces, captured
constants, workgroup size, helper specializations, and derivative-transform
identity. Diagnostics include source path, one-based line and column, and a
stable reason string.

## 7. Stable syntax target and deferred ergonomics

The current core statement subset is `pass`, expression statements, simple local or
indexed assignment, annotated assignment, augmented assignment, `return`,
`if`, `range` loops, `while`, `break`, and `continue`. Return may terminate a
nested structured region; break and continue target the nearest enclosing loop.

The expression subset includes names, numeric and Boolean literals, arithmetic,
unary operations, one comparison, calls, supported attributes, indexing,
Tensor/Tuple/Struct construction, and constant Tuple indexing.

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

The complete normative design is [`../autodiff.md`](../autodiff.md).
Autodiff transforms specialized, validated typed IR and never executes Python
to trace a function.

The initial public reverse-mode surface is one transform:
`vd.ad.vjp(program, wrt=..., rules=...)`. It produces a program whose execution
returns primal outputs and a pullback. Applying the pullback to explicit output
cotangents returns gradients for the canonical input paths selected by `wrt`.
One floating Scalar output may omit its cotangent and uses seed `1`.
Tensor/aggregate outputs may not omit cotangents and are never implicitly
reduced.

Authored Kernel and graphics entry signatures do not change. The language has
no `grad_or_not`, `requires_grad`, implicit `.grad`, global gradient clearing,
or context that silently executes backward work. `pipeline_asset()` remains the
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
use versioned built-in VJP rules. Graphics pipelines require a named versioned
custom-rule set for rasterization, visibility, depth, blending, and texture
sampling. Missing rules are errors. Texture rules may differentiate
coordinates and texel data explicitly exposed as Storage; Texture handles and
sampler state remain non-differentiable.

`VernonExecutionGraph` remains host orchestration. Applying `vd.ad.vjp` to an
ExecutionGraph composes already cooked node VJP profiles into a reverse
execution plan; it does not turn the graph into another shader asset.

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
| Effects | Typed read/write records | Region-aware reads/writes, relaxed i32/u32 atomics, and typed barriers | Additional atomic types/orderings and full race model |
| Autodiff | Not implemented | First-order VJP over typed programs, stateful Kernels, and versioned graphics custom rules | JVP, full Jacobians, convenience aliases, and higher-order AD |
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
- first-order transforms, custom rules, rejection of higher-order transforms,
  and non-differentiable boundaries have deterministic tests;
- v3 migration diagnostics and compatibility aliases are versioned;
- unsupported target combinations fail explicitly;
- cache identity and generated symbols remain deterministic;
- fluid simulation, PBR/SSR, and first-order autodiff examples compile within
  their stated boundaries.

Until every gate in this document passes, unchecked sections remain unavailable
as end-to-end guarantees even though the frontend reports the current compiler
contract version.
