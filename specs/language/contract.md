# Vernon DSL language v4 draft

> **Status: normative target, not implemented.**
>
> The compiler and Python frontend remain language version 3. The implemented
> v3 contract remains available in Git history. No syntax or behavior in this
> document is available merely because it is specified here. Frontend version 4
> may be selected only after the acceptance gates in this document pass.

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

`TensorView[T, rank, access]` is a non-owning Storage handle used by kernel
parameters, slices, and field projections. `access` is `read`, `write`, or
`read_write`. A view descriptor has:

- a logical shape of `rank` runtime extents;
- one signed stride per dimension;
- an offset into its owner;
- an owner and lifetime;
- an address space and access mode.

The source annotation carries rank and access, while layout is runtime
metadata. A host-side view is constructed explicitly, for example:

```python
positions = storage.view(
    shape=(vertex_count, 3),
    strides=(8, 1),
    offset=0,
    access="read",
)
```

The corresponding kernel parameter is annotated
`TensorView[f32, 2, read]`. Shape, strides, and offset are not generic type
arguments and therefore do not create distinct source types.

`T` may be any recursively ABI-stable Value element: Scalar, fixed Tensor,
Tuple, or Struct. Aggregate elements retain their canonical product layout.
Backend lowering may expand them into Scalar leaf descriptors over the same
owner allocation, but that expansion is not observable in source typing,
ownership, access, or dispatch semantics.

An implementation may specialize a compute artifact for a concrete runtime
layout. In that case shape, signed element strides, and offset are semantic
cache inputs, and every logical index is projected to the owner's physical
element index before backend lowering. Specialization must not expose those
values as source type arguments.

Language-level typed shape, strides, and offset use units of the recursively
resolved leaf element. Runtime descriptors and reflection additionally record
byte strides and byte offsets after Tensor and Struct layout is resolved.
External APIs must state whether supplied layout values are element or byte
units; implicit unit conversion is forbidden.

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

Loading through a view materializes the logical Value type independent of
physical layout. Storing performs the inverse projection:

```text
TensorView load  : TensorView[T, rank, read] -> Tensor[T, logical_shape]
TensorView store : TensorView[T, rank, write] x Tensor[T, logical_shape] -> ()
```

The notation describes semantics, not a promise that a whole dynamic view is
copied at once; scalar and sub-Tensor indexing obey the same rule.

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
`vd.interop.RawBuffer`. Typed backend-facing IR uses `!vernon.tensor_view`;
only opaque runtime allocation handles retain device-buffer terminology.

The v4 baseline does not claim byte-address device operations, unsized
trailing-array layouts, or CUDA/Vulkan external-memory import. Those require
separate ownership, synchronization, bounds, and lifetime contracts before
they become supported interop features.

### 3.5 Dense-only core

Core v4 Storage is dense. V4 has no SNode/layout tree, transparent sparse
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

V4 initially may reject atomics and barriers, but accepted programs and typed
IR reserve these effect categories. Read/write effects include projected
regions where statically known. Unknown overlap is conservatively aliasing.

Pure `@func` code consumes and produces Values and has no Storage or Resource
effects. Kernels and graphics entries may perform effects allowed by their
stage and parameter access modes. Host runtime allocation, uploads, dispatch,
downloads, and resource lifetime are not parsed device-language expressions.
Multi-program ordering, render-pass state, resource transitions, and
cross-backend synchronization require a future host orchestration contract.
They are not language-v4 semantics. The archived proposal in
`specs/backup/execution_graph_design.md` is non-normative.

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

A persistent `PipelineAsset` wraps exactly one executable pipeline. Its
`program=` is either one `@kernel` entry or a non-empty tuple containing only
graphics entries. Kernel programs are compute-only; graphics stage tuples form
graphics-only Pipelines. Compute and graphics entries cannot be mixed in one
pipeline.

Each graphics entry carries an explicit stage kind. A versioned,
target-independent stage registry validates tuple topology and ordering.
Vertex-plus-fragment is the currently implemented topology, not a permanent
language limit. Future registered graphics stages do not require a new
PipelineAsset shape. A language-valid topology may still fail target capability
validation when a backend has not implemented it.

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

Semantic cache identity includes frontend version, all source dependency
digests, entry, enabled features, concrete shapes and interfaces, captured
constants, workgroup size, helper specializations, and derivative-transform
identity. Diagnostics include source path, one-based line and column, and a
stable reason string.

## 7. Stable syntax target and deferred ergonomics

The v4 core statement subset is `pass`, expression statements, simple local or
indexed assignment, annotated assignment, augmented assignment, `return`,
`if`, `range` loops, `while`, `break`, and `continue`. Return may terminate a
nested structured region; break and continue target the nearest enclosing loop.

The expression subset includes names, numeric and Boolean literals, arithmetic,
unary operations, one comparison, calls, supported attributes, indexing,
Tensor/Tuple/Struct construction, and constant Tuple indexing.

Tuple destructuring, short-circuit `and`/`or`, conditional expressions,
dynamic `range`, `break`, `continue`, and nested/early return are added by v4.
Chained comparisons, recursion, dynamic allocation, exceptions, generators,
arbitrary classes, Python list/dict semantics, and Python object mutation
remain deferred.

`range(start, stop, step)` evaluates its arguments once before entering the
loop. Integer literals and dynamic `i32` bounds are accepted; `u32` values
require an explicit cast. Positive and negative steps follow Python's exclusive
stop semantics. A literal zero step is a compile error. A dynamic zero step is
a runtime contract violation and must never silently execute zero iterations.
Fixed-width overflow must terminate the range rather than wrap into an
unbounded loop. A target without a legal contract-violation mechanism must
reject dynamic step values during capability validation. Phase 5B control flow
is outside the accepted autodiff domain until separate derivative and tape
policies are specified.

Frontend acceptance guarantees well-typed Vernon IR, not that every target
implements every operation. Unsupported target/type/stage combinations fail
explicitly and never silently narrow or change semantics.

## 8. First-order autodiff

Autodiff transforms specialized, validated typed IR; it never executes Python
to trace a function. V4 defines:

- `jvp(f, primals, tangents)` for first-order forward mode;
- `vjp(f, primals)` for a primal result and first-order pullback;
- `grad(f)` for scalar-output reverse-mode gradients;
- `value_and_grad(f)` for a primal value and gradient;
- `stop_gradient(value)` as an explicit zero-tangent boundary;
- versioned custom JVP and VJP rules with validated primal, tangent, and
  adjoint signatures.

The future autodiff graph is a compiler-internal `ProgramGraph` for one
specialized Kernel or graphics Pipeline program. Its nodes and edges represent
typed value flow, control flow, Storage effects, and differentiation
dependencies inside that program. It is not a host orchestration graph and
does not order PipelineAssets, dispatches, render passes, or backend
transitions.

Floating-point Scalar leaves are differentiable. Tensor, Tuple, and Struct
Values derive tangent and adjoint structure recursively from their leaves.
Integer, Boolean, Storage, Resource, sampler, and opaque leaves are
non-differentiable unless a custom operation rule explicitly handles them.
Derivative Values have ordinary deterministic Value types; dual numbers are
not embedded into Tensor element types.

`TensorStorage.grad` denotes separately allocated companion storage managed by
the runtime. It is never an autodiff object embedded in a Tensor dtype and
never changes primal storage identity.

The initial accepted domain is pure, non-recursive `@func` code with validated
numeric operations and structured first-order control flow explicitly covered
by derivative rules. Arithmetic, casts, Tensor construction, `matmul`, and
supported math intrinsics must define behavior at non-differentiable points.
Analytical results are checked against finite differences and supported
backends are compared with CPU reference behavior.

Nested or higher-order transforms are rejected in v4. This includes
`grad(grad(f))`, Hessians, Hessian-vector products, and differentiating a
generated pullback. Generated derivative IR remains typed so a future language
version may lift this restriction without changing the Value/Storage model.

Stateful-kernel autodiff is not part of the initial v4 implementation. It
requires all of the following before acceptance:

- functionalization of local mutation and Storage writes;
- alias and race validation;
- gather/scatter adjoints and deterministic or atomic accumulation;
- branch/loop tape layout, bounded-loop rules, checkpointing, and
  recomputation policy;
- explicit primal and gradient storage bindings;
- a typed program-internal graph capable of effect-preserving reverse traversal
  and tape planning.

Texture sampling requires custom gradient rules that distinguish coordinate,
texel, and sampler inputs. Sampler state is non-differentiable. Rasterization,
visibility, depth tests, blending decisions, and discontinuous material
branches are non-differentiable unless explicit custom primitives define their
derivatives. Differentiating fragment arithmetic alone does not imply
differentiable rendering.

## 9. V3 to v4 migration

- Host runtime `Tensor` ownership becomes `TensorStorage`.
- Kernel storage parameters become `TensorView[T, rank, access]`.
- Pure fixed-shape computation values remain `Tensor[T, static_shape]`.
- Removed `Buffer` becomes a typed `TensorView` or runtime-only
  `vd.interop.RawBuffer`.
- `Array[T, N]` is not introduced; use `Tensor[T, (N,)]`.
- `Vector` and `Matrix` are rank aliases; v3 `vec`, `mat`, `vec*`, and `mat*`
  spellings are rejected.
- V3 addressability metadata becomes explicit Storage typing and access mode.
- Physical shape/stride/offset metadata moves from Tensor Value concepts to
  TensorView descriptors.

Removed v3 aliases must not create a fourth semantic category or distinct
cache identity.

## 10. Feature status

| Area | V3 implementation | V4 normative target | Deferred |
| --- | --- | --- | --- |
| Tensor | Fixed numeric aggregate; entry addressability metadata | Immutable Value with recursively ABI-stable Value elements and logical shape only | Dynamic-shape Value |
| Ownership | Runtime `Tensor`; removed `Buffer` | `TensorStorage`, borrowed `TensorView`, runtime-only `vd.interop.RawBuffer` escape hatch | General allocator model |
| Layout | Backend/runtime details | Dense strided views, AoS field projections, explicit alias rules | Transparent sparse layouts and SNode trees |
| Aggregates | Nominal immutable Struct; Vector/Matrix constructors | Tensor, structural Tuple, nominal Struct; no Array | Enums and tagged unions |
| Effects | Typed read/write records | Region-aware read/write boundary; reserved atomic/barrier effects | Full memory-order model |
| Autodiff | Not implemented | First-order pure typed-IR JVP/VJP/grad | Higher-order and stateful-kernel AD |
| Control flow | Restricted structured subset | V3 subset plus Tuple destructuring and AD-covered flow | Early return, dynamic range, break/continue, unrestricted recursion |
| Rendering | Typed graphics stages, textures, samplers | Same model with explicit Resource and derivative boundaries | General differentiable rasterization |

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

## 12. V4 acceptance gates

Frontend version 4 may be declared implemented only when:

- every public term in this document maps to one typed semantic category;
- Tensor element legality, nested normalization, and Struct boundaries have
  parser, inference, ABI, cache-identity, and backend tests;
- TensorStorage ownership, TensorView layout units, lifetime, projection,
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

Until every gate passes, tools and artifacts must identify the frontend as
language version 3.
