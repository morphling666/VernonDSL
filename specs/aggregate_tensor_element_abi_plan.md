# Aggregate Tensor Element ABI Implementation Plan

## Goal

Allow `Tensor` and `TensorView` element types to be any finite ABI-stable
Value, including nested `Struct`, `Tuple`, and static `Tensor` Values:

```python
@vd.struct
class Vertex:
    position: vd.Vector[vd.f32, 3]
    object_id: vd.u32
    uv: vd.Vector[vd.f16, 2]

vertices: vd.TensorView[Vertex, 1, vd.read]
weights: vd.Tensor[tuple[vd.Vector[vd.f32, 4], vd.i32], (2, 3)]
```

The logical element type must remain visible to user code and reflection while
all compiler and runtime boundaries use one canonical recursive Value ABI.
Generated names, scalar leaf bindings, and backend-specific input variables
must not become public parameter names.

## Existing foundation

- The frontend already classifies scalar, static Tensor, Tuple, and finite
  non-recursive Struct Values as ABI-stable.
- Aggregate-element Tensor Values already use `!vernon.tensor`; CPU/CUDA and
  SPIR-V lowering have aggregate reconstruction paths.
- Aggregate `TensorStorage` host packing and Struct field projection already
  use canonical Value ABI metadata.
- Vertex attributes already use explicit numeric leaves, and aggregate
  TensorView lowering already has scalar resource-leaf expansion.

## Current implementation status

### Completed

- Python and C++ now have matching recursive Value ABI and attribute planners
  for scalars, static Tensors, Tuples, and nominal Structs. They emit canonical
  sizes, alignments, field offsets, element strides, semantic leaves, paths,
  logical dtypes, attribute leaves, and layout hashes.
- The old scalar-only `StaticAttributePlan`, `element_abi_size`, unnamed Struct
  field fallback, and obsolete schema compatibility branches have been
  removed.
- The Python frontend emits canonical ABI metadata, preserves logical aggregate
  element types, allocates expanded attribute locations, and carries logical
  integer signedness separately from MLIR's signless integer storage types.
- Direct MLIR validation recomputes layouts and rejects inconsistent sizes,
  offsets, leaves, paths, dtypes, and hashes.
- CPU, CUDA/GPU, and SPIR-V lowering can construct, decompose, reconstruct,
  load, and store aggregate Values. Storage projection preserves each
  semantic leaf's scalar count, including nested static Tensor fields.
- Graphics lowering expands mixed-dtype Struct attributes into hardware leaves,
  reconstructs the source Value, and reflects one source-level parameter.
- Compiler reflection schema 4 emits canonical `element_layout`,
  `attribute_leaves`, Struct layouts, and logical dtypes. Multidimensional
  Tensor layout hashes are identical between Python and C++.
- Runtime reflection parsing requires `element_layout` instead of
  `element_abi_size`. Python canonical Struct storage packing, field
  projection, round-trip conversion, and aggregate TensorView dispatch work on
  CPU, CUDA, Vulkan, OpenGL, OpenGL ES, and DirectX in the current test
  environment.
- Pipeline manifest schema 4 and invocation ABI 5 use canonical recursive
  `element_layout` descriptors. `VernonTensorView`,
  `VernonPipelineParameterView`, native Python bindings, launch planners, and
  `tensor_bridge` no longer depend on a scalar Tensor dtype.
- OpenGL, OpenGL ES, Vulkan, and DirectX render the same nested Struct vertex
  attribute containing mixed scalar dtypes and static Tensor fields while
  reading `Tensor[ComplexAggregateVertex, (2, 3, 4)]` through `vd.uniform()`.
  Descriptor-backed aggregate uniforms use the canonical Value ABI as a
  read-only graphics storage resource, avoiding backend-specific uniform
  repacking. Language coverage includes multidimensional aggregate Tensor
  indexing and field projection.
- The public C++ `PipelineInvocationBuilder` resolves reflected field/index
  paths under one source parameter name and packs structured Values, Tensor
  structure-of-arrays payloads, and logical-element callbacks into
  runtime-owned canonical storage. It validates dtypes, scalar static-Tensor
  shapes, outer Tensor shapes, duplicate bindings, and completeness without
  reinterpreting native C++ Struct padding.
- OpenGL, OpenGL ES, Vulkan, and DirectX use one deterministic expanded
  attribute capability policy before backend-specific device format checks.
  The test matrix covers every supported aggregate leaf dtype/component
  family, bool/u8 rejection, backend f16/f64 restrictions, invalid component
  counts, and expanded location overflow.
- Current verification passes:
  - 289 Python tests and 297 subtests;
  - all 86 configured CTest tests, including compiler, runtime, backend, and
    integration coverage.

## Required semantics

### Type and operation rules

- An element is accepted only when `is_abi_stable_value` proves a finite,
  deterministic layout. Recursive Structs, runtime-sized members, Storage, and
  Resource members are rejected.
- Nested static Tensors compose logical shape in row-major order. Structs stay
  nominal and Tuples stay positional.
- Full Tensor indexing returns one aggregate element. Struct field access and
  Tuple indexing operate on that element without exposing ABI leaves.
- Numeric Tensor operations (`+`, `-`, `*`, `/`, broadcasting, and `matmul`)
  continue to require a scalar numeric element type. They must not implicitly
  map over Struct or Tuple fields. Users explicitly select a numeric field
  before applying numeric operations.
- Aggregate Values remain valid for construction, extraction, assignment,
  function arguments/results, TensorView load/store, and stage interfaces.

### Canonical Value ABI

Use one recursive layout algorithm for Python, MLIR/compiler, reflection, and
runtime:

- Scalars use their existing size and natural alignment.
- Static Tensor elements are contiguous row-major with aligned element stride.
- Tuple fields use declaration order, natural field alignment, and final
  product alignment.
- Struct fields use declaration order and the same product-layout rule.
- Padding bytes are part of the physical ABI but never become semantic leaves.
- The layout produces:
  - total byte size and alignment;
  - ordered numeric leaves;
  - each leaf's field/index path, scalar dtype, scalar byte offset, and count;
  - a canonical layout hash derived from type structure, offsets, sizes, and
    dtypes, never from source aliases or backend representation.

For vertex/instance attributes, partition each numeric leaf independently into
hardware leaves of at most four components and at most 128 bits. A hardware
leaf may not cross a Struct/Tuple field boundary, padding, Tensor element
boundary, or dtype boundary. `bool` remains ABI-stable for storage but is
rejected as a vertex format.

## Phase 1: Canonical recursive planners

### Python planner

Extend `python/vernon_dsl/frontend/abi.py`:

- Replace the scalar-only `attribute_layout(dtype, shape)` contract with a
  recursive planner accepting a concrete Value type and a Struct-field
  resolver.
- Add immutable descriptors for Value leaves and hardware attribute leaves.
- Record logical paths using field names for Structs and integer indices for
  Tuples/static Tensors.
- Keep `value_abi_layout` and attribute planning backed by one recursive walk;
  do not maintain separate size/alignment algorithms.
- Reject recursive or unresolved nominal Structs deterministically.

Update `python/vernon_dsl/module_graph.py` so pre-feature-pruning location
allocation resolves Struct/Tuple element types and uses this planner. Remove
the remaining assumption that an attribute has one scalar dtype.

### C++ planner

Move the current numeric attribute planning out of
`VernonTensorShapeSemantics` into a dedicated
`VernonValueAbi.{h,cpp}`/`VernonAttributeAbi.{h,cpp}` implementation under
`source/include/mlir/Dialect/Vernon/Transforms/` and
`source/lib/Dialect/Vernon/Transforms/`.

- Resolve nominal Struct declarations from the containing module.
- Recursively plan Vernon Tensor, Tuple, Struct, and scalar types.
- Return the same size, alignment, paths, offsets, dtypes, and attribute
  location spans as Python.
- Detect overflow, invalid extents, unresolved Structs, and recursion.
- Delete the old scalar-only attribute planner after all callers migrate.

Add differential fixtures shared by Python and C++ covering mixed dtypes,
padding, nested Tuples, nested static Tensors, and Structs.

## Phase 2: Frontend and Vernon IR normalization

Update:

- `python/vernon_dsl/frontend/type_parser.py`
- `python/vernon_dsl/frontend/inference.py`
- `python/vernon_dsl/frontend/lowering.py`
- Vernon type verification and validation

Required changes:

- Emit `vernon.dtype` only for scalar-element Tensors. Aggregate elements emit
  canonical element ABI metadata and their logical type identity.
- Ensure Tensor construction/extraction and Struct/Tuple projection preserve
  the aggregate element type.
- Validate that numeric operators and `matmul` reject aggregate elements with
  a clear source diagnostic.
- Validate direct MLIR by recomputing the canonical layout instead of trusting
  frontend offsets, hashes, or leaf metadata.
- Verify source-assigned vertex location ranges against the recursively planned
  hardware leaves.

## Phase 3: Compiler lowering and reflection

### CPU/CUDA and SPIR-V Values

Audit and consolidate aggregate paths in:

- `VernonLowerCPUTensors.cpp`
- `VernonLowerGPUTensors.cpp`
- `VernonSharedValuePatterns.cpp`
- `VernonToGPU.cpp`
- `VernonToSpirv.cpp`

Use the canonical planner for all aggregate construction, extraction, dynamic
selection, load/store decomposition, and reconstruction. Remove duplicate
rank-specific or Struct-specific flattening algorithms.

CUDA keeps its current restriction on static Tensor-by-value entry arguments
until a separate launch-parameter ABI is specified; aggregate TensorView
elements must still work through storage leaf expansion.

### Graphics attributes

For each aggregate vertex/instance source parameter:

- emit one scalar/vector shader input per planned hardware leaf;
- assign consecutive locations from the source base location;
- use each leaf's own dtype and ABI byte offset;
- reconstruct nested Tensor/Tuple/Struct Values in the entry prologue;
- retain one source-level parameter in reflection;
- never expose generated leaf variable names.

### Reflection

Bump compiler reflection schema and emit one `element_layout` object for every
Tensor/TensorView:

- logical element type and nominal Struct name where applicable;
- size, alignment, and canonical layout hash;
- ordered semantic leaves with paths and byte offsets;
- ordered attribute leaves with locations, dtypes, component counts, and byte
  offsets when used as a vertex/instance input.

Remove scalar-only `dtype` assumptions from aggregate parameters. Scalar
parameters may still serialize through the same one-leaf `element_layout`.

## Phase 4: Public runtime and manifest ABI

Perform a hard schema/ABI cutover. Update all in-tree producers and consumers
together; do not add legacy readers or infer aggregate layouts from shape.

### Manifest

Bump the pipeline manifest schema and replace the Tensor parameter's single
dtype contract with the canonical `element_layout`. Preserve:

- `parameter.name` as the public source binding name;
- generated `uniform_name`, resource leaves, and attribute variable names only
  inside backend-use records.

Python keyword binding and C++ parameter lookup must continue to use names such
as `"vertices"` or `"transform"`, independent of generated suffixes like
`"transform._m0"`.

### C runtime descriptor

`VernonTensorView` currently has one `VernonDataType`, so it cannot describe a
mixed-dtype element. Bump the pipeline invocation ABI and replace that field
with a uniform Value-layout view:

```c
typedef struct VernonValueLeafView {
    uint32_t dtype;
    uint32_t scalar_count;
    uint32_t byte_offset;
} VernonValueLeafView;

typedef struct VernonValueLayoutView {
    uint32_t struct_size;
    uint32_t byte_size;
    uint32_t alignment;
    VernonStringView layout_hash;
    const VernonValueLeafView *leaves;
    size_t leaf_count;
} VernonValueLayoutView;
```

All scalar Tensor views use a predefined one-leaf layout; avoid a separate
scalar-vs-aggregate invocation path. Runtime validation compares the supplied
layout hash and full leaf contract against reflection.

Update `tensor_bridge` to:

- use element byte size rather than scalar dtype size for bounds and strides;
- copy a complete aggregate element, including canonical padding;
- pack/unpack leaves without reinterpretation or narrowing;
- validate arbitrary positive leading record stride and canonical inner
  element layout.

### Python runtime

Connect `TensorStorage`'s existing canonical packer and element type metadata to
the native `VernonValueLayoutView`. Tuple/Struct storage must bind by the source
parameter name exactly like scalar storage.

### C++ structured binding

Do not make direct native Struct binding the default C++ API. Even a
standard-layout C++ type is not a portable declaration of the Vernon canonical
ABI: compiler, target, packing pragmas, member types, and alignment rules may
change its padding and offsets.

Keep one public parameter name and pass a structured field tree as its payload:

```cpp
builder.bindValue(
    "value",
    fields(
        field("a", int32_t{2}),
        field("b", 3.0f)
    )
);
```

The runtime matches `a` and `b` to reflection paths below `"value"` and writes
them at canonical offsets. Field names are data inside one binding operation;
`"value.a"` is not a separately registered public parameter.

For Tensor parameters, support SoA field binding:

```cpp
builder.bindTensor(
    "t",
    shape{2, 2},
    fields(
        field("a", std::span<const int32_t>{a}),
        field("b", std::span<const float>{b})
    )
);
```

Each field span covers the complete outer Tensor domain. Nested Struct fields,
Tuples, and static Tensor fields use nested `fields(...)`, positional
`elements(...)`, and shaped scalar views rather than one binding call per
runtime Tensor element.

Also support a packing callback when values are generated or must be gathered
from an application-specific representation:

```cpp
builder.bindTensor("t", shape{2, 2}, [](auto element) {
    element.field("a", /* scalar value for this logical element */);
    element.field("b", /* scalar value for this logical element */);
});
```

The callback iterates the logical Tensor domain and fills one structured
element at a time into runtime-owned canonical storage. It does not create
public names such as `"t[0, 0].a"` and does not register one binding per
element.

Both forms validate reflected field/tuple paths, scalar dtypes, static Tensor
shapes, outer Tensor shape, duplicate fields, and completeness. The runtime
packs canonical bytes and padding from reflection. Backend-generated names such
as `"t.a"`, storage-leaf bindings, and shader variables remain internal
implementation details.

An explicitly specialized native `std::span<T>` overload may be offered as an
optional zero-copy fast path. It is accepted only when its descriptor validates
`std::is_standard_layout_v<T>`, `sizeof(T)`, `alignof(T)`, every reflected
field's `offsetof`, nested member layout, and scalar dtype against the complete
canonical layout. A total-size match alone is insufficient. If validation
fails, callers use structured binding or explicit canonical packing; the
runtime must not silently reinterpret the native Struct.

The C API remains available for callers that already own canonical bytes and
provide an explicit `VernonValueLayoutView`.

## Phase 5: Backend consumption

Keep one attribute-leaf iteration path in each adapter:

- OpenGL selects floating, integer, or long pointer APIs per leaf.
- Vulkan and DirectX map each `(dtype, component_count)` through data-driven
  format tables.
- Instance divisors affect fetch rate only.
- Base buffer offset, record stride, and leaf-relative byte offset remain
  independent.
- Device format and location limits are checked after recursive expansion.
- Unsupported leaves, such as `f64` on DirectX or `bool` everywhere, produce a
  deterministic source-parameter/path diagnostic.

Uniform/storage buffer backends consume canonical byte offsets and must not
derive aggregate layout from rank or shape.

## Phase 6: Verification

### Planner and language tests

- Python/C++ differential cases for:
  - `Tensor[Tuple[f32, i32], (2, 3)]`;
  - Structs with padding and mixed `f16/f32/f64/i32/u32`;
  - nested Struct -> Tuple -> static Tensor;
  - nested Tensor shape composition;
  - recursive Struct and non-ABI member rejection.
- Numeric operators and `matmul` reject aggregate-element Tensors.
- Explicit field projection followed by numeric operations succeeds.

### Compiler tests

- Construction, extraction, field projection, function calls, and returns.
- CPU/CUDA/SPIR-V aggregate reconstruction.
- Dynamic extraction over SPIR-V aggregate arrays.
- Reflection preserves one source parameter and exact semantic/attribute
  leaves.
- Automatic location allocation and explicit overlap rejection use expanded
  aggregate spans.

### Runtime tests

- Python `TensorStorage` round-trip for Struct/Tuple elements.
- C++ structured Value binding, Tensor SoA binding, callback packing, optional
  validated native fast path, and field/type/shape mismatch diagnostics.
- Binding continues to use source names in Python and C++ when generated
  uniforms/resources use internal names.
- Ordinary, offset, interleaved, and padded record storage.
- TensorView read/write for aggregate elements on CPU, CUDA, Vulkan, OpenGL,
  OpenGL ES, and DirectX where available.

### Graphics tests

- Mixed-dtype Struct attributes with field use in vertex shaders.
- Tuple and nested static Tensor attributes.
- Per-vertex and `divisor > 1` instance inputs.
- Non-square numeric fields and attributes spanning multiple locations.
- OpenGL, OpenGL ES, Vulkan, and DirectX rendering parity.
- Expected capability rejection for unsupported dtype/format combinations and
  oversized location spans.

Run the complete Python and C++ suites after focused tests. Update generated
fixtures/assets to the new schemas in the same change.

## Completion criteria

- Every accepted ABI-stable Tensor element has identical Python, compiler,
  reflection, and runtime layout.
- No public API requires generated leaf or uniform names.
- C++ aggregate binding keeps one public parameter name, resolves its
  structured payload against reflection, and never assumes a native Struct has
  the canonical Vernon layout.
- No backend derives layout from Tensor rank, Matrix aliases, or a single
  parameter dtype.
- Aggregate storage, static Values, and graphics attributes use the same
  recursive layout and hash.
- All old scalar-only aggregate metadata and compatibility branches are
  deleted.
- Focused differential/backend tests and the complete project test suites pass.
