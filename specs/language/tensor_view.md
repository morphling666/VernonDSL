# Tensor and TensorView contract

## Status

This document records the accepted Tensor-family design and the in-progress v4
migration of the TensorView shape model and workgroup storage model. The old
`workgroup_array` and `!vernon.workgroup` forms are removed directly; no
aliases, parser fallbacks, IR translations, reflection readers, or runtime
compatibility paths are added.

The released Python frontend remains language version 3 while this v4 design is
implemented and validated; the current code has no numeric `FRONTEND_VERSION`
constant. Source syntax, typed IR, and compile-time tests cover unified
TensorView load/store/atomic operations, workgroup address space, and
`workgroup_storage`. The 0.1.2 Runtime descriptor ABI accepts dynamic shape,
signed stride, and offset as invocation data without layout-specific
recompilation. Broader language-v4 parity and validation remain tracked in
the [project roadmap](../roadmap.md#language-v4). Serialized reflection,
pipeline, and invocation ABI versions are bumped wherever this migration
changes their records.

## 1. Tensor family and semantic categories

The public names deliberately form one Tensor family:

```text
Tensor[T, shape]                  immutable Tensor Value
TensorView[T, shape, access]      non-owning Tensor-shaped Storage borrow
TensorStorage[T]                  host owner, not a device-language type
```

The shared `Tensor` name communicates that both device types have element,
shape, rank, and indexing. The `View` suffix communicates that a TensorView is
not a Value and does not own memory.

The semantic distinction remains strict:

```text
Value
├── bool, i32, u32, f16, f32, f64
├── Tensor
├── Tuple
└── Struct

Storage
└── TensorView

Resource
├── Texture
└── Sampler
```

`builtin()`, `attribute()`, `uniform()`, and `resource()` are entry-interface
metadata, not types.

```python
a = weights[0]  # pure Tensor Value extraction
b = values[0]   # TensorView Storage load effect
```

## 2. Source forms

`Tensor` is fixed-shape because it is an immutable Value:

```python
Tensor[f32, (4, 4)]
```

The canonical Storage annotation is:

```python
TensorView[element_type, shape, access]
```

Examples:

```python
TensorView[f32, (), read_write]
TensorView[f32, (4, 4), read]
TensorView[f32, (vd.dyn, 4), read]
TensorView[Particle, (vd.dyn,), read_write]
```

`access` is `read`, `write`, or `read_write`.

`vd.dyn` is one immutable type-level marker for one runtime-resolved
TensorView dimension. It is not callable. Every static extent is a positive
integer; a runtime extent resolving `vd.dyn` is non-negative. Rank is explicit
because every dimension occupies one shape entry.

The empty shape `()` is the canonical rank-zero form. It models one scalar
storage cell rather than an empty collection. Device code loads and stores it
with `view[()]`; host code uses the same `view[()]` spelling or NumPy's 0-d
`array[()]`/`item()` APIs. Its runtime descriptor has rank zero, null
shape/stride arrays, and still carries the storage offset. Rank-zero
`workgroup_storage` remains invalid because workgroup allocation requires an
explicit positive physical extent.

Shape is part of the source TensorView contract. Strides and offset are
concrete view-layout metadata and do not participate in core source type
identity. Runtime binding verifies each static extent and resolves each
`vd.dyn` extent from the supplied view descriptor.

## 3. Address-space inference

Address space is part of the compiler's concrete semantic TensorView type but
is not written in ordinary source:

```text
kernel TensorView parameter -> device
workgroup_storage result    -> workgroup
future private constructor  -> private
```

Users cannot forge address-space strings. Uniform and host-visible interfaces
have separate mutability and binding rules; they are not alternate spellings
for device Storage.

The internal descriptor is conceptually:

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

## 4. Host binding and graphics interfaces

Direct compute binding is strict:

```text
Tensor parameter
  <- compatible NumPy array or immutable host Tensor Value
  -> snapshotted and packed by value
  -> never accepts TensorStorage or TensorView

TensorView parameter
  <- TensorStorage owner, its canonical full view, or an explicit subview
  -> always proves that host Storage exists
  -> borrows the owner for the complete asynchronous dispatch
  -> never accepts a raw NumPy array by implicit allocation
```

Passing a `TensorStorage` owner to a TensorView parameter normalizes it to its
canonical full view with owner shape, row-major strides, offset zero, and the
requested compatible access.

Graphics interfaces distinguish semantic type from transport:

```python
position: Annotated[Tensor[f32, (3,)], attribute()]
transform: Annotated[Tensor[f32, (4, 4)], uniform()]
particles: Annotated[
    TensorView[Particle, (vd.dyn,), read_write],
    resource(),
]
```

A host vertex buffer supplies many `position` records, but each vertex
invocation receives one immutable Tensor Value. A uniform is also a Value even
when a backend physically spills it to a buffer. Only TensorView exposes
addressable Storage operations to shader code.

## 5. Owners, subviews, and layout

An entry accepts either a `TensorStorage` owner as its full view or an explicit
subview. One row-major owner of shape `(4, 4)` can provide two disjoint `(2, 2)`
views:

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

Their indices project as:

```text
physical_index = offset + sum(index[d] * stride[d])
```

Shape, signed element strides, and element offset are dispatch descriptor
data. They do not participate in artifact specialization or cache identity.
Static source extents, bounds, internal injectivity, owner lifetime, and
overlapping read/write borrows are validated. Multiple views can share an
owner when their accessed regions are compatible.

Autodiff creates a separate host-owned tangent allocation rather than writing
through a primal owner. Scalar elements use the promoted derivative dtype.
Aggregate elements use the structural `TangentLayout` defined by
[`autodiff.md`](../autodiff.md#3-value-storage-and-resource-gradients):
Vector, Matrix, Tensor, Tuple, and Struct dimensions and paths remain element
structure, while f16 leaves promote to f32 and non-differentiable leaves become
zero tangent nodes. The tangent layout may have different offsets and stride
from the primal canonical Value ABI and must not be used to reinterpret primal
bytes.

All compatible differentiated views of one owner scatter-add into one fresh
tangent owner. Separately bound overlapping read/write views remain invalid;
one legal read-write binding is handled through explicit Storage versions.

## 6. Workgroup storage

The source constructor is:

```python
shared = vd.workgroup_storage(T, shape=(d0, d1, ...))
```

It returns a read-write TensorView whose address space is inferred as
workgroup. Rules:

- every extent is a positive compile-time integer in the source type;
- literals and captured host constants are allowed, and captured constants
  participate in cache identity;
- kernel arguments and other device-runtime values cannot determine allocation
  size;
- layout is contiguous row-major and exposes no stride or offset argument;
- every workgroup receives an independent allocation for its execution;
- all recursively ABI-stable Value elements are accepted and lower through
  canonical ABI leaves;
- the initial portable allocation limit is 16 KiB after ABI layout;
- CUDA-only dynamic shared memory is not part of the core language.

No `workgroup_array` alias remains.

## 7. Indexing, atomics, and synchronization

Ordinary TensorView indexing requires one index per rank:

```python
value = shared[y, x]
shared[y, x] = value
```

Atomic indexing follows Python indexing ergonomics:

```python
vd.atomic_add(rank1, i, value)
vd.atomic_add(rank2, (y, x), value)
vd.atomic_add(rank3, (z, y, x), value)
```

Rank one accepts a scalar index. Rank greater than one requires a tuple with
exactly one integer index per dimension. `atomic_add(storage[y, x], value)` is
invalid because the subscript denotes an ordinary load, not an lvalue
reference.

The first atomic element set remains `i32` and `u32`. Wider integers and
floating-point atomics require explicit backend capability contracts and are
not emulated implicitly.

Atomic scope is inferred from address space:

```text
workgroup TensorView -> workgroup scope
device TensorView    -> device scope
```

Atomic scope is not a public argument. Atomic ordering remains represented in
typed effects; initial public operations request `relaxed`, and a backend may
strengthen but not weaken it. Barrier scope remains explicit because a barrier
has no storage operand.

## 8. Vernon IR contract

Vernon IR preserves the public TensorView term and uses one
address-space-parameterized Storage type:

```text
!vernon.tensor_view<element, shape, access, address_space>
```

Dynamic dimensions use the canonical MLIR dynamic extent. Workgroup dimensions
must be static. Device shape, signed element strides, and element offset never
appear as concrete layout attributes; internal descriptor arguments carry them
to physical index projection.

IR has unified typed TensorView allocation, `vernon.load`, `vernon.store`, and
atomic operations with ranked indices. Workgroup-specific load/store operations
and the previous stringly intrinsic spellings are removed. Load, store, and
atomic use one checked index projection. Atomic scope is derived from the
TensorView type and is not stored independently. Barrier scope remains an
operation attribute.

Lowering maps inferred address space to target memory:

```text
CPU device       -> host memref/storage descriptor
CUDA device      -> global memory
CUDA workgroup   -> shared memory
SPIR-V device    -> StorageBuffer
SPIR-V workgroup -> Workgroup
GLSL workgroup   -> shared
HLSL workgroup   -> groupshared
```

## 9. Reflection and runtime ABI

Reflection describes canonical TensorView records with element layout, source
shape constraints, rank, access, externally visible address space, storage-leaf
bindings, and the offset/extent/stride descriptor binding sequence. Concrete
dispatch values are never reflected. The capability name remains
`tensor_views`.

Native runtime records use `VernonTensorView`; the invocation ABI version is
bumped when descriptor fields change. Reflection and pipeline schemas are
bumped together, and old schema readers are removed rather than translated.

Workgroup TensorViews are compile-time kernel state, not host-bound arguments.
Required workgroup bytes and synchronization features are reflected only where
backend/device capability validation needs them.

## 10. Acceptance tests

Acceptance requires:

- static, dynamic, and mixed TensorView shape parsing and runtime matching;
- strict Tensor versus TensorView host binding;
- full-owner and explicit subview dispatch;
- disjoint and overlapping views sharing one owner;
- graphics Tensor attributes sourced from bound vertex storage;
- multidimensional workgroup load/store and row-major projection;
- workgroup storage of Scalar, Tensor, Tuple, and Struct Value elements;
- rank-one and tuple atomic indexing with inferred scope;
- independent allocations across multiple workgroups;
- barrier-visible writes within one workgroup;
- backend-independent verifier tests plus runtime parity on every available
  CUDA, Vulkan, OpenGL, and DirectX backend;
- rejection of old workgroup source names, old split workgroup IR, and old
  serialized schemas.
