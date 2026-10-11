# Graphics storage reads

Status: accepted, `program = 1` additive extension.

Vertex and fragment stages may declare device `TensorView` parameters and read
them with ordinary TensorView indexing. This is the contract behind
storage-buffer vertex pulling (fetch functions that load vertex records by
`vertex_index`), bone palettes indexed by per-vertex joint indices, and
material or instance tables indexed by a row carried in a uniform, varying, or
draw command.

It complements, and does not replace, the existing graphics interfaces:

- `attribute()` Tensor parameters still receive one immutable record per
  vertex from a host vertex buffer;
- `uniform()` parameters remain Values, even when a backend spills them to a
  buffer;
- only a `TensorView` parameter exposes addressable Storage to shader code.

## 1. Source rules

```python
@vd.vertex
def fetch(
    positions: vd.TensorView[vd.Vector[vd.f32, 3], (vd.dyn,), vd.read],
    vertex_index: Annotated[vd.u32, vd.builtin("vertex_index")],
) -> Annotated[vd.Vector[vd.f32, 4], vd.builtin("position")]:
    p = positions[vertex_index]
    return vd.Vector([p.x, p.y, p.z, 1.0])
```

- A graphics TensorView parameter has `read` access. `write`, `read_write`,
  atomics, and `workgroup_storage` are rejected in vertex and fragment stages.
- Explicit `resource(set=..., binding=...)` on a graphics TensorView is
  rejected; its physical locations are always compiler-generated (section 3).
- Element types, static and `vd.dyn` extents, subviews, and indexing follow
  [`language/tensor_view.md`](../language/tensor_view.md) unchanged. Out of
  bounds reads are undefined, as in compute.
- The same Program Value may be bound to TensorView parameters of both stages.

## 2. Per-stage ownership

Each module role owns its own physical interface. A graphics module with at
least one device TensorView parameter owns:

- one storage-buffer binding per storage leaf of each TensorView parameter
  (the leaf split is the compute ValueAbi storage layout: every leaf binds the
  whole record buffer, viewed as an array of that leaf's scalar type);
- exactly one metadata carrier with the TensorView offset, extents, and
  strides of that module's TensorView parameters, in the
  `portable_shader_metadata_i32` profile.

A Program parameter read by both stages therefore has two reflected resource
endpoints and fields in both carriers. Its storage leaves share one set of
bindings (section 3), so Runtime binds the buffer once per descriptor and
writes the same descriptor fields into each carrier.

Per-stage carriers were chosen over a single pipeline-wide carrier because a
stage can be reused with a different partner stage, and a carrier in one stage
never forces a field into the other.

## 3. Binding allocation

All generated graphics bindings use descriptor set 0, and the compiler is the
only component that generates them: the Python frontend never invents a
binding for a vertex or fragment parameter. Bindings are assigned for the
pipeline as a whole, before any of its modules is compiled:

1. *Collect.* Each module lists its generated-binding requests in argument
   order: a texture without an authored `resource(set=..., binding=...)` and
   an implicit sampler take one binding each; a uniform spilled to a buffer
   takes one; a TensorView parameter takes one per storage leaf, in leaf
   order; the module's metadata carrier takes one, after all arguments. An
   implicit sampler is listed right after the texture it samples. Each
   request carries the parameter's pipeline name, kind, type, and outermost
   binding group, if any. Authored bindings are listed separately.

   A Struct, Tuple, or Tensor uniform always spills to a buffer, because
   OpenGL, OpenGL ES, and Metal have no native inline uniform for aggregates.
   Scalar and Vector uniforms stay inline while they fit the 128-byte inline
   budget.
2. *Link* (`vernonCompilerLinkGraphicsBindings`). Every authored set-0 binding
   of every module is reserved. Requests are then assigned in module order and,
   within a module, in request order. A request with the same pipeline name,
   kind, type, and count as a request of an earlier module reuses that
   request's bindings, since a pipeline keyword names one Program Value. Every
   other request, including every metadata carrier, takes the lowest free
   bindings.

   Requests of one binding group occurrence are allocated together instead:
   the link forms the union of the group's requests over all modules, in
   field order (a request seen only in a later module, such as an implicit
   sampler of one stage, joins after the request that precedes it there),
   and assigns the whole union the lowest free consecutive range. Each module
   takes its own requests' bindings from that range, so both stages agree on
   every leaf even when one stage uses only some of them.
3. *Compile.* Each module is compiled with its own layout from the link. The
   compiler checks that the layout covers exactly that module's requests,
   writes texture and sampler bindings onto their arguments before reflection
   and lowering, and uses the same layout for reflection and SPIR-V, so shader
   and reflection binding numbers cannot diverge.

A module compiled without a layout is linked as a one-module pipeline. The
layout is a compile-request input, not a target option: it does not enter
target identity or the runtime target-option vocabulary.

Link-time sharing replaced an earlier scheme where each module continued
after the previous module's highest binding. That scheme depended on compile
order and gave a value read by both stages two descriptors.

A texture must declare both set and binding or neither. Explicit bindings are
authored per stage. The link rejects two modules' authored endpoints that share
a `(set, binding)` under different pipeline names, and Program finalization
rejects any `(set, binding)` shared by different Program Values; both
diagnostics name both endpoints. Leaving graphics bindings unassigned avoids
these failures.

Backends that build one descriptor layout per pipeline merge a binding that
several stages use into one descriptor visible to each of those stages.

An explicit `Sampler` parameter binds through the images it samples. A stage
that declares one but never samples with it, which happens when a stage uses
only part of a binding group, contributes no sampler slot in that stage. An
implicit sampler always pairs with its texture.

The OpenGL-family Runtime binds named uniforms and textures by the shader's
sanitized interface name (`material.albedo` becomes `material_albedo`). The
compiler and Runtime share one sanitizer (`runtime/interface_names.h`), so the
two spellings cannot diverge.

## 4. Manifest representation

The normative field definitions are in
[`execution_manifest.md`](execution_manifest.md) section 11.1; this section
summarizes them.

Graphics TensorViews use the same manifest shape as compute TensorViews;
nothing in it is specific to graphics:

- StageContract reflection carries `metadata_carriers`, keyed by module role
  (`vertex`, `fragment`), present exactly when a module owns TensorView
  metadata. Field `argument` is the argument endpoint index within that
  module. Compute uses the same member with the single key `compute`.
- The target implementation carries `metadata_carriers` with the same keys.
  Shader backends use profile `portable_shader_metadata_i32` with top-level
  `set` and `binding` for every module role.
- A TensorView endpoint is a `resource` endpoint with role `storage`, a buffer
  layout with `descriptor = true`, and one `storage_leaf` binding with its own
  `resource_slot` per leaf. Its implementation endpoint lists the native
  `storage_leaves` (`binding`, `byte_offset`, `element_size`).

Required feature `tensor_views` is set when any module declares a TensorView
parameter.

Unifying the compute `metadata_carrier` member with the graphics
`metadata_carriers` map and giving every storage leaf its own portable slot
changed the manifest schema, so `PROGRAM_VERSION` is 2. A version-1 Runtime
rejects these bundles at the version check instead of misreading them.

## 5. Runtime binding

- Every storage-leaf binding of a TensorView binds the base of the record
  buffer. The view's offset, extents, and strides are written to the carrier in
  logical elements, so a subview never needs a buffer offset that satisfies the
  backend's storage-offset alignment.
- The carrier is one `Block` struct whose members are the `i32` fields at their
  canonical offsets. SPIR-V forbids a `Block` struct nested inside another
  `Block`, and SPIRV-Cross drops the nested declaration from GLSL.
- Endpoints share one Stage parameter exactly when they project the same
  Program Value leaf in the same binding role; each endpoint is then one use
  of that parameter. This holds for compute and graphics alike: a value read
  by both graphics modules is one parameter with two uses. A compute value
  bound to two endpoints shares a parameter only when every endpoint reads
  it, because a shared writable parameter would alias two kernel arguments.
  Value identity alone is not enough, because one value may be projected as
  several struct leaves or several autodiff roles (tape, replay segment,
  replay status).
- Each Stage parameter records the target binding that created it, and the
  Node's endpoint projections are built from those records, so projections
  and parameters cannot be indexed differently.
- A draw that pulls all vertex data from TensorViews has no vertex attribute to
  infer a count from, so it must pass `vd.draw(vertex_count=...)`.

## 6. Targets

- OpenGL requires GLSL 430 or newer (OpenGL ES: GLSL ES 310) for shader
  storage buffers; compiling a graphics TensorView for an older version,
  including the graphics default GLSL 330, fails with
  `PROGRAM_GRAPHICS_STORAGE_UNSUPPORTED`. The fallback tier of the Vernon
  render system targets OpenGL 4.5.
- Vulkan uses the same SPIR-V lowering (StorageBuffer runtime arrays and a
  Uniform carrier block).
- DirectX 12 compiles every storage block, including the `NonWritable`
  graphics reads, as `RWByteAddressBuffer` and binds it as a raw UAV. A
  `NonWritable` block would otherwise become a `t`-register SRV, and the D3D12
  provider's binding model has one descriptor kind per storage buffer for
  compute and graphics alike. The root signature is one table visible to all
  stages, so a descriptor that both stages use is one range (section 3).
  Inline values stay per-stage root constants. Storage leaves after the first,
  extra samplers, and metadata carriers take provider slots after the highest
  parameter slot, because D3D12 places external parameters at their parameter
  slot.
- Metal rejects graphics TensorView bindings when the Stage loads, with
  `PROGRAM_GRAPHICS_STORAGE_UNSUPPORTED`, until its draw binding, meaning
  per-stage buffer indices for storage leaves and carriers, is implemented
  and tested on Apple hardware.

## 7. Acceptance

- vertex pulling by `vertex_index` from a dynamic-extent TensorView renders the
  same pixels as the equivalent vertex-attribute draw;
- a fragment stage reads a row of an aggregate (struct) table;
- one Program Value read by both stages binds correctly in both modules;
- a non-canonical subview (offset and stride) reads the selected records;
- `write` access, explicit bindings, and GLSL < 430 are rejected with stable
  diagnostics;
- cross-stage explicit-binding collisions are rejected at finalization;
- the cooked manifest round-trips through Runtime manifest parsing.
