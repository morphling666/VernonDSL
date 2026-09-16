# Unified TensorView metadata ABI

Status: implemented architecture record.

TensorView metadata uses one compiler-owned metadata carrier per compute
entry. Shader targets consume one uniform/constant buffer instead of
per-field storage-buffer bindings. CUDA consumes one aggregate kernel
parameter and CPU consumes one call-frame aggregate. This architecture applies
to CPU, CUDA, Vulkan, DirectX 12, Metal, OpenGL, and OpenGL ES.

The language semantics remain defined by
[`../language/tensor_view.md`](../language/tensor_view.md). Program field
definitions remain authoritative in
[`../program/execution_manifest.md`](../program/execution_manifest.md). This
document records the implemented ABI architecture and is not a second manifest
schema.

## 1. Decision

The architecture has one canonical TensorView metadata model, one
entry-scoped aggregate transport plan, and a small set of physical carrier
profiles:

```text
SemanticMetadataPlan
  ordered fields identified by (argument, kind, dimension)
  all offsets and strides are measured in logical elements
        |
        v
PhysicalMetadataPlan
  member representation, offset, size, alignment, and carrier location
  -> PortableShaderMetadataI32
       -> Vulkan uniform buffer
       -> DirectX constant buffer
       -> Metal constant buffer
       -> OpenGL/OpenGL ES uniform buffer
  -> CudaKernelMetadata<i64>
       -> one by-value CUDA kernel parameter record
  -> HostMetadata
       -> the existing CPU call-frame representation
```

The semantic plan, field ordering, source mapping, runtime value collection,
and validation algorithm are shared. A physical plan maps every semantic field
ordinal to exactly one member of exactly one carrier. Only integer
representation, member layout, and native carrier encoding vary by physical
profile.

Portable reflection and concrete target ABI are deliberately separate.
Portable reflection has one cross-target semantic shape. Compiled target
reflection selects one physical profile and records its complete layout and
native location. The implementation must not force one byte-identical layout
across CUDA, CPU, and shader targets.

The `PortableShaderMetadataI32` profile uses signed 32-bit descriptor
arithmetic because OpenGL ES cannot require 64-bit integer support. CUDA uses
signed 64-bit descriptor arithmetic.

## 2. Guarantees

- TensorView metadata consumes zero SSBOs on every shader backend.
- Each compute entry produces exactly one metadata uniform/constant buffer
  that has device TensorView arguments.
- CUDA passes metadata as one aggregate kernel parameter instead of one
  parameter per descriptor field.
- Dynamic offsets, extents, and signed strides do not require recompilation.
- Every target uses the same semantic field identity and ordering.
- Descriptor units, signedness, projection arithmetic, and overflow validation
  have one definition.
- Physical transport uses the compiler-owned
  `InterfacePlan`, `TransportNode`, and prepared-binding architecture.
- Backend code is limited to native buffer or kernel-parameter encoding.
- There is no OpenGL-specific metadata lowering, GLSL text rewriting, or
  backend-local reconstruction of metadata layouts.
- Reflection schemas are structurally identical across targets. A carrier may
  select a different physical profile, but no OpenGL-only or CUDA-only
  metadata field is permitted.
- Genuine storage resources and metadata carriers are accounted independently.
  Removing metadata SSBOs must not hide a kernel whose real storage leaves
  exceed a target limit.

## 3. Boundaries

- TensorView storage resources are not packed together. Aggregate element
  storage leaves retain their compiler-selected resource carriers.
- This architecture does not make more than the target-supported number of
  independent storage resources representable. Such an entry fails capability
  validation.
- Concrete invocation descriptor values never enter reflection or artifact
  identity.
- This plan does not change TensorView source semantics, aliasing, ownership,
  access validation, or autodiff behavior.
- Compatibility with unpublished intermediate artifacts is outside the ABI;
  Compiler and Runtime use the aggregate contract together.
- A common semantic ABI does not imply one native API mechanism. CUDA kernel
  parameter space and graphics constant buffers remain different carriers.

## 4. Canonical semantic record

The canonical semantic quantities are:

- `offset`: a non-negative logical-element offset from the beginning of the
  bound owner, encoded in the selected signed physical representation;
- `extent[d]`: a non-negative logical-element count;
- `stride[d]`: a signed logical-element stride.

Metadata never stores byte offset or byte stride. Program `ViewDescriptor`
remains authoritative in the byte units defined by
`program/execution_manifest.md`; the common materializer validates exact
divisibility by the compiler-owned element size and normalizes byte offset and
byte strides to logical elements exactly once. No backend performs this
conversion. Conversion back to a byte address,
when required by a physical storage leaf, uses that leaf's compiler-owned
element layout after logical index projection. All storage leaves of one
aggregate TensorView therefore share the same logical descriptor.

For each device TensorView argument, metadata fields occur in this order:

```text
offset
extent[0]
...
extent[rank - 1]
stride[0]
...
stride[rank - 1]
```

Entry metadata concatenates these records in ascending reflected argument
index. Rank-zero views contribute only `offset`. Every field retains:

- owner argument identity;
- semantic kind: offset, extent, or stride;
- dimension when applicable;
- canonical ordinal within the entry metadata record;
- units, exactly `logical_elements`.

Signedness is not repeated as a per-field reflection property. The selected
physical profile already defines the signed `i32` or `i64` representation;
semantic validation additionally requires offset and extent to be
non-negative while permitting negative stride.

The shipped field identity is the tuple `(argument, kind, dimension)`;
rank-independent fields use no dimension. This identity and order are defined
once in `SemanticMetadataPlan`. Lowering passes, reflection generation,
Program cooking, prepared plans, and backends must not independently recreate
them.

Device indexing has one normative mathematical definition:

```text
logical_index = offset + sum(index[d] * stride[d])
```

Compiler lowering and Runtime validation both use ascending dimension order
for multiplication and accumulation. Before provider mutation, Runtime uses
checked widened host arithmetic to prove that, for every legal index in every
dimension:

- each extent and runtime index is representable by the selected profile;
- every multiplication and accumulation intermediate in the emitted order is
  representable by the selected profile;
- the final logical index is within the bound owner's legal element interval;
- conversion of that index through every physical storage leaf remains within
  the leaf's byte range and native buffer-address limits.

Checking each descriptor field independently is insufficient and is forbidden
as the only overflow validation. For a zero dynamic extent the legal index set
is empty: validation must not evaluate `extent - 1`, and projection bounds are
vacuously satisfied after descriptor representation, owner, and storage
validity have been checked. A GPU view, including an empty logical view, still
requires a non-empty valid native resource range because descriptor APIs cannot
portably bind a zero-byte range; CPU host storage follows its host-pointer
contract. For non-empty views, the validator computes the range of each
multiplication and every partial sum in the compiler-recorded accumulation
dimension order; checking only the final minimum and maximum is insufficient.

Reflection continues to expose the existing logical TensorView descriptor
semantics. Program compute reflection owns one semantic `metadata_carrier`;
the selected implementation owns one matching physical carrier with
compiler-declared member offsets and sizes. Shader implementations use
`constant_region`, CUDA uses `kernel_parameter`, and CPU uses
`cpu_call_frame`. Argument endpoints do not redeclare the carrier. Scalar
child `i` of `interface_plan.root` is index-aligned with physical member `i`,
whose explicit semantic ordinal links to the canonical field and source
argument.

This entry-owned carrier relation is defined in
`program/execution_manifest.md`. Runtime never infers the carrier, field order,
source argument, or offsets from neighboring bindings. The entry-level
semantic carrier and member-source mapping are target-neutral; the physical
carrier tag is selected by profile.

Compiler-internal reflection uses `semantic_fields`, `argument_index`, and
nested `physical_layouts` before target selection. Target reflection and the
Program manifest project that intermediate form into the authoritative flat
`metadata_carrier` schema with `fields[].argument`; Runtime does not parse the
intermediate form.

## 5. Physical profiles

### 5.1 PortableShaderMetadataI32

All current SPIR-V shader targets use one `Uniform` storage-class block:

```text
struct Metadata {
    i32 field_0;
    i32 field_1;
    ...
}
```

The SPIR-V struct carries explicit member offsets and `Block` decoration.
Fields are contiguous signed 32-bit words. Scalar member alignment is 4 bytes;
the encoded field size is `4 * field_count`; and the ABI block size is rounded
up to the 16-byte portable uniform-block struct alignment. These values are
recorded by the physical plan. The layout planner, not SPIRV-Cross output, is
authoritative for member offsets, encoded size, block size, and alignment.

ABI block alignment is not native upload-placement alignment. Vulkan
`minUniformBufferOffsetAlignment`, OpenGL
`GL_UNIFORM_BUFFER_OFFSET_ALIGNMENT`, DirectX allocation rules, and Metal
buffer-offset rules are device/provider facts. Runtime applies them when
suballocating the immutable invocation payload from a transient or ring
buffer; they never change member offsets or compiled block size.

The baseline profile has a compiled block-size ceiling of 16 KiB, matching the
portable minimum uniform-block/range budget of the required OpenGL,
OpenGL ES, and Vulkan profiles. Compilation fails when the rounded block size
exceeds that ceiling. OpenGL and OpenGL ES pipeline preparation query the
device's uniform-buffer range, per-stage uniform-block count, binding count,
and offset alignment. Vulkan relies on the compiled portable ceiling and its
provider alignment. DirectX creates a 256-byte-aligned CBV range; Metal uses
the same reflected size/alignment through its constant-buffer binding path.

One SPIR-V representation is shared by Vulkan, DirectX, Metal, OpenGL, and
OpenGL ES:

- Vulkan consumes the SPIR-V uniform buffer directly;
- SPIRV-Cross maps it to an HLSL constant buffer;
- SPIRV-Cross maps it to an MSL constant buffer;
- SPIRV-Cross maps it to a GLSL/ESSL uniform block.

SPIRV-Cross must not merge resources, infer metadata semantics, or repair
layout after compilation. It only translates the explicit SPIR-V interface.

Every invocation performs the complete projection proof from section 4 for
signed `i32` arithmetic. Failure occurs before allocation, upload, descriptor
update, or other provider mutation.

### 5.2 CudaKernelMetadata

CUDA uses one by-value aggregate kernel parameter with 64-bit fields:

```text
struct Metadata {
    i64 field_0;
    i64 field_1;
    ...
}
```

Members are contiguous signed 64-bit words with 8-byte alignment and size
`8 * field_count`. The generated PTX/NVVM kernel ABI and Runtime launch
parameter list must agree on one aggregate ordinal, member offsets, size, and
alignment. Runtime supplies one pointer to the invocation-owned packed record
in `cuLaunchKernel`'s parameter array. The record remains alive through the
launch call.

Do not emulate a graphics UBO with a CUDA device allocation. That would add an
allocation, upload, pointer lifetime, and synchronization path without
improving the semantic ABI.

The generated parameter record and Runtime launch ordinal are validated
against the same flattened CUDA memref ABI. The compiled aggregate size is
limited to 4096 bytes. An oversized record fails compilation; Runtime also
validates the reflected parameter layout before launch.

### 5.3 HostMetadata

CPU passes TensorView storage as a host pointer and places the aggregate
metadata fields in the compiler-owned call frame. Field width is selected from
the reflected host index width rather than inferred in Runtime. CPU consumes
the same semantic field sequence and common invocation
validator/materializer; no graphics buffer is introduced.

## 6. Compiler design

Two backend-neutral metadata plan types live beside the Value ABI types.

`SemanticMetadataPlan` owns:

- ordered `(argument, kind, dimension)` field identities;
- canonical ordinal and first-field location for each TensorView argument;
- signed logical-element field semantics and dimension-order projection.

`PhysicalMetadataPlan`, constructed for one selected profile, owns:

- scalar representation;
- one member mapping for every semantic ordinal;
- member offsets, encoded size, block size, and ABI alignment;
- carrier kind;
- native set/binding or kernel-parameter ordinal where applicable;
- canonical layout hash and compiled carrier-size ceiling.

The plans are immutable compiler products and are validated as a bijection:
every semantic field occurs exactly once, every physical member has one
semantic source, regions do not overlap, and size/alignment are internally
consistent.

`VernonToGPU` creates one metadata tuple argument rather than projected scalar
arguments. Target compilation selects exactly one physical metadata profile.
Shader lowering materializes that tuple as a SPIR-V `Uniform` block; CUDA keeps
it as one aggregate kernel argument.

Shader lowering constructs the SPIR-V `Uniform` block before
`SPIRVLowerABIAttributesPass`. CUDA lowering constructs one aggregate kernel
parameter through the NVVM path. Reflection is emitted from the retained typed
Vernon model plus the compiler-owned metadata plan, never reverse-engineered
from GLSL, HLSL, MSL, PTX, or native driver reflection.

Native resource allocation must reserve the one metadata carrier through the
same binding allocator used for all other physical resources. Its binding must
not be derived from the first old descriptor binding, from subtraction between
per-field bindings, or from traversal order outside the retained plans.

Resource accounting is class-specific. For shader targets the compiler records
at least:

```text
storage_buffer_count = genuine storage leaves
uniform_buffer_count = has_metadata ? 1 : 0
metadata_storage_buffer_count = 0
```

The conservative OpenGL/OpenGL ES `<= 16` gate applies only to
`storage_buffer_count`. Uniform-buffer size, count, and binding limits are
validated separately. An entry that still has too many genuine storage leaves
is rejected with a diagnostic that reports the real count and target limit.

## 7. Runtime design

`PreparedComputeBindingPlan` contains one typed, entry-scoped metadata carrier
whose children reference invocation sources by canonical semantic identity:

```text
MetadataCarrier
  field 0 -> argument A offset
  field 1 -> argument A extent 0
  field 2 -> argument A stride 0
  field 3 -> argument B offset
  ...
```

A common materializer:

1. reads each bound `VernonTensorView`;
2. validates rank and static extent constraints;
3. resolves logical-element offset, extents, and signed strides;
4. performs the complete projection, owner-range, storage-leaf, and selected
   representation proof from section 4;
5. writes each field exactly once according to the semantic-to-physical
   mapping;
6. verifies that every physical member was initialized;
7. stores one immutable `metadataPayload` and its ABI alignment in the planned
   launch.

Shader providers consume that payload as a normal uniform/constant buffer.
CUDA consumes it as one kernel parameter. CPU copies the same semantic fields
to the reflected call-frame lanes. Providers may apply native
allocation-placement alignment without repacking the payload. No backend walks
logical reflection to rediscover fields, calculate offsets, or merge carriers.

The immutable invocation value owns its payload until the provider has copied
it into a submission-retained allocation or the native launch API has
completed reading the argument record. Ring-buffer reuse is synchronized with
submission completion and is not implied merely by return from command
encoding.

OpenGL and OpenGL ES use the generic uniform-buffer provider path and bind the
payload with `GL_UNIFORM_BUFFER`. Non-zero resource offsets use
`glBindBufferRange` with range and alignment validation.

## 8. Validation

Current compiler regression coverage includes:

- aggregate tuple lowering and absence of legacy descriptor attributes in
  `source/tests/mlir/GPU/tensor-view-metadata-aggregate.mlir`;
- cross-target physical profile, field order, native location, and absence of
  per-field reflection in
  `source/tests/compiler/compiler_graphics_output_test.cpp`;
- independent CUDA 4096-byte and shader 16 KiB ceiling rejection in the same
  compiler suite;
- strict semantic/physical manifest parsing in compiler and Runtime manifest
  tests.

Current Runtime regression coverage includes:

- rank-zero materialization, zero dynamic extents, negative strides, repeated
  descriptors, and projection-intermediate overflow in
  `source/tests/runtime/compute_launch_planner_test.cpp`;
- structured TensorView execution through the CPU, CUDA, Vulkan, DirectX 12,
  OpenGL, and OpenGL ES language-contract matrix;
- aggregate publication, signed-stride VJP, fan-out VJP, and GPU autodiff
  acceptance matrices.

Hardware-dependent cases run only where the corresponding backend is
available. Metal compilation is covered cross-target; Metal execution requires
macOS verification.
