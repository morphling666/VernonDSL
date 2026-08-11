# Image resource architecture

## Status

This document defines the 0.1.2 image-resource architecture for Texture,
sampled-image, storage-image, image-view, Runtime provider, and RHI
integration. The ABI changes were made before the 0.1.2 release without
changing the compiler or pipeline contract version: the existing reflection
already contains the required image binding role, dimension, sample-result
class, storage format, and access constraints.

Provider-neutral logical texture enums live in `VernonTextureTypes.h`;
`VernonRuntimeProvider.h` and RuntimeCore do not depend on `VernonRHI.h`.
Invocation attachments and image arguments carry image-view references rather
than caller-authored copies of image format or extent. Provider descriptor
queries return the immutable parent descriptor, the selected view descriptor,
the parent hazard identity, and whether the reference denotes an owner or a
view.

## 1. Semantic model

The architecture separates four concepts:

```text
Image       owns image memory and its immutable descriptor
ImageView   selects a shader-visible subresource interpretation
Binding     assigns one view a sampled, storage, or attachment role
Sampler     supplies filtering and address policy to a sampled binding
```

Python may continue to expose the owning resource as `Texture`; the lower-level
architecture calls it `Image` to distinguish resource ownership from shader
binding roles. `sampled` and `storage` are not different owner classes.

One image may be created with several usage capabilities and used in different
roles over its lifetime:

```text
compute storage write -> synchronization -> fragment sampled read
render attachment write -> synchronization -> compute storage read
transfer upload -> synchronization -> sampled read
```

Usage flags grant capabilities; they do not select the current role. A binding
and the active pipeline select the role for one invocation.

## 2. Image and image-view descriptors

Every image has one authoritative immutable descriptor owned by the resource
layer:

```text
ImageDescriptor
  dimension
  extent
  physical format
  mip-level count
  array-layer count
  sample count
  usage capabilities
```

Invocation, launch-plan, and provider-binding records must not carry independent
copies of these properties as authoritative metadata. An adapter resolves the
retained resource reference to its resource record and reads the descriptor
there. Imported native images provide the complete descriptor at import time.

An image view contains:

```text
ImageViewDescriptor
  parent image
  view dimension
  compatible view format
  base mip and mip count
  base layer and layer count
  aspect selection
```

Views retain their parent image. Destroying a public owner handle invalidates
that handle immediately, while prepared bindings and recorded commands retain
the logical resource record until completion. All views of one image share the
parent image's execution-graph hazard identity even when their native view
objects differ.

A view is not a transfer region. Upload and download regions select copy
coordinates for one operation; a view changes the subresources and
interpretation visible to shader or attachment bindings.

## 3. Sampled and storage bindings

A sampled binding:

- permits filtering, normalized coordinates, address modes, and mip selection;
- combines an image view with a separate sampler;
- exposes a shader result component class such as float, signed integer, or
  unsigned integer;
- cannot write the image.

A storage binding:

- performs typed texel load/store with integer image coordinates;
- has an exact storage format and explicit `read`, `write`, or `read_write`
  access;
- does not perform filtering, address-mode handling, implicit derivatives, or
  ordinary texture sampling.

Storage texture operations therefore do not replace sampled texture
operations. A storage binding cannot call `sample`. The same compatible image
may be rebound as sampled in a later pass after the execution graph inserts the
required synchronization. Simultaneously sampling and writing an overlapping
subresource in one dispatch is rejected unless a future contract defines a
portable, explicitly synchronized feedback model.

The source language may retain one `Texture[...]` annotation family. Its
arguments describe the shader binding role rather than creating separate
sampled-owner and storage-owner types. Reflection must encode the normalized
facts explicitly:

```text
resource kind: image
binding role: sampled | storage
dimension
sample result class, or exact storage format
access
```

Only storage bindings require an exact shader-visible image format. Sampled
bindings require a compatible component class and obtain the concrete physical
format from the bound view descriptor.

## 4. Difference from TensorView

`TensorView` and a storage image are both mutable device interfaces, but they
represent different hardware resources and semantics.

`TensorView` is a projection over linear buffer storage:

- shape, signed byte strides, and offset define addressing;
- element layout may be scalar, Tensor, Tuple, or Struct;
- indexing lowers to byte-addressed or structured buffer operations;
- it has no sampler, mip chain, image format, attachment role, or normalized
  coordinate behavior.

A storage image is a typed image resource:

- integer image coordinates select texels;
- the physical format controls channel count, packing, and conversion;
- the resource can participate in sampled, storage, attachment, and transfer
  workflows according to its usage flags;
- mip levels, layers, aspects, and backend image layouts remain meaningful.

Use `TensorView` for general structured computation and use a storage image
when the result must remain an image consumed by rendering, sampling, image
processing, or native graphics interop.

## 5. RuntimeCore planning

RuntimeCore remains independent of VernonRHI and native graphics SDK types. It
validates invocation arguments against reflection and produces typed internal
plans. C++ plans use discriminated payloads instead of a kind plus unrelated
nullable fields:

```text
ComputeArgument =
  TensorArgument
  | ImageArgument
  | ScalarArgument
  | SamplerArgument
```

`ImageArgument` carries one retained opaque image-view reference. Samplers are
separate pipeline arguments and are never embedded in an image argument.
Image descriptor metadata is not
converted to RHI enum values in RuntimeCore. Once validation is complete,
physical format and dimension are obtained from the provider's resource
record, not trusted from repeated invocation fields.

Reflection owns pipeline constraints. Resource records own concrete resource
metadata. Planning compares the two and reports mismatches before encoding.
Neither side silently overwrites the other.

## 6. Provider contract

The provider API is backend-independent and must not include `VernonRHI.h` or
expose `VernonRhiFormat`. A foreign engine can implement RuntimeCore's provider
without adopting VernonRHI.

Provider binding values use a C tagged union with role-specific payloads:

```text
ProviderBindingValue
  slot
  kind
  flags
  payload =
    InlineValue
    | BufferReference
    | ImageReference
    | SamplerReference
```

An image payload contains only the opaque retained image-view reference
required to resolve the provider-owned resource record. Owner references are
valid for ownership and transfer operations, but shader and attachment
bindings require views. Format,
dimension, extent, and usage do not appear as loose fields on every binding.
The prepared binding layout already records sampled versus storage role,
shader access, and exact storage-format constraints derived from reflection.

The provider must be able to resolve a resource reference to immutable
descriptor metadata while the reference is retained. For the VernonRHI
adapter, this resolution uses the generation-checked logical RHI record. It
must continue working after public handle destruction when a prepared binding
or recorded command still retains the record.

## 7. RHI contract

VernonRHI owns physical image descriptors, native images and views, resource
state, command encoding, submission, and completion. Runtime-to-RHI format
conversion occurs at the backend-provider boundary from the authoritative
logical view format.

The RHI adapter must have an internal descriptor query over a retained resource
record. A public `get image descriptor` function is optional; the architectural
requirement is authoritative internal lookup, not a particular public API.

Backend mappings are explicit functions rather than arithmetic enum casts:

```text
logical storage constraint -> RHI format compatibility check
RHI format                 -> Vulkan/D3D12/Metal/OpenGL native format
RHI view dimension         -> backend native view target
```

OpenGL 4.3 or `ARB_texture_view` is required for true shader-visible aliased
views. A canonical identity view whose format, dimension, aspects, and complete
subresource range equal its parent may resolve directly to the parent's texture
object. Older OpenGL and OpenGL ES otherwise continue to support transfer-region
slicing without claiming aliased image-view support. D3D12 mip generation uses
a build-time compiled and embedded utility compute shader; it does not add a
runtime shader compiler dependency.

## 8. Synchronization and hazards

ExecutionGraph tracks hazards by parent image identity plus overlapping
subresource ranges. Views of disjoint mip or layer ranges may avoid false
dependencies when the backend can express the required barriers. Overlapping
uses preserve RAW, WAR, and WAW ordering.

Each declared use records:

```text
role: sampled | storage | color attachment | depth attachment | transfer
access: read | write | read_write
subresource range
pipeline stages
```

The RHI translates those declarations into image layouts, resource states,
barriers, and cache visibility operations. Backend code must not infer hazards
from shader code or rely on implicit whole-device synchronization.

## 9. Contract invariants

The implementation must preserve these invariants:

1. RuntimeCore and Provider headers remain free of RHI image types;
2. provider retain/release callbacks are mandatory;
3. shader and attachment bindings accept image views only;
4. descriptors are queried from retained resource records;
5. ExecutionGraph retains every imported view and its parent and tracks hazards
   by parent identity plus subresource overlap;
6. backend state tracking and barriers preserve per-mip and per-layer state,
   with aspect planes tracked separately where the native API supports it;
7. no flattened compatibility records, implicit image-owned sampler field, or
   parallel binding path is reintroduced.

## 10. Acceptance requirements

The architecture is accepted only when:

- RuntimeCore builds and tests without VernonRHI headers or libraries;
- a foreign mock provider binds sampled and storage images using only the
  provider contract;
- descriptor metadata has one authoritative resource-owned source;
- invalid format, dimension, usage, access, and view compatibility fail before
  native command encoding;
- sampled, storage, attachment, and transfer transitions pass on every enabled
  graphics backend;
- one image written as storage can be synchronized and sampled in a dependent
  pass;
- TensorView and storage-image reflection and binding paths remain distinct;
- public-handle destruction cannot invalidate retained prepared or recorded
  image/view references;
- no compatibility parser, enum alias, duplicate binding path, or
  test-specific backend specialization remains.
