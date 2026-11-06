# Image resource architecture

Status: current image-resource architecture.

This document defines Texture, sampled-image, storage-image, image-view,
schedule and HAL integration. Program reflection contains the required image
binding role, dimension, sample-result class, storage format, and access
constraints.

Logical texture enums live in `VernonTextureTypes.h`, and HAL images take
the same descriptors, so a HAL image's descriptor is the one a schedule
argument declares. Attachments and image arguments carry image-view
references rather than caller-authored copies of image format or extent. A
view's descriptor query returns the immutable parent descriptor, the
selected view descriptor and the parent hazard identity.

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

Schedule arguments, pass plans and binding records must not carry independent
copies of these properties as authoritative metadata. A schedule resolves a
view reference through the HAL view registry (`hal/native_images.h`), whose
entry holds the image's descriptor, and reads the descriptor there.

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

Image dimensions are `2D`, `3D`, `CUBE`, `2D_ARRAY`, and `CUBE_ARRAY`. `2D`
and `3D` have exactly one layer, `CUBE` exactly six, `2D_ARRAY` any positive
count with depth 1, and `CUBE_ARRAY` a positive multiple of six where layer
`cube * 6 + face` holds face `face` of cube `cube`. Cube faces are square.

A view reinterprets layers but never crosses the 3D/layered split:

| View | Parent | Layer range |
|---|---|---|
| `2D` | any non-3D image | exactly one layer |
| `2D_ARRAY` | any non-3D image | any |
| `3D` | `3D` | base 0, count 1 |
| `CUBE` | `CUBE` or `CUBE_ARRAY` | base a multiple of 6, count 6 |
| `CUBE_ARRAY` | `CUBE` or `CUBE_ARRAY` | base a multiple of 6, count a multiple of 6 |

These shape and view rules are header-only `constexpr` functions in
`runtime/image_rules.h`, so every HAL device and the graphics planner apply
the same rules. Shader bindings resolve through the view, so dimension checks
against reflection use the view dimension.

An image with views refuses destruction with `VERNON_HAL_INVALID_ARGUMENT`,
so a view never outlives its image. A destroyed image or view is freed after
the device's latest submitted point, and recordings that name it are evicted
with it (`architecture_v2.md` section 7). All views of one image share the
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
may be rebound as sampled in a later pass after the linked schedule inserts
the required synchronization. Simultaneously sampling and writing an
overlapping subresource in one dispatch is rejected unless a future contract
defines a portable, explicitly synchronized feedback model.

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
format from the bound view descriptor. The sample result class is `float`
for an `f32` sample type, `sint` for `i32` and `uint` for `u32`; `float` and
`unorm` formats provide `float`. Because the MLIR texture element type is
signless, the frontend carries the sample type as the argument's
`vernon.dtype`, and binding rejects a view whose format provides another
class. The Program image constraint query reports that class, and
`VERNON_IMAGE_SAMPLE_NONE` for an image that is never sampled.

The query's binding role comes from the image Storage usage, not from the
parameter's access: an attachment usage wins, then `storage`, then `sampled`,
then transfer. A storage texture that a shader only reads is therefore still
a storage binding, and needs storage usage and its exact format. The query
reports the storage format whenever the usage includes `storage`.

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

## 5. Schedule planning

The linker and pass planner are independent of native graphics SDK types.
They validate schedule arguments against reflection and produce typed
internal plans. C++ plans use discriminated payloads instead of a kind plus
unrelated nullable fields:

```text
ComputeArgument =
  TensorArgument
  | ImageArgument
  | ScalarArgument
  | SamplerArgument
```

`ImageArgument` carries one opaque image-view reference. Samplers are
separate pipeline arguments and are never embedded in an image argument.
Once validation is complete, physical format and dimension are obtained from
the view registry, not trusted from repeated argument fields.

Reflection owns pipeline constraints. Resource records own concrete resource
metadata. Planning compares the two and reports mismatches before encoding.
Neither side silently overwrites the other.

## 6. Binding values

Binding values (`VernonRuntimeProvider.h`) are backend-independent and use a
C tagged union with role-specific payloads:

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

An image payload contains only the opaque image-view reference that the view
registry resolves. Shader and attachment bindings require views. Format,
dimension, extent, and usage do not appear as loose fields on every binding.
The prepared binding layout already records sampled versus storage role,
shader access, and exact storage-format constraints derived from reflection.

A view's reference has identity kind 6 with a serial no later view reuses as
its resource value (`architecture_v2.md` section 16.3), so a recording that
names a destroyed view cannot bind a new one through a reused handle.

## 7. HAL contract

The HAL device owns physical image descriptors, native images and views,
resource state, command recording, submission and completion. Native formats
come from the authoritative logical view format at the backend boundary.
`vernonHalImageGetDescriptor` and `vernonHalImageViewGetImage` expose the
descriptors; the schedule reads them through the view registry instead.

Backend mappings are explicit functions rather than arithmetic enum casts:

```text
logical storage constraint -> format capability check
texture format             -> Vulkan/D3D12/Metal/OpenGL native format
view dimension             -> backend native view target
```

Format facts have one source, the X-macro table
`source/include/VernonFormats.def`. Each row gives the public name, the
texture enum value, component class, pixel size, host data format and type,
the native GL, Vulkan, D3D12, Metal and SPIR-V formats, and the baseline
capabilities its class allows (renderable, blendable, filterable, storage,
resolvable, multisample, copyable). Every C++ and Python consumer reads it
(the reasons are in the [runtime design](design.md) section 12). Integer
formats (`r32_uint`, `rg32_uint`) are never blendable, filterable or
resolvable.

A device narrows the baseline: `vernonHalDeviceGetFormatCapabilities` reports
the capabilities it actually supports, and Program load rejects, with
`PROGRAM_OPERATION_UNSUPPORTED`, a target format the device cannot render,
store or multisample, and blending on an integer-only attachment.

Host image data follows the row's host data format and type, tightly packed
row by row, with no conversion. Integer formats transfer one host `uint32`
per component, so uploads and readbacks are exact.

Color clears take a `VernonClearColor` (`VernonGraphicsState.h`), read
according to the attachment's component class, so integer clears are exact
on every backend.

OpenGL 4.3 or `ARB_texture_view` is required for true shader-visible aliased
views. A canonical identity view whose format, dimension, aspects, and complete
subresource range equal its parent may resolve directly to the parent's texture
object. Older OpenGL and OpenGL ES otherwise continue to support transfer-region
slicing without claiming aliased image-view support. The phase 4 D3D12 device
generates mips with a build-time compiled and embedded utility compute
shader, so it adds no runtime shader compiler dependency.

Layered dimension mappings with backend-specific constraints:

- D3D12 `TEXTURE2D` and `TEXTURECUBE` views cannot start past layer 0, so a
  `2D` or `CUBE` view with a nonzero base layer uses the one-element
  `TEXTURE2DARRAY` or `TEXTURECUBEARRAY` form. RTV and DSV views honour the
  view's mip and layer range.
- Vulkan creates `CUBE_ARRAY` images only when the device enables the
  `imageCubeArray` feature, and reports Unsupported otherwise. SPIR-V modules
  that sample arrayed cube images declare `SampledCubeArray`, and modules with
  size queries declare `ImageQuery`; MLIR's capability deduction misses both.
- OpenGL stores arrays and cube arrays through the 3D storage and sub-image
  entry points with z = layer, binds storage images as layered unless the
  dimension is `2D`, and reads one layer back through `framebufferTextureLayer`.
- Metal uses `MTLTextureTypeCubeArray`, whose `arrayLength` counts cubes.

Surface images ([`presentable_surfaces.md`](presentable_surfaces.md)) are
ordinary `2D` images with one mip level and one layer, so their descriptors,
views, hazards and readback follow this document. Three rules differ:

- The surface owns the image and its view. Destroying either through the image
  API fails with `VERNON_HAL_INVALID_ARGUMENT`; configuring or destroying the
  surface destroys them after the submissions that use them, as for any
  destroyed image.
- An acquired image's contents are undefined until a submitted schedule
  writes it, with a render scope that loads it with `CLEAR` or `DISCARD`, or
  a host write does. Until then a submission that preserves or reads it
  fails, and so does a host read. A recording may read the image after its
  own earlier write.
- Present copies the image to the window itself, so no caller declares a
  final transition.

## 8. Synchronization and hazards

A linked schedule tracks hazards by parent image identity plus overlapping
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

The HAL device translates those declarations into image layouts, resource states,
barriers, and cache visibility operations. Backend code must not infer hazards
from shader code or rely on implicit whole-device synchronization.

## 9. Contract invariants

The implementation must preserve these invariants:

1. HAL images and schedule arguments share the `VernonTextureTypes.h`
   descriptors;
2. shader and attachment bindings accept image views only;
3. descriptors are read from the view registry;
4. a linked schedule tracks hazards by parent identity plus subresource
   overlap;
5. backend state tracking and barriers preserve per-mip and per-layer state,
   with aspect planes tracked separately where the native API supports it;
6. no flattened compatibility records, implicit image-owned sampler field, or
   parallel binding path is reintroduced.

## 10. Acceptance requirements

The architecture is accepted only when:

- descriptor metadata has one authoritative resource-owned source;
- invalid format, dimension, usage, access, and view compatibility fail before
  native command encoding;
- sampled, storage, attachment, and transfer transitions pass on every enabled
  graphics backend;
- one image written as storage can be synchronized and sampled in a dependent
  pass;
- TensorView and storage-image reflection and binding paths remain distinct;
- destroying an image or view cannot let a recording bind a later object
  through a reused handle;
- no compatibility parser, enum alias, duplicate binding path, or
  test-specific backend specialization remains.
