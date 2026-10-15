# Integer color targets

Status: implemented, awaiting the coordinated `COMPILER_CONTRACT_VERSION` and
`PROGRAM_VERSION` cut (Engine extension E5, ID output). Multisampled parts are
deferred to the MSAA extension (section 7).

Picking, outlines and selection need an exact per-pixel identifier written
next to the scene color in one pass, with or without MSAA. Float targets
cannot carry 32-bit IDs exactly. This document records the design; the rules
themselves live in the contracts listed in section 6.

## 1. Rejected alternatives

- **IDs packed into `rgba8_unorm` or `r32_float`.** Unorm quantizes, blending
  and conversions can perturb values, and float targets hold only 24 exact
  integer bits.
- **A separate ID pass.** It re-rasterizes every object; MRT costs one more
  target write in the pass that already draws it.
- **Rejecting multisampled integer targets.** A forward MSAA renderer writes
  IDs at sample rate in the same pass and resolves them itself; only the
  fixed-function resolve is meaningless for integers.
- **Implicit flat integers as the only interpolation rule.** Flat floats
  (per-primitive data), screen-linear varyings and centroid or sample
  placement under MSAA need the same qualifier, so it is general from the
  start.
- **Formats added file by file.** Format facts live in about 22 files across
  Python, compiler, Runtime and four RHI backends; each new format would
  repeat that edit. One X-macro table is the source instead
  ([`image_resources.md`](../runtime/image_resources.md) section 7).

## 2. Formats

`r32_uint` and `rg32_uint` are rows of the format table. Both have the
baseline flags renderable, copyable, storage and multisample, and are never
blendable, filterable or resolvable. Signed and 8/16-bit integer formats are
rows added when a user needs them. Program load rejects a target the device
cannot render, store or multisample, and blending on an integer-only
attachment, with `PROGRAM_OPERATION_UNSUPPORTED`
([runtime design](../runtime/design.md) section 12).

## 3. Interpolation

- `vernon.interpolation` carries the manifest spelling (`smooth`,
  `noperspective`, `flat`) and is emitted only when the author chose a mode.
  `resolveInterpolation` (`VernonAttrs.h`) is the single place that derives
  the default from the type, so validation, SPIR-V decoration and reflection
  cannot disagree, and raw MLIR follows the same rule as Python.
- Each stage is decorated from its own declaration, so a Stage artifact's
  identity never depends on the stage it is paired with. Agreement still has
  to be checked, because GLSL before 4.30 (macOS OpenGL 4.1) requires matching
  qualifiers. No single place sees both stages on every path, so it is checked
  twice: the validator checks modules that contain both entries, and the
  compiler's graphics plan checks separately compiled stages when it builds
  the linkage records, together with the varying types.
- All modes lower to SPIR-V decorations (`Flat`, `NoPerspective`), which
  SPIRV-Cross maps to every target.

## 4. Fragment outputs and sampled texel fetch

- The compiler's graphics plan checks each fragment output against its
  attachment format; the component count comes from the format table's data
  format (`Format::components()`).
- `texture_load` on a sampled Texture is a fetch rather than `texture_sample`
  because integer formats are not filterable, IDs need exact texel
  addressing, and multisampled images can only be fetched; WGSL `textureLoad`
  and HLSL `Load` draw the same line.
- Lowering: `OpImage` extracts the image from the combined image-sampler and
  `OpImageFetch` reads it without a `Lod` operand, which reads the base
  level. An explicit `Lod` would be equally valid SPIR-V, but the pinned
  MLIR's `spirv.ImageFetch` verifier rejects it because the op does not
  implement `FetchOpInterface`.
- Every sampled Texture binds as a combined image-sampler, so a texture that
  is only fetched or measured still needs a sampler. The frontend plans an
  implicit sampler for every sampled graphics texture that no
  `texture_sample` pairs with an explicit sampler; the Runtime binds its
  default sampler, and reflection pairs it with the texture its
  `vernon.implicit_texture` names.
- The MLIR texture element type is signless, so the frontend records the
  sample type as `vernon.dtype` on the argument. Reflection derives the
  sample result class from it (`float` for `f32`, `sint` for `i32`, `uint`
  for `u32`), and every sampled endpoint layout and sampled image Storage
  carries it. Binding rejects a view whose format provides another class with
  `Program borrowed image view format does not provide the <class> sample result class of its sampled Texture`.
  A Storage with a concrete format that another sampled use reads as a
  different class fails at compile time. The Runtime parses the class once,
  at manifest load; the image constraint query reports it, or
  `VERNON_IMAGE_SAMPLE_NONE` for a Storage without sampled usage. `NONE` is 3,
  so `FLOAT`, `SINT` and `UINT` keep the values 0 to 2.
- Every sampled-texture effect records its sampler mode (`implicit`,
  `explicit`, or none for fetches and size queries), and helper effects map
  onto entry parameters, so the rule that one Texture is never sampled
  through both modes is checked on the entry's effects and covers helpers.
  The implicit-sampler plan is derived from the same effects.

## 5. Acceptance

On every backend with graphics support:

1. A pass writes `rgba8_unorm` at location 0 and `r32_uint` at location 1;
   reading back location 1 returns exact IDs, including values above 2^24.
2. `rg32_uint` round-trips two independent 32-bit values per pixel.
3. Flat, linear and perspective float varyings across a perspective triangle
   match their analytic values.
4. Clearing integer targets to `0xFFFFFFFF` reads back exactly.
5. Each diagnostic of the contract rules in section 6 is produced.
6. Every pre-existing format test passes after the move to the format table.

## 6. Contracts

- [`../language/contract.md`](../language/contract.md): storage formats,
  fragment output types, `varying(interpolation)`, sampled `texture_load`,
  and the `texture_load`-only rule for integer Textures.
- [`execution_manifest.md`](execution_manifest.md): the sampled image
  `sample_result_class`.
- [`../runtime/image_resources.md`](../runtime/image_resources.md): the
  format table, device capabilities, host data layout and typed clears.
- `VernonGraphicsState.h`: the `VernonClearColor` union taken by
  `VernonRhiColorAttachment.clear_color` and `VernonColorAttachment.clear_color`
  (`PROGRAM_VERSION`).

## 7. Deferred to MSAA

These parts need multisampled images and arrive with the MSAA extension:

- Texture dimension `2d_ms`, read by `texture_load(texture, coordinates,
  sample)` and measured by `texture_samples(texture)`, for every
  multisample-capable format including depth. `texture_sample` on it fails at
  typing.
- `vd.varying(sampling=...)` with `vd.Sampling.CENTER` (default), `CENTROID`
  or `SAMPLE`, lowered to the SPIR-V `Centroid` and `Sample` decorations.
  `SAMPLE` runs the fragment stage per sample, and `sampling` does not apply
  to `FLAT`.
- A resolve target on an integer attachment fails with
  `PROGRAM_ATTACHMENT_RESOLVE_FORMAT`; a custom resolve reads the
  multisampled image as `2d_ms` instead.
- Acceptance: a 4x MSAA pass writes `r32_uint` IDs, and a custom resolve
  reading sample 0 through `2d_ms` returns the IDs of a 1x render on interior
  pixels; centroid and sample placement differ from center only on edge
  pixels.
