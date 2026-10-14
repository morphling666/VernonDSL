# Integer color targets

Status: proposed (Engine extension E5, ID output). On acceptance it amends
the contracts listed in section 9 under one coordinated
`COMPILER_CONTRACT_VERSION` and `PROGRAM_VERSION` cut.

Picking, outlines and selection need an exact per-pixel identifier written
next to the scene color in one pass, with or without MSAA. Float targets
cannot carry 32-bit IDs exactly, and the RHI, language and Runtime have no
integer color format, no interpolation control and no multisampled texture
reads.

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
  repeat that edit. Section 2 makes one table the source.

## 2. Format table

One table, `source/include/VernonFormats.def` (a public X-macro list), holds
per format: public name, RHI and texture enum values, component class
(`float`, `unorm`, `uint`, `sint`), pixel size, host data format and type, the
GL, Vulkan, D3D12, Metal and SPIR-V storage formats, and the baseline
capability flags the format class allows (renderable, blendable, filterable,
storage, resolvable, multisample, copyable). Aspects follow from the data
format. The RHI and texture enums, backend mappings, image data layout,
Runtime format validation, compiler storage-format checks and the Python
`TextureFormat` instances all read it; `VernonFormats.hpp` is the C++
accessor. Python reads a checked-in module that `tools/generate_formats.py`
writes from the table, and `VernonGeneratedFormatsCheck` fails when it is
stale. Existing formats move into the table with their enum values; each
format has one name (`d32_float` and `d32_float_s8_uint`, not the former
`depth32_*` manifest spellings).

Devices narrow the baseline: support for a format as a multisampled or
storage image, for example, varies by GPU and API version. The RHI reports a
device's actual flags through `vernonRhiDeviceGetFormatCapabilities`, and
resolve validates Program targets against them with the existing
unsupported-target diagnostics.

This cut adds:

| Name | Class | GL | Vulkan | D3D12 | Metal |
|---|---|---|---|---|---|
| `r32_uint` | uint x1 | `GL_R32UI` | `VK_FORMAT_R32_UINT` | `DXGI_FORMAT_R32_UINT` | `MTLPixelFormatR32Uint` |
| `rg32_uint` | uint x2 | `GL_RG32UI` | `VK_FORMAT_R32G32_UINT` | `DXGI_FORMAT_R32G32_UINT` | `MTLPixelFormatRG32Uint` |

Both have the baseline flags renderable, copyable, storage and multisample;
they are never blendable, filterable or resolvable. Signed and 8/16-bit integer
formats are table rows added when a user needs them.

## 3. Fragment outputs

A fragment output's element type follows its target's component class:
`f32` for `float` and `unorm`, `u32` for `uint`, `i32` for `sint`, with the
target's component count (`vd.u32` for `r32_uint`, `vd.Vector[vd.u32, 2]` for
`rg32_uint`). A mismatch fails with
`fragment output type does not match color target format`.

## 4. Interpolation

`vd.varying(interpolation=..., sampling=...)` qualifies a varying where it is
declared: a field of the vertex output Struct and the matching fragment
input.

- `interpolation`: `vd.Interpolation.PERSPECTIVE` (default for floats),
  `LINEAR` (screen-space, SPIR-V `NoPerspective`) or `FLAT`.
- `sampling`: `vd.Sampling.CENTER` (default), `CENTROID` or `SAMPLE`.
  `SAMPLE` makes the fragment stage run per sample.
- Integer varyings are `FLAT` by default, and any other interpolation on an
  integer fails with `integer varyings must be flat`. `sampling` does not
  apply to `FLAT`.
- Each stage is decorated from its own declaration, so a Stage artifact's
  identity never depends on the stage it is paired with. Pipeline creation
  checks that both sides agree, because GLSL before 4.30 (macOS OpenGL 4.1)
  requires matching qualifiers; a mismatch fails with
  `varying interpolation differs between stages`. Integer defaults are
  derived from the type, so integer varyings agree without annotations.

All modes lower to SPIR-V decorations (`Flat`, `NoPerspective`, `Centroid`,
`Sample`), which SPIRV-Cross maps to every target.

## 5. Multisampled reads

- Texture dimension `2d_ms` denotes a multisampled 2D image. It supports
  `texture_load(texture, coordinates, sample)` with integer coordinates and an
  integer sample index, and `texture_samples(texture)`; `texture_sample` on it
  fails at typing.
- Any multisample-capable format may be bound as `2d_ms`, including depth,
  so MSAA depth reads use the same path.
- Integer textures of any dimension read with `texture_load` only.

## 6. Pipeline and render pass rules

- Blending enabled on an integer location fails at pipeline creation with
  `integer color targets cannot blend`.
- A resolve target on an integer attachment fails at resolve with a new
  `PROGRAM_ATTACHMENT_RESOLVE_FORMAT`; a custom resolve reads the
  multisampled image as `2d_ms` instead.

## 7. Clear values

Clear values are typed by the target's component class:

- RHI: `VernonRhiColorAttachment.clear` and the in-scope clear command take
  `VernonClearColor` (`VernonGraphicsState.h`), a union of `float f32[4]`,
  `uint32_t u32[4]` and `int32_t i32[4]` read according to the attachment
  format. GL uses `glClearBufferfv`/`uiv`/`iv`, Vulkan the matching
  `VkClearColorValue` member, and Metal `MTLClearColor` built from the
  member. D3D12 `ClearRenderTargetView` takes floats only, which cannot carry
  every 32-bit integer, so integer color attachments are created with
  unordered access and cleared with `ClearUnorderedAccessViewUint` through a
  render-target to unordered-access round trip.
- Runtime: `VernonColorAttachment.clear_color` (`VernonRuntime.h`) becomes
  the same union, and the finite check applies only to non-integer formats.
- Engine: the OpenGL immediate backend picks the `glClearBuffer` variant
  from the bound attachment's `GL_FRAMEBUFFER_ATTACHMENT_COMPONENT_TYPE`.
- Manifest: unchanged. A render pass declares only its attachment
  signature; clear values arrive with each invocation, and the C union
  cannot carry a wrong class, so there is no resolve-time clear diagnostic.
- Python: a render pass clear on an integer target takes four ints in the
  format's range; any other target takes four finite numbers. Either
  mismatch fails when the pass is built.

## 8. Acceptance

On every backend with graphics support:

1. A pass writes `rgba8_unorm` at location 0 and `r32_uint` at location 1;
   reading back location 1 returns exact IDs, including values above 2^24.
2. `rg32_uint` round-trips two independent 32-bit values per pixel.
3. A 4x MSAA pass writes `r32_uint` IDs; a custom resolve reading sample 0
   through `2d_ms` returns the IDs of a 1x render on interior pixels.
4. Flat, linear and perspective float varyings across a perspective triangle
   match their analytic values; centroid and sample placement differ from
   center only on edge pixels under MSAA.
5. Clearing integer targets to `0xFFFFFFFF` reads back exactly.
6. Each diagnostic of sections 3 to 7 is produced.
7. Every pre-existing format test passes after the move to the format table.

## 9. Contract changes on acceptance

- [`../language/contract.md`](../language/contract.md): the format list,
  output type rule, interpolation qualifiers, `2d_ms` and the
  `texture_load`-only rule for integers.
- [`execution_manifest.md`](execution_manifest.md):
  `PROGRAM_ATTACHMENT_RESOLVE_FORMAT`.
- [`../runtime/image_resources.md`](../runtime/image_resources.md): the
  format table, capabilities and readback layout.
- `VernonRHI.h` and `VernonRuntime.h` clear-color unions (`PROGRAM_VERSION`).
