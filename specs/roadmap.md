# VernonDSL roadmap

## Current position

VernonDSL 0.1.1 is the first stable cross-platform release target. The current
version axes are:

- release `0.1.1`;
- compiler contract 10;
- pipeline contract 13;
- released frontend version 3, with language v4 remaining a future target.

The supported wheel matrix is CPython 3.11–3.14 on Windows x64, Linux x64, and
Apple Silicon macOS 15 or newer. The detailed product boundary lives in
[`RELEASE_NOTES.md`](../RELEASE_NOTES.md), [`PUBLIC_API.md`](../PUBLIC_API.md),
[`COMPATIBILITY.md`](../COMPATIBILITY.md), and [`SUPPORT.md`](../SUPPORT.md).
Unchecked publication and hardware gates remain in
[`RELEASE_READINESS.md`](../RELEASE_READINESS.md).

This is the single project-wide roadmap. Git history retains the detailed
implementation plans and milestone checklists that it replaces.

## Completed capabilities and work

### Stable Runtime and compatibility contracts

- Runtime invocation, owned RHI submission, and `ExecutionGraph.execute()` are
  synchronous and complete backend work before returning.
- At most one owned submission may be in flight per device.
- Unsupported target capabilities fail explicitly instead of silently changing
  backend or program semantics.
- Public Python behavior and the application-facing C ABI are defined for the
  0.1 release series. Compiler and pipeline incompatibilities are rejected by
  their independent version axes.
- Graphics execution is offscreen with host readback. Swapchain and window
  presentation are not part of the Runtime contract.

### Compiler and TensorView foundations

- One TensorView dispatch descriptor is shared by Python binding, reflection,
  manifests, C/C++ Runtime input, compiler projection, and enabled backends.
- Dynamic shape, signed stride, and offset are invocation data. One artifact
  supports contiguous, transposed, sliced, offset, and negative-stride views
  without layout-specific recompilation.
- Rank, element type, access mode, bounds, static extents, and backend index
  limits are validated before dispatch.
- Temporary `physical_index` pass state was replaced by typed physical
  load/store/atomic operations, and Vernon-to-GPU lowering uses MLIR
  one-to-many conversion instead of an unrealized-cast bridge.
- Cross-language Value ABI layout is derived from one canonical layout model.
  Retired per-use interface fields are rejected rather than accepted through a
  compatibility parser.
- Aggregate workgroup storage, barriers, relaxed atomics, nested values,
  padding, control flow, and independent-workgroup behavior have compiler and
  available-runtime coverage for the synchronous 0.1.1 subset.

### Backend capability baseline

- CPU provides reference compute execution and no software rasterizer.
- CUDA provides compute and buffers; images and samplers are rejected.
- Vulkan provides compute and offscreen graphics.
- OpenGL and OpenGL ES provide compute and graphics with a compatible supplied
  or Python-owned context.
- DirectX 12 provides compute and offscreen graphics on Windows.
- Metal provides compute and offscreen graphics on supported Apple Silicon
  devices. Argument Buffer pipelines are rejected when the device cannot
  provide the required tier or encoder.

CPU graphics, CUDA image/sampler resources, f16/f64 vertex attributes,
non-relaxed atomics, and backend-independent window presentation are outside
the 0.1.1 capability baseline.

### Cross-platform release foundations

- Windows, Linux, and Apple Silicon macOS CI build and test compiler, Runtime,
  Python, and CPython 3.11–3.14 wheels.
- Linux wheels are repaired with `auditwheel`; macOS wheels are arm64 with a
  15.0 deployment target.
- Installed-wheel smoke covers CPU dispatch/readback, frontend compilation,
  pipeline cooking, and bundled Runtime-source presence/version checks.
- Public API, compatibility, support, security, release-artifact, and release
  readiness policies are published in the repository.
- The release workflow validates the 12-wheel matrix, generates SHA256 sums and
  an SPDX SBOM, creates provenance attestations, stages a GitHub Release,
  publishes through PyPI Trusted Publishing, and runs post-publication smoke
  before finalizing the GitHub Release.

### GPU resource lifetime phase 1

- VernonRHI owns native resources, queues, submission, and completion.
- RuntimeCore retains opaque provider references without depending on native
  graphics SDK types.
- The RHI adapter maps provider references to generation-checked logical
  records. Public handle destruction invalidates the handle while retained
  prepared bindings keep the native object alive.
- Binding replacement retains the new record before releasing the old record.
  Failed or abandoned commands release unsubmitted references.
- Borrowed resources remain owned and synchronized by their external owner.

This model is complete for synchronous single-submission execution. It does not
permit removing queue waits or letting recorded work outlive its owners.

### Cross-backend graphics state

- Shared graphics-state declarations and named stencil-face state are used
  across public RHI and provider boundaries.
- Dynamic stencil reference is represented as draw state and wired through
  Metal, DirectX 12, OpenGL, and Vulkan command paths.
- Runtime graphics variants use a canonical field-wise key and shared variant
  preparation path rather than struct-byte hashing.
- Vulkan indexed draws propagate the index binding, and Vulkan image aspect
  selection distinguishes color, depth, and packed depth/stencil formats.
- DirectX 12 state conversion uses explicit enum mappings rather than
  arithmetic enum offsets.
- OpenGL graphics programs, vertex arrays, and framebuffers are retained by a
  shared native graphics bundle. Core variant-cache hit, miss, reuse, and
  partial-preparation cleanup paths have tests.
- Packed D32S8 readback uses one eight-byte depth/stencil layout across enabled
  backends. Metal and Vulkan tests assert rendered depth/stencil values; the
  OpenGL adapter has a packed-layout readback test.
- The pipeline contract was advanced through version 13 with regenerated
  manifests and fixtures.

### Metal acceptance-tested subset

- The compiler emits platform-specific MSL 2.4, deterministic buffer/texture/
  sampler slots, and hash-covered Apple platform, MSL version, minimum OS, and
  feature requirements.
- Metal Runtime rejects bundles cooked for the wrong Apple platform before
  pipeline creation.
- The Metal RHI owns device/queue state, buffers, base color/depth textures,
  mipmaps, views, samplers, command buffers, retained resources, and
  hazard-tracked synchronization.
- Public Runtime compute and offscreen graphics paths compile cooked MSL,
  prepare reflection-driven bindings and pipeline state, dispatch or draw, and
  read back results.
- Metal D32S8 readback separates depth and stencil planes and normalizes them
  into the shared packed layout. Provider-level Argument Buffer coverage
  includes sampled textures, samplers, and writable storage textures.
- Portable workgroup storage, barriers, and relaxed workgroup atomics compile
  for Metal. Device-scope TensorView atomics are rejected with an explicit
  capability diagnostic because they are limited to CPU, CUDA, and Vulkan.
- Metal is exposed through the Python architecture surface and packaged only in
  Apple builds; non-Apple source graphs remain free of Apple SDK dependencies.
- macOS resource, retention, compute, graphics, and barrier tests exist.
  Packaged iOS simulator smoke and an iOS arm64 Runtime plus smoke-host
  final-link build are also present.

### CPU structured Storage autodiff

- `vd.ad.vjp` emits deterministic primal, forward-with-tape, and backward
  profiles with checked dynamic tape and explicit Storage objectives.
- Direct and cooked CPU execution share one `dynamic_v2` executable,
  derivative-group validation path, and reusable pullback implementation.
- Recursive Scalar, Tensor, Tuple, and Struct Storage elements use packed
  structural tangent layouts with f16-to-f32 promotion and zero nodes for
  non-differentiable leaves.
- Dynamic, offset, strided, and reversed TensorViews preserve their descriptors
  through cotangent gathering and gradient scattering. Compatible `wrt` views
  sharing one owner accumulate into one fresh tangent owner.
- Stateful branches, loops, runtime gather/scatter, scratch overwrite,
  aggregate output cotangents, and multi-invocation carriers have direct,
  cooked, and finite-difference regression coverage.
- Compiler contract 10 and pipeline contract 13 freeze the current transform,
  profile, tape, and manifest boundary without compatibility readers.

## Future work

### Complete release verification

- Obtain a fully green exact release commit for every required CI and wheel job.
- Execute Metal compute, graphics, Argument Buffer, dispatch, and readback gates
  without capability skips on release hardware.
- Run the trusted workflow dry run, verify the complete payload, create the
  immutable `v0.1.1` tag, publish, and verify the PyPI and GitHub artifacts.
- Add fuzzing for source, manifest, reflection, TensorView, and ExecutionGraph
  inputs.
- Publish benchmark history and stable regression thresholds.

### Host language and native interop

The proposed [`Host language design`](host_language.md) adds a restricted
`@vd.host` domain that calls schema-defined C++ APIs. Development uses a typed
interpreter and generated nanobind bindings; deployment lowers the same Host IR
to native desktop objects or Emscripten-compatible WebAssembly without a Python
dependency. The first milestone is one shared gameplay demo with identical
state and checksums in interpreted, desktop, and browser execution.

This work is post-0.1.1 and does not change the current device-language or
Runtime contracts.

### Language v4

The detailed language gates remain in
[`language/future_language_roadmap.md`](language/future_language_roadmap.md).
The accepted autodiff architecture is defined in
[`autodiff.md`](autodiff.md).
The principal work is:

- extend the implemented CPU structured Storage VJP contract to GPU dynamic
  tape and aggregate tangent execution without backend-specific grouping;
- expose composed ExecutionGraph VJP beyond the current C++ GPU surface;
- define versioned custom VJPs for rasterization, visibility, depth, blend, and
  texture sampling before claiming cross-stage graphics differentiation;
- complete cross-backend acceptance for workgroup storage, barriers, relaxed
  atomics, and representative language-v4 programs;
- retain deterministic cache identity and explicit target diagnostics;
- decide which remaining autodiff gates block declaration of frontend v4.

JVP, full-Jacobian materialization, convenience `grad` aliases, nested
transforms, Hessians, and HVPs are outside the initial public autodiff surface.

### Metal completion and Apple packaging

- Extend the existing Metal stencil readback test from explicit clear to the
  full attachment load/store/clear and stencil-replace matrix.
- Promote the provider-level storage-texture compute path to a public
  cooked-bundle binding and acceptance test.
- Add Metal to the runtime synchronization acceptance suite so workgroup
  storage, barriers, and relaxed workgroup atomics run on hardware rather than
  being covered only by compiler lowering and RHI barrier tests.
- Expand texture/sampler, depth, blend, cull, indexed draw, instancing, owned/
  borrowed encoder lifetime, and ExecutionGraph coverage on Metal hardware.
- Run the already final-linked iOS arm64 smoke host on a physical device when a
  suitable device runner is available.
- Finish provider layering cleanup where backend headers or cache state are
  still pulled through common adapter structures.
- Reduce device-wide locking and synchronous `waitUntilCompleted` only as part
  of a future asynchronous execution contract.

Window presentation, tvOS, visionOS, Python-on-iOS, top-level iOS
cross-compilation, persistent `metallib` caching, and cross-generation GPU
tuning are not current Metal roadmap commitments.

### Remaining graphics-state cleanup

- Finish OpenGL ownership cleanup for the remaining raw layout/device
  relationships and verify every native pipeline preparation failure path.
- Replace the OpenGL mock-seeded D32S8 assertion with a post-render value test,
  and add equivalent DirectX 12 packed depth/stencil readback acceptance.
- Remove backend-specific naming and duplicated attachment/render-pass helpers
  that remain in shared code.
- Strengthen tests for OpenGL capability stubs and state calls, extend DirectX
  12 mapping coverage beyond blend/compare/stencil/cull, and add deterministic
  native-object leak assertions around the existing variant-cache tests.
- Remove hardcoded native Metal format values from tests and prevent parallel
  backend builds from overwriting the same development Python module.

Any further public RHI, Provider, or RuntimeCore layout cleanup requires an
intentional pipeline contract bump and migration of every enabled backend.

### Asynchronous GPU resource lifetime

Deferred reclamation activates only when at least one product requirement is
accepted:

1. recorded ExecutionGraph work may execute after the declaring call returns;
2. multiple frames or submissions may remain in flight;
3. queue synchronization is a measured material bottleneck;
4. prepared bindings intentionally outlive their public wrappers.

Phase 2 adds per-device submission serials, touched-resource collection,
backend completion queries, batched reclamation, observable pending memory, and
deterministic shutdown draining. Completion sources should use DirectX 12
fences, Vulkan timeline semaphores or fences, CUDA stream events, and `GLsync`
where applicable.

Phase 3 lets compiled or recorded graphs own logical records, releases them on
cull/replacement/failure/abandonment, permits multiple submissions in flight,
and moves synchronization to explicit readback, compatibility, and shutdown
boundaries.

Before asynchronous execution is exposed, every enabled backend must implement
one record and completion contract. Do not reserve an ABI or maintain parallel
raw-pointer and stable-record lifetime paths before these activation gates are
met.

### Measurement-driven performance

- Target warm frontend compilation below 5% of cold frontend time.
- Make cooking cost proportional to unique specialized stages rather than
  `variants * stages`.
- Keep warm dispatch allocation-free when pipelines and bindings are reused.
- Report transfer, queue wait, execution, readback, pending bytes, and
  reclamation latency separately.

### Optional product tracks

- Scheduled or self-hosted real-GPU jobs for long-term regression detection;
  these are not a near-term feature or release priority.
- Native Vulkan, DirectX 12, or Metal validation/debug layers for those GPU
  jobs; these are also operational hardening rather than near-term product work.
- AMD/ROCDL compiler and Runtime support.
- Constrained and const generics.
- Enums, `Option[T]`, exhaustive `match`, and compile-time data structures.
- Broader platform, presentation, and hardware-backed backend coverage.
