# VernonDSL roadmap

## Current position

VernonDSL 0.1.1 is the first stable cross-platform release target. The current
version axes are:

- release `0.1.1`;
- compiler contract 9;
- pipeline contract 12;
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
- The pipeline contract was advanced to version 12 with regenerated manifests
  and fixtures.

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
- Metal is exposed through the Python architecture surface and packaged only in
  Apple builds; non-Apple source graphs remain free of Apple SDK dependencies.
- macOS resource, retention, compute, graphics, and barrier tests exist.
  Packaged iOS simulator smoke and the iOS arm64 Runtime build are also present.

## Future work

### Complete release verification

- Obtain a fully green exact release commit for every required CI and wheel job.
- Execute Metal compute, graphics, Argument Buffer, dispatch, and readback gates
  without capability skips on release hardware.
- Run the trusted workflow dry run, verify the complete payload, create the
  immutable `v0.1.1` tag, publish, and verify the PyPI and GitHub artifacts.
- Add scheduled or self-hosted GPU jobs with native validation/debug layers.
- Add fuzzing for source, manifest, reflection, TensorView, and ExecutionGraph
  inputs.
- Publish benchmark history and stable regression thresholds.

### Language v4

The detailed language gates remain in
[`language/future_language_roadmap.md`](language/future_language_roadmap.md).
The principal work is:

- first-order JVP, VJP, `grad`, `value_and_grad`, and `stop_gradient` for
  specialized pure helper functions, validated against finite differences;
- complete cross-backend acceptance for workgroup storage, barriers, relaxed
  atomics, and representative language-v4 programs;
- retain deterministic cache identity and explicit target diagnostics;
- decide which remaining autodiff gates block declaration of frontend v4.

Stateful-kernel differentiation, nested transforms, Hessians, and general
differentiable rendering are outside the initial language-v4 target unless the
language contract is amended.

### Metal completion and Apple packaging

- Complete and verify stencil attachment load/store/clear behavior.
- Add public cooked-bundle storage-texture compute binding and acceptance.
- Expand texture/sampler, depth, blend, cull, indexed draw, instancing, owned/
  borrowed encoder lifetime, and ExecutionGraph coverage on Metal hardware.
- Complete the packaged iOS arm64 device final-link smoke in CI.
- Finish provider layering cleanup where backend headers or cache state are
  still pulled through common adapter structures.
- Reduce device-wide locking and synchronous `waitUntilCompleted` only as part
  of a future asynchronous execution contract.

Window presentation, tvOS, visionOS, Python-on-iOS, top-level iOS
cross-compilation, persistent `metallib` caching, and cross-generation GPU
tuning are not current Metal roadmap commitments.

### Remaining graphics-state cleanup

- Replace OpenGL raw shared native graphics pointers with explicit ref-counted
  ownership and verify every native pipeline preparation failure path.
- Consolidate packed D32S8 sizing and readback normalization and assert actual
  depth/stencil values in OpenGL tests.
- Remove backend-specific naming and duplicated attachment/render-pass helpers
  that remain in shared code.
- Strengthen tests for OpenGL capability stubs and state calls, complete
  DirectX 12 state-mapping coverage, variant-key hit/miss behavior, and native
  object leak handling.
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

- AMD/ROCDL compiler and Runtime support.
- Constrained and const generics.
- Enums, `Option[T]`, exhaustive `match`, and compile-time data structures.
- Broader platform, presentation, and hardware-backed backend coverage.
