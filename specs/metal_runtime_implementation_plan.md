# Native Metal runtime implementation plan

## Status

This document defines the active implementation plan for adding a native
Apple Metal runtime for macOS and packaged iOS 15+ device/simulator builds.
Metal is available as an experimental compute and graphics Runtime backend on
macOS. Graphics support covers the acceptance-tested cooked-bundle path;
stencil attachments, explicit in-scope clears, and several advanced draw features
remain pending.

Implementation progress:

- Step 1 is complete: platform-explicit macOS/iOS MSL 2.4 options, deterministic per-entry resource
  slots, and the `metal_resource_slots` manifest sidecar are compiler-tested.
- Step 2 is complete on macOS: native device/queue ownership, command submission,
  buffer memory classes, base 2D/3D/cube/depth texture round-trips and mipmap
  generation, compatible-format image views, sampler filter mapping, native
  handles, and retained logical resources are implemented and covered by macOS
  tests. Failed submission ownership is explicit and leaves the retained command
  buffer for abandonment.
- The provider and public Runtime compute and graphics paths consume cooked
  resource slots, compile MSL, bind resources, dispatch or draw, synchronize,
  and read back through the existing APIs. Acceptance-tested capabilities are
  advertised through the public Runtime and Python surfaces.
- Basic color/depth attachment load/store encoding, depth testing, and color
  readback pass through a native render encoder with render-scope validation.
  Stencil attachments, explicit in-scope clears, storage-texture compute
  coverage, and several advanced draw features remain pending. Barriers validate
  resource states and rely on Metal's hazard tracking between provider encoders.

## Goals

- Add Metal as a first-class VernonRHI and VernonRuntime backend.
- Load existing cooked MSL artifacts without user-side shader processing.
- Support compute and offscreen graphics through the existing RuntimeCore,
  provider, RHI, and ExecutionGraph contracts.
- Match the Vulkan and DirectX 12 runtime surface where Metal exposes an
  equivalent operation.
- Run `examples/mandelbulb_showcase.py` with `--architecture metal --headless`.
- Return an explicit unsupported error for capabilities that are not
  implemented; never silently execute with altered semantics.

## Non-goals

- Window-system presentation or swapchain ownership.
- tvOS, visionOS, or Python-on-iOS support.
- Building or cross-compiling iOS from the top-level VernonDSL project. iOS is
  supported only by the standalone packaged Runtime source project.
- Universal wheel signing, notarization, or release automation.
- Asynchronous multi-frame optimization beyond the current runtime contract.
- Persistent binary `metallib` caching and cross-generation GPU tuning.

## Architecture

The implementation preserves the existing runtime layering:

```text
PipelineAsset manifest and MSL
              |
         RuntimeCore
              |
    RuntimeProvider adapter
              |
          VernonRHI
              |
  MTLDevice / MTLCommandQueue
```

RuntimeCore continues to own reflection-driven planning and pipeline caching.
The Metal provider adapter translates prepared layouts, bindings, and
invocations. The Metal RHI owns native devices, resources, command encoding,
submission, synchronization, and native object lifetimes.

## 1. Define the MSL binding contract

- [x] Configure platform-explicit macOS/iOS MSL 2.4 compilation.
- [x] Assign deterministic per-entry buffer, texture, and sampler indices.
- [x] Emit and compiler-test the `metal_resource_slots` sidecar.
- [x] Emit and validate cooked `apple_platform`, MSL version, and minimum OS
  requirements from structured compiler reflection.
- [x] Consume the resource-slot sidecar in the Metal compute provider path.
- [x] Reject a cross-platform cooked bundle in the Metal Runtime before pipeline
  creation.

Update `source/lib/compiler/compiler_spirv_cross.cpp` to configure explicit
macOS or iOS MSL 2.4 options and a deterministic mapping from Vernon descriptor
`(set, binding, kind)` to Metal buffer, texture, and sampler indices.

The contract must cover:

- compute, vertex, and fragment entry points;
- uniform, storage, and inline constant buffers;
- sampled and storage textures;
- samplers and texture/sampler pairs;
- resources visible to more than one shader stage;
- vertex buffers without colliding with shader buffer slots.

Add compiler tests that lock entry-point names, stages, resource indices, and
constant-buffer layouts. If the existing reflection cannot express the final
indices without reproducing SPIRV-Cross internals, extend the cooked manifest
with a Metal resource-slot sidecar and consume that sidecar in the adapter.
Runtime code must not parse generated MSL source to discover bindings.

The cooked target contract records `apple_platform` as `macos` or `ios`.
An iOS artifact requires iOS 15 or newer and is valid for both physical-device
and simulator runtimes; CPU architecture and SDK environment are properties of
the packaged native Runtime build, not of MSL. A bundle cooked for one Apple
platform must be rejected by the other before pipeline creation.

## 2. Add the Metal RHI backend

- [x] Add the gated public Metal RHI backend and backend dispatch.
- [x] Implement macOS device enumeration, iOS default-device discovery, and
  command-queue ownership.
- [x] Implement device/upload/readback buffers, transfers, native handles, and
  validity checks.
- [x] Implement base 2D/3D/cube/mipmapped color/depth texture allocation,
  upload/download, and mipmap generation.
- [x] Implement image views and compatible Metal texture format reinterpretation.
- [x] Implement sampler creation and filter/address modes.
- [x] Retain logical buffer, image, and sampler resources.
- [x] Implement command-buffer creation, submission, waiting, abandonment, and
  error capture.
- [x] Implement barrier semantics for owned hazard-tracked resources.
- [x] Implement color and D32 depth attachment load/store encoding and readback.
- [ ] Implement stencil attachments and explicit in-scope attachment clears.

Extend `source/include/VernonRHI.h`,
`source/lib/rhi/backend_dispatch.h`, and `source/lib/rhi/rhi.cpp` with
`VERNON_RHI_BACKEND_METAL` and a gated `metalBackendDispatch()`.

Add Objective-C++ Metal backend files under `source/lib/rhi/` implementing:

- default-device discovery, owned-device creation, and command queues;
- buffer creation for device, upload, and readback memory classes;
- buffer upload, download, native handles, and validity checks;
- 2D, 3D, cube, mipmapped, color, and depth textures;
- image upload/download, mipmap generation, views, and format conversion;
- sampler creation and the existing filter/address-mode surface;
- retained resource handles used by owned and borrowed command encoders;
- command buffer creation, submission, waiting, abandonment, and error capture;
- render attachment clearing and the barrier semantics required by
  ExecutionGraph.

Map Vernon formats and state enums explicitly. Unsupported formats or state
combinations must fail during resource or pipeline preparation rather than at
draw time.

## 3. Implement the Metal provider adapter

- [x] Add `adapter_metal.mm` and connect it to common adapter ownership.
- [ ] Implement Metal shader libraries/functions and compute/render pipeline
  states (compute and a basic color-only render pipeline are implemented).
- [ ] Implement provider layouts, immutable slot mappings, binding sets, and
  render-target variants.
- [ ] Implement reflected constant-buffer packing and in-flight binding-set
  lifetime retention.

Add `source/lib/runtime/rhi_adapter/adapter_metal.mm`, using
`adapter_common.cpp` for common ownership and cleanup behavior.

Implement prepared objects for:

- `MTLLibrary` and resolved `MTLFunction` entry points;
- compute and render pipeline states;
- depth/stencil state;
- provider layouts and immutable Metal slot mappings;
- binding sets with retained buffers, textures, samplers, and inline storage;
- render-target variants keyed by attachment formats and sample count.

Inline values and graphics uniforms use constant buffers packed according to
the reflected `metal_constant_buffer` physical layout. Binding-set snapshots
must remain alive until their command buffer completes.

## 4. Implement compute execution

- [ ] Bind reflected buffers, textures, samplers, and inline values.
- [ ] Plan and dispatch legal Metal threadgroups.
- [ ] Support owned and borrowed command encoders.
- [x] Pass public Runtime bundle load/dispatch/synchronize/readback coverage.

Use the existing compute launch planner and tensor bridge. The Metal adapter
must:

- bind storage, uniform, and inline-value buffers at their compiled indices;
- bind sampled/storage textures and samplers;
- derive a legal threads-per-threadgroup shape from reflected local size and
  device limits;
- dispatch the planned grid with exact Vernon invocation semantics;
- support both owned submissions and borrowed command encoders;
- make synchronized host readback observable through the existing API.

The compute gate is a cooked bundle that completes
load, resolve, dispatch, synchronize, and readback through the public Runtime
API.

## 5. Implement graphics execution

- [ ] Implement Metal vertex/index input and primitive topology.
- [ ] Implement color/depth attachments and viewport/scissor state.
- [ ] Implement raster, blend, color-mask, and depth state.
- [ ] Implement texture/sampler binding and all draw variants.
- [ ] Pass attachment readback and offscreen graphics tests.

Use the existing graphics invocation planner and provider interface. Implement:

- vertex descriptors for all advertised Vernon vertex attribute types;
- vertex and index buffers, offsets, and instance stepping;
- triangle, line, and point topology supported by the public API;
- color and depth attachments with clear, preserve, and discard operations;
- viewport and scissor state;
- cull mode, front face, blend, color write mask, and depth state;
- sampled textures, storage textures, cube maps, samplers, and mipmaps;
- indexed, non-indexed, and instanced draw calls;
- attachment readback after command completion.

Metal render encoders should be created directly from the planned attachment
set. Do not reproduce Vulkan render-pass or framebuffer objects inside the
Metal backend.

## 6. Integrate the public runtime

- [x] Add and dispatch `VERNON_RUNTIME_METAL`.
- [x] Add Metal Runtime backend and pipeline resolve/invoke implementations for
  acceptance-tested compute and graphics bundles.
- [x] Advertise only acceptance-tested Metal capabilities.
- [x] Expose Metal as a Python runtime architecture.
- [x] Enable Python kernel and pipeline resolution through the Metal Runtime.
- [x] Emit and schema-validate Metal bundle runtime requirements.
- [x] Preserve compile-only Metal artifact APIs.

Add `VERNON_RUNTIME_METAL` to `source/include/VernonRuntime.h` and route it
through:

- `source/lib/runtime/runtime_backend_dispatch.cpp`;
- `source/lib/runtime/runtime_pipeline_dispatch.cpp`;
- a Metal backend implementation;
- a Metal pipeline resolve/invoke implementation.

Report compute, graphics, storage-buffer, texture, and native-interop
capabilities only after their acceptance tests pass.

Update the Python runtime and bundle path:

- add the Metal architecture to
  `python/vernon_dsl/_runtime/session.py` and architecture parsing;
- allow kernel and pipeline resolution for the Metal target;
- emit and validate Metal runtime requirements in
  `python/vernon_dsl/bundle/requirements.py`;
- preserve compile-only Metal artifact APIs.

## 7. Integrate the Apple build

- [x] Enable Objective-C++ and Metal sources only for Apple Metal builds.
- [x] Gate the Metal RHI with `VERNON_HAS_METAL_RHI`.
- [x] Add the provider adapter and `VERNON_HAS_METAL_RUNTIME`.
- [x] Discover and link Metal and Foundation frameworks through CMake.
- [x] Keep Windows/Linux build source graphs free of Apple languages, sources,
  headers, and frameworks.
- [x] Include Objective-C++ sources and Metal headers in packaged Runtime
  sources.
- [x] Keep the top-level DSL host-oriented and reject iOS cross-builds there.
- [x] Support static iOS 15+ device/simulator builds only from the standalone
  packaged Runtime source project.

Update `source/lib/runtime/VernonRuntimeTarget.cmake` and the top-level build
options to:

- enable Objective-C++ only on Apple platforms;
- compile the Metal RHI and adapter only when Metal runtime support is enabled;
- define `VERNON_HAS_METAL_RHI` and `VERNON_HAS_METAL_RUNTIME`;
- link `Metal` and `Foundation`, and link `QuartzCore` only if required by
  public native interop;
- keep non-Apple builds and source packages free of Apple header dependencies.

Installed Runtime source packages must include the new headers and
Objective-C++ implementation files.

## 8. Verification

Add Metal-gated native and Python tests in this order:

- [ ] Device creation, capability query, and error reporting (device creation
  is covered; capability/error coverage remains).
- [x] Buffer and image upload/download round trips.
- [x] Compute pipeline creation, storage-buffer binding, dispatch, and readback.
- [x] Basic vertex/fragment rendering to a color texture through the provider.
- [ ] Texture sampling, storage textures, mipmaps, and samplers (resource,
  mipmap, and sampler creation is covered; pipeline binding remains).
- [ ] Depth testing, blending, culling, indexing, and instancing.
- [ ] Owned and borrowed command encoder resource lifetimes.
- [ ] ExecutionGraph barriers and render-scope execution.
- [ ] Pipeline bundle selection and runtime-requirements rejection (manifest
  schema/platform validation is covered; Runtime rejection remains).
- [ ] Mandelbulb smoke render with nonempty, finite output.
- [x] macOS native Metal resource, retention, and barrier-validation regression
  tests.
- [x] Packaged iOS simulator build, launch, and Metal buffer/image/sampler
  round-trip smoke.
- [x] Packaged iOS arm64 device Runtime build.
- [ ] Packaged iOS arm64 device final-link smoke (the CI gate is added but has
  not yet completed in CI).
- [x] Compiler, manifest, RuntimeCore, Python metadata, and non-Metal regression
  coverage.

Run compiler, manifest, RuntimeCore, and non-Metal backend regression tests
after the Metal-specific tests pass.

## Completion criteria

Metal can be advertised as an experimental complete runtime when:

- [ ] All verification items above pass on an Apple Silicon macOS host.
- [ ] Compute and graphics public APIs require no Metal-specific user code.
- [ ] Binding indices are consumed as deterministic compiler/runtime contract
  data (compiler production is complete; provider consumption remains).
- [ ] Unsupported features fail before encoding across the complete provider.
- [ ] Non-Apple CI passes with the Metal changes.
- [ ] Runtime and compiler design documents no longer describe Metal as
  cook-only.

If compute passes while graphics remains incomplete, expose only the validated
compute capability and retain explicit unsupported results for graphics.
Do not claim graphics parity until the graphics and Mandelbulb gates pass.

## Architecture audit follow-up

The July 2026 cross-backend review found the following work that must be
resolved before Metal graphics is advertised. These items compare Metal with
the Vulkan, DirectX 12, OpenGL, and CUDA adapter/runtime paths.

### Correctness and capability consistency

- [x] Reject graphics pipeline objects in Metal `encodeDispatch`.
  `adapter_metal.mm` currently checks that the pipeline handle exists but does
  not require a non-graphics pipeline with a valid `MTLComputePipelineState`.
  Invalid direct-provider usage must fail before creating a compute encoder.
- [x] Establish one graphics capability contract across
  `adapter_metal.mm`, `runtime_backend_dispatch.cpp`,
  `runtime_pipeline_metal.cpp`, and `runtime_pipeline_dispatch.cpp`.
  The provider currently reports graphics support while the public Runtime is
  compute-only and the public graphics resolve/invoke path is unsupported.
  Until the complete graphics path is connected, do not advertise provider
  graphics merely because the adapter-level color/depth smoke test passes.
- [x] Record and validate an active Metal render-scope signature.
  The first draw creates the native render encoder, but later draws currently
  reuse it without checking whether attachment identities, locations, formats,
  load/store operations, sample count, or depth target changed. A mismatch must
  fail rather than silently draw into the first attachment set.
- [x] Set a specific Runtime error when a Metal bundle's MSL version is
  rejected. The current version-mismatch branch returns false without updating
  `context.error`.
- [x] Validate the cooked `minimum_os_version` against the host macOS/iOS
  version during Runtime requirement validation.
- [x] Cache Metal device compute limits in `MetalContextState` and reject
  reflected workgroup sizes that exceed per-dimension or total-thread limits,
  matching Vulkan and DirectX 12's early validation.

### Type safety and layering

- [x] Replace `void *VernonRuntimeRhiAdapter::metalDevice` and
  `createBorrowedMetalRhiAdapter(void *)` with a forward-declared
  `vernon::rhi::metal::DeviceState *` and typed reference.
  The opaque pointer was introduced to avoid including `metal_backend.h` from
  ordinary C++, because that header imports Metal and contains Objective-C
  object types. A C++ forward declaration preserves that isolation without
  discarding type safety. Include the full backend header only in Objective-C++
  implementation files.
- [x] Separate the portable Metal device forward declaration from the
  Objective-C++ native resource declarations in `metal_backend.h`.
- [x] Review Runtime source-package install globs. Generic packages currently
  include private `lib/rhi/*.h` files, including a header that imports
  `<Metal/Metal.h>`. Exclude Apple-only private headers from non-Apple packages
  or split them so unpacked Windows/Linux source graphs remain Apple-header
  free.
- [x] Replace the current use of generic command color/depth resource arrays as
  storage for a retained `MTLRenderCommandEncoder`. Add a distinct typed opaque
  backend rendering object and cleanup/end mechanism so attachment resource
  slots retain their documented meaning.

### Incomplete public execution surface

- [ ] Extend `runtime_pipeline_metal.cpp` resource-slot resolution beyond
  compute `storage_buffer` entries. Consume and test uniform buffers, sampled
  images, storage images, and samplers through cooked public Runtime bundles.
  Uniform buffers, inline constants, vertex buffers, sampled images, and
  samplers are wired and covered by native tests; storage-texture compute
  binding remains open.
- [x] Connect the adapter-level graphics implementation to a Metal graphics
  `MetalPipelineState` resolve/invoke path, or keep that implementation
  explicitly experimental and unreachable until vertex attributes, graphics
  bindings, render variants, and state are implemented.
- [x] Add native interop parity tests and an end-to-end cooked graphics bundle
  test. The existing graphics test exercises the provider directly and does not
  prove public Runtime graphics support.
- [x] Implement Metal explicit in-scope color/depth clear semantics or return a
  public unsupported status. Load-action clears alone do not cover the explicit
  RHI clear operations.

### Lifetime, synchronization, and performance debt

- [x] Keep the current barrier behavior documented as conditional on one Metal
  command queue and hazard-tracked resources. Validation-only barriers are
  correct for that model; introducing untracked resources, multiple queues,
  fences, or asynchronous overlap requires real synchronization or explicit
  rejection. Do not classify the current implementation as a confirmed no-op
  bug without a failing ordering case.
- [x] Implement `completeBorrowedCommands` before making Metal submission
  asynchronous. It is currently null and safe only because owned submission
  waits for completion.
- [ ] Reduce device-wide mutex scope and synchronous `waitUntilCompleted` use
  when asynchronous execution is introduced. The current behavior is correct
  but serializes CPU callers and GPU work.
- [x] Decide whether device-local Metal images should use private storage with
  staging blits. All textures currently use `MTLStorageModeShared`, unlike
  device-memory buffers and the Vulkan/DirectX 12 image policy.

### Cleanup

- [x] Remove the unused `PreparedPipeline::device` field in
  `adapter_metal.mm`.
- [x] Remove stale direct includes such as `<optional>` where no longer used.
- [ ] Replace backend definitions pulled transitively through
  `adapter_internal.h` with forward declarations and explicit implementation
  includes. Move `adapter_test_hooks.h` and other implementation-only includes
  out of `adapter_common.h`. Metal now uses `metal_backend_fwd.h` and
  `adapter_test_hooks.h` moved to `adapter_common.cpp`; CUDA/OpenGL/Vulkan full
  backend headers remain transitively included.
- [ ] Consider moving unconditional OpenGL cache state out of
  `VernonRuntimeRhiAdapter` into backend-specific state. This is low-priority
  structural cleanup rather than a Metal correctness issue.
