# Native Metal runtime implementation plan

## Status

This document defines the active implementation plan for adding a native
Apple Metal runtime for macOS and packaged iOS 15+ device/simulator builds.
Metal remains a cook-only compiler target until the
acceptance gates in this document pass.

Implementation progress:

- Step 1 is complete: platform-explicit macOS/iOS MSL 2.4 options, deterministic per-entry resource
  slots, and the `metal_resource_slots` manifest sidecar are compiler-tested.
- Step 2 is in progress: native device/queue ownership, command submission,
  buffer memory classes, base 2D/3D/cube/depth texture round-trips and mipmap
  generation, sampler filter mapping, native handles, and retained logical
  resources are implemented and covered by macOS tests. Failed submission
  ownership is explicit and leaves the retained command buffer for abandonment.
- Image views, render attachment encoding, full barrier handling, and the
  provider/runtime layers remain pending. Metal must not yet be advertised as a
  runtime architecture. Nonempty barriers return an explicit unsupported status
  until their encoding semantics are implemented.

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
- [ ] Consume the resource-slot sidecar in the pending Metal provider adapter.
- [ ] Reject a cross-platform cooked bundle in the pending Metal Runtime
  provider before pipeline creation.

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
- [ ] Implement image views and any required texture format conversion.
- [x] Implement sampler creation and filter/address modes.
- [x] Retain logical buffer, image, and sampler resources.
- [x] Implement command-buffer creation, submission, waiting, abandonment, and
  error capture.
- [ ] Implement render attachment encoding and complete barrier semantics.

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

- [ ] Add `adapter_metal.mm` and connect it to common adapter ownership.
- [ ] Implement Metal shader libraries/functions and compute/render pipeline
  states.
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
- [ ] Pass public Runtime bundle load/dispatch/synchronize/readback coverage.

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

- [ ] Add and dispatch `VERNON_RUNTIME_METAL`.
- [ ] Add Metal Runtime backend and pipeline resolve/invoke implementations.
- [ ] Advertise only acceptance-tested Metal capabilities.
- [ ] Expose Metal as a Python runtime architecture.
- [ ] Enable Python kernel and pipeline resolution through the Metal Runtime.
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
- [ ] Add the provider adapter and `VERNON_HAS_METAL_RUNTIME` once the provider
  exists.
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
- [ ] Compute pipeline creation, binding, dispatch, and readback.
- [ ] Basic vertex/fragment rendering to a color texture.
- [ ] Texture sampling, storage textures, mipmaps, and samplers (resource,
  mipmap, and sampler creation is covered; pipeline binding remains).
- [ ] Depth testing, blending, culling, indexing, and instancing.
- [ ] Owned and borrowed command encoder resource lifetimes.
- [ ] ExecutionGraph barriers and render-scope execution.
- [ ] Pipeline bundle selection and runtime-requirements rejection (manifest
  schema/platform validation is covered; Runtime rejection remains).
- [ ] Mandelbulb smoke render with nonempty, finite output.
- [x] macOS native Metal resource, retention, and unsupported-barrier regression
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
