# VernonRHI and RuntimeCore execution plan

## Goal

Create one hardware implementation shared by Vernon Engine and VernonDSL
standalone execution:

```text
Vernon Engine ─┬─ VernonRHI
               ├─ VernonRuntimeCore
               └─ VernonRuntimeRHIAdapter

Python ────────┴─ the same three targets

Foreign Engine ── VernonRuntimeCore + its own RuntimeDeviceProvider

CPU ───────────── VernonRuntimeCore + RuntimeCpuProvider
```

RuntimeCore owns PipelineAsset parsing, reflection, binding plans, and prepared
pipeline caches. VernonRHI owns devices, resources, commands, barriers, and
completion. Engine-only RenderGraph, RenderPass, Material, Scene, and asset
policy remain outside both libraries.

## Non-negotiable boundaries

- RuntimeCore must not include VernonRHI or native SDK types.
- VernonRHI must not parse PipelineAsset or shader reflection.
- `VernonRuntimeDeviceProvider` is a versioned opaque-handle C SPI. The official
  GPU adapter implements it with VernonRHI; RuntimeCpuProvider and foreign
  engines may implement it independently without linking VernonRHI.
- RHI capabilities are facets: Compute, Graphics, and NativeInterop. CUDA is
  Compute-only; no backend implements fake unsupported operations.
- Engine and Python GPU resources use the same RHI handles. CPU host buffers
  remain Runtime-owned provider resources; there is no fake CPU RHI backend.
- The invocation hot path performs no JSON parsing, name lookup, shader/layout
  creation, or unbounded heap allocation.
- RHI does not define a high-level Framebuffer or RenderGraph. Rendering uses
  image views plus an attachment descriptor.

## Initial source layout

Keep the first implementation in VernonDSL while preserving dependency
independence:

```text
source/include/VernonRHI.h
source/include/VernonRHI.hpp
source/include/VernonRuntimeProvider.h
source/lib/rhi/
source/lib/runtime/
source/lib/runtime/rhi_adapter/
source/tests/rhi/
```

`VernonRHI` must not link `VernonRuntimeCore`. The adapter links both. This
allows a later repository split without changing public APIs.

## Phase 1: contract and build targets

- [x] Add CMake targets `VernonRHI`, `VernonRuntimeCore`, and
   `VernonRuntimeRHIAdapter`.
- [x] Define generational opaque handles and descriptors for adapter/device,
   queue, buffer, image, image view, sampler, shader module, pipeline layout,
   compute/graphics pipeline, binding set, command encoder, and completion.
- [x] Define capability queries and explicit resource usage/access/state values.
- [x] Define the versioned `VernonRuntimeDeviceProvider` function table for:
   - capability and device identity;
   - shader, layout, and pipeline preparation;
   - provider-owned resource references;
   - prepared binding creation/update;
   - draw and dispatch encoding.
- [x] Add a mock foreign provider test that links RuntimeCore without VernonRHI.

Acceptance:

- [x] The dependency graph is enforced by CMake and include-boundary tests.
- [x] A synthetic PipelineAsset prepares and encodes through the mock provider.
- [x] Unsupported capability combinations fail before resource or pipeline work.

## Phase 2: two vertical slices

Implement one Compute and one Graphics path before broad migration:

### CUDA Compute

- [x] Move CUDA device/context, buffer, module/function, stream, event, and transfer
  code from `source/lib/runtime/backend_cuda*` into the CUDA RHI backend.
- [x] Keep modules and functions persistent; use stream/event and pinned staging
  pools. Dispatch only marshals a precomputed argument layout and enqueues.
- [x] RuntimeCore loads PTX and reflection, creates a provider compute pipeline, and
  encodes through the provider.

### OpenGL Graphics

- [x] Move the current context callback and texture path into the OpenGL RHI
  backend.
- [x] Extend the RHI slice only as required for existing Engine texture, pipeline,
  binding, attachment, and draw behavior.
- [x] Adapt Vernon's current `render/backend/graphics_device.h` to RHI handles
  without moving RenderPass or RenderGraph into RHI.
  The public texture handle is `VernonRhiImage`; VernonRHI owns the
  generational slot table and native OpenGL texture lifetime.

Acceptance:

- [x] CUDA compute and an existing OpenGL Engine pipeline pass through RuntimeCore,
  the official adapter, and VernonRHI.
- [x] Python compute and graphics use those same paths.
- [x] Repeated invocation does not recreate modules, layouts, pipelines, or stable
  bindings.

## Phase 3: D3D12 and Vulkan

### D3D12

- [x] Move device/resource/command ownership into the RHI backend.
- [x] Use per-frame command allocators/lists, upload/readback rings, and
  shader-visible descriptor rings.
- [x] Cache root signatures, PSOs, input layouts, and descriptor layouts during
  pipeline preparation.
- [x] Support borrowed native device, queue, command list, resource, and descriptor
  handles through NativeInterop.

### Vulkan

- [x] Move instance/device/resource/command ownership into the RHI backend.
- [x] Use device-local resources, staging rings, reusable command/descriptor pools,
  and persistent shader/pipeline layouts.
- [x] Prefer dynamic rendering; use a render-pass compatibility cache when the
  capability is unavailable.
- [x] Support borrowed device, queue, command buffer, buffer, image, and image view
  handles through NativeInterop.

Acceptance:

- [x] Existing CUDA, D3D12, Vulkan, and OpenGL Runtime tests pass through RHI.
- [x] Stable frame loops perform no per-frame pipeline/layout/command-pool creation.
- [x] Borrowed handles are never submitted, synchronized, or destroyed against the
  owner's contract.

## Phase 4: RuntimeCore hot path

Split RuntimeCore execution into:

- [x] load and validate PipelineAsset;
- [x] prepare provider pipeline and immutable binding layout;
- [x] prepare/update stable bindings;
- [x] encode invocation with pre-resolved numeric slots.
- [x] Represent direct artifacts, CPU entries, and PipelineAssets with the same
  `VernonLoadedPipeline` compute/graphics handle.
- [x] Remove `VernonLoadedKernel`, `vernonRuntimeLaunch`, and
  `runtime_kernel_dispatch`; direct loaders synthesize compute pipeline variants.
- [x] Route CPU preparation, binding, and synchronous dispatch through
  RuntimeCpuProvider without a VernonRHI dependency.

Cache keys include artifact hash, provider device identity, specialization,
attachment compatibility, topology, vertex layout, and pipeline layout.
Provider resource identity participates only in prepared-binding caches.

Expose reflected stage/access/usage requirements so Vernon RenderGraph or a
foreign engine can plan barriers. RuntimeCore never maintains a competing
resource-state tracker.

## Phase 5: Python and Engine migration

- [x] Python Tensor/Texture allocate RHI Buffer/Image through the standalone host.
  CUDA, Vulkan, DirectX 12, OpenGL, and OpenGL ES use the shared RHI host.
  DirectX 12 image transfer is tested on WARP.
- [x] Remove unconditional synchronization after pipeline invocation.
  Download only on `to_numpy()`, conflicting host access, explicit synchronize,
  or final cleanup.
- [ ] Vernon TextureStorage and GPU buffer wrappers hold RHI generational handles.
  Python wrappers use generational handles; Engine wrappers remain to migrate.
- Vernon RenderGraph owns transitions and submission; RuntimeCore only encodes
  into its RHI command encoder through the adapter.
- Delete the old context-owned Runtime backend after backend parity and remove
  temporary Engine native-handle bridges.

## Verification

- Backend conformance tests share definitions for allocation, transfer,
  compute, graphics, barriers, capability rejection, and borrowed ownership.
- [x] Architecture tests prove RuntimeCore links and executes with a foreign mock
  provider and has no VernonRHI dependency.
- [x] Counters assert that warm invocation does not recreate shader modules,
  layouts, PSOs, descriptor pools/heaps, command allocators, or staging
  allocations.
- Benchmarks separate compilation, cold preparation, warm CPU encode time, GPU
  completion time, and explicit readback. Use `fractal.py` for Compute and
  `unified_pipeline.py` for Graphics; do not use absolute CI timing thresholds.

## Migration rule

Migrate one complete backend slice at a time. Do not add new behavior to both
the old Runtime backend and RHI backend. Until a slice reaches parity, route
production through the old path and test the RHI path explicitly; after parity,
switch callers and delete the old implementation in the same phase.

## Remaining legacy removal

Complete these steps after profiling and optimizing the shared RHI execution
path:

1. Migrate Vernon Engine to RHI handles.
2. Migrate all C++ backend tests to owned RHI devices and resources.
3. Change Python CPU invocation to use host tensors directly.
4. Delete Runtime resource creation/import APIs, legacy resource structures,
   and pipeline fallback branches.
5. Delete legacy backend resource implementations and temporary Engine bridges.

## Performance gate before legacy removal

Use `examples/fractal.py` without UI work as the Compute benchmark. Measure
frontend/cached-specialization lookup, resource residency, native encode and
completed GPU execution, and readback separately.

The July 2026 Windows baseline showed that warm Python dispatch spent about
105 ms rebuilding the frontend specialization key and another roughly 100 ms
constructing and exhaustively validating full-size TensorViews. Completed native
GPU execution was below 0.4 ms on the measured system. Optimize the Python
specialization lookup and TensorView/borrow validation hot paths before migrating
the Engine, so migration performance comparisons measure RHI integration rather
than avoidable frontend overhead.

After adding dependency-aware fast specialization lookup, cached full-storage
views, and an O(rank) injectivity proof for ordinary layouts, warm high-level
dispatch measured 0.19-0.51 ms across the GPU backends on the same system.
