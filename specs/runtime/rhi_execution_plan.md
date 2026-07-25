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
```

RuntimeCore owns PipelineAsset parsing, reflection, binding plans, and prepared
pipeline caches. VernonRHI owns devices, resources, commands, barriers, and
completion. Engine-only RenderGraph, RenderPass, Material, Scene, and asset
policy remain outside both libraries.

## Non-negotiable boundaries

- RuntimeCore must not include VernonRHI or native SDK types.
- VernonRHI must not parse PipelineAsset or shader reflection.
- `VernonRuntimeDeviceProvider` is a versioned opaque-handle C SPI. The official
  adapter implements it with VernonRHI; a foreign engine may implement it
  independently.
- RHI capabilities are facets: Compute, Graphics, and NativeInterop. CUDA is
  Compute-only; no backend implements fake unsupported operations.
- Engine and Python resources use the same RHI handles. Do not retain a second
  Runtime-owned backend implementation after migration.
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

1. Add CMake targets `VernonRHI`, `VernonRuntimeCore`, and
   `VernonRuntimeRHIAdapter`.
2. Define generational opaque handles and descriptors for adapter/device,
   queue, buffer, image, image view, sampler, shader module, pipeline layout,
   compute/graphics pipeline, binding set, command encoder, and completion.
3. Define capability queries and explicit resource usage/access/state values.
4. Define the versioned `VernonRuntimeDeviceProvider` function table for:
   - capability and device identity;
   - shader, layout, and pipeline preparation;
   - provider-owned resource references;
   - prepared binding creation/update;
   - draw and dispatch encoding.
5. Add a mock foreign provider test that links RuntimeCore without VernonRHI.

Acceptance:

- The dependency graph is enforced by CMake and include-boundary tests.
- A synthetic PipelineAsset prepares and encodes through the mock provider.
- Unsupported capability combinations fail before resource or pipeline work.

## Phase 2: two vertical slices

Implement one Compute and one Graphics path before broad migration:

### CUDA Compute

- Move CUDA device/context, buffer, module/function, stream, event, and transfer
  code from `source/lib/runtime/backend_cuda*` into the CUDA RHI backend.
- Keep modules and functions persistent; use stream/event and pinned staging
  pools. Dispatch only marshals a precomputed argument layout and enqueues.
- RuntimeCore loads PTX and reflection, creates a provider compute pipeline, and
  encodes through the provider.

### OpenGL Graphics

- Move the current context callback and texture path into the OpenGL RHI
  backend.
- Extend the RHI slice only as required for existing Engine texture, pipeline,
  binding, attachment, and draw behavior.
- Adapt Vernon's current `render/backend/graphics_device.h` to RHI handles
  without moving RenderPass or RenderGraph into RHI.

Acceptance:

- CUDA compute and an existing OpenGL Engine pipeline pass through RuntimeCore,
  the official adapter, and VernonRHI.
- Python compute and graphics use those same paths.
- Repeated invocation does not recreate modules, layouts, pipelines, or stable
  bindings.

## Phase 3: D3D12 and Vulkan

### D3D12

- Move device/resource/command ownership into the RHI backend.
- Use per-frame command allocators/lists, upload/readback rings, and
  shader-visible descriptor rings.
- Cache root signatures, PSOs, input layouts, and descriptor layouts during
  pipeline preparation.
- Support borrowed native device, queue, command list, resource, and descriptor
  handles through NativeInterop.

### Vulkan

- Move instance/device/resource/command ownership into the RHI backend.
- Use device-local resources, staging rings, reusable command/descriptor pools,
  and persistent shader/pipeline layouts.
- Prefer dynamic rendering; use a render-pass compatibility cache when the
  capability is unavailable.
- Support borrowed device, queue, command buffer, buffer, image, and image view
  handles through NativeInterop.

Acceptance:

- Existing CUDA, D3D12, Vulkan, and OpenGL Runtime tests pass through RHI.
- Stable frame loops perform no per-frame pipeline/layout/command-pool creation.
- Borrowed handles are never submitted, synchronized, or destroyed against the
  owner's contract.

## Phase 4: RuntimeCore hot path

Split RuntimeCore execution into:

1. load and validate PipelineAsset;
2. prepare provider pipeline and immutable binding layout;
3. prepare/update stable bindings;
4. encode invocation with pre-resolved numeric slots.

Cache keys include artifact hash, provider device identity, specialization,
attachment compatibility, topology, vertex layout, and pipeline layout.
Provider resource identity participates only in prepared-binding caches.

Expose reflected stage/access/usage requirements so Vernon RenderGraph or a
foreign engine can plan barriers. RuntimeCore never maintains a competing
resource-state tracker.

## Phase 5: Python and Engine migration

- Python Tensor/Texture allocate RHI Buffer/Image through the standalone host.
- Remove unconditional synchronization after Kernel/Pipeline invocation.
  Download only on `to_numpy()`, conflicting host access, explicit synchronize,
  or final cleanup.
- Vernon TextureStorage and GPU buffer wrappers hold RHI generational handles.
- Vernon RenderGraph owns transitions and submission; RuntimeCore only encodes
  into its RHI command encoder through the adapter.
- Delete the old context-owned Runtime backend after backend parity and remove
  temporary Engine native-handle bridges.

## Verification

- Backend conformance tests share definitions for allocation, transfer,
  compute, graphics, barriers, capability rejection, and borrowed ownership.
- Architecture tests prove RuntimeCore links and executes with a foreign mock
  provider and has no VernonRHI dependency.
- Counters assert that warm invocation does not recreate shader modules,
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
