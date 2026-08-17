# Scalable Execution and GPU Autodiff Plan

## Problem

The RHI exposes encoder/completion objects, but execution is mostly synchronous:

- only one active encoder is allowed per device;
- Vulkan and DirectX 12 reuse device-global command state;
- GPU AD repeatedly submits, waits, and reads back;
- graph cotangents cross pass boundaries through host memory.

This causes the Vulkan graph VJP failure and makes replay overhead scale with workgroup count.

## Ownership

- Execution Graph owns dependency analysis, subresource hazards, render fusion, queue selection, and admission policy.
- RHI owns independent recording contexts, native queues, completion tokens, bounded pools, and resource lifetime.
- Runtime AD emits replay and derivative nodes; it does not submit or wait behind the graph scheduler.
- Compiler and pipeline artifacts describe kernels and bindings; they do not own runtime scheduling policy.

## Required architecture

1. **Independent encoders**
   - Each encoder owns its command recording, resource journal, and backend allocator.
   - Finishing an encoder produces an immutable command list.
   - No upload, download, or second encoder may reset another recording.

2. **Explicit command dependencies**
   - Compute, render, transfer, checkpoint, and AD replay are command nodes.
   - Dependencies track read/write access, resource versions, image subresources, aliasing, and cross-queue ownership.
   - Submission returns a completion token; resources are recycled only after completion.
   - Encode callbacks never perform hidden device-wide submit or wait.

3. **Bounded scheduling**
   - Default to one ordered compute lane.
   - Overlap work only when backend capability and measured occupancy/bandwidth predict a gain.
   - Use bounded pools and backpressure for command, descriptor, staging, and temporary memory.
   - Preserve a low-overhead single-submission fast path.
   - Schedule compiled scopes, not individual passes. A fused render scope is an indivisible DAG node.
   - Coalesce adjacent same-queue scopes into one command list when no dependency or state transition requires a boundary.

4. **Backend mapping**
   - Select behavior from reported capabilities; unsupported concurrency or timeline features use an ordered fallback.
   - Vulkan: command buffer per encoder and timeline semaphore, with a fence-pool fallback.
   - DirectX 12: command allocator/list per encoder and fence value.
   - Metal: independent command buffers and completion handlers/events.
   - CUDA: streams and events.
   - OpenGL/OpenGL ES: multi-producer software recording, serialized execution on the context thread.
   - CPU: task DAG and futures.

5. **Device-resident graph backward**
   - Keep compatible cotangents and gradients in RHI buffers across passes.
   - Accumulate compatible fan-in on device; use the existing deterministic host path otherwise.
   - Static replay may encode forward, barrier, and backward in one submission.
   - Dynamic replay keeps an explicit status dependency before backward until a versioned artifact contract provides safe device-side conditional execution.
   - Read back only dynamic Tape status and requested final host gradients.

## Compatibility

- Do not change compiler reflection or serialized pipeline contracts.
- Existing synchronous APIs retain their wait, error, and result semantics.
- Submit/completion APIs retain their documented asynchronous semantics; compatibility helpers wait only where they already promise synchronous results.
- CPU and unsupported GPU dtype/layout paths retain current behavior.
- Deterministic mode preserves stable accumulation order.
- Native/borrowed interop retains caller-owned completion semantics.
- Existing render-pass fusion remains compile-time authoritative: one fused scope maps to one native rendering scope with no inserted transfer, barrier, submit, or encoder boundary.
- Device loss, cancellation, failed submission, and teardown release every retained command/resource exactly once.

## Implementation order

1. Remove the host-visible launch-metadata workaround; retain safe metadata lifetime and bounded replay batching.
2. Add backend capability reporting and shared dependency/completion invariants.
3. Replace device-global command state with independent encoder recordings and completion-based recycling.
4. Convert internal transfers to dependency-producing command nodes.
5. Fuse static replay and retain the explicit dynamic-status boundary.
6. Add device-resident cotangent chaining and accumulation.
7. Remove superseded synchronous AD paths after fallback coverage is complete.

## Acceptance gates

- Vulkan 128² `min_memory` graph VJP succeeds without restoration failure.
- Metal 1024² replay boundaries scale with bounded batches, not workgroups.
- CPU and GPU numerical, deterministic, failure, memory-limit, and rollback tests pass.
- Non-AD compute, render, transfer, interop, and resource-lifetime tests pass on every compiled backend.
- Existing compatible render passes still produce one rendering scope; `PassNoMerge`, attachment changes, hazards, and incompatible load/store behavior still prevent fusion.
- Concurrent recording has no device-global lock spanning command construction; short registry and queue-submit locks are permitted.
- Saturating kernels are not overlapped by default.
- Resource hazards and results match ordered execution under cross-queue fan-out/fan-in, aliasing, and image-subresource access.
- Command, descriptor, staging, and completion pool growth is bounded by configured in-flight limits.
- Single-dispatch latency, transfer latency, and peak in-flight memory stay within measured regression limits.
- Device loss, cancellation, failure injection, borrowed interop, and teardown leave no live submissions or leaked resources.
- Full CTest and Python suites pass from `osx_build/`; CI provides CUDA and DirectX 12 runtime coverage unavailable on macOS.
