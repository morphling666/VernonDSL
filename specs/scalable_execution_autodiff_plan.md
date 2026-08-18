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

## Execution model

The execution path has five explicit layers:

1. **Logical Pass DAG**
   - Contains semantic dependencies, resource versions, aliases, and requested outputs.
   - Remains independent of backend queues, command allocators, and submission boundaries.
2. **Compiled Scopes**
   - Applies dead-pass elimination, render fusion, and legal compute-scope coalescing.
   - A fused render scope is an indivisible unit in every later layer.
3. **Command DAG**
   - Represents compute, render, transfer, checkpoint, replay, status, and derivative work as typed nodes.
   - Carries normalized resource accesses and explicit completion dependencies.
4. **Queue Schedule**
   - Assigns nodes to logical lanes using capabilities, hazards, cost estimates, and admission limits.
   - Defaults to ordered execution; overlap requires measured evidence.
5. **Submission Plan**
   - Coalesces adjacent compatible nodes into immutable command lists.
   - Allocates bounded recording, descriptor, staging, temporary, and completion resources.
   - Maps dependencies to native synchronization without changing graph semantics.

No layer may infer a dependency from command-list order that is absent from the preceding layer. Queue assignment and
submission coalescing may change without invalidating compiler or serialized pipeline artifacts.

## Required architecture

1. **Independent encoders**
   - Each encoder owns its command recording, resource journal, and backend allocator.
   - Finishing an encoder produces an immutable command list.
   - No upload, download, or second encoder may reset another recording.
   - Encoder creation does not reserve a device-wide execution lock; only bounded-pool acquisition and short registry
     operations may block.
   - An encoder transitions once through `recording -> finished -> submitted`, or from `recording`/`finished` to
     `abandoned`. Submission consumes a finished recording exactly once.

2. **Explicit command dependencies**
   - Compute, render, transfer, checkpoint, and AD replay are command nodes.
   - Dependencies track read/write access, resource versions, buffer byte ranges, image subresources, alias groups, and
     cross-queue ownership.
   - Imported resources carry a stable physical alias identity when available. Unknown external aliasing uses a
     conservative ordered domain rather than assuming independence.
   - Accesses are normalized before hazard analysis. Empty ranges are rejected, integer overflow is an error, and
     overlapping aliases share one version domain.
   - Resource state transitions and queue ownership transfers are generated from the Command DAG, never injected by an
     encode callback.
   - Submission returns a completion token; resources are recycled only after completion.
   - Encode callbacks never perform hidden device-wide submit or wait.
   - Upload, download, map, and readback are transfer/status nodes with explicit dependencies. A synchronous
     compatibility helper may wait for its terminal token but may not bypass the DAG.

3. **Bounded scheduling**
   - Default to one ordered compute lane.
   - Overlap work only when backend capability and measured occupancy/bandwidth predict a gain.
   - Use bounded pools and backpressure for command, descriptor, staging, and temporary memory.
   - Admission reserves the resources needed to record and submit a plan before externally visible state changes.
   - Pool exhaustion blocks, rejects, or reduces concurrency according to an explicit policy; it never grows an
     unbounded emergency pool.
   - Backpressure applies before command construction when the configured in-flight submission or byte limit is reached.
   - Preserve a low-overhead single-submission fast path.
   - Schedule compiled scopes, not individual passes. A fused render scope is an indivisible DAG node.
   - Coalesce adjacent same-queue scopes into one command list when no dependency or state transition requires a boundary.
   - Queue overlap is disabled for saturating kernels, deterministic accumulation, unknown external synchronization, and
     workloads below a measured crossover threshold.

4. **Backend mapping**
   - Select behavior from reported capabilities; unsupported concurrency or timeline features use an ordered fallback.
   - Vulkan: command buffer per encoder and timeline semaphore, with a fence-pool fallback.
   - DirectX 12: command allocator/list per encoder and fence value.
   - Metal: independent command buffers and completion handlers/events.
   - CUDA: streams and events.
   - OpenGL/OpenGL ES: multi-producer software recording, serialized execution on the context thread.
   - CPU: task DAG and futures.
   - Capability reporting distinguishes independent recording, concurrent submission, timeline synchronization,
     queue-family topology, timestamp support, and host-coherent transfer support.
   - Independent recording does not imply concurrent execution; the scheduler treats them as separate capabilities.

5. **Device-resident graph backward**
   - Keep compatible cotangents and gradients in RHI buffers across passes.
   - Accumulate compatible fan-in on device; use the existing deterministic host path otherwise.
   - Graph derivative values have an explicit storage kind and physical layout. Compatible device values do not pass
     through the public host value ABI between pullbacks.
   - Device fan-in uses a compiler-selected legal accumulation strategy. Deterministic mode uses a stable staged
     reduction or the ordered host fallback, never silently selecting nondeterministic atomics.
   - Static replay may encode forward, barrier, and backward in one submission.
   - Dynamic replay keeps an explicit status dependency before backward until a versioned artifact contract provides safe device-side conditional execution.
   - Read back only dynamic Tape status and requested final host gradients.

6. **Completion and resource lifetime**
   - A completion token has one terminal state: `succeeded`, `failed`, `cancelled`, or `device_lost`.
   - Completion observation is idempotent. Every retained resource and pool entry is released exactly once on every
     terminal path, including failed submit and teardown.
   - Cancellation prevents unsent dependents from being submitted. Submitted native work is still observed and retired;
     cancellation never permits early resource reuse.
   - Device loss stops admission, fails pending plans, drains host ownership records without waiting forever, and
     preserves the first actionable diagnostic.
   - Borrowed interop declares runtime-owned, caller-signalled, or externally waited completion. Unknown completion
     forbids reuse and cross-queue overlap.

7. **Failure and publication**
   - Recording and submission are transactional with respect to externally visible graph state.
   - Partial submission is represented explicitly. Dependents of a failed submission are not issued, while independent
     submitted work is retired safely.
   - Forward outputs, gradients, checkpoints, and caller-owned resources are published only after their terminal
     dependency succeeds.
   - Retry is limited to replay-safe operations. Dynamic Tape resize abandons the old attempt, restores declared state,
     and creates a newly admitted plan.

8. **Observability and cost model**
   - Telemetry records CPU recording time, queue delay, GPU duration when supported, submissions, waits, readbacks,
     command-list count, pool high-water marks, temporary bytes, and overlap decisions.
   - Every unavailable measurement reports a capability or skip reason; host wall time is not labelled GPU time.
   - Scheduling thresholds come from versioned benchmark evidence keyed by backend, device class, workload shape, and
     policy. Missing evidence selects ordered execution.
   - Telemetry has low-overhead production and detailed diagnostic modes. Performance gates use production mode unless
     explicitly measuring instrumentation overhead.

## Core invariants

- The same Command DAG produces the same observable result as ordered execution for every legal queue assignment.
- A resource version has one writer domain; all readers observe that writer or a declared predecessor version.
- No command list references a resource, descriptor, allocator, staging range, or temporary allocation after its
  completion-bound lifetime ends.
- No synchronous compatibility call holds a registry, pool, scheduler, or queue-submit lock while waiting.
- No backend callback recursively submits work through a device-wide API.
- Fused rendering begins and ends exactly once and contains no hidden transfer, wait, or ownership boundary.
- Deterministic mode fixes accumulation order and disables transformations that would change it.
- Admission and rollback are side-effect free until all required bounded resources are reserved.

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

1. Define the Command DAG, normalized access model, completion state machine, and failure/publication invariants; add
   backend-independent model tests before changing native submission.
2. Add backend capability reporting and the ordered queue-schedule fallback.
3. Convert every transfer that can occur during recording, including launch metadata and retained-primal copies, to
   dependency-producing command nodes. Reject hidden device-level submit or wait while an encoder is active.
4. Replace device-global command state with independent encoder recordings and completion-based recycling, one backend
   at a time.
5. Add bounded command, descriptor, staging, temporary, and completion pools with admission and backpressure.
6. Route Runtime AD replay through the graph scheduler; fuse static replay and retain the explicit dynamic-status
   boundary.
7. Add device-resident cotangent chaining and accumulation while preserving deterministic fallback behavior.
8. Enable measured queue overlap only after ordered execution passes correctness, memory, and latency gates.
9. Remove superseded synchronous AD paths only after fallback and failure coverage is complete.

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
- Model tests cover buffer byte-range overlap, image subresources, physical aliases, resource versions, cross-queue
  ownership, fan-out/fan-in, failed predecessors, and partial submission.
- Completion tests cover every state transition, repeated observation, failed submit, cancellation before and after
  submit, device loss, and teardown with in-flight work.
- Backpressure tests prove that configured command-count and byte limits are never exceeded under producer overload.
- Static GPU replay uses one submission when no dynamic status dependency is required.
- Multi-pass GPU graph backward performs no intermediate full-gradient host readback; readbacks are limited to declared
  dynamic status and requested final host values.
- Production telemetry reports queue delay and GPU duration where timestamps are supported, and an explicit skip reason
  otherwise.
- Dedicated GPU performance jobs compare median and p95 latency against versioned same-device baselines; unexplained
  regression greater than 10% fails the gate.
- For grids 256² and larger in the reference workload, an available GPU backend must outperform the same-policy CPU
  baseline on the benchmark machine; unsupported or unavailable hardware is recorded as a skip, not a pass.
