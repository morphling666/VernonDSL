# Deferred GPU resource lifetime plan

## Status

Phase 1 is active because prepared bindings have exercised resource-owner
destruction and RHI slot reuse. Phases 2 and 3 remain deferred and
non-normative. The current synchronous submission contract remains unchanged
while phase 1 replaces unstable provider references with stable logical
records.

## Goal

Allow prepared bindings and `ExecutionGraph` command recording to outlive the
Python/C++ wrapper that supplied a resource, and allow submitted GPU work to
remain in flight without keeping stale RHI slot pointers or synchronizing every
submission.

The design must preserve these boundaries:

- VernonRHI owns native resources, queues, submission, and completion.
- RuntimeCore retains opaque provider references but does not depend on
  VernonRHI or native SDK types.
- `VernonRuntimeRHIAdapter` translates provider references to stable RHI
  resource records.
- Foreign providers may implement equivalent lifetime tracking without
  adopting VernonRHI.
- Borrowed native resources remain owned and synchronized by their external
  owner.

## Activation gates

The plan is activated when at least one of these becomes required:

1. `ExecutionGraph` records work that may execute after the declaring call
   returns.
2. More than one frame or submission may remain in flight.
3. A profiler shows queue synchronization on the invocation path is a material
   bottleneck.
4. Bindings intentionally persist after their source resource wrapper is
   released.

Gate 4 has been reached by the current prepared-binding and slot-reuse path.
This activates stable logical records only; it does not authorize asynchronous
submission or multiple frames in flight.

Do not change an ABI version merely to reserve fields for this plan.

## Current constraint

The current Python path retains resource owners through encode/execute and
updates prepared bindings before each invocation. CUDA, Vulkan, and OpenGL
provider `retain_resource` callbacks therefore validate references but do not
extend native lifetime. DirectX 12 additionally retains COM objects because
releasing a stale RHI slot can otherwise become a CPU use-after-free.

This is valid only while encoding and submission consume every reference before
the owner can disappear. A future deferred graph would violate that condition.
Backend-native values are not stable resource identities:

- an OpenGL integer name can be deleted and reused;
- a Vulkan handle can be destroyed while a descriptor still contains it;
- a CUDA device address can be returned to the allocator and reused;
- a DirectX COM pointer can remain alive while the RHI slot containing its
  tracked state is invalid or reused.

## Resource record model

Every owned RHI resource will have one stable resource record. Public handles
refer to the record by slot index and generation; provider references retain
the record, never the address of a recyclable slot or a backend-native value.

Conceptually each record contains:

```text
ResourceRecord
  public_generation
  owner_alive
  binding_reference_count
  last_submission_serial
  native_resource
  backend_state
  ownership: owned | borrowed
```

Two independent conditions control reclamation:

1. **Logical lifetime:** the public owner is gone and no prepared binding or
   unsubmitted graph command retains the record.
2. **Execution lifetime:** the device completion serial has reached the
   record's last submission serial.

An owned native resource is destroyed only when both conditions hold. Destroying
the public handle immediately invalidates its generation, but its slot cannot be
recycled until the stable record is reclaimable.

Borrowed records never destroy the native object. Their external owner must keep
the object alive through the declared completion point.

## Provider and binding behavior

`VernonRuntimeProviderResourceReference` remains opaque to RuntimeCore. The
official adapter resolves it to a stable resource key and implements:

- `retain_resource`: increment the record's binding reference count;
- `release_resource`: decrement that count without dereferencing a public
  wrapper or backend-native handle;
- binding update: retain all new references, update the backend binding, then
  release the previous references;
- binding destruction: release all retained records.

The existing provider SPI layout should be kept if its opaque values can encode
the stable key. Change the provider ABI only if the implementation proves that
the current fields cannot represent it; do not maintain raw-pointer and stable
record paths in parallel.

`ExecutionGraph` retains the records referenced by recorded commands until the
commands are submitted or discarded. Pass culling and failed compilation must
release unsubmitted references.

## Submission and completion

Command encoders collect a deduplicated list of resource records touched by the
recorded work. Submission performs these steps:

1. assign a monotonically increasing device submission serial;
2. submit backend commands;
3. set each touched record's `last_submission_serial` to that serial;
4. release command-recording references;
5. periodically reclaim records whose logical and execution lifetimes ended.

The completion source is backend-specific:

- DirectX 12: the existing queue fence values;
- Vulkan: a timeline semaphore when available, otherwise fence serials;
- CUDA: stream events mapped to device serials;
- OpenGL/OpenGL ES: `GLsync` fences for asynchronous mode; the current
  synchronous mode may mark the serial complete immediately.

Externally borrowed queues or command lists require an owner-supplied completion
contract. VernonRHI must not submit, wait, or infer completion for them.

## Performance constraints

- Do not perform global locking or native reference counting per draw call.
  Retention changes occur on binding update, graph recording, and submission.
- Command encoders use preallocated small vectors and deduplicate resource keys
  before submission.
- Reclamation is batched once per frame/submission or when pending bytes cross a
  threshold; it is not a full slot-table scan.
- Slot lookup remains generation-checked and O(1).
- Warm invocation must not allocate solely because resource references are
  unchanged.
- Track pending resource count and bytes so memory growth is observable.

The expected steady-state cost is O(unique resources touched by a submission),
not O(draw calls). Removing unconditional queue waits should outweigh this
bookkeeping once multiple submissions are in flight.

## Implementation phases

### Phase 1: stable logical records

- Add stable owned/borrowed resource records for buffer, image, and sampler
  slots.
- Make public destruction invalidate handles immediately while deferring slot
  reuse until binding references reach zero.
- Replace DirectX adapter COM bookkeeping and CUDA/Vulkan/OpenGL no-op retention
  with record retention.
- Keep current synchronous submission behavior.

Acceptance:

- destroying an owner before a binding set never reuses or aliases its record;
- all backends pass slot-reuse tests equivalent to the DirectX regression test;
- repeated binding updates perform no unbounded allocation.

### Phase 2: in-flight serial tracking

- Add per-device submission/completion serials.
- Have command encoders collect touched records.
- Add backend fence/event completion queries and a deferred-destruction queue.
- Add explicit synchronization and deterministic shutdown draining.

Acceptance:

- resources released immediately after submit remain valid until GPU
  completion;
- destruction occurs after the correct fence/event without device-wide idle;
- borrowed resources follow the external completion contract.

### Phase 3: deferred `ExecutionGraph`

- Let compiled/recorded graphs own resource-record references.
- Release references for culled, replaced, failed, or abandoned graph work.
- Permit asynchronous submit and multiple frames in flight.
- Remove backend-local synchronous waits that are no longer required for ring
  or command allocator reuse.

Acceptance:

- at least two submissions can remain in flight;
- graph execution remains deterministic and does not retain resources after its
  final completion;
- CPU encode time, queue wait time, pending bytes, and reclamation latency are
  benchmarked separately.

## Verification

- Unit tests cover owner destruction before binding destruction, slot
  generation reuse, transactional binding replacement, and graph cancellation.
- Backend tests cover buffer, image, sampler, attachment, and imported-resource
  lifetimes.
- Stress tests repeatedly create/destroy resources while multiple submissions
  remain in flight.
- ASan validates host-side records; Vulkan validation, D3D12 debug layer,
  CUDA memcheck, and OpenGL debug output validate backend use.
- Shutdown tests prove that owned pending resources are drained and borrowed
  resources are not destroyed.

## Migration rule

Implement one stable-record contract across every enabled GPU backend before
enabling deferred graph execution. Delete raw slot-pointer retention and no-op
retention in the same change; do not add compatibility branches or silently
fall back to synchronous behavior.
