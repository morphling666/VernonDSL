# Deferred GPU resource lifetime plan

## Status

Phase 1 stable logical resource records are implemented for the enabled GPU
backends. Public destruction invalidates a generation-checked handle while a
`LogicalResourceRecord` keeps the native object alive for retained prepared
bindings. Current submission remains synchronous.

Phases 2 and 3 are deferred and non-normative. They become active only when
Vernon supports asynchronous submission, multiple frames in flight, or
recorded work that outlives the declaring call.

## Current contract

- VernonRHI owns native resources, queues, submission, and completion.
- RuntimeCore retains opaque provider references and does not depend on
  VernonRHI or native graphics SDK types.
- `VernonRuntimeRHIAdapter` maps provider references to stable logical records.
- Public handles use slot index plus generation; recyclable slot addresses and
  backend-native handles are not stable identities.
- Prepared binding updates retain new records before releasing old records.
- Borrowed native resources remain owned and synchronized by their external
  owner.
- Synchronous submission consumes references before execution returns.

This contract is valid for the current single-submission lifetime. It does not
permit removing queue waits or allowing work to remain in flight after owners
are released.

## Activation gates

Implement the deferred phases when at least one condition becomes a supported
product requirement:

1. `ExecutionGraph` records work that may execute after the declaring call
   returns.
2. More than one frame or submission may remain in flight.
3. Queue synchronization is a measured material bottleneck.
4. Prepared bindings intentionally persist independently from public wrappers.

Do not bump an ABI merely to reserve fields before one of these gates is
accepted.

## Target resource model

Each owned RHI resource keeps one stable logical record:

```text
LogicalResourceRecord
  public_generation
  owner_alive
  binding_reference_count
  last_submission_serial
  native_resource
  backend_state
  ownership: owned | borrowed
```

Two conditions control reclamation:

1. Logical lifetime ended: the public owner is gone and no prepared binding or
   unsubmitted command retains the record.
2. Execution lifetime ended: the device completion serial reached the record's
   last submission serial.

Owned native resources are destroyed only when both conditions hold. Borrowed
records never destroy the native object and require an external completion
contract.

## Phase 2: in-flight serial tracking

Required work:

- add monotonically increasing per-device submission serials;
- make command encoders collect deduplicated touched records;
- stamp touched records after successful submission;
- query backend completion without device-wide idle;
- reclaim records in bounded batches;
- drain owned pending resources deterministically at shutdown.

Completion sources:

- DirectX 12 queue fences;
- Vulkan timeline semaphores where available, otherwise fence serials;
- CUDA stream events;
- OpenGL/OpenGL ES `GLsync` in asynchronous mode.

Acceptance:

- resources released immediately after submit remain valid until completion;
- destruction occurs after the correct serial without an unconditional
  device-wide wait;
- failed or abandoned commands release unsubmitted references;
- borrowed resources follow the external completion contract;
- pending resource count and bytes are observable.

## Phase 3: deferred ExecutionGraph

Required work:

- let compiled or recorded graphs own logical record references;
- release references for culled, replaced, failed, or abandoned graph work;
- permit multiple submissions in flight;
- move synchronization to explicit readback, compatibility, and shutdown
  boundaries;
- preserve deterministic scheduling and hazard behavior.

Acceptance:

- at least two owned submissions can remain in flight;
- graph completion releases all retained records;
- owner destruction and slot reuse remain safe for buffers, images, samplers,
  attachments, and imported resources;
- CPU encode time, queue wait, pending bytes, and reclamation latency are
  benchmarked separately.

## Performance constraints

- no global lock or native reference count update per draw call;
- unchanged warm bindings allocate nothing solely for lifetime tracking;
- retention changes occur at binding update, recording, and submission;
- record lookup remains generation-checked and O(1);
- reclamation is batched rather than scanning every slot each frame;
- steady-state work is proportional to unique resources touched by a
  submission, not draw count.

## Verification

- unit tests cover owner destruction, generation reuse, transactional binding
  replacement, cancellation, and shutdown;
- backend tests cover owned and borrowed buffers, images, samplers, and
  attachments;
- stress tests repeatedly create and destroy resources across overlapping
  submissions;
- ASan, Vulkan validation, D3D12 debug layer, CUDA memcheck, and OpenGL debug
  output validate the applicable layers.

Implement one record and completion contract across every enabled backend
before enabling deferred execution. Do not keep raw-pointer and stable-record
paths in parallel.
