# Asynchronous submission

Status: accepted. Phase 1 (section 11) is implemented on every GPU backend
and its contract changes (section 12) are applied; phases 2 to 5 are
proposed. Replaces the earlier pipelined-mode draft. Prerequisite for Engine
frames in flight, E8 timing readback, asynchronous training loops and the
multi-graph scheduler (E9). The contract changes ship under one
`PROGRAM_VERSION` cut.

Execute ends in `executeCommandPlanAndWait`, then reads back and copies
commit-after-success results on the host before it returns. Every invocation
therefore alternates CPU and GPU work: a renderer loses about half its frame
throughput, and a training loop serializes forward, backward and optimizer
steps behind host waits. This document makes every Execute return after
submission. Effects that depend on completion are attached to the
submission instead of being performed inside Execute, so the same mechanism
serves graphics frames, compute and autodiff. Section 13 relates it to the
future distributed compiler; section 14 shows the layers and how a renderer
and a training loop run.

## 1. Rejected alternatives

- **An opt-in pipelined mode limited to plans whose writes are in place and
  device resident** (the earlier draft). It fits graphics frames but excludes
  commit-after-success results, host readbacks and retained pullbacks, which
  is almost every training step. A later extension would have to replace its
  core.
- **Publishing completion-defined effects before completion.** Commit-after-
  success results and host readbacks would expose unfinished or failed data.
- **Host commit copies after a CPU wait** (today). Correct, but every
  commit-after-success boundary costs a host round trip.
- **Readback into caller memory filled at completion.** Every caller would
  have to keep the destination alive and untouched until completion, a rule
  that fails silently when broken.
- **Accepting host memory for a writable boundary of an asynchronous
  invocation and leaving it untouched**, with the bytes only in
  `vernonSubmissionReadBoundary`. A binding that looks like an output but is
  never written fails silently.
- **Runtime guards**: a device status word that device-side checks set,
  commit copies and later dispatches predicated on it, and readers that
  inherit the words of the writers they read. No check needs it: a failed
  submission is almost always device loss, which no predicate recovers, and
  the one data-dependent check, residual carrier overflow, is confined to
  `WAIT` (section 4). Predication would need indirect draws, which OpenGL
  3.3, OpenGL ES 3.0 and WebGL2 lack, and CUDA graph conditional nodes,
  which need the CUDA device runtime and so a toolkit-dependent build.
  Render-pass clears cannot be predicated on any API, and inheritance would
  need caller-resource tracking and acknowledgement rules. A conditional
  update, such as skipping an optimizer step whose gradients are not finite,
  is dataflow in the Program (`where(all_finite(g), params - lr * g,
  params)`), which every backend executes in queue order.
- **Automatic Runtime retry of a failed asynchronous invocation.** Later
  submissions that read its results are already queued; re-executing it
  after them breaks queue order.
- **Runtime-owned frame or step pacing.** How many frames or steps run ahead
  is application latency policy. Runtime bounds only its own resources
  (section 7).
- **Engine-owned fences or queues.** Runtime owns recording and submission
  (`design.md` section 8); callers observe completion only through
  `VernonSubmission`.

## 2. Model

- Execute validates, plans, allocates, records and submits, then returns.
  Every failure before submission is synchronous and leaves every boundary
  unchanged, as today.
- Every submitting Execute hands out one `VernonSubmission` through
  `vernonRuntimeProgramInvocationTakeSubmission`. Its state is `PENDING`,
  `SUCCEEDED` or `FAILED`; `vernonSubmissionWait` and state queries are
  unchanged. Completion facts (the completion outcome, readback bytes, GPU
  timings of [`gpu_profiling.md`](gpu_profiling.md)) belong to the
  submission, so the invocation can be destroyed right after Commit.
- The outcome has two phases:
  - **Execute outcome**: overall `NOT_SUBMITTED` or `SUBMITTED`; each
    writable boundary or RenderPass control `UNCHANGED` (it failed before
    submission or is not written) or `SUBMITTED`.
  - **Completion outcome**, read from the submission once it completed:
    each written boundary `COMMITTED`, `IN_PLACE_COMMITTED` or
    `INDETERMINATE` (section 3.5).
- A `WAIT` invocation (the C API default) waits before Execute returns and
  reports both phases at once, so existing callers see today's outcomes.
  It is submission followed by a wait, not a separate execution path.
- Commit publishes binding revisions immediately; they are CPU state and
  immutable once published.
- Device work of one RHI device executes in submission order on its queue,
  so a later invocation that reads what an earlier one wrote needs no CPU
  wait. Ordering across queues arrives with E9 as submission dependencies on
  queue timelines (section 9).
- Dependencies are points on queue timelines, keyed by (device, queue,
  value). Pending owners (section 5) record the point after which the
  access that last wrote or read them is complete, and a later access
  depends only on the points of the owners it touches. A
  `VernonSubmission` is a host future over the points it signals. On one
  queue a submission signals one point; when E9 splits an invocation's
  nodes across queues, each node group signals its own point, so
  independent nodes of consecutive frames or steps overlap without changing
  owner bookkeeping.
- Graphics nodes keep their semantics: primal-only, attachments written in
  place, recorded draws in recording order. Compute nodes keep value
  semantics and both publication modes. This document changes when effects
  happen, never what a node means.

## 3. Completion effects

### 3.1 In-place writes

Queue-ordered device writes, including render-pass attachments
(`VERNON_MUTATION_RENDER_PASS_CONTROL`). Completion reports
`IN_PLACE_COMMITTED`.

### 3.2 Commit-after-success writes

Results are produced into Runtime staging, so the invocation's own reads of
a destination see its previous contents. The commit is a buffer or image
copy from staging into the destination, recorded in the same submission
after the work. Commands of a submission execute in order, so the copy runs
only after the work it publishes; completion reports `COMMITTED`. Every
image format supports a transfer copy, so no destination requires `WAIT`.

### 3.3 Host readbacks

A writable boundary whose result the caller reads on the host is bound as a
readback slot: a Tensor argument with storage `VERNON_TENSOR_READBACK`, its
dtype and shape, and no memory. The submission copies the result into
host-visible staging it owns. After successful completion,
`vernonSubmissionReadBoundary(submission, slot, destination, byte_size)`
copies the slot's bytes, dense in row-major order, into caller memory. It
returns:

- `VERNON_STATUS_NOT_READY` while the submission is pending;
- `VERNON_STATUS_INTERNAL_ERROR` once it failed;
- `VERNON_STATUS_INVALID_ARGUMENT` for a slot that is not a readback slot
  or a `byte_size` other than the dense size.

The staging is released with the submission. Readback slots work in both
modes. An asynchronous Execute that binds host memory to a writable
boundary fails before submission with `VERNON_STATUS_INVALID_ARGUMENT`,
naming the boundary. The rule holds on every backend, including the CPU
backend whose work finishes inside Execute, so an invocation that runs
asynchronously on one backend runs on all of them. `WAIT` invocations keep
host-memory destinations. Python presents readback slots as values that
synchronize on first host access (section 8).

### 3.4 Pullbacks

A retained pullback is published when Execute returns; its residuals are
device storage it retains. Applying it reads them in queue order, so the
forward, backward and optimizer invocations of a training step queue back
to back without a CPU wait. The apply is itself an asynchronous invocation
with its own submission.

A static Tape writes compiler-assigned fixed residual offsets and needs no
validation. A Tape with dynamic regions may overflow its residual carrier,
which only its lane-status readback reveals, so its plan requires `WAIT`
(section 4) and keeps today's automatic replay.

### 3.5 Failure after submission

A submission that fails after the device accepted it reports every boundary
it writes `INDETERMINATE`, in place or commit-after-success alike, in both
modes: its commands stop at an unknown point, which may be inside a commit
copy. Its readbacks are unreadable. The owners it writes are poisoned
(section 5), and a pending submission that reads one of them reports its own
writes `INDETERMINATE` too, because it may have run on the unfinished data.
Retirement resolves outcomes in submission order, so that propagation needs
no device work. A failure after submission is normally device loss, which
also poisons the context.

## 4. When an invocation requires `WAIT`

Resolve records on the `ResolvedExecutionPlan` whether the plan can complete
asynchronously. An asynchronous Execute of a plan that cannot fails before
submission with `PROGRAM_SUBMISSION_REQUIRES_WAIT`, naming the first
blocking Tape; it never silently waits. A plan requires `WAIT` when it
writes a Tape with dynamic regions (section 3.4).

Everything else is asynchronous: graphics frames, compute with either
publication mode, readbacks, retained pullbacks of static Tapes and their
applies.

## 5. Owners and host access

- Every owner an asynchronous invocation accesses records the timeline
  point of the accessing node as a pending writer or reader until the
  submission's completion outcome is resolved. This
  covers Runtime-owned storage (instance Storage, pullback residuals,
  staging) and Python `ResourceControlBlock`s, whose claims stay held until
  completion instead of being released at Execute.
- A host read waits for the owner's pending writers. A host write waits for
  its pending writers and readers. A later device claim of the same session
  and queue waits for nothing: queue order already orders it.
- A completion outcome of `INDETERMINATE` poisons exactly the owners the
  submission writes, through the existing poison path (`design.md`
  section 1).
- Caller-owned RHI resources are not tracked; their owner learns a failure
  from the completion outcome. RHI host uploads follow the memory class of
  the destination:
  - `VERNON_RHI_MEMORY_DEVICE`: queue-ordered. The upload takes effect after
    all work submitted earlier on the device and before all work submitted
    later, with no CPU wait. CUDA, Vulkan, D3D12 and Metal enqueue the
    staged copy and retire the staging with that submission; OpenGL has this
    from `glBufferSubData`.
  - `VERNON_RHI_MEMORY_UPLOAD` and Metal shared storage: an immediate CPU
    write into mapped memory, which cannot be ordered behind GPU work. The
    caller keeps a region per submission in flight and reuses it after that
    submission completes.
- Resources an invocation uses stay retained until its submission completes
  (`design.md` section 13).

## 6. Per-submission storage

Everything an Execute allocates for its own use belongs to its submission and
is retired when the submission completes: the execution workspace (instance
workspaces become a pool), endpoint carriers, upload staging, inline uniform
storage, transient descriptors, commit staging and readback staging. No
storage that a pending submission reads is reachable by a later invocation.

## 7. Resource bounds

Context creation sets two bounds on Runtime's own resources:

- `transient_budget_bytes`: bytes of per-submission storage across pending
  submissions. The default is provisional, sized so that two frames of the
  Engine FrameGraph fit, and is revisited with measurements; it is a tuning
  value, not part of the contract.
- `max_pending_submissions`: submissions not yet retired, bounding retained
  objects and how late a failure surfaces.

A third bound belongs to the RHI device: its `max_in_flight_submissions`
command limit. One Runtime submission may hold several RHI submissions, so
the context counts the RHI submissions its pending submissions hold, and an
Execute is charged with the count its instance's previous Execute made (one
before the first). The limit is read at every admission, because the
device's owner may change it. A device shared with other contexts or with
the Engine has submissions this count does not see; their owners split the
limit between them.

When an Execute would exceed any of the three, it first waits for the oldest
pending submission. This is the only implicit wait and it never changes an
outcome.
Per-context counters report the waits of each kind and their total wait
time, so a renderer or trainer can tell that it is pacing on Runtime bounds
rather than on its own limit. How far frames or steps run ahead stays caller
policy.

## 8. Python

Python is the training front end, so it becomes asynchronous by default:

- Program, Kernel and pipeline calls return after submission. Host access to
  any written owner (`numpy()`, item reads, host writes) synchronizes
  through the claims of section 5.
- `vd.synchronize()` waits for every pending submission of the session.
- An asynchronous failure raises at the first synchronization that observes
  it, with the invocation's diagnostic.
- `VERNON_SYNCHRONOUS_EXECUTION=1` makes every call wait, so a failure
  raises at its call site while debugging.

Python switches its default only after the claim tests of section 11 pass on
every backend; until then asynchronous execution is opt-in.

## 9. RHI requirements

- Completions are values on a per-queue timeline (Vulkan timeline
  semaphores, D3D12 fence values, Metal shared events, OpenGL fence syncs),
  so retirement and later cross-queue dependencies (E9) compare values
  instead of polling objects.
- Device-memory uploads are queue-ordered without a CPU wait (section 5).
- Buffer, image and image-to-buffer copies recorded into a submission, for
  commit copies and readbacks (section 3).
- Staging and upload rings retire by completion value instead of resetting
  after a wait.
- An OpenGL or OpenGL ES device is bound to the thread that owns its
  context, because a context is current on one thread at a time: its
  submissions, completion polls and waits come from that thread. CUDA,
  Vulkan, D3D12 and Metal devices accept them from any thread.

## 10. API

- `VernonProgramInvocationOptions` (C) and `ProgramInvocationOptions` (C++)
  with `completion_mode` (`WAIT` default, `ASYNCHRONOUS`).
- `VernonInvocationSubmissionState` gains `VERNON_INVOCATION_SUBMITTED`;
  `VernonBoundaryMutationState` gains `VERNON_BOUNDARY_MUTATION_SUBMITTED`.
- `vernonRuntimeProgramInvocationTakeSubmission(invocation, output)`, valid
  once after a submitting Execute and before destroy, in both modes.
- `vernonSubmissionGetMutationOutcome` and `vernonSubmissionReadBoundary`;
  both return `VERNON_STATUS_NOT_READY` while the submission is pending.
- `VernonTensorStorage` gains `VERNON_TENSOR_READBACK` (section 3.3).
- `VernonCommon.h` gains `VERNON_STATUS_NOT_READY`.
- `VernonRuntimeCreateOptions` gains `transient_budget_bytes` and
  `max_pending_submissions`; `vernonRuntimeContextGetSubmissionTelemetry`
  reports the pending RHI submissions and the waits of section 7.
- Diagnostic `PROGRAM_SUBMISSION_REQUIRES_WAIT`.

## 11. Phases and acceptance

Each phase runs on every GPU backend and keeps the previous phases green.

1. **Core**: submissions for every Execute, per-submission storage, resource
   bounds, queue-ordered uploads, pending owners. Asynchronous plans are
   those whose writes are all in place; other plans report
   `PROGRAM_SUBMISSION_REQUIRES_WAIT` until phase 2.
   - Two asynchronous invocations, the second sampling the image the first
     renders, produce the same pixels as two `WAIT` invocations, with no CPU
     wait between them.
   - A per-frame uniform updated for 64 asynchronous frames with no caller
     waits produces its own frame's value in every frame; so does a
     device-memory vertex buffer rewritten every frame with
     `vernonRhiDeviceUploadBuffer`, which returns without waiting.
   - Execute of a GPU-heavy frame returns while its submission is pending.
   - Exceeding either bound waits for the oldest submission and increments
     its counter.
   - A forced submission failure poisons exactly the owners it writes.
2. **Deferred effects**: commit-after-success copies and readbacks recorded
   in the submission.
   - A commit-after-success compute invocation followed by one reading its
     result matches `WAIT` execution with no CPU wait between them, for
     buffer and image destinations.
   - A forced failure after submission reports every boundary it writes
     `INDETERMINATE`, poisons them, and makes a queued reader of one of
     them report its own writes `INDETERMINATE`; a queued invocation that
     reads none of them completes. A failure before submission leaves every
     boundary `UNCHANGED`.
   - A readback is `NOT_READY` while pending, exact after completion and
     unreadable after a failure.
   - An asynchronous Execute that binds host memory to a writable boundary
     fails before submission and names the boundary.
3. **Autodiff**: asynchronous pullbacks and applies of static Tapes.
   - 64 training steps (forward, retained pullback, apply, in-place
     optimizer) queued without waits match the same steps under `WAIT`.
   - An asynchronous Execute of a plan with a dynamic Tape fails with
     `PROGRAM_SUBMISSION_REQUIRES_WAIT` naming it, and the same plan under
     `WAIT` still replays after an overflow.
4. **Python default**: claims held until completion.
   - A host read of a buffer written by a pending submission returns the
     completed contents; a host write to a buffer a pending submission
     reads waits for it; a host read of a buffer it only reads does not.
5. **E9**: cross-queue submission dependencies on queue timelines.

## 12. Contract changes on acceptance

- [`design.md`](design.md) sections 1, 2, 6, 8 and 13: two-phase outcomes,
  pending owners, deferred commits, per-submission storage and resource
  bounds.
- [`../program/execution_manifest.md`](../program/execution_manifest.md):
  `PROGRAM_SUBMISSION_REQUIRES_WAIT`, the asynchronous publication of
  commit-after-success boundaries, and section 3.5's failure rule, which
  replaces restoring a borrowed destination after a failure that follows
  submission.
- [`gpu_profiling.md`](gpu_profiling.md): timings are completion facts of
  the submission.
- `VernonRuntime.h`, `VernonCommon.h`, the C++ wrapper and the Python
  runtime (`PROGRAM_VERSION`).

## 13. Relation to the distributed compiler

[`../future/architecture.md`](../future/architecture.md) assigns queue,
engine, lifetime and residency to the compiler's asynchronous task graph
(section 4.4) and forbids Runtime from inventing synchronization or
schedules (section 4.7). This design keeps that split:

- Within one invocation, the order and queue of nodes come from the
  `ResolvedExecutionPlan`. Today Runtime derives them (one device, the
  graphics or compute queue class); with the task graph they are its
  declared dependencies and placements, which Runtime validates and
  executes.
- Runtime owns only what no compile-time plan can know: the order between
  invocations the host issues (frames, training steps), through pending
  owners, plus the retirement of its own storage. It chooses no schedule,
  pipeline order, micro-batching or staleness; those belong to the plan or
  the caller.
- Timeline points are keyed by (device, queue), so a submission can later
  span the queues of several devices, and placed transfer and collective
  tasks join the same dependency model. One context per device is today's
  topology, not part of this contract.
- This document is the single-device part of the ownership,
  synchronization and failure contract that
  [`../roadmap.md`](../roadmap.md) section 4 requires before distributed
  execution reaches Runtime; topology, transport and deployment are added
  by that later contract.

## 14. Architecture and execution (non-normative)

### 14.1 Layers

```text
   Engine FrameGraph                    Python training loop
     Execute(ASYNCHRONOUS)                train_step(params, batch)
     TakeSubmission, Wait                 host access synchronizes
            |                                       |
            v                                       v
+------------------------------------------------------------------------+
| Runtime API                                                            |
|   invocation  -> Execute outcome (SUBMITTED)                           |
|   submission  -> completion outcome, ReadBoundary, GPU timings         |
+------------------------------------------------------------------------+
| RuntimeCore                                                            |
|   ResolvedExecutionPlan (requires WAIT?), publication modes,           |
|   pending owners (readers, writers), poison propagation                |
+------------------------------------------------------------------------+
| Runtime provider                                                       |
|   record and submit, per-submission storage, resource bounds,          |
|   retirement by completion value                                       |
+------------------------------------------------------------------------+
| VernonRHI                                                              |
|   queues with completion timelines, queue-ordered device uploads       |
+------------------------------------------------------------------------+
            |  command buffers                      ^  completion values
            v                                       |
                  GPU queue, executing in submission order
```

Graphics and compute share every layer. They differ only in node semantics
(section 2), which no layer below RuntimeCore sees.

### 14.2 One Execute and its submission

```text
CPU, inside Execute                                          returns
  validate -> plan -> allocate -> record -> submit  ------>  Execute outcome
     |                                                       SUBMITTED
     +-- any failure before submit: NOT_SUBMITTED, every boundary UNCHANGED

GPU, one submission, in queue order
  +-----------+------------------+-------------------+--------------------+
  | staged    | node work        | commit copies     | readback copies    |
  | uploads   | (dispatches,     | (staging into     | (into submission   |
  |           | draws)           | destinations)     | staging)           |
  +-----------+------------------+-------------------+--------------------+
                                                               -> completion value

On completion
  retire per-submission storage -> resolve the completion outcome
  -> release pending owners -> readback bytes and timings become readable
```

### 14.3 Renderer, two frames in flight

```text
time ------------------------------------------------------------------------>
CPU   [rec 0][rec 1][wait 0          ][rec 2][wait 1          ][rec 3]
GPU          [======= frame 0 =======][======= frame 1 =======][== frame 2 ...
              depth  forward  display
```

- `rec f`: update the view constants, record the depth, forward and display
  passes, Execute, keep the submission in a ring of N = 2.
- `wait f - 2` before recording frame `f`: it returns at once when that
  frame already finished, so the CPU blocks only when it is two frames
  ahead. E8 reads frame `f - 2`'s GPU timings at this point.
- Consecutive frames share the FrameGraph attachments. Queue order puts
  frame `f + 1`'s writes after frame `f`'s reads, so neither a CPU wait nor
  a copy separates them. Per-frame view constants live in per-submission
  storage, so recording frame `f + 1` never overwrites what frame `f` still
  reads.
- The drawing is GPU-bound: the GPU never idles. A CPU-bound renderer shows
  gaps between GPU frames and waits of zero length instead.

### 14.4 Training loop

```python
for k, batch in enumerate(loader):
    loss = train_step(params, batch)   # forward F, pullback apply B, optimizer O
    if k % 100 == 0:
        print(loss.item())             # first host access: waits for F of step k
```

```text
time ------------------------------------------------------------------------>
CPU   [F0][B0][O0][F1][B1][O1][F2][B2][O2] ... [F100][B100][O100][loss.item()....]
GPU       [ F0 ][ B0 ][ O0 ][ F1 ][ B1 ][ O1 ][ F2 ][ B2 ][ O2 ] ... [F100]
                                                                         ^ waits here only
```

Data flow of step `k`, all on the device:

```text
params_k, batch_k --F_k--> loss_k (readback slot), pullback_k (residuals retained)
pullback_k        --B_k--> grads_k      commit-after-success
params_k, grads_k --O_k--> params_k+1   in place
```

The CPU runs ahead until it reads the loss or reaches a resource bound
(section 7). Each read of `loss.item()` waits for the forward of that step
only; the backward and optimizer of the same step keep running.

### 14.5 Skipping a step

Skipping an optimizer step whose gradients are not finite is part of the
optimizer Program, not a Runtime feature:

```text
params_k, grads_k --O_k--> params_k+1 = where(all_finite(grads_k),
                                              params_k - lr * grads_k,
                                              params_k)
```

`F_k+1` reads `params_k+1` in queue order as usual. A skipped step leaves
the parameters unchanged, so later steps need no rejection and the caller
re-executes nothing. The optimizer can also return `all_finite(grads_k)`
in a readback slot, so the caller counts skipped steps without waiting for
each one.

### 14.6 Host access (Python, phase 4)

```text
x = model(batch)    submission S1 writes x       x:     pending writer S1
y = other(batch)    submission S2 reads batch    batch: pending reader S2
x.numpy()           host read of x               waits for S1 only
batch.write(new)    host write of batch          waits for S2, its pending reader
z = third(x)        device read of x, same queue no wait: queue order
```
