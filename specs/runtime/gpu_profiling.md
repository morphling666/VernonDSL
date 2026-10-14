# GPU labels and timestamps

Status: proposed (Engine extension E8, profiling). Depends on
[`pipelined_submission.md`](pipelined_submission.md) for timing readback. On
acceptance it amends the contracts listed in section 7 under one
`PROGRAM_VERSION` cut.

A renderer is profiled by pass and inspected by draw: a capture tool
(RenderDoc, PIX, Xcode, Nsight) must show each pass under its name with its
draws grouped by what the renderer considers meaningful (material, object,
cascade), and the renderer must read each pass's GPU time without a capture
tool. The Runtime owns command recording (`design.md` section 8), so only it
can emit either.

## 1. Rejected alternatives

- **Engine-side native markers.** The Engine has no command encoder; adding
  one violates the Runtime's ownership of recording.
- **Labels only per Node.** Captures of a forward pass with hundreds of draws
  are unreadable without groups inside the pass, and adding recording
  commands later would change the recording format.
- **Timestamps on every invocation by default.** Query writes and readback
  cost GPU and CPU time; profiling is opt-in.
- **Disabling scope fusion while profiling.** It would measure a different
  schedule from the one that ships.
- **Labels as identity.** Node names are diagnostic labels and never identity
  (`../program/execution_manifest.md` section 1).

## 2. Labels

Node groups:

- A Node's label is its manifest `name`. `vernonRuntimeProgramGraphAddProgram`
  gains an optional label argument; without one the name defaults to the
  child Program's asset id. Labels never enter specialization, artifact or
  resolve identity.
- Execute wraps each Node's commands, including its barriers and render
  scope, in one debug group carrying the label. When graphics Nodes fuse into
  one render scope (`../program/graphics_execution.md` sections 7 and 9), the
  fused scope's group carries the member labels joined with `+`, and each
  member's draws are nested in a group with its own label.

Recording groups:

- `ProgramGraphicsRecording` gains `pushDebugGroup(label)`,
  `popDebugGroup()` and `insertDebugMarker(label)`. They execute in recording
  order inside the Node's group and have no effect on rendering.
- Pushes and pops must balance within one recording; otherwise the recording
  fails with `PROGRAM_RECORDING_INVALID`.

Backends: GL `glPushDebugGroup`, `glPopDebugGroup` and
`glDebugMessageInsert` (GL 4.3 or `KHR_debug`; no-ops without them, as on
macOS OpenGL 4.1); Vulkan `vkCmdBeginDebugUtilsLabelEXT`,
`vkCmdEndDebugUtilsLabelEXT` and `vkCmdInsertDebugUtilsLabelEXT` when
`VK_EXT_debug_utils` is enabled; D3D12 `BeginEvent`, `EndEvent` and
`SetMarker` on the command list; Metal `pushDebugGroup`, `popDebugGroup` and
`insertDebugSignpost` on the active encoder.

RHI: `vernonRhiCommandEncoderPushDebugGroup(encoder, label)`,
`vernonRhiCommandEncoderPopDebugGroup(encoder)` and
`vernonRhiCommandEncoderInsertDebugMarker(encoder, label)`. Unsupported
backends accept all three as no-ops. Labels are copied at the call, so callers
may pass temporaries.

## 3. Timestamps

- `ProgramInvocationOptions` gains `gpu_timings`. When set, Execute writes a
  timestamp at the start and end of each timed scope.
- A timed scope is one native render scope or one compute Node, which is the
  finest unit every backend can time: tile GPUs sample counters only at
  render-scope boundaries (Metal `MTLCounterSamplingPointAtStageBoundary`).
- `vernonSubmissionGetGpuScopes(submission, scopes, capacity, count)` fills
  `VernonGpuScopeTiming { first_node, node_count, nanoseconds }` in execution
  order. An unfused Node is a scope with `node_count` 1; fused Nodes appear as
  one scope. The result describes the schedule that ran.
- Timings are a completion fact, so they belong to the submission
  (`pipelined_submission.md` section 2), not the invocation: the getter
  returns `VERNON_STATUS_NOT_READY` while the submission is pending and stays
  valid until the submission is destroyed. A renderer with frames in flight
  destroys each invocation after commit and reads frame `f - N` from the
  submission it waits on.
- RHI: `vernonRhiDeviceCreateTimestampQueries(device, count, output)`,
  `vernonRhiCommandEncoderWriteTimestamp(encoder, queries, index)` and
  `vernonRhiDeviceGetTimestampResults(device, queries, first, count, ns)`,
  converting ticks to nanoseconds inside the RHI. Runtime allocates queries
  per submission and recycles them on completion.
- Backends: GL `glQueryCounter(GL_TIMESTAMP)` (GL 3.3, ES via
  `EXT_disjoint_timer_query`); Vulkan timestamp query pools when
  `timestampComputeAndGraphics` holds; D3D12 timestamp query heaps resolved
  with `ResolveQueryData` and `GetTimestampFrequency`; Metal counter sample
  buffers at stage boundaries.
- A backend without timestamps reports `VERNON_STATUS_UNSUPPORTED_TARGET`
  from the getter; Execute never fails for lack of timing support.

## 4. Telemetry

`VernonProgramGraphicsNodeTelemetry` is unchanged: counts are CPU facts known
at Execute; times are GPU facts known at completion and have their own getter.

## 5. Engine use (non-normative)

The FrameGraph labels its nodes `depth`, `forward` and `display`, groups
forward draws by material instance, and reports the timed scopes of frame
`f - 2`, the submission it waits on before recording frame `f`, in its
telemetry.

## 6. Acceptance

On every backend with graphics support:

1. A capture of a three-Node graph shows three labelled groups, each
   containing exactly its Node's render scope, and a recording with nested
   groups shows them nested inside its Node.
2. An unbalanced recording fails with `PROGRAM_RECORDING_INVALID`.
3. With `gpu_timings`, each scope reports a positive time, and a scope that
   draws 100 times the work of another reports a larger time.
4. Without `gpu_timings`, Execute creates no queries.
5. Fused Nodes appear as one scope with `node_count` greater than 1.
6. An asynchronous submission's scopes report `NOT_READY` until it completes
   and remain readable after its invocation is destroyed.

## 7. Contract changes on acceptance

- [`design.md`](design.md): Execute emits debug groups and optional
  timestamps; the RHI functions of sections 2 and 3.
- [`../program/recorded_draws.md`](../program/recorded_draws.md): the
  recording gains the three debug commands and the balance rule.
- [`../program/execution_manifest.md`](../program/execution_manifest.md):
  `name` is also the debug label.
- `VernonRuntime.h` and `VernonRHI.h` (`PROGRAM_VERSION`).
