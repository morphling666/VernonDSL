# Debug labels and timings

Status: labels (section 2) and timings (section 3) implemented (Engine
extension E8, profiling), and the engine uses them as section 5 describes;
awaiting the `PROGRAM_VERSION` cut. Metal records labels, is unverified on
hardware, and reports no timings yet. The
cut amends the contracts listed in section 7.

A renderer is profiled by pass and inspected by draw: a capture tool
(RenderDoc, PIX, Xcode, Nsight) must show each pass under its name with its
draws grouped by what the renderer considers meaningful (material, object,
cascade), and the renderer must read each pass's device time without a
capture tool. The Runtime owns command recording (`design.md` section 8), so
only it can emit either. Both are defined per Node, so they cover compute
Nodes and every backend, not only graphics.

## 1. Rejected alternatives

- **Engine-side native markers.** `VernonHal.h` has no command encoder,
  because commands are recorded only from linked schedules; adding one
  violates the Runtime's ownership of recording.
- **Labels only per Node.** Captures of a forward pass with hundreds of draws
  are unreadable without groups inside the pass, and adding recording
  commands later would change the recording format.
- **Timestamps on every call by default.** Query writes and readback
  cost GPU and CPU time; profiling is opt-in.
- **Disabling scope fusion while profiling.** It would measure a different
  schedule from the one that ships.
- **Labels as identity.** Node names are diagnostic labels and never identity
  (`../program/execution_manifest.md` section 1).
- **NVTX push/pop ranges on CUDA.** They belong to the calling thread, and a
  call's ranges may end on another thread than the one that began them;
  start/end ranges belong to the process.
- **CPU labels.** No capture tool reads them; CPU profilers (Tracy, ETW,
  Instruments) have their own zones, and the Runtime takes no profiler
  dependency until a consumer needs one.
- **GPU-only timings.** A compute Node on CUDA or CPU is as much a pass as a
  render scope, and both backends time it cheaply (events, host clock).

## 2. Labels

Node groups:

- A Node's label is its manifest `name`. `vernonRuntimeProgramGraphAddProgram`
  gains an optional label argument; without one the name defaults to the
  child Program's asset id. Labels never enter specialization, artifact or
  resolve identity.
- A linked child with one Node names it with the label alone; a child with
  several Nodes names each `label/<name>`, falling back to the Node's stage
  when it has no name. Nesting under the label keeps captures readable when
  one child contributes several passes; the previous `node/<id>/<name>`
  prefix carried no information a capture reader can use.
- On a device created with `VERNON_HAL_DEVICE_LABELS`
  ([`hot_path_diagnostics.md`](hot_path_diagnostics.md) section 4), Execute
  wraps each Node's commands, including its barriers and render scope, in one
  debug group carrying the label. When graphics Nodes fuse into
  one render scope (`../program/graphics_execution.md` sections 7 and 9), the
  fused scope's group carries the member labels joined with `+`, and each
  member's draws are nested in a group with its own label.

Recording groups:

- `ProgramGraphicsRecording` gains `pushDebugGroup(label)`,
  `popDebugGroup()` and `insertDebugMarker(label)`. They execute in recording
  order inside the Node's group and have no effect on rendering.
- Pushes and pops must balance within one recording; otherwise the recording
  fails with `PROGRAM_RECORDING_INVALID`.
- In the C API they are the graphics command kinds `PUSH_DEBUG_GROUP`,
  `POP_DEBUG_GROUP` and `INSERT_DEBUG_MARKER`, with the label copied at
  record. Each keeps its position relative to the recording's draws.

Backends: GL `glPushDebugGroup`, `glPopDebugGroup` and
`glDebugMessageInsert` (GL 4.3, ES 3.2 or `KHR_debug`, whose ES entry points
carry the `KHR` suffix; no-ops without them, as on macOS OpenGL 4.1); Vulkan
`vkCmdBeginDebugUtilsLabelEXT`, `vkCmdEndDebugUtilsLabelEXT` and
`vkCmdInsertDebugUtilsLabelEXT` when `VK_EXT_debug_utils` is enabled; D3D12
`BeginEvent`, `EndEvent` and `SetMarker` on the command list; Metal
`pushDebugGroup`, `popDebugGroup` and `insertDebugSignpost` on the render
encoder inside a rendering scope and on the command buffer outside one; CUDA
NVTX `nvtxDomainRangeStartEx` and `nvtxDomainRangeEnd` in the `Vernon` domain
around each labelled dispatch's launch, at submission because CUDA enqueues
commands as they are submitted. CPU Programs have no labels.

- NVTX is header-only and loads a tool only when one sets
  `NVTX_INJECTION64_PATH` (Nsight Systems does), so it adds no runtime
  dependency.
- A labelled CUDA device launches on the stream rather than replaying a CUDA
  graph, as a timed call does (section 3): one graph launch would merge the
  ranges of its dispatches.
- OpenGL groups live on the context, and a recording's groups balance, so a
  call never leaves a group open there; other backends keep groups in the
  native command buffer.

- GL limits the group depth and the message length. A push at the context's
  maximum depth is skipped together with its pop, and a label is cut at the
  last UTF-8 code point that fits, so deep or long labels never raise GL
  errors.
- A Metal command buffer has no signposts; a marker outside rendering is an
  empty group with the marker's label.
- A Metal render encoder that restarts inside one rendering scope pops its
  open groups before it ends and pushes them again on the new encoder.

The HAL has no label functions of its own: labels reach a device only through
schedules, whose recordings are checked for balance as above, and a backend
without debug groups drops them when it builds its recording.

## 3. Timings

- `vernonScheduleSetTimings(schedule, enabled)` makes the schedule's later
  calls measure the device time of each timed scope; timing is off by
  default.
- A timed scope is one native render scope or one compute Node, which is the
  finest unit every backend can time: tile GPUs sample counters only at
  render-scope boundaries (Metal `MTLCounterSamplingPointAtStageBoundary`).
- `vernonScheduleGetScopeTimings(schedule, point, scopes, capacity, count)`
  fills `VernonScopeTiming { first_node, node_count, begin_nanoseconds,
  end_nanoseconds }` in execution order for the call that signalled `point`.
  An unfused Node is a scope with `node_count` 1; fused Nodes appear as one
  scope, and Node ids are contiguous in execution order
  (`../program/execution_manifest.md` section 10). The result describes the
  schedule that ran.
- A call is one device submission, so begin and end are measured from the
  begin of the call's first scope on one clock. Calibrating that clock to the
  host clock is a later extension.
- Timings are a completion fact, so they belong to the call's timeline point
  (`architecture_v2.md` section 8): the getter returns
  `VERNON_STATUS_NOT_READY` until a `vernonScheduleWait` for the point
  returned, and a device keeps only its latest few calls' timings. A failed
  call reports no scopes: scope timings describe only work that succeeded. A
  renderer with frames in flight reads frame `f - N` from the point it waits
  on.
- Each scope has a begin and an end edge, recorded outside rendering
  scopes. The device's query storage belongs to the recording's frame slot
  and is read when the call retires.
- Edge pairs instead of raw timestamps: Metal samples counters only when an
  encoder begins or ends (`MTLCounterSamplingPointAtStageBoundary` on tile
  GPUs), so a timestamp inside a rendering scope has no Metal equivalent,
  while a begin/end pair outside rendering scopes maps to every backend.
  Spans instead of elapsed times: overlap and gaps between scopes are what a
  frame timeline shows.
- Backends: GL `glQueryCounter(GL_TIMESTAMP)` on queries from the device's
  pool (GL 3.3 or `ARB_timer_query`; ES via `EXT_disjoint_timer_query`,
  whose disjoint flag fails the read where the driver accepts
  `GL_GPU_DISJOINT_EXT`); Vulkan timestamp query pools, reset on the command buffer
  before their first write, when the queue's `timestampValidBits` is not 0;
  CUDA `cuEventRecord`, read with `cuEventElapsedTime` from the first event,
  whose half-microsecond resolution bounds the span. D3D12 uses timestamp
  query heaps resolved with `ResolveQueryData` and `GetTimestampFrequency`;
  Metal timings are to use an empty blit encoder per edge, sampled
  at its start into a timestamp counter sample buffer and scaled to
  nanoseconds with `sampleTimestamps` pairs; a GPU that drops the empty
  encoder reports an error sample, which fails the read.
- A span covers the time the device reached its end edge after its begin
  edge. Commands are recorded before the call (section 8), so a span does not
  cover host recording time; on GL and CUDA, which execute as the call issues
  them, it covers the driver's per-command cost of the call.
- A backend without timing support (GL ES without
  `EXT_disjoint_timer_query`, Vulkan without timestamp bits, Metal without
  stage-boundary sampling) reports `VERNON_STATUS_UNSUPPORTED_TARGET` from the
  getter; a call never fails for lack of timing support. A call made while
  timing was off reports `VERNON_STATUS_UNSUPPORTED_TARGET` too.
- The CPU device times each dispatch with the host steady clock. A timed CUDA
  call launches on the
  stream with an event pair around each dispatch rather than replaying its
  CUDA graph, so its spans include the per-launch gaps the graph replay
  avoids. Event record nodes in the graph would time the replay itself; they
  wait until a profile needs that precision.

## 4. Telemetry

`vernonScheduleGetGraphicsTelemetry` reports the render scopes and draws of a
schedule's last call: counts are CPU facts known when the call returns; times
are device facts known at completion and have their own getter.

## 5. Engine use (non-normative)

The FrameGraph labels its nodes `depth`, `forward` and `display`, groups
forward draws by material instance, and reports the timed scopes of frame
`f - 2`, the call it waits on before calling frame `f`, in its telemetry.

## 6. Acceptance

Labels, on every backend with graphics support and on CUDA for compute:

1. A capture of a three-Node graph shows three labelled groups, each
   containing exactly its Node's render scope or dispatches, and a recording
   with nested groups shows them nested inside its Node.
2. An unbalanced recording fails with `PROGRAM_RECORDING_INVALID`.
3. A recording with debug groups draws the same pixels with and without
   `LABELS`.

Timings, on every backend:

4. With timings enabled, each scope reports `begin <= end`, and a scope that
   does 100 times the work of another reports a longer span.
5. With timings disabled, a call creates no queries or events.
6. Fused Nodes appear as one scope with `node_count` greater than 1.
7. A call's scopes report `NOT_READY` until a wait for its point returns.

## 7. Contract changes on acceptance

- [`design.md`](design.md): schedule calls emit debug groups and optional
  timings (applied in section 8).
- [`../program/recorded_draws.md`](../program/recorded_draws.md): the
  recording gains the three debug commands and the balance rule (applied).
- [`../program/execution_manifest.md`](../program/execution_manifest.md):
  `name` is also the debug label (applied).
- `VernonRuntime.h` and `VernonSchedule.h` (`PROGRAM_VERSION`): the label
  functions, command kinds and `AddProgram` label are in place, as are
  `vernonScheduleSetTimings` and `vernonScheduleGetScopeTimings`.

## 8. Recorded spans on GL and CUDA

The OpenGL device executes a tape recorded when the schedule is built, and
the CUDA device launches a linked schedule's dispatches at the call
(`architecture_v2.md` section 16.3). Scope edges therefore reach the device
back to back with the commands between them, as on Vulkan, and a span does
not cover host recording time. Validation stays at link and record, so a
call that breaks nesting, scope or binding rules fails before it runs; native
driver errors fail the call's completion
([`hot_path_diagnostics.md`](hot_path_diagnostics.md) section 2).

What remains is the driver's per-command cost while the call runs, and on
every backend a span is device wall time, which includes device stalls.
Device busy time needs vendor counters (CUPTI activity records, GL
performance monitor extensions) and stays out of scope.
