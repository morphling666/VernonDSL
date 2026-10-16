# Debug labels and timings

Status: labels (section 2) and timings (section 3) implemented (Engine
extension E8, profiling), and the engine uses them as section 5 describes;
awaiting the `PROGRAM_VERSION` cut. Metal is unverified on hardware. Timings
depend on [`pipelined_submission.md`](pipelined_submission.md) for timing
readback. The
cut amends the contracts listed in section 7.

A renderer is profiled by pass and inspected by draw: a capture tool
(RenderDoc, PIX, Xcode, Nsight) must show each pass under its name with its
draws grouped by what the renderer considers meaningful (material, object,
cascade), and the renderer must read each pass's device time without a
capture tool. The Runtime owns command recording (`design.md` section 8), so
only it can emit either. Both are defined per Node, so they cover compute
Nodes and every backend, not only graphics.

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
- **NVTX push/pop ranges on CUDA.** They belong to the calling thread, and an
  encoder may pop or be abandoned on another thread than the one that pushed;
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
NVTX `nvtxDomainRangeStartEx`, `nvtxDomainRangeEnd` and `nvtxDomainMarkEx` in
the `Vernon` domain, at record time because CUDA enqueues commands as they are
recorded. CPU Programs have no labels.

- NVTX is header-only and loads a tool only when one sets
  `NVTX_INJECTION64_PATH` (Nsight Systems does), so it adds no runtime
  dependency.
- OpenGL groups live on the context and NVTX ranges on the process, so an
  encoder destroyed with open groups closes them; other backends discard them
  with the native recording.

- GL limits the group depth and the message length. A push at the context's
  maximum depth is skipped together with its pop, and a label is cut at the
  last UTF-8 code point that fits, so deep or long labels never raise GL
  errors.
- A Metal command buffer has no signposts; a marker outside rendering is an
  empty group with the marker's label.
- A Metal render encoder that restarts inside one rendering scope pops its
  open groups before it ends and pushes them again on the new encoder.

RHI: `vernonRhiCommandEncoderPushDebugGroup(encoder, label)`,
`vernonRhiCommandEncoderPopDebugGroup(encoder)` and
`vernonRhiCommandEncoderInsertDebugMarker(encoder, label)`. Unsupported
backends accept all three as no-ops. Labels are copied at the call, so callers
may pass temporaries. The RHI validates nesting on every backend: a pop needs
an open group, a group pushed inside a rendering scope closes before the scope
ends, and Finish fails while a group is open.

## 3. Timings

- `ProgramInvocationOptions` gains `timings`. When set, Execute measures the
  device time of each timed scope.
- A timed scope is one native render scope or one compute Node, which is the
  finest unit every backend can time: tile GPUs sample counters only at
  render-scope boundaries (Metal `MTLCounterSamplingPointAtStageBoundary`).
- `vernonSubmissionGetScopeTimings(submission, scopes, capacity, count)` fills
  `VernonScopeTiming { first_node, node_count, device_submission,
  begin_nanoseconds, end_nanoseconds }` in execution order. An unfused Node is
  a scope with `node_count` 1; fused Nodes appear as one scope, and Node ids
  are contiguous in execution order (`../program/execution_manifest.md`
  section 10). The result describes the schedule that ran.
- Begin and end are measured from the first scope's begin of the same
  `device_submission`, the index of the device submission among those the
  Execute made. A plain Execute makes one; an autodiff pullback replays and
  retries its tape in further device submissions, whose scopes carry the Node
  ids of the graph each ran. Clocks of different device submissions are not
  related; calibrating them to the host clock is a later extension.
- Timings are a completion fact, so they belong to the submission
  (`pipelined_submission.md` section 2), not the invocation: the getter
  returns `VERNON_STATUS_NOT_READY` while the submission is pending and stays
  valid until the submission is destroyed. A failed submission reports
  `VERNON_STATUS_INTERNAL_ERROR` instead of the scopes of the device
  submissions that finished before the failure: like the RHI's timers, scope
  timings describe only work that succeeded. A renderer with frames in flight
  destroys each invocation after commit and reads frame `f - N` from the
  submission it waits on.
- RHI: `vernonRhiCommandEncoderBeginTimer(encoder, &timer)` and
  `vernonRhiCommandEncoderEndTimer(encoder, timer)` bracket commands outside
  rendering scopes, and `vernonRhiCompletionGetTimers(completion, spans,
  capacity, count)` reads `VernonRhiTimerSpan { begin_nanoseconds,
  end_nanoseconds }` per timer in begin order, measured from the first
  timer's begin, once the completion succeeded. Timers may nest, Finish fails
  while one is open, and the encoder owns the backend's query storage, which
  moves to the completion and is released when the completion is destroyed.
  A backend without timing accepts the calls and records nothing; the getter
  reports `VERNON_RHI_STATUS_UNSUPPORTED`.
- Timers instead of raw timestamps: Metal samples counters only when an
  encoder begins or ends (`MTLCounterSamplingPointAtStageBoundary` on tile
  GPUs), so a timestamp inside a rendering scope has no Metal equivalent,
  while a begin/end pair outside rendering scopes maps to every backend.
  Spans instead of elapsed times: overlap and gaps between scopes are what a
  frame timeline shows.
- Backends: GL `glQueryCounter(GL_TIMESTAMP)` (GL 3.3 or
  `ARB_timer_query`; ES via `EXT_disjoint_timer_query`, whose disjoint flag
  fails the read); Vulkan timestamp query pools, reset on the command buffer
  before their first write, when the queue's `timestampValidBits` is not 0;
  D3D12 timestamp query heaps resolved with `ResolveQueryData` and
  `GetTimestampFrequency`; CUDA `cuEventRecord`, read with
  `cuEventElapsedTime` from the first event, whose half-microsecond
  resolution bounds the span; Metal an empty blit encoder per timer edge,
  sampled at its start into a timestamp counter sample buffer and scaled to
  nanoseconds with `sampleTimestamps` pairs. No encoder is open at a timer
  edge, since edges lie outside rendering scopes, so the marker encoder needs
  no change to the encoders that record work; a GPU that drops the empty
  encoder reports an error sample, which fails the read. CPU Programs bypass
  the RHI and time each Node with the host steady clock around its dispatch.
- A span covers the time the device reached its end edge after its begin
  edge. GL and CUDA run commands as they are recorded, so an idle device
  reaches each edge as soon as it is recorded and a span there also covers
  the host time recording its commands; the spans of short scopes are host
  bound on these backends. Section 8 records the deferred replay that would
  remove this.
- A backend without timing support (GL ES without
  `EXT_disjoint_timer_query`, Vulkan without timestamp bits, Metal without
  stage-boundary sampling) reports `VERNON_STATUS_UNSUPPORTED_TARGET` from the
  getter; Execute never fails for lack of timing support. An Execute that did
  not request `timings` reports no scopes.

## 4. Telemetry

`VernonProgramGraphicsNodeTelemetry` is unchanged: counts are CPU facts known
at Execute; times are device facts known at completion and have their own
getter.

## 5. Engine use (non-normative)

The FrameGraph labels its nodes `depth`, `forward` and `display`, groups
forward draws by material instance, and reports the timed scopes of frame
`f - 2`, the submission it waits on before recording frame `f`, in its
telemetry.

## 6. Acceptance

Labels, on every backend with graphics support and on CUDA for compute:

1. A capture of a three-Node graph shows three labelled groups, each
   containing exactly its Node's render scope or dispatches, and a recording
   with nested groups shows them nested inside its Node.
2. An unbalanced recording fails with `PROGRAM_RECORDING_INVALID`.
3. An encoder destroyed with open groups leaves no group open.

Timings, on every backend:

4. With `timings`, each scope reports `begin <= end`, and a scope that does
   100 times the work of another reports a longer span.
5. Without `timings`, Execute creates no queries or events.
6. Fused Nodes appear as one scope with `node_count` greater than 1.
7. An asynchronous submission's scopes report `NOT_READY` until it completes
   and remain readable after its invocation is destroyed.

## 7. Contract changes on acceptance

- [`design.md`](design.md): Execute emits debug groups and optional timings
  (applied in section 8); the RHI functions of sections 2 and 3.
- [`../program/recorded_draws.md`](../program/recorded_draws.md): the
  recording gains the three debug commands and the balance rule (applied).
- [`../program/execution_manifest.md`](../program/execution_manifest.md):
  `name` is also the debug label (applied).
- `VernonRuntime.h` and `VernonRHI.h` (`PROGRAM_VERSION`): the label
  functions, command kinds and `AddProgram` label are in place, as are the
  `timings` option, which takes a reserved slot of
  `VernonProgramInvocationOptions` so that its version stays 1,
  `vernonSubmissionGetScopeTimings` and the RHI timer functions.

## 8. Future work: deferred replay on GL and CUDA

Status: not scheduled. Eager spans are accepted for pass comparison and
regression tracking; this section applies once spans must mean the same on
every backend.

- GL and CUDA encoders record into a host command list instead of calling
  the driver, and Submit replays the list in order on the context or stream.
  Timer edges then reach the device back to back with the commands between
  them, as on Vulkan, D3D12 and Metal, so a span no longer covers host
  recording time.
- RHI validation stays at record, so the call that breaks nesting, scope or
  binding rules still fails. Native driver errors move to Submit and fail the
  submission instead of the recording call.
- Every resource a recorded command references stays retained from record
  to replay, and buffer contents that a command copies from host memory are
  copied at record, so callers may still pass temporaries.
- GL debug groups and NVTX ranges are emitted at replay, so capture tools see
  them around the replayed commands; the record-time NVTX rule in section 2
  changes accordingly.
- CUDA may later replay through stream capture into a CUDA graph, cached by
  recording shape, to cut launch overhead; plain in-order replay comes first.
- What remains: the driver's per-command cost during replay, and on every
  backend a span is device wall time, which includes device stalls. Device
  busy time needs vendor counters (CUPTI activity records, GL performance
  monitor extensions) and stays out of scope.
- Acceptance: on GL and CUDA, a scope whose recording sleeps on the host
  between two commands reports a span no longer than the same scope recorded
  without the sleep, within timer resolution.
