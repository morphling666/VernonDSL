# Diagnostics off the hot path

Status: implemented. The device flags exist on every backend; the GL error,
timer and label changes are verified on OpenGL and OpenGL ES (NVIDIA), the
label flag on CUDA. Metal and the CUDA and Metal flag plumbing are unbuilt
here. The contracts of section 7 are amended.

Threaded GL drivers (NVIDIA's by default) queue calls for a driver thread.
A call that returns state, such as `glGetError`, makes the calling thread
wait until that thread has run every queued call. The OpenGL backend reads
`glGetError` before and after every rendering scope, clear and transfer,
creates a query object with `glGenQueries` on every timer edge under the
device lock, and pushes a debug group around every Node and material run.
In the Editor drawing 1000 boxes these take about 15% of Execute, while the
draws themselves take about 2%. The same pattern exists wherever a backend
checks per operation what it could check per submission.

## 1. Validation mode

Device creation (the `flags` of `VernonRhiOwnedDeviceDescriptor` and of the
Vulkan and D3D12 borrowed descriptors, and a `flags` argument of
`vernonRhiCreateOpenGLDevice`) gains the flags
`VERNON_RHI_DEVICE_VALIDATION`, `VERNON_RHI_DEVICE_LABELS` and, for OpenGL,
`VERNON_RHI_DEVICE_OPENGL_OWNS_DEBUG_OUTPUT`.

With `VALIDATION`, the device keeps today's behaviour: an error scope around
every operation that can raise a native error, attributed to that
operation, and the adapter's per-command checks. Tests of exact error
attribution and developers chasing a driver error use it, as they would
Vulkan's validation layers. Without it, sections 2 to 4 apply. RHI-level
validation (handles, ranges, nesting, scope rules) does not depend on the
mode, since it is not a driver round trip.

## 2. OpenGL errors without validation

- When the host grants the device its context's debug output
  (`OPENGL_OWNS_DEBUG_OUTPUT`) and the context has `KHR_debug`
  (GL 4.3, ES 3.2 or the extension), the device enables `GL_DEBUG_OUTPUT`
  with `GL_DEBUG_OUTPUT_SYNCHRONOUS` disabled and installs a callback that
  keeps messages of type `GL_DEBUG_TYPE_ERROR` in a device queue. The driver
  may call it on any thread, at any later time.
- Every completion check (poll or wait) whose fence signalled drains the
  queue. A queued error fails the completion being checked with
  `BackendFailure` and `OpenGL error <code>: <driver message>`; the device
  stays usable. An error is reported by the first completion checked after it
  arrives, which may belong to a later submission than the one that raised
  it. Only a debug context's message names the GL function: NVIDIA's
  non-debug contexts send `No detailed debug message due to a non-debug`
  `context`, so a developer chasing one error uses `VALIDATION`.
- The callback receives every error of the context, including the host's
  own GL calls, so granting debug output makes host errors fail Vernon
  submissions. A host that installs its own callback does not grant it.
- A context without `KHR_debug`, or whose debug output the host keeps
  (macOS OpenGL 4.1, ES before 3.2 without the extension, external
  contexts), reads `glGetError` once per completion check instead. That read
  waits for the driver thread once per submission instead of twice per
  scope.
- Operations that return a result to the caller synchronously (image and
  buffer downloads, mapping a readback, and mipmap generation, which reports
  an unsupported format only through `glGetError`) already wait for the
  device and keep their error scope, so a failed `glReadPixels` still fails
  its own call instead of returning zeroed data. The scope first queues the
  errors already on the context, so that it fails only for its own, and drops
  its own errors from the queue, since it reported them.
- The adapter's per-command `glGetError` drain discarded errors the RHI's
  own calls raised. Without it, ES devices on NVIDIA failed every timed
  submission: their contexts list `EXT_disjoint_timer_query` but reject
  `GL_GPU_DISJOINT_EXT`. An ES device reads the flag once at creation and,
  when the read fails, times without disjoint detection, as desktop GL does.

## 3. Timers

- Each device keeps a pool of timer query objects. An encoder takes queries
  from it at each timer edge and returns them when the completion that read
  them is destroyed, so a steady state creates no query objects. The pool
  only grows.
- Timer edges are RHI commands recorded between packet batches
  (`stored_plans.md` section 11.2), one per scope edge.
- Results are read only after the completion's fence has signalled, when
  `GL_QUERY_RESULT` no longer waits. Vulkan, D3D12 and Metal already pool
  their query storage per encoder; they reuse it the same way.

Scope timings stay opt-in per schedule through `vernonScheduleSetTimings`
(`gpu_profiling.md` section 3).

## 4. Labels

- Debug groups and markers reach the native API only on a device created
  with `LABELS`; `VALIDATION` implies it.
- Without `LABELS`, recording debug commands are still validated
  (balance, `PROGRAM_RECORDING_INVALID`) and then dropped when the packet
  batch is built, so recordings are portable between the two modes.
- A capture tool launch sets `LABELS` through the host: the Engine exposes it
  as a command-line flag. Detecting capture tools from inside the Runtime was
  rejected: each tool has its own injection mechanism, and NVTX already
  loads its tool only when one is injected.

## 5. Rejected alternatives

- **Per-operation error scopes in every build**: correct attribution, but a
  driver-thread round trip per scope is a fixed cost that grows with the
  number of passes and transfers.
- **`glGetError` once per Submit**: one round trip per submission still
  waits for the driver thread while it runs the previous frame.
- **Synchronous debug output**: it makes the driver run every call on the
  calling thread, which disables the threaded driver altogether.
- **No error reporting outside validation mode**: a release build would draw
  wrong results silently.

## 6. Acceptance

On the OpenGL backend:

1. A steady-state Execute without `VALIDATION` calls no `glGetError`,
   creates no query objects and, without `LABELS`, pushes no debug groups;
   test hooks count the calls.
2. An invalid GL operation injected through a test hook fails the next
   completion, with the driver's message under granted debug output and the
   error code without it; the device stays usable.
3. With `VALIDATION`, the same operation fails its own call, as today.
4. A failed download fails its own call in both modes.
5. Timers report the same scopes with the query pool as before, and a
   thousand frames create no more query objects than the first frames did.

On every backend, a recording with debug groups draws the same pixels with
and without `LABELS`.

## 7. Contract changes (applied)

- [`binding_groups.md`](binding_groups.md) section 8: `glGetError` is read
  per render scope only in validation mode.
- [`design.md`](design.md) section 10: transfers run inside an error scope in
  validation mode; synchronous downloads always do.
- [`gpu_profiling.md`](gpu_profiling.md) section 2: Execute wraps Nodes in
  debug groups when the device emits labels; section 3: GL timers come from
  the device's query pool.
