# Platform and execution evolution plan

## Status and scope

This document sequences four post-release tracks:

1. measurement-driven performance work and a backend-neutral submission model;
2. statically linked CPU WebAssembly support for browser applications;
3. composed `ExecutionGraph` VJP, beginning with CPU and WebAssembly;
4. structured Storage autodiff and composed graph VJP on GPU backends.

It is a planning document, not a released Runtime, compiler, pipeline, or
language contract. Compiler contract 11 and pipeline contract 15 remain
unchanged until an intentional release boundary.

## Recommended order

1. Establish performance measurements and shared platform boundaries.
2. Support cooked wasm32 CPU objects linked into a browser application.
3. Separate immutable graph plans from per-execution submission state and make
   `submit()` the backend-neutral execution primitive.
4. Implement composed `ExecutionGraph` VJP on CPU and WebAssembly using that
   submission and lifetime model.
5. Allow CPU and GPU backends to complete submissions asynchronously and keep
   multiple submissions in flight.
6. Extend structured Storage autodiff from CPU to a narrow GPU backend slice.
7. Compose GPU pipeline pullbacks into `ExecutionGraph` VJP.

Measurement begins before the browser work and continues through every phase.
The existing synchronous behavior remains available as a compatibility wrapper
throughout. The internal submission model is introduced before graph VJP so CPU
and GPU graph execution do not acquire different orchestration architectures.
Making a submission genuinely concurrent is a later backend capability, not a
different graph API.

## Long-term execution architecture

### Graph, plan, and submission

Keep three distinct ownership layers:

1. `ExecutionGraph` is a mutable builder containing passes, dependencies, and
   logical resources.
2. `CompiledExecutionGraph` is an immutable, shareable plan containing the
   schedule, scopes, barriers, resource requirements, and backend executable
   bindings.
3. `ExecutionSubmission` is per-execution state containing completion, retained
   resources, command encoders or CPU jobs, transient allocations, diagnostics,
   and profiling data.

Compiling snapshots the builder. Later builder mutation cannot change an
already compiled plan or an in-flight submission. A submission retains its plan
and every resource it touches, so destroying the builder or dropping the caller
handle cannot reclaim live work.

`submit()` is the canonical internal operation for every provider. It returns a
submission in one of pending, running, succeeded, or failed states. Polling,
waiting, and completion callbacks observe the same state machine. `execute()`
remains equivalent to `submit().wait()` and must report the same final status
and diagnostics.

The first CPU and WebAssembly implementation may finish inline and return an
already-complete submission. Native CPU can later complete the same submission
from the worker pool. A browser build may use inline execution initially and
Web Workers later. Returning a JavaScript Promise without a Worker or
cooperative scheduling does not by itself make CPU work non-blocking.

### Backend execution boundary

Graph scheduling, dependency validation, graph-level VJP construction, and
logical resource lifetime are backend-neutral. A backend executor is
responsible only for:

- materializing per-submission resources;
- recording or launching the compiled plan;
- producing and observing a completion primitive;
- retaining native objects until completion;
- recycling pools and transient allocations after completion;
- translating backend failures into the common submission state.

Completion is per-submission, never inferred from device-wide idle. Native
implementations should use fence or timeline values, command-buffer completion,
CUDA events, GL sync objects, or CPU job state as appropriate. Browser
completion is integrated with the event loop and must not introduce a blocking
desktop wait on the main thread.

Cancellation cannot promise to stop GPU work already submitted. It stops
dependent scheduling and notifications where possible, while resource
reclamation still waits for observed backend completion.

### Resource and shutdown rules

Every submission owns a retention set for imported and graph-owned resources,
views, pipelines, binding sets, tape storage, cotangent storage, and backend
recording objects. Handle generations prevent a recycled object from being
mistaken for an object retained by older work.

Pools are generation-aware and recycle only completed entries. Concurrent
submissions do not share mutable command buffers, fences, descriptor pools,
binding caches, argument frames, or tape arenas without explicit
synchronization.

Shutdown first rejects new submissions, then deterministically drains accepted
work, releases completion-owned resources, and finally destroys devices and
schedulers. User callbacks are never invoked while a backend or device lock is
held.

### Graph VJP

Graph VJP is a transformation over a compiled logical graph, not a CPU- or
GPU-specific graph implementation. It:

- records differentiable inputs, objectives, and saved primal values;
- traverses differentiable passes in reverse topological order;
- connects cotangents and accumulates contributions at fan-out joins;
- treats non-differentiable effects as explicit validation boundaries;
- produces a forward plan and a backward plan with a shared derivative
  signature.

Running the forward plan returns a pullback that owns the forward completion
and retained tape state. Applying that pullback schedules the backward plan,
waiting on or depending on forward completion without forcing a device-wide
wait. The synchronous VJP wrapper waits for both submissions.

CPU, WebAssembly, and GPU share graph transformation, derivative signatures,
dependency construction, and pullback semantics. They differ only below the
backend execution boundary: native CPU retains host tape snapshots, WebAssembly
retains linear-memory regions, and GPU retains device tape and cotangent
buffers.

## Phase 0: shared foundation and performance baseline

- Measure CPU and GPU invocation, command recording, queue wait, execution,
  readback, allocation, and pipeline/binding preparation independently.
- Keep warm dispatch allocation-free where pipelines and bindings are reused,
  and remove measured redundant preparation without changing completion
  semantics.
- Isolate dynamic-library loading, filesystem bundle access, CPU thread
  scheduling, browser context ownership, and backend availability behind
  explicit platform boundaries.
- Record the pre-migration synchronous behavior of Runtime invocation, owned
  RHI submission, and `ExecutionGraph.execute()` as the comparison baseline.

This phase provides evidence for later asynchronous work without forcing the
browser or autodiff tracks to depend on an unfinished completion model.

## Phase 1: statically linked browser WebAssembly objects

### Deployment model

The native compiler and cooker produce relocatable CPU code. The final browser
application links that code with the web Runtime:

```text
Vernon source
  -> native compiler emits a wasm32 relocatable object and reflection
  -> cooker packages kernel.wasm.o, reflection, and pipeline metadata
  -> the application build links kernel.wasm.o with the Runtime library
  -> Emscripten/wasm-ld produces the final application module
  -> startup registers the linked kernel symbol with the Runtime
  -> the Runtime CPU scheduler invokes the registered entry
```

The cooker does not invoke `wasm-ld`. Linking belongs to the final application
build. Adding a wasm32 target triple and `object_format: "wasm"` is necessary
but not sufficient: the compiler must emit a compatible relocatable object,
and cooking must preserve its symbol, reflection, hash, target requirements,
and Vernon ABI metadata.

### Runtime responsibilities

The Runtime library and cooked objects become one WebAssembly instance. The
Runtime remains responsible for:

- reflection and invocation validation;
- packed arguments and resource ownership;
- workgroup scheduling and barrier progress;
- Runtime helper functions and diagnostics;
- optional CPU autodiff tape and pullback execution.

JavaScript starts the application and bridges browser APIs; it does not
reimplement Vernon execution semantics. Generated application glue registers
each linked kernel through `vernonRuntimeRegisterStaticCpuEntry`.

### Initial acceptance slice

- Accept and lower one agreed wasm32 CPU target.
- Cook one CPU pipeline to `.wasm.o`, reflection, and pipeline metadata.
- Build the Runtime as a static library with
  `VERNON_RUNTIME_PROFILE=web`.
- Link the cooked object, Runtime library, registration glue, and application
  through Emscripten.
- Load embedded or in-memory pipeline metadata without `dlopen` or a native
  filesystem.
- Execute through a single-thread CPU scheduler.
- Register and select multiple cooked CPU pipelines in one application.

Changing the cooked pipeline set requires relinking the application.
Post-deployment loading of independent WebAssembly modules is outside the
initial scope.

### Implementation evidence

The `wasm32-unknown-emscripten` compiler and cooker path emits content-addressed
`.wasm.o` artifacts while preserving compiler contract 11 and pipeline contract
15. The web Runtime profile uses static entry resolution, metadata-only external
artifact validation, and the calling-thread policy of the existing CPU
scheduler. Focused tests cover real WebAssembly object emission, multiple
filesystem-free static pipelines, and range-phase execution without workers.

The baseline runner reports cold invocation, warm dispatch, host upload,
host readback, and combined transfer/invocation independently. One reference
CPU result is recorded in
[`baselines/2026-08-11-macos-arm64-cpu.json`](baselines/2026-08-11-macos-arm64-cpu.json).

The external-engine example now embeds both bundles in one executable. Its
shared frame loop displays the animated Julia-set CPU Execution Graph beside
an RHI Execution Graph-encoded Mandelbulb through native OpenGL or browser
WebGL2, and retains a CPU-only headless Node checksum smoke. GLSL ES stages and
the CPU wasm object are linked without browser filesystem access. Reproducible
commands are described in [`web_wasm.md`](web_wasm.md).

The implemented follow-on adds OpenGL ES/WebGL execution. The browser build does
not expose CUDA, Vulkan, DirectX 12, or Metal; those remain available to
platform-specific native Runtime builds. WebGPU is a separate future backend.

## Phase 2: unified submission and lifetime foundation

- Split the mutable graph builder, immutable compiled plan, and per-run
  `ExecutionSubmission`.
- Use the same pending/succeeded/failed state model for CPU and RHI providers.
- Replace the unreleased synchronous Runtime invocation, RHI synchronization,
  and `ExecutionGraph.execute()` APIs instead of retaining compatibility
  wrappers.
- Make successful RHI submission consume its command encoder and transfer
  command stats, cleanup actions, and retained resources to a generation-safe
  completion handle.
- Drain unfinished completions during submission destruction and device
  shutdown.
- Keep compiler contract 11 and pipeline contract 15 unchanged.

This phase does not require concurrent execution. Existing backends may produce
an already-complete submission while the ownership model and failure semantics
stabilize. The existing `VernonRhiCompletion` handle declaration is the natural
future RHI surface; an unrelated completion model should not be introduced.

Acceptance requires safe builder and compiled-plan destruction after
submission, reusable immutable plans, no early resource release, deterministic
shutdown draining, and tests for success, recording failure, submission
failure, stale completion handles, and backend execution failure.

### Implementation evidence

The C RHI now returns `VernonRhiCompletion`; Runtime returns
`VernonSubmission`; and C++/Python execution graphs compile once into immutable
plans and submit per-run state. Existing backends may report an already-complete
submission. The CPU baseline remains the regression oracle until Phase 4 has
evidence that true overlap improves end-to-end workloads.

On the reference macOS arm64 Release build, a 20-iteration warmup followed by a
20-sample post-migration run at 65,536 elements measured 1.1780 ms median warm
dispatch and 1.1353 ms median
upload-dispatch-readback, versus 1.3586 ms and 1.3229 ms in the recorded
baseline. The ownership migration therefore introduces no measured steady-state
regression; cold compilation remains tracked separately because process startup
and compiler cache state dominate it. The full result is recorded in
[`baselines/2026-08-11-macos-arm64-cpu-phase2.json`](baselines/2026-08-11-macos-arm64-cpu-phase2.json).

## Phase 3: CPU and WebAssembly ExecutionGraph VJP

- Define graph-level differentiable inputs, objectives, cotangents, gradients,
  and saved-primal resources independently of physical storage.
- Build reverse graph dependencies and deterministic gradient accumulation at
  fan-out joins.
- Adapt existing structured CPU pipeline forward/pullback execution as
  differentiable compute-pass implementations rather than duplicating their
  tape logic in the graph layer.
- Retain forward submissions and tape state in the graph pullback until
  backward completion or explicit pullback destruction.
- Support native CPU and statically linked wasm32 entries through the same graph
  transformation and submission state machine.

The initial browser path may execute inline. Its pullback API and ownership must
still match the eventual Worker-backed implementation.

Acceptance requires multi-pass chains, branches with gradient accumulation,
multiple differentiable inputs, explicit and implicit scalar cotangents,
structured Storage objectives, graph destruction before pullback application,
and numerical agreement with pipeline VJP and finite differences on native CPU
and WebAssembly.

## Phase 4: asynchronous multi-submission execution

- Expose non-blocking completion from each backend executor without changing
  graph or VJP semantics.
- Let the native CPU scheduler return job completion instead of waiting inside
  `dispatch`; keep calling-thread execution as a supported policy.
- Replace the one-active-encoder registry and device-wide execution-session
  lock with per-submission backend state and submission-order synchronization.
- Resolve Vulkan command-buffer, fence, descriptor-pool, upload/readback ring,
  and cached-binding generations, with equivalent per-submission ownership on
  DirectX 12, Metal, CUDA, and OpenGL.
- Integrate browser GPU completion with the event loop. Add Web Worker CPU
  execution only when its memory and shutdown model are explicit.
- Permit dependencies between submissions without turning each dependency into
  a host wait.

Acceptance requires at least two independent CPU jobs and two independent GPU
submissions in flight where the backend supports it, dependent forward/backward
submissions, resource reclamation only after observed completion, deterministic
shutdown draining, and unchanged explicit-wait behavior.

## Phase 5: GPU autodiff

Begin with one narrow backend slice, preferably CUDA:

- scalar, static-shape compute;
- no workgroup barrier;
- invocation-private gradients;
- CPU finite-difference and structured VJP results as the numerical oracle.

Reuse backend-neutral autodiff analysis, differentiation rules, tape planning,
logical `vernon.ad.*` operations, derivative signatures, and accumulation
capability descriptions. GPU lowering, device tape storage, executable
dispatch, completion, and pullback resource ownership remain backend executor
work.

Asynchronous resource lifetime precedes this phase because tape and cotangent
buffers, forward submission, and backward submission must safely outlive
individual host calls. SPIR-V and Metal support, barrier-aware pullbacks, and
graphics custom VJPs follow only after the narrow slice passes numerical
acceptance.

Acceptance requires numerical agreement with the CPU `ExecutionGraph` VJP
oracle, forward and backward submissions with no device-wide wait between them,
correct device tape reclamation, and deterministic failure cleanup.

## Phase 6: GPU ExecutionGraph VJP

- Connect differentiable GPU compute passes to the graph-level VJP
  transformation introduced in Phase 3.
- Keep forward tape, saved primal resources, cotangents, and gradient
  accumulation on device unless an explicit graph edge requests readback.
- Represent forward-to-backward ordering as submission dependencies.
- Reject mixed-device or unsupported differentiable effects explicitly rather
  than silently synchronizing or falling back to CPU.
- Add graphics custom VJPs only after compute graph composition is stable.

No second GPU graph scheduler or GPU-only pullback object is introduced. This
phase supplies backend pass implementations and storage policies to the same
compiled graph and pullback architecture used by CPU and WebAssembly.

Acceptance requires multi-pass GPU chains and fan-out accumulation, multiple
forward/backward pairs in flight, numerical agreement with CPU and finite
differences, no unintended host readback, and identical synchronous wrapper
results.

## Dependency rationale

- Browser CPU support comes before asynchronous execution because the current
  synchronous contract maps to a smaller first WebAssembly application.
- The submission ownership model comes before graph VJP so CPU and GPU share
  one execution and pullback lifetime architecture.
- CPU and WebAssembly graph VJP come before GPU autodiff because they establish
  graph composition semantics and provide a numerical oracle without requiring
  device tape allocation.
- Asynchronous resource lifetime comes before GPU autodiff because GPU
  pullbacks require tape and gradient resources to survive across submissions.
- GPU pipeline autodiff comes before GPU graph VJP so backend tape,
  accumulation, and completion behavior are validated before graph composition.
- Measurement starts first because queue synchronization should only be
  removed after it is shown to be a material bottleneck.
- Full compiler-in-browser deployment and dynamic loading of independent
  WebAssembly modules are not part of this plan.
