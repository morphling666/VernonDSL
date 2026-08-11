# Platform and execution evolution plan

## Status and scope

This document sequences three post-release tracks:

1. measurement-driven performance work, including future asynchronous backend
   execution;
2. statically linked CPU WebAssembly support for browser applications;
3. structured Storage autodiff on non-CPU backends.

It is a planning document, not a released Runtime, compiler, pipeline, or
language contract. Compiler contract 11 and pipeline contract 15 remain
unchanged until an intentional release boundary.

## Recommended order

1. Establish performance measurements and shared platform boundaries.
2. Support cooked wasm32 CPU objects linked into a browser application.
3. Introduce asynchronous GPU submission and resource lifetime.
4. Extend structured Storage autodiff from CPU to GPU backends.

Measurement begins before the browser work and continues through every phase.
Only performance changes that preserve the current synchronous execution
contract belong in the first phase. Record, submit, and completion semantics
change only in the asynchronous phase.

## Phase 0: shared foundation and performance baseline

- Measure CPU and GPU invocation, command recording, queue wait, execution,
  readback, allocation, and pipeline/binding preparation independently.
- Keep warm dispatch allocation-free where pipelines and bindings are reused,
  and remove measured redundant preparation without changing completion
  semantics.
- Isolate dynamic-library loading, filesystem bundle access, CPU thread
  scheduling, browser context ownership, and backend availability behind
  explicit platform boundaries.
- Preserve the synchronous behavior of Runtime invocation, owned RHI
  submission, and `ExecutionGraph.execute()`.

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

After CPU execution works, add OpenGL ES/WebGL execution. The browser build
does not expose CUDA, Vulkan, DirectX 12, or Metal; those remain available to
platform-specific native Runtime builds. WebGPU is a separate future backend.

## Phase 2: asynchronous GPU execution

- Separate command recording, submission, and waiting while preserving a
  synchronous compatibility wrapper.
- Return an internal submission/completion object and retain every touched
  resource until backend completion is observed.
- Replace the one-active-encoder and device-wide execution-session assumptions
  only after completion and deterministic shutdown draining are implemented.
- Resolve Vulkan command-buffer, fence, descriptor-pool, and cached-binding
  generations before allowing multiple submissions in flight.
- Map browser completion to its event loop rather than importing a desktop
  blocking-fence model into WebAssembly.

This phase follows the synchronous browser slice so initial WebAssembly support
does not also have to define Promise completion, deferred linear-memory
reclamation, and multi-submission ownership.

Acceptance requires at least two independent GPU submissions in flight,
resource reclamation only after observed completion, deterministic shutdown
draining, and unchanged behavior through the synchronous compatibility path.

## Phase 3: GPU autodiff

Begin with one narrow backend slice, preferably CUDA:

- scalar, static-shape compute;
- no workgroup barrier;
- invocation-private gradients;
- CPU finite-difference and structured VJP results as the numerical oracle.

Reuse backend-neutral autodiff analysis, differentiation rules, tape planning,
logical `vernon.ad.*` operations, and accumulation capability descriptions.
GPU lowering, device tape storage, executable dispatch, and pullback resource
ownership are backend-specific work.

Asynchronous resource lifetime precedes this phase because tape and cotangent
buffers, forward submission, and backward submission must safely outlive
individual host calls. SPIR-V and Metal support, barrier-aware pullbacks,
graphics custom VJPs, and composed ExecutionGraph VJP follow only after the
narrow slice passes numerical acceptance.

## Dependency rationale

- Browser CPU support comes before asynchronous execution because the current
  synchronous contract maps to a smaller first WebAssembly application.
- Asynchronous resource lifetime comes before GPU autodiff because GPU
  pullbacks require tape and gradient resources to survive across submissions.
- Measurement starts first because queue synchronization should only be
  removed after it is shown to be a material bottleneck.
- Full compiler-in-browser deployment and dynamic loading of independent
  WebAssembly modules are not part of this plan.
