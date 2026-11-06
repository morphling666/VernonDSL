# Performance: compiled launches, compiled frames, SIMD kernels

Status: accepted future design. It covers compute and graphics, the
runtime first and the compiler second. The runtime architecture is
[`runtime/architecture_v2.md`](runtime/architecture_v2.md); section 3 here
holds only the two runtime details it refers to. Sections 4 to 6 are the
compiler and the engine. The engine-side scene work is Vernon
`specs/render/gpu_scene.md`. Measurements:
[`reports/benchmarks/numba_comparison.md`](../reports/benchmarks/numba_comparison.md),
[`reports/benchmarks/training_step.md`](../reports/benchmarks/training_step.md)
and
[`reports/benchmarks/planet_walker_frame.md`](../reports/benchmarks/planet_walker_frame.md).

## 1. Goal

One rule for compute and graphics: **structure is compiled once; a
steady-state launch or frame costs O(changed bytes) of CPU time plus O(1)
submissions, and its kernels run at the target's hardware bound**, memory
bandwidth or arithmetic throughput.

Structure is what decides code and commands: the kernel and its
specialization, argument dtypes, ranks, access and aliasing, pipelines,
resources and capacities. Payload is everything else: scalars, extents,
grids, instance counts, visibility and per-object rows. Nothing that
changes every frame or every call is structure.

| Workload | Today | Target |
|---|---|---|
| warm CPU launch from Python (`launch_64`) | 600 µs | under 5 µs; under 1 µs from Host code and AOT |
| `axpy`, 8M f64, CPU, 32 threads | 24.8 ms (numba 6.1) | memory bandwidth, at most numba |
| `relief`, CPU, 32 threads | 29.0 ms (numba 2.06) | faster than numba |
| `axpy`, CUDA, resident | 5.37 ms (numba 0.24) | memory bandwidth, at most numba |
| warm CUDA launch | 1.39 ms (numba 0.022) | under 10 µs |
| `PlanetWalker --walk`, 1280×720 | 1.40 ms of CPU per frame, GPU-bound (36.8 ms frame, CPU-bound, before v2) | under 2 ms of CPU per frame; GPU-bound |
| `PlanetWalker`, standing | 0.77 ms of CPU per frame, GPU-bound (15.2 ms frame before v2) | under 1 ms of CPU per frame |

## 2. Where the time goes

- **Compute launch.** Python revalidates the kernel's sources on every call
  (reads and SHA-256-hashes each dependency, twice), evaluates annotations
  twice, validates every argument and crosses into native code 11 to 19
  times.
- **Write staging.** Every write output is copied into a carrier and back on
  every launch, on CPU and device, and a runtime-owned output is zero-filled
  on the host even when the kernel runs on the device.
- **Blocking submission.** Each Execute waits for each of its 2 to 4
  submissions.
- **Host copies.** An upload copies the array three times on the host; a
  download returns Python `bytes` at 2 GB/s.
- **CPU code.** One scalar call per invocation, which reloads its arguments;
  no SIMD across invocations; generic x86-64 code with no SSE4.1, AVX2 or
  FMA, so `floor` is a library call.
- **Graphics.** A frame whose draw structure changes, as walking's tile and
  ground-cover streaming changes it, plans, validates and hazard-checks
  every draw of the recording again: 80% of a walking frame. A standing
  frame still culls and decodes every draw on the CPU.

## 3. Runtime

[`runtime/architecture_v2.md`](runtime/architecture_v2.md) is the runtime:
schedule fragments the compiler emits per Program and the runtime links per
graph, a HAL, recordings that do not depend on shapes, calls that write
changed bytes and submit, buffers with one home, in-place outputs, and
draws as data. Two details it refers to are here.

### 3.1 Source epoch

The frontend checks a kernel's sources against a process-wide source epoch,
so a call compares one integer. In development a file watcher bumps the
epoch when a dependency changes, and the next call revalidates; in a
release the epoch never changes. Per-call `stat` was rejected: it costs
microseconds per dependency. Annotations are resolved once per
specialization.

### 3.2 CPU execution

Workers are persistent. A kernel with a chunk entry (section 4.1) runs as
spans of whole workgroups, one call of the chunk entry per span with the
argument record's address; any other kernel runs one range per workgroup.
A runner takes whole workgroups: splitting one across threads runs the
barrier-phase protocol, which costs more than the work of small launches.
The CPU device gives each runner at least
`VERNON_CPU_INVOCATIONS_PER_RUNNER` invocations, 4096 by default, until a
per-machine measurement of the cost of waking a worker replaces it.

- **Claims.** Runners claim spans of about an eighth of a runner's share
  as they finish the last, because a kernel whose cost varies per lane
  (`nearest_segment`) leaves a static partition unbalanced. The calling
  thread is one of the runners.
- **No spinning.** Idle workers sleep rather than poll for the next
  launch: a polling thread takes issue slots from its SMT sibling, which
  is still running the launch's last spans.
- **Pinned workers, opt-in.** With `VERNON_CPU_PIN_WORKERS=1` (Windows),
  worker N runs only on logical processor N, and a span launch splits its
  groups into one contiguous region per runner. Worker N claims spans of
  region N first, then of the others, and the calling thread, which may
  move, only waits. Repeated launches over the same data then find each
  region in the caches of the processor that ran it last; on a chip whose
  last-level cache is split between core complexes, unpinned workers move
  most of the data between caches on every launch. An affinity hint
  (`SetThreadIdealProcessor`) lets workers drift and loses most of the
  gain. Pinning is off by default because it competes with an
  application's own threads for fixed processors.

## 4. Compiler

### 4.1 Chunk entries and SIMT-to-SIMD

A CPU kernel compiles to `entry(record, begin, end)`, which loads its
arguments once, computes uniform values once, and runs the body for
invocations `[begin, end)` in gangs of W lanes, as ISPC does. W is the
target's native vector width for the kernel's widest varying type: 8
32-bit lanes on AVX2, 16 on AVX-512, 4 on NEON, with 64-bit values in
register pairs.

- **Uniformity analysis.** A value is uniform when it depends only on
  scalar arguments, constants, workgroup ids and other uniform values;
  uniform values stay scalar and are hoisted. Every other value is varying
  and becomes a vector of W.
- **Divergence is masks.** A branch on a varying condition executes both
  sides under the execution mask; a loop with a varying trip count runs
  while any lane is active. A uniform branch stays a branch.
- **Memory.** An index that is a uniform base plus the lane index is a
  vector load or store; any other is a gather or scatter, masked. Atomics
  run lane by lane under the mask, in lane order, so results are
  deterministic.
- **Workgroups.** A workgroup of N invocations runs as N/W gangs; a barrier
  splits the body into phases that loop over the gangs (the work-item loops
  of PoCL), and workgroup storage is a stack array per chunk.
- **Where.** A Vernon-dialect pass before CPU lowering, producing the
  vector dialect, so LLVM receives explicit vector code. The ABI wrapper's
  per-invocation loop (`VernonCpuAbiWrapper.cpp`) and its per-lane
  argument reloads are removed.

The same uniformity analysis feeds GPU targets: uniform values become
scalar registers and uniform loads.

Until that pass exists, the ABI wrapper provides the chunk entry and
leaves vectorization to LLVM:

- **Chunk entry.** `<wrapper>_chunk(const VernonCpuChunk *)`
  (`VernonCommon.h`) runs every lane of workgroups
  `[group_begin, group_end)`. The wrapper emits it for kernels without
  barriers, workgroup helpers, results or texture callbacks; the others
  keep the range entry only. In AOT objects it is weak and in its own
  COMDAT like the range entry, since identical helper Programs in several
  objects of one application must fold.
- **Lane loop.** Both entries call one outlined loop that copies the
  argument record once, walks each workgroup row with a linear induction
  variable and publishes range progress once on exit. Its buffer pointers
  are `noalias` parameters (section 4.3); the loop is never inlined into
  its caller, where those parameters would become per-iteration scopes the
  vectorizer cannot use.
- **Unit strides.** A TensorView's strides are runtime values, so the loop
  reads and writes views with gathers and scatters. A second copy of the
  loop stores 1 into every innermost stride after copying the record, and
  a launch whose innermost strides are all 1 runs that copy, whose
  accesses are consecutive vector loads and stores. Both copies compute
  the same addresses, so results do not change.
- **Unit strides on CUDA.** `VernonToGPU` emits a kernel's body twice
  under a uniform branch on whether every innermost stride is 1, and the
  first copy reads those strides as the constant 1, so its addresses need
  no 64-bit multiply. Kernels with barriers, workgroup storage or
  workgroup reductions keep one body: their lowering assumes the body is
  the function's, and a second copy would reserve workgroup storage twice.

### 4.2 Targets

- **JIT targets the host CPU**: its name and features, for both the
  TargetMachine that emits the object and the JIT. Only interactive
  compilation does; an AOT object without versions keeps the baseline.
- **AOT emits several versions** of each entry, x86-64-v2, v3 and v4 on
  x86-64 and the baseline with optional SVE on AArch64, and a Program picks
  one by the CPU's features when it loads. wasm32 keeps one version, with
  SIMD128 when the profile enables it.

### 4.3 Aliasing

The language contract rejects a call whose written TensorView overlaps
another argument, and every CPU dispatch path checks it: the compute
invocation planner, the Program boundary binder, provider storage
bindings and the linked-schedule stub, which compares the arguments'
byte spans, O(arguments²) over a handful. A kernel therefore compiles to
one executable whose single-leaf TensorView pointers are `noalias`. Two
executables, one per aliasing class, were rejected: no legal call reaches
the aliasing one. numba cannot assume this without proof, so `noalias` is
where VernonDSL vectorizes and reorders memory beyond it.

### 4.4 Floating point

- **Results are bit-exact with numpy by default**: IEEE operations,
  no contraction, no reassociation. A kernel opts in per kernel to FMA
  contraction or to fast math.
- **Vectorizing never changes a result.** The vector and scalar forms of
  every operation, including the elementary functions, are the same
  function; a vector math routine is accepted only if it matches the
  scalar one bit for bit.
- `floor`, `ceil`, `round` and `trunc` compile to the target's rounding
  instructions; conversions to integers compile to the conversion
  instruction with the language's defined result for out-of-range values.
- `min`, `max` and `clamp` are a compare and a select
  (`language/contract.md`), one `minsd`/`maxsd` on x86, as in compiled
  host code and numba. The IEEE `minimum`/`maximum` they lowered to before
  expand to NaN and signed-zero fixups around each one; that was the whole
  of `nearest_segment`'s 1.24x single-thread gap.

### 4.5 Fusion

Adjacent stages of a Program whose index maps match (element-wise chains,
a map followed by its reduction's epilogue) compile into one kernel, so
intermediates stay in registers. The planner fuses only where the
intermediate has no other reader and no observable publication, so fusion
never changes what a Program publishes.

### 4.6 64-bit integers in kernels

Kernels gain `i64` and `u64` on every target that has them
(`language/kernel_limitations.md`, "Types and storage"). Hashing such as
SplitMix64 then runs in the kernel instead of in host-built tables, and
the engine's ground cover, seeded per cell by SplitMix64, can be generated
on the GPU (section 6).

## 5. Graphics backend

The high tier runs on **Vulkan** on Windows and Linux, as Vernon's render
architecture plans (Vernon `specs/render/architecture.md` section 8): it
has compute, multi-draw indirect, descriptor indexing, timeline semaphores
and command buffers that replay (`runtime/architecture_v2.md` section 6).
OpenGL 3.3 remains the compatibility tier, with the tape as its floor. The
engine's `Runtime::initialize` admits Vulkan and both OpenGL tiers.

## 6. Requirements on the engine

Vernon `specs/render/gpu_scene.md` owns these; they are what the runtime
bounds assume:

- **Visibility is payload.** Draws live in persistent buckets per pipeline,
  material and geometry; GPU culling writes instance lists and indirect
  arguments, so a changed visible set changes no command.
- **Streaming is payload.** Streamed tiles and props go into shared
  geometry buffers and GPU scene rows, so a new tile adds rows, not draws,
  unless it brings a new pipeline, material or geometry kind.
- **Ground cover is generated on the GPU** from the tile's layer weights
  and spawn fields, not spawned as world entities, so walking creates no
  entities and no proxies.

## 7. Benchmarks and acceptance

- `examples/numba_benchmark` on CPU at 1 and 32 threads and on CUDA,
  resident and roundtrip, with bit-for-bit agreement with numpy.
- `PlanetWalker --no-vsync`, standing and `--walk`, with a per-frame CPU
  and GPU breakdown (`runtime/gpu_profiling.md`).
- The Editor scenes, with `--frame-stats` and `--no-vsync`
  ([`../reports/benchmarks/editor_scenes.md`](../reports/benchmarks/editor_scenes.md)):
  the Editor's `dog`, `bodies` and `planet` demos and its panels, on OpenGL
  and on every high-tier backend the Editor runs on.

  | Scene | Target |
  |---|---|
  | boxes1000 (OpenGL, about 6000 draws) | a frame's host time bounded by driver time plus the tape, about 0.3 ms |
  | cave + acrobot, simulating | no regression in any phase |
  | a pass whose structure changes every frame (the ImGui pass) | recording no slower than the old per-Execute path, `ui.end` 0.7 ms |

- Comparisons with other runtimes:

  | Workload | Compared against | Target |
  |---|---|---|
  | 10000 draws, distinct materials, static | the same scene in UE 5.7 (render and RHI thread time in Unreal Insights) | Vulkan, D3D12 and Metal: a call under 20 µs plus the changed bytes; the OpenGL tape under 30 ns per command plus driver time |
  | a chain of 1000 compute kernels | `cuLaunchKernel` in a loop; a CUDA graph built by hand | within 10% of the hand-built graph |
  | small-model LLM decode, growing sequence | PyTorch with CUDA graphs; llama.cpp | tokens per second at least equal; no new recording within a capacity class |
  | training step (forward, pullback, update) | PyTorch with CUDA graphs | CPU time per step independent of layer count |

Each phase reports before and after in `reports/`, with the provenance
`roadmap.md` section 8 requires.

## 8. Phases

Runtime: the migration of `runtime/architecture_v2.md` section 16. The
source epoch (3.1) and the persistent CPU workers with chunk calls (3.2)
belong to its phase 1, with the chunk entries of compiler phase 1.

Compiler:

1. Chunk entries, host CPU targets, rounding and conversion instructions
   (4.1 without vectors, 4.2, 4.4).
2. SIMT-to-SIMD with uniformity analysis and the aliasing classes (4.1,
   4.3).
3. Fusion (4.5).
4. 64-bit integers, GPU uniformity (4.6, 4.1).

Engine, with the runtime's phase 3: the Vulkan high tier (5), then
`gpu_scene.md` phases 2 and 3 with streaming as payload and GPU ground
cover (6).

## 9. Rejected alternatives

- **Caching in Python only.** It removes the hashing but keeps Python work
  and native crossings per argument; the call must be native for Host code
  and AOT anyway.
- **A loop over lanes left to LLVM's loop vectorizer.** It gives up on
  divergent control flow, calls, gathers and barriers, which real kernels
  have; ISPC-style masking vectorizes them by construction.
- **Fast math or FMA by default.** It changes results against numpy and
  between targets, and the benchmark shows exact results cost 10 to 15% on
  CUDA, which a kernel may opt out of.
- **OpenGL 4.6 as the high tier.** It has compute and multi-draw indirect
  but no replayable command buffers and a single-threaded driver; Vernon's
  architecture rejects a third tier (section 8.2 there).
- **D3D12 as the primary backend.** It covers Windows only; Vulkan covers
  Windows, Linux and Android with one code path.
- **Faster re-planning per frame.** Its cost still grows with draws times
  frames; making visibility and streaming payload removes the re-planning.
