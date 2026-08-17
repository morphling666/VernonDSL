# Cost-Aware Residual and Bounded Replay Autodiff

## Status

The CPU `min_memory` path is implemented and verified through the 1024 smoke
grid. It retains no Tape allocation and uses complete-workgroup bounded replay
with a fixed temporary Tape peak.

GPU compute backends now compile and execute no-Tape rematerialized Storage
pullbacks through one shared RHI execution path. Ordinary inputs and final
gradients cross the host API boundary; no residual Tape is read back. Captured
static/dynamic Tape profiles still require the backend-local complete-workgroup
replay extension below.

This document describes the architecture that exists now. The only checkboxes
are unfinished work. Historical migration notes and superseded designs are not
completion evidence.

Do not change the compiler, pipeline, reflection, or Tape allocator contracts
until their next versioned release.

## Goals

1. Select residual sources per active value instead of retaining every load in
   Tape.
2. Keep logical residual size, retained allocation, checkpoints, construction
   scratch, backward values, and total managed peak as separate quantities.
3. Make `min_memory`, `balanced`, and `min_runtime` obey explicit planning
   semantics and hard physical budgets.
4. Keep CPU replay memory bounded by one complete workgroup.
5. Preserve reusable pullbacks, rollback, mutation isolation, alias safety, and
   deterministic gradient publication.
6. Keep Python as a binding adapter; scheduling and replay belong to native
   compiler, Runtime, and execution-graph layers.

## Architectural invariants

- Residual decisions are keyed by active value and ABI leaf.
- A value may be rematerialized only when its dependency slice is pure,
  reconstructible, and exact-version safe.
- Storage reloads use logical resource versions, not current owner identity.
- Dynamic control history is retained only when structured reconstruction is
  not legal.
- Runtime enforces physical budgets but does not recreate compiler cost
  planning.
- Replay segments are complete workgroups. A running workgroup is never
  suspended to satisfy a budget.
- Replay preserves virtual global, workgroup, and local IDs.
- Backward reads immutable Tape; construction storage is recyclable only after
  that read epoch ends.
- Gradients use shared transactional staging and publish only after successful
  completion.
- Missing write-footprint metadata selects conservative whole-view restoration.
- No Tape path uses file backing or GPU-to-host spill.
- Checked arithmetic is required for every byte, shape, range, and launch-size
  calculation.

## Current architecture

### Residual source planning

The compiler classifies each active value into one of these sources:

- `none`: no backward residual is required;
- pure rematerialization;
- exact-version Storage reload;
- graph checkpoint/replay;
- invocation-static Tape;
- invocation-dynamic Tape;
- mixed capture when only part of a value can be reconstructed.

The cost model distinguishes:

- logical residual bytes;
- retained allocation bytes;
- forward construction peak;
- compacted resident and allocated Tape;
- checkpoint and transaction bytes;
- backward value and accumulation bytes;
- recomputation cost;
- Runtime-managed peak.

The selected source must satisfy legality first, then policy and budget.
`min_memory` may prefer recomputation even when capture is faster.

### Compiler and profile generation

The compiler emits explicit backward inputs for reconstructible primals and
keeps Tape arguments only for captured leaves. `none` profiles emit no Tape
traffic or Tape allocator arguments.

Static and dynamic capture use compatible logical offsets. CPU static records
and promoted dynamic records seal into immutable page-layout-v1 storage.
Page-layout-v1 is a CPU implementation detail, not a cross-backend ABI.

### Structured control

Canonical frontend ranges lower to `scf.for` before autodiff planning.
Reconstructible `if` and reverse `for` paths do not retain unconditional branch
or iteration history. Dynamic step, negative step, non-unit step, `break`,
`continue`, early return, and unsupported dynamic `while` forms retain the
general fallback.

Backend legalization happens after autodiff planning, so CPU, CUDA, Vulkan,
OpenGL, and OpenGL ES target compilation may lower structured loops without
hiding them from VJP analysis.

### CPU Runtime strategies

The loaded CPU pipeline owns immutable `CpuAutodiffProgram` and
`CpuResidualPlan` values. Pullbacks retain shared program state rather than a
complete executable.

Runtime selects one explicit strategy:

- `NoTapeCpuPullback`;
- `RetainedTapeCpuPullback`;
- `SegmentReplayCpuPullback`.

Whole-dispatch retention is allowed only for an admitted static plan and a hard
physical budget. Otherwise CPU replay executes one complete workgroup forward,
seals its Tape, runs backward, and releases or explicitly recycles construction
storage before the next segment.

Static and dynamic construction reuse use the same apply-time budget and
lifecycle:

```text
constructing -> frozen_reader -> recyclable -> released
```

Direct Runtime callers default reusable construction bytes to zero. Graph
callers derive temporary and reusable limits from the remaining compiled
checkpoint budget.

### ExecutionGraph integration

The graph tracks logical resource versions independently from backing owners.
A backward dependency may use a retained stable owner, an existing checkpoint,
or graph replay. Matching the current owner is not sufficient evidence.

Checkpoint planning accounts for residual, retained owner, restoration,
transaction, backward value, and replay memory separately. Pullback apply
passes its remaining budget through versioned Runtime apply options.

Replay restores caller-visible state and keeps gradient publication
transactional. Recoverable failures preserve pullback reuse only when retained
state and Tape reconstruction both succeed.

### Telemetry semantics

The three Tape retention metrics are:

- `logical_residual_bytes`;
- `resident_tape_bytes`;
- `allocated_tape_bytes`.

`peak_temporary_tape_bytes` includes construction and compaction scratch.
`peak_runtime_managed_bytes` additionally includes retained primal owners,
checkpoints, restoration state, backward values, accumulation, and other
Runtime-managed temporary storage.

Therefore `min_memory` means retained Tape can be zero; it does not mean total
process or graph memory is zero. Grid-sized primal, output, and gradient storage
still scales with the number of cells.

## CPU verification baseline

The committed one-step artifacts are:

- `autodiff_phase2_benchmark.json`;
- `autodiff_phase2_benchmark.md`.

They cover grids 32, 64, 128, 256, 512, and 1024 under `min_memory`.

Observed reference-host results:

- logical, resident, and allocated Tape are zero at every grid;
- peak temporary Tape is 436336 bytes at every grid;
- grid 512 completes in about 0.89 seconds;
- grid 1024 completes in about 5.5 seconds;
- grid 1024 reports about 155 MB of total Runtime-managed peak, primarily from
  grid-sized non-Tape state;
- the finite-difference gradient gate passes separately from the large-grid
  timing run.

The last complete verification baseline was 412 CTest cases and 457 Python
tests, with 52 expected backend skips.

## Completed CPU scope

- Per-value residual classification and budgeted source selection.
- Mixed and no-Tape compiler profiles.
- Canonical structured-loop reconstruction.
- CPU no-Tape, whole-dispatch, and complete-workgroup replay strategies.
- Immutable CPU program and residual-plan ownership.
- Shared Tape-enabled range execution.
- Static and dynamic construction recycling under apply-time budgets.
- Logical graph versions, checkpoint/replay, rollback, and state restoration.
- Versioned pullback apply options and temporary-memory telemetry.
- 32 through 1024 one-step `min_memory` smoke artifacts.

## Remaining work

### Can be completed without a contract release

- [x] Add direct compiler tests for measured residual-source priority and
  dynamic-`while` fallback.
- [x] Add an isolated C++ Runtime test that mutates caller-owned primal storage
  after pullback creation and proves replay uses the retained version.
- [x] Add a C++ Graph autodiff alias test that verifies transactional gradient
  publication; existing C++ alias tests cover graph hazards only.
- [x] Extend the benchmark artifact with top-level retained-allocation bytes and
  per-step RSS samples.
- [x] Commit a separate pressure-iteration 10, 32-grid, 100-step artifact. It
  must show zero retained Tape on every step, a bounded temporary Tape peak, and
  whether RSS reaches a stable allocator high-water mark.
- [x] Re-run the complete C++ and Python suites sequentially, then regenerate
  the 32/64/128/256/512/1024 one-step artifacts.

### Included in the current unreleased compiler/pipeline contract

The current contract version has not shipped, so these fields were added
without incrementing its version. Missing write-footprint metadata remains
conservative. There is no second Runtime inference path, hidden budget, mutable
global policy, or workload-size heuristic.

- [x] Serialize an explicit whole-dispatch-admission decision in the compiler
  plan and manifest, and validate the current unreleased contract.
- [x] Add versioned TensorView write-footprint reflection and validate it
  against bound shapes.
- [x] Replace whole-view replay restoration with declared dirty-range
  restoration. Missing or unknown metadata must remain conservative.
- [x] Enable end-to-end `balanced` frontend selection and commit 512/1024
  evidence for both admitted whole-dispatch retention and forced bounded replay.

### Separate backend and operator work

- [ ] Select a specialized pass/operator VJP when its measured plan is cheaper
  than generic replay.
- [x] Implement backend-local no-Tape rematerialized pullbacks for CUDA,
  Vulkan, DirectX 12, Metal, and OpenGL.
- [ ] Extend the shared GPU path with complete-workgroup bounded replay for
  captured static/dynamic Tape without GPU-to-host Tape readback.
