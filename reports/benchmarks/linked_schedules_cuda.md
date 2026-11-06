# Linked schedules on CUDA: phase 1 against numba-cuda

The CUDA slice of `specs/runtime/architecture_v2.md` phase 1. As on the CPU
([`linked_schedules_cpu.md`](linked_schedules_cpu.md)), a kernel called with an
explicit grid and no specializations links after its first call as a one-node
schedule. The fragment's record is the kernel's CUDA parameter list. The CUDA
HAL device loads the Stage's PTX once and launches it on the RHI device's
stream; a call packs the record in the native stub and returns once the launch
is enqueued. Buffer arguments are the owners' residencies in the session, used
only while they hold the owner's current version. The workloads, method and
kernel sources are those of [`numba_comparison.md`](numba_comparison.md).

## Provenance

- Source revision: VernonDSL `6fcb25d` with uncommitted work (`-dirty`);
  compiler contract 1, Program version 2, release 0.1.2.
- Command: `python examples/numba_benchmark/run.py --arch cuda --result-json reports/benchmarks/linked_schedules_cuda.json`,
  from the repository root with `PYTHONPATH=python`.
- Platform: Windows 11 (10.0.26300), AMD Ryzen 9 9950X, NVIDIA GeForce RTX
  4090, driver 610.88 (WDDM); Python 3.11.9, numpy 1.26.4, numba 0.68.1,
  numba-cuda 0.30.4 with fused multiply-adds off.
- Method: as in `numba_comparison.md`. A resident sample is one launch and a
  wait: numba waits with `cuda.synchronize()`. VernonDSL waits with the
  output's `wait()` in the last section; before it had a device wait, it
  launched `peek`, which copied one output element, and read that back.
- Raw results: `linked_schedules_cuda.json`, from the last section's run.

## Results (median ms; ratio is VernonDSL / numba)

| workload | mode | numba | VernonDSL | ratio | VernonDSL before | ratio before |
|---|---|---:|---:|---:|---:|---:|
| axpy | resident | 0.243 | 0.261 | 1.08x | 5.37 | 22x |
| axpy | roundtrip | 30.6 | 110 | 3.6x | 94.3 | 3.7x |
| launch_64 | resident | 0.025 | 0.033 | 1.31x | 1.39 | 63x |
| relief | resident | 1.29 | 1.32 | 1.02x | 4.58 | 3.6x |
| relief | roundtrip | 29.8 | 68.1 | 2.3x | 59.8 | 2.5x |
| nearest_segment | resident | 11.8 | 12.3 | 1.04x | 15.7 | 1.5x |
| nearest_segment | roundtrip | 29.4 | 87.8 | 3.0x | 78.3 | 2.9x |

"Before" is `numba_comparison_cuda.json`, measured at `ecb5a2e` on the same
machine. All three workloads still agree bit for bit with numba and numpy.
Roundtrip rows run as Programs, since each sample's fresh inputs have no
residency yet. On these rows numba was also 8 to 24% slower than in the
earlier run, so their differences are run-to-run variation.

- **Launch.** Enqueuing a linked call costs 8.4 µs; `Kernel.__call__` adds
  under 0.2 µs. Checking each buffer's owner and residency costs about
  0.2 µs, and publishing a write about 0.1 µs, measured as the same
  attribute accesses from Python. The remaining 7 µs or so is native: the
  driver's `cuLaunchKernel` under WDDM and the RHI's buffer address lookups,
  not measured apart. A `launch_64` sample is two such launches (the kernel
  and `peek`) and a one-element readback, where numba's is one launch and a
  synchronize. The `specs/performance.md` section 1 target for a warm CUDA
  launch is under 10 µs.
- **`axpy` resident.** Back to back, without the per-sample wait, an 8M
  `axpy` launch takes 0.242 to 0.248 ms against numba's 0.232 to 0.241 ms
  (two runs of 200 launches each). The 18 µs gap in the table is mostly the
  extra `peek` launch and readback of the wait; the 3 to 4% kernel gap
  remains.

## Unit-stride kernel bodies

`VernonToGPU` now emits each kernel body twice, selected by whether every
innermost stride is 1 (`specs/performance.md` section 4.1). In `axpy`'s PTX the
unit-stride copy addresses its three arrays with one `mul.wide.s32`; the other
copy keeps a `mul.lo.s64` per access. Back to back, 5 runs of 200 launches of
the 8M `axpy` with one wait per run (probe script, not saved), median per
launch: VernonDSL 0.2180 ms, numba 0.2166 ms, a 0.6% gap where it was 3 to 4%.
Outputs agree bit for bit. With the per-sample wait the table's method
measured, in one run after the change:

| workload | mode | numba | VernonDSL | ratio |
|---|---|---:|---:|---:|
| axpy | resident | 0.241 | 0.262 | 1.08x |
| launch_64 | resident | 0.024 | 0.025 | 1.03x |
| relief | resident | 1.22 | 1.31 | 1.07x |
| nearest_segment | resident | 32.8 | 11.9 | 0.36x |

The resident `axpy` row still carries the extra `peek` launch and readback of
the wait. numba's `nearest_segment` row is an outlier of that run (11.8 ms in
the table above).

## `min`, `max` and `clamp` as compare and select

These intrinsics now lower to a compare and a select on every backend instead
of `arith.minimumf`/`maximumf` (`specs/language/contract.md`). One run after
the change: `nearest_segment` resident 9.87 ms against numba's 18.8 (numba
again high; 11.8 in the first table), `relief` 1.31 against 1.29, `axpy`
0.263 against 0.244. All three agree bit for bit with numba and numpy.

## `relief` back to back

The `relief` kernels tie. In 5 interleaved runs of 200 back-to-back launches
with one wait per run (probe script, not saved), median per launch:
VernonDSL 1.1963 ms, numba 1.1954 ms, a 0.08% gap. Outputs agree bit for bit.
The table's 20 µs per-sample difference is the `peek` launch and the
one-element readback that VernonDSL's wait costs, as in the `axpy` row.

## Waiting with `wait()`

A linked call now records the timeline point its launch signals on the
residency it writes, and `wait()` waits for that point without a download
(`specs/language/contract.md` section 5). The benchmark waits with it instead
of `peek`:

| workload | mode | numba | VernonDSL | ratio |
|---|---|---:|---:|---:|
| axpy | resident | 0.243 | 0.232 | 0.96x |
| axpy | roundtrip | 25.5 | 102 | 4.0x |
| launch_64 | resident | 0.026 | 0.014 | 0.54x |
| relief | resident | 1.29 | 1.28 | 0.99x |
| relief | roundtrip | 24.4 | 63.4 | 2.6x |
| nearest_segment | resident | 10.4 | 9.82 | 0.94x |
| nearest_segment | roundtrip | 28.0 | 82.3 | 2.9x |

All three workloads agree bit for bit with numba and numpy. A `launch_64`
sample is now one launch and one stream synchronize on each side.

## At the end of phase 2

The same command after phase 2, at `6fcb25d` with uncommitted work. Raw
results: `linked_schedules_cuda_phase2.json`.

| workload | mode | numba | VernonDSL | ratio |
|---|---|---:|---:|---:|
| axpy | resident | 0.254 | 0.227 | 0.90x |
| axpy | roundtrip | 26.0 | 97.2 | 3.7x |
| launch_64 | resident | 0.025 | 0.012 | 0.47x |
| relief | resident | 1.29 | 1.27 | 0.99x |
| relief | roundtrip | 25.3 | 61.4 | 2.4x |
| nearest_segment | resident | 10.4 | 9.76 | 0.94x |
| nearest_segment | roundtrip | 28.5 | 79.7 | 2.8x |

Every resident row stays at or below numba's, and all three workloads agree
bit for bit with numba and numpy. Kernel launches take the same one-node
path; multi-node schedules replay a CUDA graph, whose per-node cost is in
[`training_step.md`](training_step.md).

## After step 5

The same command after section 16.3 step 5 of `architecture_v2.md`, at
`f840106` with uncommitted work. Raw results:
`linked_schedules_cuda_step5.json`.

| workload | mode | numba | VernonDSL | ratio |
|---|---|---:|---:|---:|
| axpy | resident | 0.242 | 0.229 | 0.95x |
| axpy | roundtrip | 25.6 | 93.3 | 3.6x |
| launch_64 | resident | 0.026 | 0.012 | 0.47x |
| relief | resident | 1.29 | 1.27 | 0.99x |
| relief | roundtrip | 23.8 | 57.5 | 2.4x |
| nearest_segment | resident | 10.4 | 9.79 | 0.94x |
| nearest_segment | roundtrip | 27.4 | 77.3 | 2.8x |

Every row matches phase 2 within 1%, or is below it, and all three
workloads agree bit for bit with numba and numpy.

## What remains

- **Gate.** Every resident row is at or below numba's, and `launch_64` is
  0.54x. Back to back, `axpy` launches are within 0.6% of numba's and
  `relief` launches within 0.1%.
- **Launch floor.** Each call is one `cuLaunchKernel`. Multi-node schedules
  replay a CUDA graph (`architecture_v2.md` section 6) and pay one launch
  for the graph.
- **First call on fresh data.** A call whose owner has no current residency
  in the session runs as a Program, which uploads, and later calls link. The
  roundtrip rows are therefore unchanged until buffers with one home replace
  residencies (`architecture_v2.md` section 8).
- **Coverage.** As on the CPU, kernels with Struct, Tensor-by-value or
  texture arguments, results, autodiff tapes, or `grid=None` still run as
  Programs.
