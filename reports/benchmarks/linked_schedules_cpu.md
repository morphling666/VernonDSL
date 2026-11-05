# Linked schedules on CPU: phase 1 against numba

The first slice of `specs/runtime/architecture_v2.md` phase 1 on CPU. A kernel
called with an explicit grid and no specializations links, after its first
call, as a one-node schedule: one fragment built from the kernel's reflection,
recorded once on the CPU HAL device. Later calls pack the argument record in a
native stub and submit it, without the Program path. The workloads, method and
both kernel sources are those of
[`numba_comparison.md`](numba_comparison.md); only VernonDSL's runtime path
changed.

## Provenance

- Source revision: VernonDSL `6fcb25d` with uncommitted work (`-dirty`);
  compiler contract 1, Program version 2, release 0.1.2.
- Command: `python examples/numba_benchmark/run.py --arch cpu --result-json reports/benchmarks/linked_schedules_cpu_32t.json`,
  from the repository root with `PYTHONPATH=python`.
- Platform: Windows 11 (10.0.26300), AMD Ryzen 9 9950X (16 cores, 32
  threads); Python 3.11.9, numpy 1.26.4, numba 0.68.1, threading layer `omp`.
- Method: as in `numba_comparison.md`, two warm-up calls then the median of 9
  samples (`--repeats 9`), and 2000 samples for `launch_64`. Both sides use 32
  threads unless a column says otherwise.
- Pinned column: the same command with `VERNON_CPU_PIN_WORKERS=1` and
  `--result-json reports/benchmarks/linked_schedules_cpu_32t_pinned.json`.
- One-thread column: `--threads 1 --repeats 5 --launch-repeats 500`, not saved.
- Raw results: `linked_schedules_cpu_32t.json`, `linked_schedules_cpu_32t_pinned.json`.

## Results (median ms; ratio is VernonDSL / numba)

| workload | mode | numba | VernonDSL | ratio | pinned | ratio | 1 thread, numba / VernonDSL | first slice | `numba_comparison.md` |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| axpy | resident | 6.56 | 8.19 | 1.25x | 7.30 | 1.10x | 7.88 / 7.36 | 7.46 | 24.8 |
| axpy | roundtrip | 9.58 | 37.0 | 3.9x | 35.2 | 3.7x | 11.0 / 36.7 | 52.5 | 53.3 |
| launch_64 | resident | 0.008 | 0.004 | 0.57x | 0.004 | 0.59x | 0.001 / 0.003 | 0.004 | 0.599 |
| relief | resident | 2.89 | 3.14 | 1.09x | 2.42 | 1.18x | 27.8 / 29.3 | 19.8 | 29.0 |
| relief | roundtrip | 15.1 | 31.6 | 2.1x | 28.8 | 1.9x | 40.7 / 54.8 | 44.6 | 53.7 |
| nearest_segment | resident | 31.4 | 28.1 | 0.90x | 28.3 | 0.93x | 397 / 399 | 47.6 | 59.3 |
| nearest_segment | roundtrip | 33.6 | 48.8 | 1.45x | 48.3 | 1.18x | 401 / 418 | 65.1 | 79.7 |

"First slice" is this report's first measurement, with the per-workgroup lane
loop and before the compiler work below. The last column is the 32-thread host
column of `numba_comparison.md`, measured at `ecb5a2e` on the same machine.
The pinned column's ratio is against that run's own numba side: 6.65 ms for
`axpy`, 2.06 ms for `relief` and 30.3 ms for `nearest_segment` resident.
The previous run, before the `min`/`max` change below, gave pinned `relief`
2.25 ms against numba's 2.24. All three workloads agree bit for bit with
numba and numpy in every run.

Medians at 32 threads move by up to 30% between runs on both sides: numba's
`relief` resident ranged from 1.6 to 2.6 ms over this work. Back-to-back
probes of 300 calls each (`relief`, 32 threads) gave numba a 1.67 ms minimum
and 2.1 to 2.2 ms median, and VernonDSL 3.19 / 3.48 ms unpinned and 2.06 / 2.39 ms
pinned.

## Compiler and scheduler changes since the first slice

The causes are `specs/performance.md` sections 3.2 and 4.1 to 4.3; this is
what each changed.

- **Lane loop outlined with `noalias` buffers, host-CPU JIT.** With these,
  the 32-thread medians were 7.2 to 8.0 ms for `axpy` and 3.4 to 3.8 ms for
  `relief`; one thread, 14.8 ms (numba 7.3) and 42.5 ms (numba 28.2).
- **Chunk entry.** Single-thread `axpy` went from 14.8 to 6.8 ms and
  `relief` from 42.5 to 32.3 ms; at 32 threads `axpy` gained about 0.5 ms
  and `relief` nothing.
- **Unit-stride copy.** `relief`'s copy reads `xs` and `ys` and writes `out`
  with vector loads and stores; only its 32 lattice-table gathers, which numba
  also has, remain. Single-thread `relief` went from 32.3 to 29.0 ms against
  numba's 28.2, and the 32-thread median from 3.5 to 3.1 ms.
- **`min`, `max` and `clamp` as compare and select.** They lowered to the
  IEEE `minimum`/`maximum`, which x86 expands into NaN and signed-zero fixups;
  `nearest_segment` calls three per candidate segment. With the same kernel
  written with conditional expressions, one-thread `nearest_segment` measured
  395 ms against 496 ms. After the lowering change, `vd.max`/`vd.clamp`
  measured 397 ms, numba 397 ms. The 32-thread median went from 35.0 to
  28.1 ms. Results did not change on this data, which has no NaN.
- **Calling thread as a runner.** Within noise.
- **Rejected: workers that poll before sleeping.** A 200 µs poll raised
  `relief` resident to 4.4 ms; a 1 µs poll measured 3.08 ms.
- **Rejected: a static partition without pinning.** `nearest_segment` rose to
  40.5 ms (its cost varies per point) and `relief` to 4.0 ms.

## Why 32-thread `relief` depends on pinning

With the two runtimes' calls interleaved, so that neither finds its data in
cache, they tie: 4.04 ms median for numba and 4.11 ms for VernonDSL at 32
threads, 3.60 and 3.99 ms at 16. numba's advantage back to back is cache reuse:
its OpenMP threads keep their static slices on the same processors between
calls, so the next call finds them in the L3 of the same core complex.
VernonDSL's workers sleep and wake on whichever processor Windows picks.
Pinning them and giving each a fixed region (`VERNON_CPU_PIN_WORKERS=1`)
recovers that reuse: `relief` resident 2.25 ms against numba's 2.24. An ideal-processor hint instead of
a hard mask measured a 3.27 to 3.33 ms median.

`axpy` resident has the same cause. With samples interleaved (40 per side,
probe script, not saved), median ms, VernonDSL / numba:

| threads | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|---:|
| default | 7.55 / 7.31 | 7.69 / 7.40 | 7.52 / 7.58 | 7.76 / 7.48 | 7.97 / 7.76 | 8.08 / 8.15 |

Both are bound by DRAM bandwidth from one thread on: the three 64 MB arrays
do not fit the two 32 MB L3s, and more threads do not help. Back to back,
numba's 6.56 ms at 32 threads is the part of the previous call's data still
in the L3 of the core complex that reads it next.

- **Launch.** A call costs about 1.9 µs: about 0.4 µs of Python in
  `Kernel.__call__` and about 1.5 µs in the native stub and the dispatch.
- **`axpy` resident** no longer stages its output through a carrier. The
  remaining 1.24x is the kernel itself (next section).
- **Roundtrip rows** are dominated by `from_numpy` and `to_numpy` copies, which
  this phase does not change.

## Runner count

Before this change, the CPU scheduler took as many runners as the thread
budget allowed. When a launch had fewer workgroups than runners, it split each
workgroup's lanes across threads and ran the barrier-phase protocol. A
1000-element `axpy` (16 workgroups) took 195 µs. The HAL device now gives each
runner whole workgroups and at least `VERNON_CPU_INVOCATIONS_PER_RUNNER`
invocations. `axpy` resident, µs per call, by that setting:

| invocations per runner | 1000 | 4096 | 16384 | 65536 | 262144 | 1M | 8M |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 5.2 | 40.6 | 50.2 | 79.6 | 129 | 367 | 6958 |
| 4096 (default) | 5.2 | 15.3 | 58.4 | 80.9 | 136 | 363 | 6968 |
| 16384 | 5.2 | 15.5 | 57.2 | 116 | 165 | 366 | 6878 |
| 65536 | 5.1 | 15.5 | 56.3 | 219 | 295 | 384 | 6912 |

From the one-runner columns, an `axpy` invocation costs about 3.5 ns, and
waking workers costs about 35 µs. The default is 4096, the best or near-best
setting at every size.

## At the end of phase 2

The same commands after phase 2 (multi-node Program fragments, the linked
VJP, CPU autodiff tapes, graph linking), at `6fcb25d` with uncommitted work.
Kernel launches take the same one-node path, so the phase must leave these
rows unchanged. Raw results: `linked_schedules_cpu_32t_phase2.json` and
`linked_schedules_cpu_32t_pinned_phase2.json`.

| workload | mode | numba | VernonDSL | ratio | pinned: numba | VernonDSL | ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| axpy | resident | 6.57 | 8.11 | 1.24x | 6.64 | 7.43 | 1.12x |
| axpy | roundtrip | 11.1 | 38.0 | 3.4x | 10.0 | 34.9 | 3.5x |
| launch_64 | resident | 0.011 | 0.004 | 0.32x | 0.007 | 0.003 | 0.48x |
| relief | resident | 1.69 | 3.10 | 1.84x | 2.67 | 2.39 | 0.90x |
| relief | roundtrip | 15.2 | 31.2 | 2.05x | 15.0 | 28.9 | 1.92x |
| nearest_segment | resident | 30.3 | 28.0 | 0.93x | 30.3 | 28.0 | 0.93x |
| nearest_segment | roundtrip | 39.4 | 48.4 | 1.23x | 41.9 | 47.8 | 1.14x |

VernonDSL's medians are within 3% of the phase 1 table. numba's `relief`
resident moved from 1.69 to 2.67 ms between the two runs, the variation the
results section describes. All three workloads agree bit for bit with numba
and numpy. The training step's measurements are in
[`training_step.md`](training_step.md).

## What remains

- **Gate.** `relief` resident at 32 threads ties numba only pinned and
  within noise: 1.00x in the previous run, 1.18x in this one. Unpinned it
  is 1.1x to 1.6x. `axpy` resident is 1.1x to 1.3x at 32 threads back to
  back and ties at every thread count interleaved; both gaps are cache
  reuse across calls, which pinning recovers.
- **`nearest_segment`.** Faster than numba at 32 threads and even on one.
  LLVM still does not vectorize its lane loop, whose trip count varies per
  lane; the SIMT-to-SIMD pass of `specs/performance.md` section 4.1
  vectorizes such loops under masks.
- **Wake cost.** Workers sleep on a condition variable, so waking them costs
  about 35 µs. A measured chunk size (`specs/performance.md` section 3.2)
  replaces the fixed setting.
- **Coverage.** Kernels with Struct, Tensor-by-value or texture arguments,
  results, or `grid=None` still run as Programs, as does a lone kernel with
  an autodiff tape; a Module VJP links its tapes (`architecture_v2.md`
  section 16.1). So do arguments with device residencies, aliasing writes,
  or non-injective writable layouts.
