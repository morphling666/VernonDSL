# VernonDSL kernels vs numba, CPU and CUDA

Workloads from the Vernon engine's procedural terrain (PCG) code, written once
per side with the same arithmetic in the same order. Every run checks both
outputs against a numpy reference before reporting a time, and records whether
they agree bit for bit.

## Provenance

- Source revision: VernonDSL `ecb5a2e` with uncommitted work (`-dirty`);
  compiler contract 1, Program version 2, release 0.1.2. The work includes a
  dense-block path in `planProgramTensorCopy`. Without it, the runtime walks
  every element of each write output on every launch: about 8 ns per element
  on CPU and CUDA, or 70 ms per call for an 8M-element f64 output on CUDA.
- Command: `python examples/numba_benchmark/run.py [--threads 1] [--numba-cpu generic] [--arch cuda [--numba-fma on]] --result-json <file>`,
  from the repository root with `PYTHONPATH=python`.
- Platform: Windows 11 (10.0.26300), AMD Ryzen 9 9950X (16 cores, 32 threads),
  NVIDIA GeForce RTX 4090 on driver 610.88 (WDDM). The 4090 runs f64 at 1/64
  of its f32 rate, and both sides pay this equally.
- Python 3.11.9, numpy 1.26.4, numba 0.68.1, llvmlite 0.50.0, numba threading
  layer `omp`, numba-cuda 0.30.4 (`cu12` wheels).
- Method: two warm-up calls, then the median of 10 samples (5 at one thread).
  `launch_64` takes 2000 samples (500 at one thread). The first call of each
  kernel in the process (compilation plus one run) is reported separately.
  - CPU numba uses `parallel=True`, `fastmath=False`, `error_model="numpy"`,
    `cache=False`. Both sides use the same thread count: VernonDSL through
    `VERNON_CPU_THREAD_BUDGET`, numba through `NUMBA_NUM_THREADS`.
  - CUDA numba uses `cuda.jit` with one thread per output in 64-thread blocks,
    VernonDSL's workgroup size.
- Raw results: `numba_comparison_32t_host.json`,
  `numba_comparison_32t_generic.json`, `numba_comparison_1t_host.json`,
  `numba_comparison_cuda.json` (numba FMA off), `numba_comparison_cuda_fma.json`
  (numba FMA on). Three CUDA runs with FMA off agree within 1%.

## Workloads

- `axpy`: `out = a * x + y` over 8M f64, memory-bound.
- `launch_64`: `axpy` over one 64-lane workgroup, the fixed cost of a call.
- `relief`: two noises of two octaves each over 4M cells, as in Vernon's
  `ground_kernels._relief`. The lattice is hashed on the host into tables the
  kernel interpolates, because VernonDSL kernels have no 64-bit integers.
  `numba_hashed` hashes the lattice in the kernel with `u64` SplitMix64.
- `nearest_segment`: each of 4M bucketed cells finds its nearest of 24,000
  segments within 30 m among its bucket's candidates (66 per bucket), as in
  Vernon's `field_kernels._distance`. It writes 4M f64 and 4M i32.

Modes:

- `resident` reuses allocated inputs and outputs. A sample is one launch and a
  wait for it:
  - CPU VernonDSL waits with `synchronize()` of its outputs.
  - CUDA numba waits with `cuda.synchronize()`.
  - CUDA VernonDSL, which has no wait that does not download, launches a
    one-lane `peek` kernel copying the output's first element and reads that
    back.
- `roundtrip` starts from numpy inputs and ends with numpy outputs. Inputs are
  uploaded and outputs read back where the target needs it, both sides allocate
  their outputs, and the table-driven noise hashes its tables.

## CPU results (median ms; ratio is VernonDSL / numba)

| workload | mode | 32 threads: numba | VernonDSL | ratio | 32 threads, numba `generic`: numba | ratio | 1 thread: numba | VernonDSL | ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| axpy | resident | 6.10 | 24.8 | 4.1x | 6.28 | 3.9x | 7.11 | 44.9 | 6.3x |
| axpy | roundtrip | 9.20 | 53.3 | 5.8x | 9.16 | 6.1x | 10.3 | 73.0 | 7.1x |
| launch_64 | resident | 0.007 | 0.599 | 87x | 0.012 | 48x | 0.001 | 0.591 | ~1000x |
| relief | resident | 2.06 | 29.0 | 14x | 3.78 | 7.7x | 28.2 | 326 | 12x |
| relief | roundtrip | 14.4 | 53.7 | 3.7x | 15.3 | 3.5x | 42.0 | 351 | 8.4x |
| nearest_segment | resident | 29.7 | 59.3 | 2.0x | 32.8 | 1.8x | 400 | 778 | 1.9x |
| nearest_segment | roundtrip | 33.5 | 79.7 | 2.4x | 35.3 | 2.2x | 400 | 798 | 2.0x |

For `relief` at 32 threads on the host CPU, `numba_hashed` takes 2.04 ms
resident and 4.21 ms roundtrip. Hashing the tables on the host alone takes
9.6 ms.

## CUDA results (median ms; ratio is VernonDSL / numba)

| workload | mode | numba, FMA off | VernonDSL | ratio | numba, FMA on | ratio |
|---|---|---:|---:|---:|---:|---:|
| axpy | resident | 0.241 | 5.37 | 22x | 0.242 | 23x |
| axpy | roundtrip | 25.5 | 94.3 | 3.7x | 24.4 | 3.9x |
| launch_64 | resident | 0.022 | 1.39 | 62x | 0.022 | 63x |
| relief | resident | 1.29 | 4.58 | 3.6x | 1.16 | 4.0x |
| relief | roundtrip | 23.9 | 59.8 | 2.5x | 23.6 | 2.5x |
| nearest_segment | resident | 10.4 | 15.7 | 1.5x | 8.79 | 1.8x |
| nearest_segment | roundtrip | 27.1 | 78.3 | 2.9x | 26.0 | 3.0x |

The "FMA on" ratios use that run's VernonDSL times, which are within 2% of
these.

- **Bit-for-bit agreement.** With FMA off, all three workloads agree bit for bit
  with numba and numpy. With numba-cuda's default FMA on, numba differs by up
  to 7e-15 while VernonDSL still matches numpy exactly.
- **Setting FMA off in numba-cuda takes two steps.** numba-cuda passes NVVM no
  `fma` option unless `fastmath` is set, and it links kernels with LTO, where
  nvJitLink fuses again. The benchmark therefore passes NVVM `fma=0` and turns
  LTO off.
- **`relief_hashed`.** It takes 1.53 ms resident with FMA off (1.35 ms with FMA
  on). Hashing the tables on the host takes 8.7 ms.

## First call, compilation plus one run (ms)

| kernel | CPU numba | CPU VernonDSL | CUDA numba | CUDA VernonDSL |
|---|---:|---:|---:|---:|
| axpy | 262 | 116 | 128 | 122 |
| relief | 513 | 120 | 269 | 83 |
| relief_hashed | 1056 | none | 503 | none |
| nearest_segment | 369 | 222 | 160 | 121 |

## CPU attribution

A one-thread micro-benchmark run against the same tree, over 4M invocations,
gives these times in ns per invocation. Each kernel writes a 4M-element f64
output, except the first row, which runs the same 4M invocations with a
64-element output:

| kernel body | VernonDSL | numba |
|---|---:|---:|
| invocations only | 2.5 | none |
| `out[i] = x[i]` | 4.9 | 0.3 |
| 16 × `i32(floor(...))` | 28.1 | 0.8 |
| 16 × f64 division | 13.8 | 1.4 |
| 16 random f64 gathers | 12.6 | 4.8 |
| 16 loop-invariant `(k, 4)` reads | 7.2 | 0.4 |

- **About 2.5 ns fixed per invocation.** The CPU range wrapper calls the lane
  entry once per invocation, re-reading the packed arguments each time
  (`VernonCpuAbiWrapper.cpp`).
- **About 2.4 ns per f64 output element at one thread, for staging the output.**
  A write output is staged into a carrier and copied back on every launch,
  whether one lane writes or all of them:
  - one thread: about 2.4 ns per f64 element;
  - 32 threads: about 16 ms per 64 MB output, which is most of `axpy`'s 24.8 ms.
- **No SIMD across invocations.** numba vectorizes `prange` bodies, while
  VernonDSL runs each lane's scalar code:
  - arithmetic-bound bodies (`floor`, division) are 10 to 37 times slower;
  - memory-bound gathers are 2.6 times slower.
- **Generic x86-64 code.** JIT kernels are compiled with an empty CPU name
  (`CpuTargetOptions.processor`), so LLVM targets generic x86-64 without
  SSE4.1, AVX2 or FMA. Forcing numba to `generic` makes its `relief` 1.8 times
  slower; on `axpy` it makes no difference.
- **Fixed launch cost of about 0.6 ms**, the same at 1 and 32 threads. A
  `cProfile` of 1000 launches puts it in Python orchestration:
  - About 40% revalidates the kernel's sources. Each call resolves every
    frontend dependency by probing parent directories, then reads and
    SHA-256-hashes the file, once for the frontend cache and once for the
    compiled-kernel cache.
  - About 15% re-evaluates string annotations through
    `inspect.get_annotations`.
  - About 20% binds arguments and computes their ABI layouts.
  - The native dispatch is a small remainder.

## CUDA attribution

From `py-spy --native` profiles of the resident loops, and from launches that
vary output size, grid and number of writing lanes independently:

- **Kernel code is close to numba's.** Subtracting the measured cost of an
  `axpy` launch with the same output size leaves these kernel times:

  | kernel | VernonDSL, estimated | numba, FMA off |
  |---|---:|---:|
  | `relief` | about 1.5 ms | 1.29 ms |
  | `nearest_segment` | about 11.5 ms | 10.4 ms |

  With FMA on, numba's `nearest_segment` gains another 15%.
- **About 3.1 ms per 32 MB and 4.5 ms per 64 MB of write output.** The cost
  tracks the size of the bound output, not the grid or the number of lanes that
  write. Each launch:
  - zero-fills a host `ownedHostBytes` backing of the output's full size in
    `materializeProgramOwnedStorages` (about a third of the time), though the
    kernel runs on the device;
  - allocates a fresh device carrier;
  - copies the output into the carrier and back.

  This is why `axpy` is 22 times slower: numba's whole kernel takes 0.24 ms.
- **Every submission blocks.** `SubmissionRecord::execute` submits and then
  waits on the completion event: for the kernel graph, and again for each copy
  batch. So one launch makes several blocking CPU–GPU round trips, and host
  work never overlaps device work. Waiting is about a third of an 8M-element
  launch.
- **No wait without a download.** `synchronize()` downloads device-dirty
  ranges. The benchmark's `peek` workaround costs a second launch: `launch_64`
  is two 0.6–0.7 ms launches, plus an 8-byte readback.
- **Transfers copy the data several times on the host.**
  - Upload: `from_numpy` copies the array, `_materialize_buffer` snapshots it
    with `bytes(...)`, and the native upload stages it.
  - Download: native `download_ranges` returns Python `bytes`, which are copied
    into the host array, and `to_numpy` copies that again.

  A 64 MB upload takes 7.2 ms; a 64 MB download takes about 33 ms (2 GB/s).
  numba's entire `axpy` roundtrip, which uploads 128 MB and downloads 64 MB,
  takes 25 ms.
- **No FMA contraction.** VernonDSL's CUDA target leaves multiply-adds unfused,
  so it matches numpy bit for bit. This costs about 10–15% on `relief` and
  `nearest_segment` against numba-cuda's default, and there is no opt-in.
- **Compilation** takes 80–120 ms per kernel, against numba-cuda's 130–270 ms.
