# Compute kernel limitations found by procedural terrain

Status: findings, non-normative. Each entry is current behavior, checked on the
CPU backend, and for the performance entries also on CUDA, at the revision of
[`reports/benchmarks/numba_comparison.md`](../../reports/benchmarks/numba_comparison.md).

These came from porting the Vernon engine's procedural terrain to VernonDSL
CPU kernels (`frontend/python/engine/location/placement/field_kernels.py` and
`ground_kernels.py` in the Vernon repository) and from writing the numba
comparison (`examples/numba_benchmark/`). Each entry gives the observed
behavior, the workaround those kernels carry, and what would remove it.
Entries marked "by design" follow [`contract.md`](contract.md) and are listed
because they shape kernel code.

## Types and storage

**No 64-bit integers.** The contract's scalars are `bool`, `i32`, `u32`, `f16`,
`f32` and `f64`. `vd.u64` and `vd.i64` exist as names, but storage rejects them
("dtype must be a Vernon scalar type") and kernels cannot construct them
("cannot infer call to 'u64'").
- Workaround: the SplitMix64 value-noise lattice is hashed on the host into
  per-query tables that the kernels interpolate (`ground_kernels._noise_inputs`).
- Cost: in the numba comparison, hashing the tables on the host takes 9.6 ms.
  That is 4.7 times as long as numba's whole noise kernel hashing in place with
  `u64` on CPU (2.0 ms), and 6 times its CUDA kernel (1.5 ms). 32-bit indices
  also cap a view at 2^31 elements.
- Wanted: `i64`/`u64` Values with wrapping arithmetic, on backends that
  advertise them.

**No 8- or 16-bit integer storage.** `TensorStorage` rejects `uint8` and
`int16` arrays ("unsupported TensorStorage dtype uint8").
- Workaround: terrain layer weights, five bytes per cell, travel packed into
  two `u32` words. They are packed on the host and unpacked in the kernel
  (`_packed`, `_unpacked`, `_layer`, `_store`).
- Wanted: `u8`/`i8`/`u16`/`i16` storage elements, widened to `i32`/`u32` on
  load.

**No implicit `i32`/`u32` mixing** (by design, contract section 2.1). The
invocation ID is `u32` while indices read from storage are `i32`, so index
arithmetic needs explicit casts.

## Literals

**Float literals passed as call arguments are rounded through `f32`.**
`vd.f64(0.1)`, `vd.max(x, 0.1)` and `vd.clamp(x, 0.1, 0.7)` with an `f64` `x`
all produce `0.10000000149011612`. The same literal inside arithmetic,
`x + 0.1`, stays exact. This contradicts the contextual typing of contract
section 2.1.
- Workaround: precision-critical constants are either kernel parameters
  (`_FAR`, `_TINY`) or chosen to be exact in `f32` (noise scales such as 320.0
  and 110.0).
- Wanted: literals typed by the call's resolved parameter type.

## Writes and output size

**Ordinary stores must be provably lane-exclusive**
([`../compiler/invocation_index_ownership.md`](../compiler/invocation_index_ownership.md)).
A scatter `out[at[i]] = v` and stores into a per-invocation range
`for k in range(first[b], first[b + 1]): out[k] = ...` are both rejected ("not
proven lane-exclusive").
- Workaround: `field_kernels._cull_fill` fills each bucket's own range of
  zeroed storage with `atomic_add`, which serves as a store. That only works
  for `i32`/`u32`; `f32`/`f64` atomic add is capability-checked.
- Wanted: a way to declare ranges as owned, such as CSR offsets from a prefix
  sum, so an ordinary store can be accepted. Or an explicitly unchecked store.

**No stream compaction or prefix sum.** A kernel cannot produce an output whose
size depends on the data. `atomic_add` does return the previous value, so an
append slot works, but the order is nondeterministic.
- Workaround: two launches. `_cull_count` counts per bucket, the host takes a
  cumulative sum, and `_cull_fill` fills.
- Wanted: a deterministic device scan and compaction in the library.

**Zero-size launches are rejected** ("grid must contain three positive
integers"). Every entry point special-cases empty input (`nearest`, `blended`,
`relief_heights`). Binding zero-length storage works.

**Outputs are not optional.** A kernel that only sometimes needs an output
still gets one bound.
- Workaround: `field_kernels.nearest` binds a one-element `segment` output and
  passes a `keep_segments` flag, because a launch pays for every element of its
  outputs (see "Performance").
- Wanted: an output disabled by static specialization (`When`) at launch,
  without a second kernel.

## Local data and intrinsics

**No mutable local arrays.** Assigning into a local Tensor, as in
`acc[j] = ...`, fails ("indexed assignment requires TensorView storage").
Reading a local Tensor at a dynamic index works.
- Workaround: `ground_kernels._store` carries five weights as five scalars with
  `if` chains for the argmax and the rounding remainder. `field_kernels._blend`
  cannot keep a fixed-capacity per-polyline table, so it scans the candidates
  twice and relies on each polyline's segments being consecutive.
- Wanted: fixed-size mutable local arrays, which the roadmap's fixed-capacity
  containers would provide.

**Missing intrinsics.** The available ones are `sin`, `cos`, `acos`, `atan2`,
`exp`, `log`, `sqrt`, `floor`, `abs`, `min`, `max`, `pow` and `clamp`, plus
the vector and atomic operations. These are absent: `ceil`, `round`/`rint`,
`trunc`, `fract`, `fmod`, `sign`, `mix`, `step`, `smoothstep`, `isinf`,
`isnan`, `isfinite`, `select`, `fma`, `tan`, `asin`, `atan`, `tanh`, `exp2`
and `log2`.
- Workaround: `ground_kernels._rint` rounds half to even through `floor`
  to match numpy. `_ease`/`_step` write out smoothstep. `_paint` tests
  finiteness with `w - w == 0.0`.

## Authoring

**Kernels and functions must be top-level definitions in a file** ("kernel
functions must be top-level definitions in files"). Kernels cannot be generated
by a factory or a closure; variation goes through static specialization.

**Type aliases cannot be used in annotations.** `F64_OUT = vd.TensorView[...]`
used as an annotation fails ("unknown DSL type 'F64_OUT'"), so every parameter
spells out its full type. The PCG kernels take 10 to 14 such parameters. The
annotation `tuple[...]` is not accepted either; it must be `vd.Tuple[...]`.

## Performance

These are measured in
[`reports/benchmarks/numba_comparison.md`](../../reports/benchmarks/numba_comparison.md),
where the outputs agree with numba bit for bit.

Common to all backends:

- **About 0.6 ms per launch**, the same at 1 and 32 threads and on CUDA,
  against 1 to 25 µs for numba. It is Python orchestration:
  - every call re-resolves, re-reads and re-hashes the kernel's source
    dependencies, once per cache layer;
  - it re-evaluates string annotations;
  - it binds the arguments.

  Callers therefore batch: the PCG kernels lay every leaf of a partition end to
  end and launch once (`field_kernels._Batch`, `ground_kernels._noise_inputs`).
- **Every write output is staged in full on every launch.** The runtime stages
  the whole bound output into a fresh carrier and copies it back, whether one
  lane writes or all of them:
  - CPU: about 16 ms per 64 MB at 32 threads;
  - CUDA: 4.5 ms per 64 MB, including a zero-filled host backing of the full
    size that the device kernel never uses.
- **100 to 250 ms first call per kernel per process on CPU, 80 to 120 ms on
  CUDA.** Vernon compiles its kernels at server start (`warm()`). Compilation
  is faster than numba's on both backends.

CPU:

- **About 2.5 ns of fixed cost per invocation.** The CPU range wrapper calls
  the lane entry once per invocation, re-reading its packed arguments.
- **No SIMD across invocations.** Each lane runs scalar code, while numba
  vectorizes the loop:
  - bodies bound by `floor` or division run 10 to 37 times slower;
  - gather-bound bodies run about 2.6 times slower;
  - the nearest-segment search runs 2 times slower.
- **Generic x86-64 code.** JIT kernels get an empty CPU name, so LLVM targets
  generic x86-64 without SSE4.1, AVX2 or FMA, and `floor` cannot use `roundsd`.

CUDA:

- **Kernel code is within about 10% of numba-cuda's** with fused multiply-add
  off on both sides. The gaps are in the runtime around it: an 8M-element
  `axpy` is 22 times slower, and the nearest-segment search 1.5 times slower.
- **Launches are synchronous.** The runtime waits for each submission to
  complete, once for the kernel and again for each copy batch. Host work never
  overlaps device work, and consecutive launches cannot queue.
- **No device wait without a download.** `synchronize()` downloads
  device-dirty ranges. Waiting for a launch without reading its output needs a
  second, one-lane kernel whose output is read back.
- **Transfers copy on the host three to four times per direction.** A 64 MB
  download takes about 33 ms (2 GB/s); numba-cuda uploads 128 MB and downloads
  64 MB in 25 ms.
- **No FMA contraction, and no opt-in.** Results match numpy bit for bit.
  numba-cuda's default contraction makes it 10 to 15% faster on the noise and
  nearest-segment kernels.
- **f64 runs at consumer-GPU rate** (1/64 of f32 on the RTX 4090).   The PCG
  kernels compute in f64, so this is a hardware cost rather than a language
  one, and numba pays it too.

## Checked and working

PCG code carries workarounds for some things that work now:

- an exponent literal such as `1.0e30` inside arithmetic, and module-level
  `f64` constants;
- binding zero-length storage, which makes the one-element padding of
  `field_kernels._nonempty` unnecessary;
- `while`, `break`, early `return`, conditional expressions, dynamic `range`
  bounds and `Tuple` returns from `@func`;
- wrapping `u32` multiplication, and `atomic_add` returning the previous value;
- bitwise operators and shifts on integers, so the byte unpacking by
  `// 256**k % 256` and an `if` chain choosing the power can become shifts and
  masks, and 32-bit hashes can be written in a kernel.
