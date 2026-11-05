"""Compares VernonDSL kernels with numba on procedural-terrain workloads.

Each workload runs the same algorithm, in the same arithmetic order, in both
(`dsl_kernels.py` against `numba_kernels.py` with `--arch cpu`, or against
`numba_cuda_kernels.py` with `--arch cuda`), and the outputs are checked
against each other and against numpy before any timing is reported. Modes:

- `resident`: inputs and outputs are allocated once on the target; a sample
  is one launch and a wait for it. On CPU, VernonDSL waits by
  `synchronize()` of its outputs. On CUDA, numba waits by
  `cuda.synchronize()`, and VernonDSL, which has no device-wait call, by
  reading back one element copied from the output by a second kernel.
- `roundtrip`: a sample starts from numpy inputs and ends with numpy outputs,
  as a caller holding numpy arrays pays: inputs are uploaded and outputs read
  back where the target needs it, outputs are allocated, and the table-driven
  noise hashes its lattice on the host.

The first call of each kernel (compilation plus one run) is reported apart.
On CPU both sides use the same thread count, `--threads`; set before either
library is imported, as both read it then.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

_SPLITMIX_GOLDEN_X = np.uint64(0x9E3779B97F4A7C15)
_SPLITMIX_GOLDEN_Y = np.uint64(0xC2B2AE3D27D4EB4F)
_SPLITMIX_A = np.uint64(0xBF58476D1CE4E5B9)
_SPLITMIX_B = np.uint64(0x94D049BB133111EB)
_UNIT = 1.0 / 9007199254740992.0
# Grid of cells 1 m apart from this corner, bucketed in squares of cells.
_ORIGIN_M = 1000.0
_BUCKET_CELLS = 16
# Nearest-segment search radius, and the distance read beyond it.
_REACH_M = 30.0
_FAR = 1.0e30
_TINY = 1.0e-12


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arch", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    parser.add_argument(
        "--numba-cpu",
        default=None,
        help="numba's target CPU (NUMBA_CPU_NAME); 'generic' matches VernonDSL's CPU target",
    )
    parser.add_argument(
        "--numba-fma",
        choices=("off", "on"),
        default="off",
        help="whether numba-cuda fuses multiply-adds (its default is on); VernonDSL does not",
    )
    parser.add_argument("--grid", type=int, default=2048, help="cells per side of the noise and distance grids")
    parser.add_argument("--axpy-size", type=int, default=1 << 23)
    parser.add_argument("--rings", type=int, default=2000, help="closed 12-gons whose edges `nearest` searches")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--launch-repeats", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--result-json", type=Path)
    arguments = parser.parse_args()
    if arguments.threads <= 0 or arguments.grid % _BUCKET_CELLS != 0:
        parser.error(f"--threads must be positive and --grid a multiple of {_BUCKET_CELLS}")
    return arguments


ARGUMENTS = _arguments()
os.environ["VERNON_CPU_THREAD_BUDGET"] = str(ARGUMENTS.threads)
os.environ["NUMBA_NUM_THREADS"] = str(ARGUMENTS.threads)
if ARGUMENTS.numba_cpu:
    os.environ["NUMBA_CPU_NAME"] = ARGUMENTS.numba_cpu
# `launch_64` is one block on purpose.
os.environ["NUMBA_CUDA_LOW_OCCUPANCY_WARNINGS"] = "0"

import dsl_kernels as dk  # noqa: E402
import numba  # noqa: E402
import vernon_dsl as vd  # noqa: E402

ON_GPU = ARGUMENTS.arch == "cuda"
if ON_GPU:
    import numba_cuda_kernels as nk
    from numba import cuda

    nk.contract_fma(ARGUMENTS.numba_fma == "on")
    _numba_put = cuda.to_device
    _numba_wait = cuda.synchronize

    def _numba_empty(n: int, dtype=np.float64):
        return cuda.device_array(n, dtype)

    def _numba_get(array) -> np.ndarray:
        return array.copy_to_host()

else:
    import numba_kernels as nk

    _numba_empty = np.empty

    def _numba_put(array: np.ndarray) -> np.ndarray:
        return array

    def _numba_get(array: np.ndarray) -> np.ndarray:
        return array

    def _numba_wait() -> None:
        pass


def _dsl_wait(*outputs) -> None:
    """Waits for the launch that wrote `outputs`, without copying them to the host."""
    for output in outputs:
        output.wait()


@dataclass
class Sample:
    workload: str
    implementation: str
    mode: str
    size: int
    median_ms: float
    min_ms: float
    repeats: int
    first_call_ms: float | None = None
    note: str = ""


@dataclass
class Check:
    workload: str
    identical: bool
    max_abs_difference: float
    reference_max_abs_difference: float


@dataclass
class Report:
    environment: dict
    samples: list[Sample] = field(default_factory=list)
    checks: list[Check] = field(default_factory=list)


def _timed(run: Callable[[], object], repeats: int) -> tuple[float, float]:
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        run()
        times.append(time.perf_counter() - start)
    return statistics.median(times) * 1e3, min(times) * 1e3


def _first(run: Callable[[], object]) -> float:
    start = time.perf_counter()
    run()
    return (time.perf_counter() - start) * 1e3


_COMPILED: set[tuple[str, str]] = set()


def _measure(
    report: Report,
    workload: str,
    size: int,
    runs: dict[tuple[str, str], Callable[[], object]],
    repeats: int,
    notes: dict[str, str] | None = None,
    kernel: str | None = None,
) -> None:
    """Times each (implementation, mode) of `runs`. The first run of an
    implementation's `kernel` in the process counts as its compilation."""
    for (implementation, mode), run in runs.items():
        first = None
        if (implementation, kernel or workload) not in _COMPILED:
            first = _first(run)
            _COMPILED.add((implementation, kernel or workload))
        for _ in range(2):
            run()
        median, least = _timed(run, repeats)
        report.samples.append(
            Sample(
                workload,
                implementation,
                mode,
                size,
                median,
                least,
                repeats,
                first,
                (notes or {}).get(implementation, ""),
            )
        )


def _groups(count: int) -> tuple[int, int, int]:
    return ((count + dk.WORKGROUP - 1) // dk.WORKGROUP, 1, 1)


def _check(report: Report, workload: str, dsl: np.ndarray, other: np.ndarray, reference: np.ndarray) -> None:
    finite = np.isfinite(reference)
    both = np.isfinite(dsl) & np.isfinite(other)
    report.checks.append(
        Check(
            workload,
            bool(np.array_equal(dsl, other)),
            float(np.max(np.abs(dsl[both] - other[both]), initial=0.0)),
            float(np.max(np.abs(dsl[finite] - reference[finite]), initial=0.0)),
        )
    )
    if not np.allclose(dsl[finite], reference[finite], rtol=1e-12, atol=1e-9):
        raise SystemExit(f"{workload}: VernonDSL disagrees with the numpy reference")
    if not np.allclose(other[finite], reference[finite], rtol=1e-12, atol=1e-9):
        raise SystemExit(f"{workload}: numba disagrees with the numpy reference")


def axpy(report: Report, rng: np.random.Generator, n: int, repeats: int, workload: str = "axpy") -> None:
    x = rng.random(n)
    y = rng.random(n)
    a = 1.5
    put = vd.storage.from_numpy
    x_view, y_view = put(x), put(y)
    dsl_out = vd.storage.zeros(dtype=vd.f64, shape=(n,))
    x_numba, y_numba = _numba_put(x), _numba_put(y)
    numba_out = _numba_empty(n)

    def dsl_resident():
        dk.axpy(x_view, y_view, dsl_out, np.float64(a), np.uint32(n), grid=_groups(n))
        _dsl_wait(dsl_out)

    def dsl_roundtrip():
        out = vd.storage.zeros(dtype=vd.f64, shape=(n,))
        dk.axpy(put(x), put(y), out, np.float64(a), np.uint32(n), grid=_groups(n))
        return out.to_numpy()

    def numba_resident():
        nk.axpy(x_numba, y_numba, numba_out, a)
        _numba_wait()

    def numba_roundtrip():
        out = _numba_empty(n)
        nk.axpy(_numba_put(x), _numba_put(y), out, a)
        return _numba_get(out)

    _measure(
        report,
        workload,
        n,
        {
            ("numba", "resident"): numba_resident,
            ("vernon", "resident"): dsl_resident,
            ("numba", "roundtrip"): numba_roundtrip,
            ("vernon", "roundtrip"): dsl_roundtrip,
        },
        repeats,
    )
    _check(report, workload, dsl_out.to_numpy(), _numba_get(numba_out), a * x + y)


def launch(report: Report, repeats: int) -> None:
    """Fixed cost per call: `axpy` over one workgroup."""
    n = dk.WORKGROUP
    x = np.ones(n)
    x_view = vd.storage.from_numpy(x)
    dsl_out = vd.storage.zeros(dtype=vd.f64, shape=(n,))
    x_numba = _numba_put(x)
    numba_out = _numba_empty(n)

    def dsl_resident():
        dk.axpy(x_view, x_view, dsl_out, np.float64(1.0), np.uint32(n), grid=_groups(n))
        _dsl_wait(dsl_out)

    def numba_resident():
        nk.axpy(x_numba, x_numba, numba_out, 1.0)
        _numba_wait()

    _measure(
        report,
        "launch_64",
        n,
        {("numba", "resident"): numba_resident, ("vernon", "resident"): dsl_resident},
        repeats,
        kernel="axpy",
    )


def _cells(grid: int) -> np.ndarray:
    """Cell centers in bucket order: each bucket's cells together, u fastest
    within it."""
    side = grid // _BUCKET_CELLS
    order = np.arange(grid * grid).reshape(side, _BUCKET_CELLS, side, _BUCKET_CELLS).transpose(0, 2, 1, 3).ravel()
    u, v = np.divmod(order, grid)
    return np.stack([_ORIGIN_M + u + 0.5, _ORIGIN_M + v + 0.5], axis=1)


def _lattice_values(seed: np.uint64, ix: np.ndarray, iy: np.ndarray) -> np.ndarray:
    """`numba_kernels.lattice_value`, vectorized."""
    z = seed ^ (ix.astype(np.uint64) * _SPLITMIX_GOLDEN_X) ^ (iy.astype(np.uint64) * _SPLITMIX_GOLDEN_Y)
    z = (z ^ (z >> np.uint64(30))) * _SPLITMIX_A
    z = (z ^ (z >> np.uint64(27))) * _SPLITMIX_B
    z = z ^ (z >> np.uint64(31))
    return (z >> np.uint64(11)).astype(np.float64) * _UNIT


_RELIEF_SCALES_M = (dk.HILL_COARSE_M, dk.HILL_FINE_M, dk.SWELL_COARSE_M, dk.SWELL_FINE_M)


def _lattice_tables(xs: np.ndarray, ys: np.ndarray, seeds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The hashed lattice the points read, one row per noise octave:
    (base, x0, y0, span_y) into the concatenated table."""
    tables, rows, base = [], [], 0
    for seed, scale in zip(seeds, _RELIEF_SCALES_M, strict=True):
        x0, x1 = (int(np.floor(v / scale)) for v in (xs.min(), xs.max()))
        y0, y1 = (int(np.floor(v / scale)) for v in (ys.min(), ys.max()))
        span_x, span_y = x1 - x0 + 2, y1 - y0 + 2
        gx, gy = np.meshgrid(np.arange(x0, x0 + span_x), np.arange(y0, y0 + span_y), indexing="ij")
        tables.append(_lattice_values(seed, gx.ravel(), gy.ravel()))
        rows.append((base, x0, y0, span_y))
        base += span_x * span_y
    return np.concatenate(tables), np.asarray(rows, np.int32)


def _relief_reference(xs: np.ndarray, ys: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    def value(seed, x, y):
        fx, fy = np.floor(x), np.floor(y)
        sx = (x - fx) * (x - fx) * (3.0 - 2.0 * (x - fx))
        sy = (y - fy) * (y - fy) * (3.0 - 2.0 * (y - fy))
        ix, iy = fx.astype(np.int64), fy.astype(np.int64)
        a = _lattice_values(seed, ix, iy)
        b = _lattice_values(seed, ix + 1, iy)
        c = _lattice_values(seed, ix, iy + 1)
        d = _lattice_values(seed, ix + 1, iy + 1)
        bottom = a + sx * (b - a)
        top = c + sx * (d - c)
        return bottom + sy * (top - bottom)

    def octaves(row):
        coarse, fine = _RELIEF_SCALES_M[row], _RELIEF_SCALES_M[row + 1]
        return 0.7 * value(seeds[row], xs / coarse, ys / coarse) + 0.3 * value(seeds[row + 1], xs / fine, ys / fine)

    return 12.0 * (octaves(0) - 0.5) + 2.0 * (octaves(2) - 0.5)


def relief(report: Report, rng: np.random.Generator, grid: int, repeats: int) -> None:
    cells = _cells(grid)
    xs = np.ascontiguousarray(cells[:, 0])
    ys = np.ascontiguousarray(cells[:, 1])
    n = xs.shape[0]
    seeds = rng.integers(0, 2**63, size=4, dtype=np.uint64)
    table, lattice = _lattice_tables(xs, ys, seeds)
    put = vd.storage.from_numpy
    xs_view, ys_view, table_view, lattice_view = put(xs), put(ys), put(table), put(lattice)
    dsl_out = vd.storage.zeros(dtype=vd.f64, shape=(n,))
    numba_inputs = [_numba_put(a) for a in (xs, ys, table, lattice)]
    seeds_numba = _numba_put(seeds)
    numba_out = _numba_empty(n)
    hashed_out = _numba_empty(n)

    def dsl_resident():
        dk.relief(xs_view, ys_view, table_view, lattice_view, dsl_out, np.uint32(n), grid=_groups(n))
        _dsl_wait(dsl_out)

    def dsl_roundtrip():
        table, lattice = _lattice_tables(xs, ys, seeds)
        out = vd.storage.zeros(dtype=vd.f64, shape=(n,))
        dk.relief(put(xs), put(ys), put(table), put(lattice), out, np.uint32(n), grid=_groups(n))
        return out.to_numpy()

    def numba_resident():
        nk.relief(*numba_inputs, numba_out)
        _numba_wait()

    def numba_roundtrip():
        table, lattice = _lattice_tables(xs, ys, seeds)
        out = _numba_empty(n)
        nk.relief(*(_numba_put(a) for a in (xs, ys, table, lattice)), out)
        return _numba_get(out)

    def hashed_resident():
        nk.relief_hashed(numba_inputs[0], numba_inputs[1], seeds_numba, hashed_out)
        _numba_wait()

    def hashed_roundtrip():
        out = _numba_empty(n)
        nk.relief_hashed(_numba_put(xs), _numba_put(ys), _numba_put(seeds), out)
        return _numba_get(out)

    _measure(
        report,
        "relief",
        n,
        {
            ("numba", "resident"): numba_resident,
            ("vernon", "resident"): dsl_resident,
            ("numba_hashed", "resident"): hashed_resident,
            ("numba", "roundtrip"): numba_roundtrip,
            ("vernon", "roundtrip"): dsl_roundtrip,
            ("numba_hashed", "roundtrip"): hashed_roundtrip,
        },
        repeats,
        {"numba_hashed": "hashes the lattice in the kernel with u64; VernonDSL has no 64-bit integers"},
    )
    report.samples.append(
        Sample(
            "relief",
            "host_lattice_hashing",
            "roundtrip",
            n,
            *_timed(lambda: _lattice_tables(xs, ys, seeds), repeats),
            repeats,
        )
    )
    reference = _relief_reference(xs, ys, seeds)
    _check(report, "relief", dsl_out.to_numpy(), _numba_get(numba_out), reference)
    if not np.array_equal(_numba_get(hashed_out), _numba_get(numba_out)):
        raise SystemExit("relief: the in-kernel hash disagrees with the host-hashed table")


def _segments(rng: np.random.Generator, grid: int, rings: int) -> tuple[np.ndarray, np.ndarray]:
    """The edges of `rings` jittered 12-gons of 5 to 25 m radius."""
    sides = 12
    centers = _ORIGIN_M + rng.random((rings, 2)) * grid
    radii = 5.0 + 20.0 * rng.random((rings, 1)) * (0.8 + 0.4 * rng.random((rings, sides)))
    angles = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False) + 0.2 * rng.random((rings, sides))
    corners = centers[:, None, :] + radii[..., None] * np.stack([np.cos(angles), np.sin(angles)], axis=-1)
    starts = corners.reshape(-1, 2)
    ends = np.roll(corners, -1, axis=1).reshape(-1, 2)
    return np.ascontiguousarray(starts), np.ascontiguousarray(ends)


def _candidates(starts: np.ndarray, ends: np.ndarray, grid: int) -> tuple[np.ndarray, np.ndarray]:
    """Per bucket, the segments whose box grown by the reach meets it:
    `pool[first[b]:first[b + 1]]`, in segment order."""
    side = grid // _BUCKET_CELLS
    lo = (np.minimum(starts, ends) - _REACH_M - _ORIGIN_M) / _BUCKET_CELLS
    hi = (np.maximum(starts, ends) + _REACH_M - _ORIGIN_M) / _BUCKET_CELLS
    b0 = np.clip(np.floor(lo).astype(np.int64), 0, side - 1)
    b1 = np.clip(np.floor(hi).astype(np.int64), 0, side - 1)
    nu = b1[:, 0] - b0[:, 0] + 1
    nv = b1[:, 1] - b0[:, 1] + 1
    counts = nu * nv
    segment = np.repeat(np.arange(starts.shape[0]), counts)
    local = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
    du, dv = np.divmod(local, np.repeat(nv, counts))
    bucket = (np.repeat(b0[:, 0], counts) + du) * side + np.repeat(b0[:, 1], counts) + dv
    order = np.argsort(bucket, kind="stable")
    first = np.zeros(side * side + 1, np.int32)
    np.cumsum(np.bincount(bucket, minlength=side * side), out=first[1:])
    return first, segment[order].astype(np.int32)


def _nearest_reference(points: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Brute-force distance to the nearest of all segments, infinite beyond
    the reach."""
    out = np.empty(points.shape[0])
    ab = ends - starts
    length2 = np.maximum(np.einsum("ij,ij->i", ab, ab), _TINY)
    for at in range(0, points.shape[0], 256):
        p = points[at : at + 256, None, :]
        t = np.clip(np.einsum("pij,ij->pi", p - starts, ab) / length2, 0.0, 1.0)
        gap = p - starts - ab * t[..., None]
        out[at : at + 256] = np.sqrt(np.einsum("pij,pij->pi", gap, gap)).min(axis=1)
    out[out > _REACH_M] = np.inf
    return out


def nearest(report: Report, rng: np.random.Generator, grid: int, rings: int, repeats: int) -> None:
    points = _cells(grid)
    n = points.shape[0]
    bucket_of = np.repeat(np.arange((grid // _BUCKET_CELLS) ** 2, dtype=np.int32), _BUCKET_CELLS**2)
    starts, ends = _segments(rng, grid, rings)
    first, pool = _candidates(starts, ends, grid)
    put = vd.storage.from_numpy
    views = [put(a) for a in (points, bucket_of, starts, ends, first, pool)]
    dsl_distance = vd.storage.zeros(dtype=vd.f64, shape=(n,))
    dsl_segment = vd.storage.zeros(dtype=vd.i32, shape=(n,))
    numba_inputs = [_numba_put(a) for a in (points, bucket_of, starts, ends, first, pool)]
    numba_distance = _numba_empty(n)
    numba_segment = _numba_empty(n, np.int32)
    scalars = (np.uint32(n), np.float64(_REACH_M), np.float64(_FAR), np.float64(_TINY))

    def dsl_resident():
        dk.nearest(*views, dsl_distance, dsl_segment, *scalars, grid=_groups(n))
        _dsl_wait(dsl_distance, dsl_segment)

    def dsl_roundtrip():
        distance = vd.storage.zeros(dtype=vd.f64, shape=(n,))
        segment = vd.storage.zeros(dtype=vd.i32, shape=(n,))
        inputs = [put(a) for a in (points, bucket_of, starts, ends, first, pool)]
        dk.nearest(*inputs, distance, segment, *scalars, grid=_groups(n))
        return distance.to_numpy(), segment.to_numpy()

    def numba_resident():
        nk.nearest(*numba_inputs, numba_distance, numba_segment, _REACH_M, _FAR, _TINY)
        _numba_wait()

    def numba_roundtrip():
        distance = _numba_empty(n)
        segment = _numba_empty(n, np.int32)
        inputs = [_numba_put(a) for a in (points, bucket_of, starts, ends, first, pool)]
        nk.nearest(*inputs, distance, segment, _REACH_M, _FAR, _TINY)
        return _numba_get(distance), _numba_get(segment)

    _measure(
        report,
        "nearest_segment",
        n,
        {
            ("numba", "resident"): numba_resident,
            ("vernon", "resident"): dsl_resident,
            ("numba", "roundtrip"): numba_roundtrip,
            ("vernon", "roundtrip"): dsl_roundtrip,
        },
        repeats,
        {"vernon": f"{starts.shape[0]} segments, {pool.shape[0] / first.shape[0]:.1f} candidates per bucket"},
    )
    subset = rng.choice(n, size=min(n, 4096), replace=False)
    reference = _nearest_reference(points[subset], starts, ends)
    dsl = dsl_distance.to_numpy()
    dsl[dsl >= _FAR] = np.inf
    other = _numba_get(numba_distance)
    other[other >= _FAR] = np.inf
    _check(report, "nearest_segment", dsl[subset], other[subset], reference)
    # Rounding differences may flip the choice between equidistant segments.
    chose = dsl_segment.to_numpy() != _numba_get(numba_segment)
    if np.any(chose & ~np.isclose(dsl, other, rtol=0.0, atol=1e-9, equal_nan=True)):
        raise SystemExit("nearest_segment: the implementations chose segments at different distances")


def _revision() -> str:
    root = Path(__file__).resolve().parents[2]
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True, text=True)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True)
    except OSError:
        return "unknown"
    return head.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")


def _gpu() -> dict:
    from importlib.metadata import version

    query = "--query-gpu=name,driver_version,driver_model.current"
    try:
        smi = subprocess.run(["nvidia-smi", query, "--format=csv,noheader"], capture_output=True, text=True)
        name, driver, model = (part.strip() for part in smi.stdout.splitlines()[0].split(","))
    except (OSError, IndexError, ValueError):
        name, driver, model = cuda.get_current_device().name.decode(), "unknown", "unknown"
    return {"gpu": name, "gpu_driver": driver, "gpu_driver_model": model, "numba_cuda": version("numba-cuda")}


def _environment() -> dict:
    import llvmlite

    environment = {
        "vernon_dsl_revision": _revision(),
        "arch": ARGUMENTS.arch,
        "command": " ".join([Path(sys.executable).name, *sys.argv]),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "threads": ARGUMENTS.threads,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "numba": numba.__version__,
        "llvmlite": llvmlite.__version__,
    }
    if ON_GPU:
        environment.update(_gpu(), numba_fma=ARGUMENTS.numba_fma)
    else:
        environment["numba_cpu"] = numba.config.CPU_NAME or "host"
        environment["numba_threading_layer"] = numba.threading_layer()
    return environment


def _print(report: Report) -> None:
    env = report.environment
    if ON_GPU:
        target = f"numba-cuda {env['numba_cuda']} (fma {env['numba_fma']}), {env['gpu']}"
        target += f" ({env['gpu_driver_model']} {env['gpu_driver']})"
    else:
        target = f"(cpu {env['numba_cpu']}, {env['numba_threading_layer']}), {env['threads']} threads"
        target += f", {env['processor']}"
    print(f"VernonDSL {env['vernon_dsl_revision']} vs numba {env['numba']} {target}")
    by_key = {(s.workload, s.implementation, s.mode): s for s in report.samples}
    print(f"{'workload':<16} {'mode':<10} {'numba ms':>10} {'vernon ms':>10} {'vernon/numba':>13}")
    for (workload, implementation, mode), sample in by_key.items():
        if implementation != "vernon":
            continue
        other = by_key[(workload, "numba", mode)]
        print(
            f"{workload:<16} {mode:<10} {other.median_ms:>10.3f} {sample.median_ms:>10.3f} "
            f"{sample.median_ms / other.median_ms:>12.2f}x"
        )
    print("first call (compile + run), ms:")
    for sample in report.samples:
        if sample.first_call_ms is not None:
            print(f"  {sample.workload:<16} {sample.implementation:<14} {sample.first_call_ms:>10.1f}")
    print("other rows, median ms:")
    for sample in report.samples:
        if sample.implementation not in ("numba", "vernon"):
            note = f"  ({sample.note})" if sample.note else ""
            print(
                f"  {sample.workload:<16} {sample.implementation:<20} {sample.mode:<10} {sample.median_ms:>9.3f}{note}"
            )
    print("checks:")
    for check in report.checks:
        print(
            f"  {check.workload:<16} identical={check.identical} max|vernon-numba|={check.max_abs_difference:.3g} "
            f"max|vernon-numpy|={check.reference_max_abs_difference:.3g}"
        )


def main() -> None:
    vd.init(arch=vd.cuda if ON_GPU else vd.cpu)
    rng = np.random.default_rng(ARGUMENTS.seed)
    report = Report({})
    axpy(report, rng, ARGUMENTS.axpy_size, ARGUMENTS.repeats)
    launch(report, ARGUMENTS.launch_repeats)
    relief(report, rng, ARGUMENTS.grid, ARGUMENTS.repeats)
    nearest(report, rng, ARGUMENTS.grid, ARGUMENTS.rings, ARGUMENTS.repeats)
    # numba names its threading layer only once a parallel function has run.
    report.environment = _environment()
    _print(report)
    if ARGUMENTS.result_json is not None:
        ARGUMENTS.result_json.parent.mkdir(parents=True, exist_ok=True)
        ARGUMENTS.result_json.write_text(json.dumps(asdict(report), indent=2) + "\n")


if __name__ == "__main__":
    main()
