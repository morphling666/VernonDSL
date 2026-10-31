"""numba-cuda twins of `dsl_kernels.py` and `numba_kernels.relief_hashed`.

Each launcher has the signature of its `numba_kernels` namesake but takes
device arrays and returns without waiting. Launches use VernonDSL's workgroup
size, one thread per output, and `fastmath` stays off; `contract_fma` decides
whether multiply-adds fuse, which VernonDSL's CUDA target does not do.
"""

from __future__ import annotations

import math

import numba
import numpy as np
from dsl_kernels import HILL_COARSE_M, HILL_FINE_M, SWELL_COARSE_M, SWELL_FINE_M, WORKGROUP
from numba import cuda
from numba.cuda.cudadrv import nvvm

_GOLDEN_X = np.uint64(0x9E3779B97F4A7C15)
_GOLDEN_Y = np.uint64(0xC2B2AE3D27D4EB4F)
_MIX_A = np.uint64(0xBF58476D1CE4E5B9)
_MIX_B = np.uint64(0x94D049BB133111EB)
_U30 = np.uint64(30)
_U27 = np.uint64(27)
_U31 = np.uint64(31)
_U11 = np.uint64(11)
_UNIT = 1.0 / 9007199254740992.0


def contract_fma(enabled: bool) -> None:
    """Sets whether `a * b + c` fuses into FMA; call before the first launch.

    numba-cuda passes NVVM no `fma` option, so NVVM fuses, and so does
    nvJitLink when kernels link with LTO, as they do by default. Disabling
    passes NVVM `fma=0` and drops LTO, whose `fma` option cuda-core spells in a
    form NVVM rejects; each kernel inlines all its device functions, so LTO has
    nothing to optimize across."""
    if enabled:
        return
    compile_ir = nvvm.compile_ir

    def compile_without_fma(llvmir, **options):
        return compile_ir(llvmir, **options, fma=False)

    nvvm.compile_ir = compile_without_fma
    for kernel in (_axpy, _relief, _relief_hashed, _nearest):
        kernel.targetoptions["lto"] = False


def _blocks(count: int) -> int:
    return (count + WORKGROUP - 1) // WORKGROUP


@cuda.jit
def _axpy(x, y, out, a):
    i = cuda.grid(1)
    if i < x.shape[0]:
        out[i] = a * x[i] + y[i]


def axpy(x, y, out, a):
    _axpy[_blocks(x.shape[0]), WORKGROUP](x, y, out, a)


@cuda.jit(device=True, inline=True)
def _ease(t):
    return t * t * (3.0 - 2.0 * t)


@cuda.jit(device=True, inline=True)
def _value(table, lattice, row, x, y):
    fx = numba.float64(math.floor(x))
    fy = numba.float64(math.floor(y))
    sx = _ease(x - fx)
    sy = _ease(y - fy)
    span_y = lattice[row, 3]
    at = lattice[row, 0] + (numba.int32(fx) - lattice[row, 1]) * span_y + (numba.int32(fy) - lattice[row, 2])
    a = table[at]
    b = table[at + span_y]
    c = table[at + 1]
    d = table[at + span_y + 1]
    bottom = a + sx * (b - a)
    top = c + sx * (d - c)
    return bottom + sy * (top - bottom)


@cuda.jit(device=True, inline=True)
def _octaves(table, lattice, row, x, y, coarse, fine):
    return 0.7 * _value(table, lattice, row, x / coarse, y / coarse) + 0.3 * _value(
        table, lattice, row + 1, x / fine, y / fine
    )


@cuda.jit
def _relief(xs, ys, table, lattice, out):
    i = cuda.grid(1)
    if i < xs.shape[0]:
        x = xs[i]
        y = ys[i]
        hill = _octaves(table, lattice, 0, x, y, HILL_COARSE_M, HILL_FINE_M)
        swell = _octaves(table, lattice, 2, x, y, SWELL_COARSE_M, SWELL_FINE_M)
        out[i] = 12.0 * (hill - 0.5) + 2.0 * (swell - 0.5)


def relief(xs, ys, table, lattice, out):
    _relief[_blocks(xs.shape[0]), WORKGROUP](xs, ys, table, lattice, out)


@cuda.jit(device=True, inline=True)
def _lattice_value(seed, ix, iy):
    z = seed ^ (numba.uint64(ix) * _GOLDEN_X) ^ (numba.uint64(iy) * _GOLDEN_Y)
    z = (z ^ (z >> _U30)) * _MIX_A
    z = (z ^ (z >> _U27)) * _MIX_B
    z = z ^ (z >> _U31)
    return numba.float64(z >> _U11) * _UNIT


@cuda.jit(device=True, inline=True)
def _hashed_value(seed, x, y):
    fx = numba.float64(math.floor(x))
    fy = numba.float64(math.floor(y))
    sx = _ease(x - fx)
    sy = _ease(y - fy)
    ix = numba.int64(fx)
    iy = numba.int64(fy)
    a = _lattice_value(seed, ix, iy)
    b = _lattice_value(seed, ix + 1, iy)
    c = _lattice_value(seed, ix, iy + 1)
    d = _lattice_value(seed, ix + 1, iy + 1)
    bottom = a + sx * (b - a)
    top = c + sx * (d - c)
    return bottom + sy * (top - bottom)


@cuda.jit(device=True, inline=True)
def _hashed_octaves(seeds, row, x, y, coarse, fine):
    return 0.7 * _hashed_value(seeds[row], x / coarse, y / coarse) + 0.3 * _hashed_value(
        seeds[row + 1], x / fine, y / fine
    )


@cuda.jit
def _relief_hashed(xs, ys, seeds, out):
    i = cuda.grid(1)
    if i < xs.shape[0]:
        x = xs[i]
        y = ys[i]
        hill = _hashed_octaves(seeds, 0, x, y, HILL_COARSE_M, HILL_FINE_M)
        swell = _hashed_octaves(seeds, 2, x, y, SWELL_COARSE_M, SWELL_FINE_M)
        out[i] = 12.0 * (hill - 0.5) + 2.0 * (swell - 0.5)


def relief_hashed(xs, ys, seeds, out):
    _relief_hashed[_blocks(xs.shape[0]), WORKGROUP](xs, ys, seeds, out)


@cuda.jit(device=True, inline=True)
def _along(px, py, ax, ay, bx, by, tiny):
    abx = bx - ax
    aby = by - ay
    return min(max(((px - ax) * abx + (py - ay) * aby) / max(abx * abx + aby * aby, tiny), 0.0), 1.0)


@cuda.jit(device=True, inline=True)
def _gap(px, py, ax, ay, bx, by, t):
    gx = px - ax - (bx - ax) * t
    gy = py - ay - (by - ay) * t
    return math.sqrt(gx * gx + gy * gy)


@cuda.jit
def _nearest(points, bucket_of, starts, ends, first, pool, distance, segment, reach, far, tiny):
    i = cuda.grid(1)
    if i >= points.shape[0]:
        return
    px = points[i, 0]
    py = points[i, 1]
    b = bucket_of[i]
    best = far
    found = -1
    for k in range(first[b], first[b + 1]):
        s = pool[k]
        ax = starts[s, 0]
        ay = starts[s, 1]
        bx = ends[s, 0]
        by = ends[s, 1]
        d = _gap(px, py, ax, ay, bx, by, _along(px, py, ax, ay, bx, by, tiny))
        if d < best:
            best = d
            found = s
    if best > reach:
        best = far
        found = -1
    distance[i] = best
    segment[i] = found


def nearest(points, bucket_of, starts, ends, first, pool, distance, segment, reach, far, tiny):
    _nearest[_blocks(points.shape[0]), WORKGROUP](
        points, bucket_of, starts, ends, first, pool, distance, segment, reach, far, tiny
    )
