"""numba twins of `dsl_kernels.py`, plus `relief_hashed`, which hashes the
noise lattice in the kernel with 64-bit integers, as VernonDSL kernels cannot.

All kernels write into caller-allocated outputs, like the VernonDSL ones.
`fastmath` is off and division follows numpy's error model, matching
VernonDSL's IEEE arithmetic without Python's zero-division checks.
"""

from __future__ import annotations

import numpy as np
from dsl_kernels import HILL_COARSE_M, HILL_FINE_M, SWELL_COARSE_M, SWELL_FINE_M
from numba import njit, prange

_FLAGS = {"fastmath": False, "error_model": "numpy", "cache": False}
_GOLDEN_X = np.uint64(0x9E3779B97F4A7C15)
_GOLDEN_Y = np.uint64(0xC2B2AE3D27D4EB4F)
_MIX_A = np.uint64(0xBF58476D1CE4E5B9)
_MIX_B = np.uint64(0x94D049BB133111EB)
_UNIT = 1.0 / 9007199254740992.0


@njit(parallel=True, **_FLAGS)
def axpy(x, y, out, a):
    for i in prange(x.shape[0]):
        out[i] = a * x[i] + y[i]


@njit(inline="always", **_FLAGS)
def _ease(t):
    return t * t * (3.0 - 2.0 * t)


@njit(inline="always", **_FLAGS)
def _value(table, lattice, row, x, y):
    fx = np.floor(x)
    fy = np.floor(y)
    sx = _ease(x - fx)
    sy = _ease(y - fy)
    span_y = lattice[row, 3]
    at = lattice[row, 0] + (np.int32(fx) - lattice[row, 1]) * span_y + (np.int32(fy) - lattice[row, 2])
    a = table[at]
    b = table[at + span_y]
    c = table[at + 1]
    d = table[at + span_y + 1]
    bottom = a + sx * (b - a)
    top = c + sx * (d - c)
    return bottom + sy * (top - bottom)


@njit(inline="always", **_FLAGS)
def _octaves(table, lattice, row, x, y, coarse, fine):
    return 0.7 * _value(table, lattice, row, x / coarse, y / coarse) + 0.3 * _value(
        table, lattice, row + 1, x / fine, y / fine
    )


@njit(parallel=True, **_FLAGS)
def relief(xs, ys, table, lattice, out):
    for i in prange(xs.shape[0]):
        x = xs[i]
        y = ys[i]
        hill = _octaves(table, lattice, 0, x, y, HILL_COARSE_M, HILL_FINE_M)
        swell = _octaves(table, lattice, 2, x, y, SWELL_COARSE_M, SWELL_FINE_M)
        out[i] = 12.0 * (hill - 0.5) + 2.0 * (swell - 0.5)


@njit(inline="always", **_FLAGS)
def lattice_value(seed, ix, iy):
    """SplitMix64's finalizer over a lattice point, as a float in [0, 1)."""
    z = seed ^ (np.uint64(ix) * _GOLDEN_X) ^ (np.uint64(iy) * _GOLDEN_Y)
    z = (z ^ (z >> np.uint64(30))) * _MIX_A
    z = (z ^ (z >> np.uint64(27))) * _MIX_B
    z = z ^ (z >> np.uint64(31))
    return np.float64(z >> np.uint64(11)) * _UNIT


@njit(inline="always", **_FLAGS)
def _hashed_value(seed, x, y):
    fx = np.floor(x)
    fy = np.floor(y)
    sx = _ease(x - fx)
    sy = _ease(y - fy)
    ix = np.int64(fx)
    iy = np.int64(fy)
    a = lattice_value(seed, ix, iy)
    b = lattice_value(seed, ix + 1, iy)
    c = lattice_value(seed, ix, iy + 1)
    d = lattice_value(seed, ix + 1, iy + 1)
    bottom = a + sx * (b - a)
    top = c + sx * (d - c)
    return bottom + sy * (top - bottom)


@njit(inline="always", **_FLAGS)
def _hashed_octaves(seeds, row, x, y, coarse, fine):
    return 0.7 * _hashed_value(seeds[row], x / coarse, y / coarse) + 0.3 * _hashed_value(
        seeds[row + 1], x / fine, y / fine
    )


@njit(parallel=True, **_FLAGS)
def relief_hashed(xs, ys, seeds, out):
    for i in prange(xs.shape[0]):
        x = xs[i]
        y = ys[i]
        hill = _hashed_octaves(seeds, 0, x, y, HILL_COARSE_M, HILL_FINE_M)
        swell = _hashed_octaves(seeds, 2, x, y, SWELL_COARSE_M, SWELL_FINE_M)
        out[i] = 12.0 * (hill - 0.5) + 2.0 * (swell - 0.5)


@njit(inline="always", **_FLAGS)
def _along(px, py, ax, ay, bx, by, tiny):
    abx = bx - ax
    aby = by - ay
    return min(max(((px - ax) * abx + (py - ay) * aby) / max(abx * abx + aby * aby, tiny), 0.0), 1.0)


@njit(inline="always", **_FLAGS)
def _gap(px, py, ax, ay, bx, by, t):
    gx = px - ax - (bx - ax) * t
    gy = py - ay - (by - ay) * t
    return np.sqrt(gx * gx + gy * gy)


@njit(parallel=True, **_FLAGS)
def nearest(points, bucket_of, starts, ends, first, pool, distance, segment, reach, far, tiny):
    for i in prange(points.shape[0]):
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
