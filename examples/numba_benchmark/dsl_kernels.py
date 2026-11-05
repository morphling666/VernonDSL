"""VernonDSL kernels of the numba comparison, launched on CPU or CUDA.

Each kernel has a twin of the same name in `numba_kernels.py` and
`numba_cuda_kernels.py` doing the same arithmetic in the same order.
"""

from __future__ import annotations

from typing import Annotated

import vernon_dsl as vd

WORKGROUP = 64
# Lattice spacings of the two noises' octaves, in meters; each is exact in f32,
# so passing it to an f64 parameter cannot round it.
HILL_COARSE_M = 320.0
HILL_FINE_M = 110.0
SWELL_COARSE_M = 60.0
SWELL_FINE_M = 21.0


@vd.kernel(workgroup_size=(WORKGROUP, 1, 1))
def axpy(
    x: vd.TensorView[vd.f64, (vd.dyn,), vd.read],
    y: vd.TensorView[vd.f64, (vd.dyn,), vd.read],
    out: vd.TensorView[vd.f64, (vd.dyn,), vd.write],
    a: vd.f64,
    count: vd.u32,
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    i = gid[0]
    if i < count:
        out[i] = a * x[i] + y[i]


@vd.func
def _ease(t: vd.f64) -> vd.f64:
    return t * t * (3.0 - 2.0 * t)


@vd.func
def _value(
    table: vd.TensorView[vd.f64, (vd.dyn,), vd.read],
    lattice: vd.TensorView[vd.i32, (vd.dyn, 4), vd.read],
    row: vd.i32,
    x: vd.f64,
    y: vd.f64,
) -> vd.f64:
    """Value noise in [0, 1] with lattice points one unit apart, its lattice
    values read from row `row` of the host-hashed table."""
    fx = vd.floor(x)
    fy = vd.floor(y)
    sx = _ease(x - fx)
    sy = _ease(y - fy)
    span_y = lattice[row, 3]
    at = lattice[row, 0] + (vd.i32(fx) - lattice[row, 1]) * span_y + (vd.i32(fy) - lattice[row, 2])
    a = table[at]
    b = table[at + span_y]
    c = table[at + 1]
    d = table[at + span_y + 1]
    bottom = a + sx * (b - a)
    top = c + sx * (d - c)
    return bottom + sy * (top - bottom)


@vd.func
def _octaves(
    table: vd.TensorView[vd.f64, (vd.dyn,), vd.read],
    lattice: vd.TensorView[vd.i32, (vd.dyn, 4), vd.read],
    row: vd.i32,
    x: vd.f64,
    y: vd.f64,
    coarse: vd.f64,
    fine: vd.f64,
) -> vd.f64:
    return 0.7 * _value(table, lattice, row, x / coarse, y / coarse) + 0.3 * _value(
        table, lattice, row + 1, x / fine, y / fine
    )


@vd.kernel(workgroup_size=(WORKGROUP, 1, 1))
def relief(
    xs: vd.TensorView[vd.f64, (vd.dyn,), vd.read],
    ys: vd.TensorView[vd.f64, (vd.dyn,), vd.read],
    table: vd.TensorView[vd.f64, (vd.dyn,), vd.read],
    lattice: vd.TensorView[vd.i32, (vd.dyn, 4), vd.read],
    out: vd.TensorView[vd.f64, (vd.dyn,), vd.write],
    count: vd.u32,
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    """Rolling hills and swells: two noises of two octaves each."""
    i = gid[0]
    if i < count:
        x = xs[i]
        y = ys[i]
        hill = _octaves(table, lattice, 0, x, y, HILL_COARSE_M, HILL_FINE_M)
        swell = _octaves(table, lattice, 2, x, y, SWELL_COARSE_M, SWELL_FINE_M)
        out[i] = 12.0 * (hill - 0.5) + 2.0 * (swell - 0.5)


@vd.func
def _along(px: vd.f64, py: vd.f64, ax: vd.f64, ay: vd.f64, bx: vd.f64, by: vd.f64, tiny: vd.f64) -> vd.f64:
    abx = bx - ax
    aby = by - ay
    return vd.clamp(((px - ax) * abx + (py - ay) * aby) / vd.max(abx * abx + aby * aby, tiny), 0.0, 1.0)


@vd.func
def _gap(px: vd.f64, py: vd.f64, ax: vd.f64, ay: vd.f64, bx: vd.f64, by: vd.f64, t: vd.f64) -> vd.f64:
    gx = px - ax - (bx - ax) * t
    gy = py - ay - (by - ay) * t
    return vd.sqrt(gx * gx + gy * gy)


@vd.kernel(workgroup_size=(WORKGROUP, 1, 1))
def nearest(
    points: vd.TensorView[vd.f64, (vd.dyn, 2), vd.read],
    bucket_of: vd.TensorView[vd.i32, (vd.dyn,), vd.read],
    starts: vd.TensorView[vd.f64, (vd.dyn, 2), vd.read],
    ends: vd.TensorView[vd.f64, (vd.dyn, 2), vd.read],
    first: vd.TensorView[vd.i32, (vd.dyn,), vd.read],
    pool: vd.TensorView[vd.i32, (vd.dyn,), vd.read],
    distance: vd.TensorView[vd.f64, (vd.dyn,), vd.write],
    segment: vd.TensorView[vd.i32, (vd.dyn,), vd.write],
    count: vd.u32,
    reach: vd.f64,
    far: vd.f64,
    tiny: vd.f64,
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    """Each point's nearest segment among its bucket's candidates, `pool`
    from `first[b]` to `first[b + 1]`, and its distance; `far` and -1 beyond
    `reach`."""
    i = gid[0]
    if i < count:
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
