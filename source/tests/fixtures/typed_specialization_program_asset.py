from __future__ import annotations

import vernon_dsl as vd

EXTENT = vd.specialization("extent", vd.u32)


@vd.kernel
def fill(output: vd.TensorView[vd.f32, (EXTENT,), vd.write]) -> None:
    output[0] = vd.f32(EXTENT)


asset = vd.program_asset(
    id="compute/typed-specialization",
    program=fill,
    variants=({EXTENT: 1}, {EXTENT: 4}),
)
