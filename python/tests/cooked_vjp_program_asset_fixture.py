from __future__ import annotations

import vernon_dsl as vd


@vd.kernel(workgroup_size=(1, 1, 1))
def square(
    source: vd.TensorView[vd.f32, (1,), vd.read],
    output: vd.TensorView[vd.f32, (1,), vd.write],
) -> None:
    output[0] = source[0] * source[0]


square_vjp_asset = vd.program_asset(
    id="tests/cooked_square_vjp",
    program=vd.ad.vjp(square, wrt=("source",), outputs=("output",)),
)
