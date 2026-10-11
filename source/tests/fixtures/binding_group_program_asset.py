from __future__ import annotations

from typing import Annotated

import vernon_dsl as vd

EXTENT = vd.specialization("extent", vd.u32)


@vd.binding_group
class Lookup:
    weights: vd.TensorView[vd.f32, (vd.dyn,), vd.read]


@vd.binding_group
class Terms:
    gain: vd.f32
    lookup: Lookup


@vd.kernel(workgroup_size=(1, 1, 1))
def scale(
    terms: Terms,
    output: vd.TensorView[vd.f32, (EXTENT,), vd.write],
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    x = gid[0]
    output[x] = terms.lookup.weights[x] * terms.gain + vd.f32(EXTENT)


asset = vd.program_asset(
    id="compute/binding-group",
    program=scale,
    variants=({EXTENT: 2}, {EXTENT: 4}),
)
