from __future__ import annotations

from typing import Annotated

import vernon_dsl as vd

MODE = vd.specialization("MODE", vd.u32)
TINTED = vd.feature("TINTED")


@vd.vertex
def mode_vertex(
    position: Annotated[vd.Vector[vd.f32, 2], vd.attribute()],
) -> Annotated[vd.Vector[vd.f32, 4], vd.builtin("position")]:
    return vd.Vector([position, 0.0, 1.0])


@vd.fragment
def mode_fragment(
    tint: vd.When[TINTED and MODE != 0, Annotated[vd.Vector[vd.f32, 4], vd.uniform()]],
) -> vd.Vector[vd.f32, 4]:
    if MODE == 0:
        color = vd.Vector([1.0, 0.0, 0.0, 1.0])
    elif MODE == 1:
        color = vd.Vector([0.0, 1.0, 0.0, 1.0])
    else:
        color = vd.Vector([0.0, 0.0, 1.0, 1.0])
    if TINTED and MODE != 0:
        color = tint
    return color


@vd.kernel(workgroup_size=(1, 1, 1))
def mode_kernel(
    output: vd.TensorView[vd.f32, (1,), vd.write],
    scale: vd.When[MODE >= 2, vd.f32],
) -> None:
    if MODE >= 2:
        output[0] = vd.f32(MODE) * scale
    else:
        output[0] = vd.f32(MODE) - 10.0
