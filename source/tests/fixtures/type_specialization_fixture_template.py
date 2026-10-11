from __future__ import annotations

from typing import Annotated

import vernon_dsl as vd

MATERIAL = vd.specialization("MATERIAL", vd.type)


@vd.kernel(workgroup_size=(1, 1, 1))
def shade(
    material: MATERIAL,
    output: vd.TensorView[vd.f32, (vd.dyn,), vd.write],
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    x = gid[0]
    output[x] = material.term(x)
