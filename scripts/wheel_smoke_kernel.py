from __future__ import annotations

from typing import Annotated

import vernon_dsl as vd


@vd.kernel(workgroup_size=(8, 1, 1))
def add_one(
    output: vd.TensorView[vd.f32, (vd.dyn,), vd.write],
    values: vd.TensorView[vd.f32, (vd.dyn,), vd.read],
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    index = gid[0]
    output[index] = values[index] + 1.0
