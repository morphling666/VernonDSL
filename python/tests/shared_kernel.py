from __future__ import annotations

from typing import Annotated

import vernon_dsl as vd


@vd.func(shared=True)
def shared_polynomial(value: vd.f32) -> vd.f32:
    return value * value + vd.f32(1.0)


@vd.kernel(workgroup_size=(1, 1, 1))
def evaluate_shared(
    output: vd.TensorView[vd.f32, (vd.dyn,), vd.write],
    value: vd.f32,
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    output[gid[0]] = shared_polynomial(value + vd.f32(gid[0]))
