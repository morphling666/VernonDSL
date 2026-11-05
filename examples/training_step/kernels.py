from typing import Annotated

import vernon_dsl as vd


@vd.kernel(workgroup_size=(64, 1, 1))
def mul(
    source: vd.TensorView[vd.f32, (vd.dyn,), vd.read],
    weight: vd.TensorView[vd.f32, (vd.dyn,), vd.read],
    output: vd.TensorView[vd.f32, (vd.dyn,), vd.write],
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    i = gid[0]
    if i < source.shape[0]:
        output[i] = source[i] * weight[i]
