from typing import Annotated

import vernon_dsl as vd


@vd.kernel(workgroup_size=(2, 2, 1))
def objective(
    values: vd.TensorView[vd.f32, (2, 2), vd.read],
    loss: vd.TensorView[vd.f32, (2, 2), vd.write],
    gid: Annotated[
        vd.Tensor[vd.u32, (3,)],
        vd.builtin("global_invocation_id"),
    ],
) -> None:
    loss[gid[1], gid[0]] = values[gid[1], gid[0]] * values[gid[1], gid[0]]


asset = vd.pipeline_asset(
    id="compute/gpu_no_tape_vjp",
    program=vd.ad.vjp(
        objective,
        wrt=("values",),
        outputs=("loss",),
    ),
)
