from typing import Annotated

import vernon_dsl as vd


@vd.kernel(workgroup_size=(96, 1, 1))
def objective(
    scale: vd.f32,
    values: vd.TensorView[vd.f32, (192,), vd.read],
    carried_loss: vd.TensorView[vd.f32, (192,), vd.write],
    shared_loss: vd.TensorView[vd.f32, (192,), vd.write],
    gid: Annotated[
        vd.Tensor[vd.u32, (3,)],
        vd.builtin("global_invocation_id"),
    ],
) -> None:
    carried_loss[gid[0]] = scale * values[gid[0]]
    shared_loss[gid[0]] = scale * values[gid[0]] * values[gid[0]]


asset = vd.program_asset(
    id="compute/gpu_non_power_of_two_vjp",
    program=vd.ad.vjp(
        objective,
        wrt=("scale",),
        outputs=("carried_loss", "shared_loss"),
    ),
)
