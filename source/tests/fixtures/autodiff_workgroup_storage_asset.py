from typing import Annotated

import vernon_dsl as vd


@vd.kernel(workgroup_size=(4, 1, 1))
def workgroup_objective(
    carriers: vd.TensorView[vd.f32, (8,), vd.read],
    scale: vd.f32,
    output: vd.TensorView[vd.f32, (vd.dyn, vd.dyn, vd.dyn), vd.write],
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
    lane_id: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("local_invocation_id")],
) -> None:
    first = vd.workgroup_storage(vd.f32, shape=(4,))
    second = vd.workgroup_storage(vd.f32, shape=(4,))
    lane = vd.i32(lane_id[0])
    global_index = vd.i32(gid[0])
    first[lane] = carriers[global_index] * scale
    vd.workgroup_barrier()
    neighbor = lane + 1
    if lane == 3:
        neighbor = 0
    second[lane] = first[neighbor] * 2.0
    vd.workgroup_barrier()
    output[gid[2], gid[1], gid[0]] = second[lane]


asset = vd.pipeline_asset(
    id="compute/structured_storage_workgroup_vjp",
    program=vd.ad.vjp(
        workgroup_objective,
        wrt=("carriers", "scale"),
        outputs=("output",),
    ),
)
