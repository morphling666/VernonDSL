from typing import Annotated

import vernon_dsl as vd  # pyright: ignore[reportMissingImports]


@vd.kernel(workgroup_size=(4, 1, 1))
def synchronize(
    output: vd.TensorView[vd.i32, (10,), vd.write],
    lane_id: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("local_invocation_id")],
    group_id: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("workgroup_id")],
) -> None:
    shared = vd.workgroup_storage(vd.i32, shape=(1,))
    lane = lane_id[0]
    group = group_id[0]
    if lane == 0:
        shared[0] = vd.i32(group) * 100
    vd.workgroup_barrier()
    previous = vd.atomic_add(shared, 0, 1)
    vd.workgroup_barrier()
    if lane == 0:
        output[8 + group] = shared[0]
    output[group * 4 + lane] = previous


asset = vd.program_asset(
    id="runtime/synchronization",
    program=synchronize,
    variants=({},),
)
