from __future__ import annotations

from typing import Annotated

import vernon_dsl as vd  # pyright: ignore[reportMissingImports]


@vd.struct
class Pair:
    left: vd.i32
    right: vd.f32


@vd.kernel(workgroup_size=(4, 1, 1))
def aggregate_publication(
    output: vd.TensorView[vd.f32, (1,), vd.read_write],
    local_id: Annotated[vd.Vector[vd.u32, 3], vd.builtin("local_invocation_id")],
) -> None:
    values = vd.workgroup_storage(Pair, shape=(2, 3))
    if local_id[0] == 0:
        values[1, 2] = Pair(7, 2.5)
    vd.workgroup_barrier()
    if local_id[0] == 0:
        vd.atomic_add(output, 0, values[1, 2].right)


@vd.kernel(workgroup_size=(4, 1, 1))
def floating_contention(
    device: vd.TensorView[vd.f32, (2,), vd.read_write],
    local_id: Annotated[vd.Vector[vd.u32, 3], vd.builtin("local_invocation_id")],
) -> None:
    shared = vd.workgroup_storage(vd.f32, shape=(1,))
    if local_id[0] == 0:
        shared[0] = 0.0
    vd.workgroup_barrier()
    vd.atomic_add(shared, 0, 1.0)
    vd.workgroup_barrier()
    if local_id[0] == 0:
        vd.atomic_add(device, 1, shared[0])
    vd.atomic_add(device, 0, 1.0)


aggregate_asset = vd.program_asset(
    id="runtime/language-contract-aggregate-publication",
    program=aggregate_publication,
)

floating_asset = vd.program_asset(
    id="runtime/language-contract-floating-contention",
    program=floating_contention,
)
