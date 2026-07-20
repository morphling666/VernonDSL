from __future__ import annotations

from typing import Annotated

from vernon_dsl import Tensor, builtin, f32, kernel, u32


@kernel(workgroup_size=(8, 1, 1))
def scale(
    output: Tensor[f32, (8, )],
    factor: f32,
    gid: Annotated[Tensor[u32, (3, )],
                   builtin("global_invocation_id")],
) -> None:
    output[gid[0]] = f32(gid[0]) * factor
