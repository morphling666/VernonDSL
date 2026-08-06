from typing import Annotated

import vernon_dsl as vd


@vd.kernel(workgroup_size=(4, 1, 1))
def gather(
    values: vd.TensorView[vd.f32, (3,), vd.read],
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> vd.f32:
    return values[gid[0]]


asset = vd.pipeline_asset(
    id="compute/scatter-vjp",
    program=vd.ad.vjp(gather, wrt=("values",), protocol="legacy_fixed"),
)
