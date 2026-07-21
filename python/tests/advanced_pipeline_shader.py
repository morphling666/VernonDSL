from __future__ import annotations

from typing import Annotated

import vernon_dsl as vd

PICKING = vd.feature("PICKING")


@vd.struct
class VertexData:
    position: Annotated[vd.vec4[vd.f32], vd.builtin("position")]
    local_color: Annotated[vd.vec2[vd.f32], vd.location(0)]


@vd.struct
class GBuffer:
    color: Annotated[vd.vec4[vd.f32], vd.location(0)]
    object_id: Annotated[vd.vec4[vd.f32], vd.location(1)]


@vd.kernel(workgroup_size=(2, 1, 1))
def feature_compute(
    offset: vd.Tensor[vd.f32, (None, None)],
    gid: Annotated[vd.Tensor[vd.u32, (3, )],
                   vd.builtin("global_invocation_id")],
) -> None:
    if PICKING:
        offset[gid[1], gid[0]] = offset[gid[1], gid[0]]


@vd.vertex
def advanced_vertex(
    position: Annotated[vd.vec2[vd.f32], vd.location(0)],
    offset: Annotated[vd.vec2[vd.f32],
                      vd.instance(location=1)],
) -> VertexData:
    return VertexData(vd.vec4(position + offset, 0.0, 1.0),
                      position + vd.vec2(0.5, 0.5))


@vd.fragment
def advanced_fragment(
    local_color: Annotated[vd.vec2[vd.f32],
                           vd.varying(),
                           vd.location(0)],
) -> GBuffer:
    color = vd.vec4(local_color, 1.0, 1.0)
    object_id = vd.vec4(0.0, 0.0, 0.0, 1.0)
    if PICKING:
        object_id = vd.vec4(1.0, 0.25, 0.0, 1.0)
    return GBuffer(color, object_id)
