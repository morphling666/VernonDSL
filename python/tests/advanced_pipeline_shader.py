from __future__ import annotations

from typing import Annotated

import vernon_dsl as vd

PICKING = vd.feature("PICKING")


@vd.struct
class VertexData:
    position: Annotated[vd.Vector[vd.f32, 4], vd.builtin("position")]
    local_color: Annotated[vd.Vector[vd.f32, 2], vd.location(0)]


@vd.struct
class GBuffer:
    color: Annotated[vd.Vector[vd.f32, 4], vd.location(0)]
    object_id: Annotated[vd.Vector[vd.f32, 4], vd.location(1)]


@vd.kernel(workgroup_size=(2, 1, 1))
def feature_compute(
    offset: vd.TensorView[vd.f32, 2, vd.read_write],
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    if PICKING:
        offset[gid[1], gid[0]] = offset[gid[1], gid[0]]


@vd.vertex
def advanced_vertex(
    position: Annotated[vd.Vector[vd.f32, 2], vd.location(0)],
    offset: Annotated[vd.Vector[vd.f32, 2], vd.instance(location=1)],
) -> VertexData:
    return VertexData(vd.Vector([position + offset, 0.0, 1.0]), position + vd.Vector([0.5, 0.5]))


@vd.fragment
def advanced_fragment(
    local_color: Annotated[vd.Vector[vd.f32, 2], vd.varying(), vd.location(0)],
) -> GBuffer:
    color = vd.Vector([local_color, 1.0, 1.0])
    object_id = vd.Vector([0.0, 0.0, 0.0, 1.0])
    if PICKING:
        object_id = vd.Vector([1.0, 0.25, 0.0, 1.0])
    return GBuffer(color, object_id)
