from typing import Annotated

import vernon_dsl as vd


@vd.kernel(workgroup_size=(2, 1, 1))
def translate_vertices(
    position: vd.Tensor[vd.f32, (None, 2)],
    offset: vd.f32,
    gid: Annotated[vd.Tensor[vd.u32, (3, )],
                   vd.builtin("global_invocation_id")],
) -> None:
    component = gid[0]
    vertex = gid[1]
    if component == 0:
        position[vertex, component] = position[vertex, component] + offset


@vd.vertex
def triangle_vertex(
    position: Annotated[vd.vec2[vd.f32], vd.location(0)],
) -> Annotated[vd.vec4[vd.f32], vd.builtin("position")]:
    return vd.vec4(position, 0.0, 1.0)


@vd.vertex
def translated_vertex(
    position: Annotated[vd.vec2[vd.f32], vd.location(0)],
    offset: Annotated[vd.vec2[vd.f32], vd.uniform()],
) -> Annotated[vd.vec4[vd.f32], vd.builtin("position")]:
    return vd.vec4(position + offset, 0.0, 1.0)


@vd.fragment
def solid_fragment() -> Annotated[vd.vec4[vd.f32], vd.location(0)]:
    return vd.vec4(1.0, 0.25, 0.0, 1.0)
