from typing import Annotated

import vernon_dsl as vd


@vd.kernel(workgroup_size=(2, 1, 1))
def translate_vertices(
    position: vd.TensorView[vd.f32, 2, vd.read_write],
    offset: vd.f32,
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    component = gid[0]
    vertex = gid[1]
    if component == 0:
        position[vertex, component] = position[vertex, component] + offset


@vd.vertex
def triangle_vertex(
    position: Annotated[vd.Vector[vd.f32, 2], vd.location(0)],
) -> Annotated[vd.Vector[vd.f32, 4], vd.builtin("position")]:
    return vd.Vector([position, 0.0, 1.0])


@vd.vertex
def translated_vertex(
    position: Annotated[vd.Vector[vd.f32, 2], vd.location(0)],
    offset: Annotated[vd.Vector[vd.f32, 2], vd.uniform()],
) -> Annotated[vd.Vector[vd.f32, 4], vd.builtin("position")]:
    return vd.Vector([position + offset, 0.0, 1.0])


@vd.fragment
def solid_fragment() -> Annotated[vd.Vector[vd.f32, 4], vd.location(0)]:
    return vd.Vector([1.0, 0.25, 0.0, 1.0])


@vd.fragment
def colored_fragment(
    color: Annotated[vd.Vector[vd.f32, 4], vd.uniform()],
) -> Annotated[vd.Vector[vd.f32, 4], vd.location(0)]:
    return color
