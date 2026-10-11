from __future__ import annotations

from typing import Annotated

import vernon_dsl as vd

MATERIAL = vd.specialization("MATERIAL", vd.type)
ELEMENT = vd.specialization("ELEMENT", vd.type)
SURFACE = vd.specialization("SURFACE", vd.type)


@vd.kernel(workgroup_size=(1, 1, 1))
def shade(
    material: MATERIAL,
    output: vd.TensorView[vd.f32, (vd.dyn,), vd.write],
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    x = gid[0]
    output[x] = material.term(x)


@vd.kernel(workgroup_size=(1, 1, 1))
def convert(
    values: vd.TensorView[vd.f32, (vd.dyn,), vd.read],
    output: vd.TensorView[ELEMENT, (vd.dyn,), vd.write],
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    x = gid[0]
    output[x] = ELEMENT(values[x] * 2.0)


@vd.vertex
def surface_vertex(
    position: Annotated[vd.Vector[vd.f32, 2], vd.attribute()],
) -> Annotated[vd.Vector[vd.f32, 4], vd.builtin("position")]:
    return vd.Vector([position, 0.0, 1.0])


@vd.fragment
def surface_fragment(surface: SURFACE) -> vd.Vector[vd.f32, 4]:
    return surface.shade()
