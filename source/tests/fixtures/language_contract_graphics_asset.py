from __future__ import annotations

from typing import Annotated

import vernon_dsl as vd  # pyright: ignore[reportMissingImports]

DOUBLE = vd.feature("DOUBLE")


@vd.func
def specialize_value(value):
    if DOUBLE:
        return value * 2.0
    return value


@vd.vertex
def raw_contract_vertex(
    vertex_index: Annotated[vd.u32, vd.builtin("vertex_index")],
    instance_index: Annotated[vd.u32, vd.builtin("instance_index")],
) -> Annotated[vd.Vector[vd.f32, 4], vd.builtin("position")]:
    x = -0.75 if vertex_index == 0 else (0.75 if vertex_index == 1 else 0.0)
    y = -0.75 if vertex_index < 2 else 0.75
    return vd.Vector([x if instance_index == 0 else 2.0, y, 0.0, 1.0])


@vd.fragment
def raw_contract_fragment(
    coordinate: Annotated[vd.Vector[vd.f32, 4], vd.builtin("frag_coord")],
    front_facing: Annotated[bool, vd.builtin("front_facing")],
) -> vd.Vector[vd.f32, 4]:
    valid = front_facing and coordinate.x >= 0.0 and coordinate.y >= 0.0
    return vd.Vector([1.0 if valid else 0.0, 0.25, 0.0, 1.0])


@vd.vertex
def generated_contract_vertex() -> Annotated[vd.Vector[vd.f32, 4], vd.builtin("position")]:
    vertex_index = vd.vertex_id()
    instance_index = vd.instance_id()
    x = -0.75 if vertex_index == 0 else (0.75 if vertex_index == 1 else 0.0)
    y = -0.75 if vertex_index < 2 else 0.75
    return vd.Vector([x if instance_index == 0 else 2.0, y, 0.0, 1.0])


@vd.fragment
def generated_contract_fragment() -> vd.Vector[vd.f32, 4]:
    coordinate = vd.fragment_coord()
    front_facing = vd.front_facing()
    size = vd.resolution()
    valid = front_facing and coordinate.x >= 0.0 and size.x >= 1.0 and size.y >= 1.0
    return vd.Vector([1.0 if valid else 0.0, 0.25, 0.0, 1.0])


@vd.fragment
def specialization_fragment() -> vd.Vector[vd.f32, 4]:
    return vd.Vector([specialize_value(0.25), 0.0, 0.0, 1.0])


raw_asset = vd.program_asset(
    id="runtime/language-contract-graphics-raw",
    program=vd.pipeline(
        raw_contract_vertex,
        raw_contract_fragment,
        targets=vd.target_formats(colors={0: vd.rgba8_unorm}),
    ),
)

generated_asset = vd.program_asset(
    id="runtime/language-contract-graphics-generated",
    program=vd.pipeline(
        generated_contract_vertex,
        generated_contract_fragment,
        targets=vd.target_formats(colors={0: vd.rgba8_unorm}),
    ),
)

specialization_asset = vd.program_asset(
    id="runtime/language-contract-specialization",
    program=vd.pipeline(
        generated_contract_vertex,
        specialization_fragment,
        targets=vd.target_formats(colors={0: vd.rgba8_unorm}),
    ),
    variants=({}, {DOUBLE: True}),
)
