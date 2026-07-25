from typing import Annotated

import vernon_dsl as vd

INSTANCE = vd.feature("INSTANCE")
SKIN = vd.feature("SKIN")


@vd.vertex
def mesh_vertex(
    position: vd.Vector[vd.f32, 3],
    instance_transform: vd.When[INSTANCE, Annotated[vd.Matrix[vd.f32, 4, 4], vd.instance()]],
    joints: vd.When[SKIN, vd.Vector[vd.u32, 4]],
    weights: vd.When[SKIN, vd.Vector[vd.f32, 4]],
) -> Annotated[vd.Vector[vd.f32, 4], vd.builtin("position")]:
    result = vd.Vector([position, 1.0])
    if INSTANCE:
        result = vd.matmul(instance_transform, result)
    if SKIN:
        result = result + weights
    return result


@vd.fragment
def mesh_fragment(
    tint: Annotated[vd.Vector[vd.f32, 4], vd.uniform()],
) -> Annotated[vd.Vector[vd.f32, 4], vd.location(0)]:
    return tint


mesh_asset = vd.pipeline_asset(
    id="shaders/variant_mesh",
    program=(mesh_vertex, mesh_fragment),
    variants=((), (INSTANCE,), (SKIN,), (INSTANCE, SKIN)),
)
