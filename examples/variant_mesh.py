from typing import Annotated

import vernon_dsl as vd

INSTANCE = vd.feature("INSTANCE")
SKIN = vd.feature("SKIN")


@vd.vertex
def mesh_vertex(
    position: vd.vec3[vd.f32],
    instance_transform: vd.When[INSTANCE, Annotated[vd.mat4[vd.f32],
                                                    vd.instance()]],
    joints: vd.When[SKIN, vd.vec4[vd.u32]],
    weights: vd.When[SKIN, vd.vec4[vd.f32]],
) -> Annotated[vd.vec4[vd.f32], vd.builtin("position")]:
    result = vd.vec4(position, 1.0)
    if INSTANCE:
        result = vd.matmul(instance_transform, result)
    if SKIN:
        result = result + weights
    return result


@vd.fragment
def mesh_fragment(
    tint: Annotated[vd.vec4[vd.f32], vd.uniform()],
) -> Annotated[vd.vec4[vd.f32], vd.location(0)]:
    return tint


mesh_asset = vd.pipeline_asset(
    id="shaders/variant_mesh",
    vertex=mesh_vertex,
    fragment=mesh_fragment,
    variants=((), (INSTANCE, ), (SKIN, ), (INSTANCE, SKIN)),
    targets={
        "opengl": {
            "glsl_version": 330
        },
        "vulkan": {},
    },
)
