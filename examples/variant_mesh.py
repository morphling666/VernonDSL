from vernon_dsl import *

INSTANCE = feature("INSTANCE")
SKIN = feature("SKIN")


@vertex
def mesh_vertex(
    position: vec3[f32],
    instance_transform: When[INSTANCE, Annotated[mat4[f32],
                                                 instance()]],
    joints: When[SKIN, vec4[u32]],
    weights: When[SKIN, vec4[f32]],
) -> Annotated[vec4[f32], builtin("position")]:
    result = vec4(position, 1.0)
    if INSTANCE:
        result = matmul(instance_transform, result)
    if SKIN:
        result = result + weights
    return result


@fragment
def mesh_fragment() -> Annotated[vec4[f32], location(0)]:
    return vec4(1.0, 1.0, 1.0, 1.0)
