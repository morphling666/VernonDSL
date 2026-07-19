from vernon_dsl import *


@vertex
def static_vertex(
    position: Annotated[vec3[f32], location(0)],
    model_view_projection: Annotated[mat4[f32],
                                     uniform(set=1, binding=0)],
) -> vec4[f32]:
    return matmul(model_view_projection, vec4(position, 1.0))


@vertex
def instanced_vertex(
    position: Annotated[vec3[f32], location(0)],
    instance_offset: Annotated[vec3[f32], instance(location=4)],
    instance_scale: Annotated[vec3[f32], instance(location=5)],
    custom_tint: Annotated[vec4[f32], instance(location=6)],
    view_projection: Annotated[mat4[f32], uniform(set=1, binding=0)],
) -> vec4[f32]:
    world_position = position * instance_scale + instance_offset
    return matmul(view_projection, vec4(world_position, 1.0))


@vertex
def skinned_vertex(
    position: Annotated[vec3[f32], location(0)],
    weights: Annotated[vec2[f32], location(3)],
    bone0: Annotated[mat4[f32], uniform(set=2, binding=0)],
    bone1: Annotated[mat4[f32], uniform(set=2, binding=1)],
    view_projection: Annotated[mat4[f32], uniform(set=1, binding=0)],
) -> vec4[f32]:
    local_position = vec4(position, 1.0)
    transformed0 = matmul(bone0, local_position)
    transformed1 = matmul(bone1, local_position)
    weight0 = vec4(weights.x, weights.x, weights.x, weights.x)
    weight1 = vec4(weights.y, weights.y, weights.y, weights.y)
    skinned_position = transformed0 * weight0 + transformed1 * weight1
    return matmul(view_projection, skinned_position)
