from vernon_dsl import *

from examples.blinn_phong_types import BlinnPhongVertexData


@vertex
def static_vertex(
    position: Annotated[vec3[f32], location(0)],
    normal: Annotated[vec3[f32], location(1)],
    model_view_projection: Annotated[mat4[f32], uniform()],
) -> BlinnPhongVertexData:
    clip_position = matmul(model_view_projection, vec4(position, 1.0))
    return BlinnPhongVertexData(clip_position, normal, position, vec4(1.0, 1.0, 1.0, 1.0))


@vertex
def instanced_vertex(
    position: Annotated[vec3[f32], location(0)],
    normal: Annotated[vec3[f32], location(1)],
    instance_offset: Annotated[vec3[f32], instance(location=4)],
    instance_scale: Annotated[vec3[f32], instance(location=5)],
    custom_tint: Annotated[vec4[f32], instance(location=6)],
    view_projection: Annotated[mat4[f32], uniform()],
) -> BlinnPhongVertexData:
    world_position = position * instance_scale + instance_offset
    clip_position = matmul(view_projection, vec4(world_position, 1.0))
    return BlinnPhongVertexData(clip_position, normal, world_position, custom_tint)


@vertex
def skinned_vertex(
    position: Annotated[vec3[f32], location(0)],
    normal: Annotated[vec3[f32], location(1)],
    weights: Annotated[vec2[f32], location(3)],
    bone0: Annotated[mat4[f32], uniform()],
    bone1: Annotated[mat4[f32], uniform()],
    view_projection: Annotated[mat4[f32], uniform()],
) -> BlinnPhongVertexData:
    local_position = vec4(position, 1.0)
    transformed0 = matmul(bone0, local_position)
    transformed1 = matmul(bone1, local_position)
    weight0 = vec4(weights.x, weights.x, weights.x, weights.x)
    weight1 = vec4(weights.y, weights.y, weights.y, weights.y)
    skinned_position = transformed0 * weight0 + transformed1 * weight1
    clip_position = matmul(view_projection, skinned_position)
    return BlinnPhongVertexData(clip_position, normal, skinned_position.xyz, vec4(1.0, 1.0, 1.0, 1.0))
