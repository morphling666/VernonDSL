from vernon_dsl import *

from examples.blinn_phong_types import BlinnPhongVertexData


@vertex
def static_vertex(
    position: Annotated[Vector[f32, 3], location(0)],
    normal: Annotated[Vector[f32, 3], location(1)],
    model_view_projection: Annotated[Matrix[f32, 4, 4], uniform()],
) -> BlinnPhongVertexData:
    clip_position = matmul(model_view_projection, Vector([position, 1.0]))
    return BlinnPhongVertexData(clip_position, normal, position, Vector([1.0, 1.0, 1.0, 1.0]))


@vertex
def instanced_vertex(
    position: Annotated[Vector[f32, 3], location(0)],
    normal: Annotated[Vector[f32, 3], location(1)],
    instance_offset: Annotated[Vector[f32, 3], instance(location=4)],
    instance_scale: Annotated[Vector[f32, 3], instance(location=5)],
    custom_tint: Annotated[Vector[f32, 4], instance(location=6)],
    view_projection: Annotated[Matrix[f32, 4, 4], uniform()],
) -> BlinnPhongVertexData:
    world_position = position * instance_scale + instance_offset
    clip_position = matmul(view_projection, Vector([world_position, 1.0]))
    return BlinnPhongVertexData(clip_position, normal, world_position, custom_tint)


@vertex
def skinned_vertex(
    position: Annotated[Vector[f32, 3], location(0)],
    normal: Annotated[Vector[f32, 3], location(1)],
    weights: Annotated[Vector[f32, 2], location(3)],
    bone0: Annotated[Matrix[f32, 4, 4], uniform()],
    bone1: Annotated[Matrix[f32, 4, 4], uniform()],
    view_projection: Annotated[Matrix[f32, 4, 4], uniform()],
) -> BlinnPhongVertexData:
    local_position = Vector([position, 1.0])
    transformed0 = matmul(bone0, local_position)
    transformed1 = matmul(bone1, local_position)
    skinned_position = transformed0 * weights.x + transformed1 * weights.y
    clip_position = matmul(view_projection, skinned_position)
    return BlinnPhongVertexData(clip_position, normal, skinned_position.xyz, Vector([1.0, 1.0, 1.0, 1.0]))
