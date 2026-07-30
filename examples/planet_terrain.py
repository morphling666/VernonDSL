from vernon_dsl import *


@vertex
def planet_terrain_vertex(
    position: Annotated[Vector[f32, 3], attribute()],
    normal: Annotated[Vector[f32, 3], attribute()],
    vertex_height: Annotated[f32, attribute()],
    planet_radius: Annotated[f32, uniform(set=0, binding=0)],
    height_scale: Annotated[f32, uniform(set=0, binding=1)],
    view_projection: Annotated[Matrix[f32, 4, 4], uniform(set=0, binding=2)],
) -> Vector[f32, 4]:
    radius = planet_radius + vertex_height * height_scale
    displaced_position = normalize(normal) * radius + position
    return matmul(view_projection, Vector([displaced_position, 1.0]))
