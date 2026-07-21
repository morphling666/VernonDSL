from vernon_dsl import *


@vertex
def planet_terrain_vertex(
    position: Annotated[vec3[f32], location(0)],
    normal: Annotated[vec3[f32], location(1)],
    vertex_height: Annotated[f32, location(2)],
    planet_radius: Annotated[f32, uniform(set=0, binding=0)],
    height_scale: Annotated[f32, uniform(set=0, binding=1)],
    view_projection: Annotated[mat4[f32], uniform(set=0, binding=2)],
) -> vec4[f32]:
    radius = planet_radius + vertex_height * height_scale
    radius_vector = vec3(radius, radius, radius)
    displaced_position = normalize(normal) * radius_vector + position
    return matmul(view_projection, vec4(displaced_position, 1.0))
