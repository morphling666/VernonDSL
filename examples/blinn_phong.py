from vernon_dsl import *


@fragment
def blinn_phong_fragment(
    normal: Annotated[Vector[f32, 3], varying()],
    world_position: Annotated[Vector[f32, 3], varying()],
    custom_tint: Annotated[Vector[f32, 4], varying()],
    albedo: Annotated[Vector[f32, 3], uniform()],
    specular_color: Annotated[Vector[f32, 3], uniform()],
    ambient_color: Annotated[Vector[f32, 3], uniform()],
    light_position: Annotated[Vector[f32, 3], uniform()],
    camera_position: Annotated[Vector[f32, 3], uniform()],
    shininess: Annotated[f32, uniform()],
) -> Vector[f32, 4]:
    unit_normal = normalize(normal)
    light_direction = normalize(light_position - world_position)
    view_direction = normalize(camera_position - world_position)
    half_direction = normalize(light_direction + view_direction)
    diffuse = max(dot(unit_normal, light_direction), 0.0)
    specular = pow(max(dot(unit_normal, half_direction), 0.0), shininess)
    tinted_albedo = albedo * custom_tint.xyz
    lit_color = ambient_color + tinted_albedo * diffuse + specular_color * specular
    return Vector([lit_color, 1.0])
