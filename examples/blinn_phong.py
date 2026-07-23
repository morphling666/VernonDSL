from vernon_dsl import *


@fragment
def blinn_phong_fragment(
    normal: Annotated[vec3[f32], varying(), location(0)],
    world_position: Annotated[vec3[f32], varying(), location(1)],
    custom_tint: Annotated[vec4[f32], varying(), location(2)],
    albedo: Annotated[vec3[f32], uniform()],
    specular_color: Annotated[vec3[f32], uniform()],
    ambient_color: Annotated[vec3[f32], uniform()],
    light_position: Annotated[vec3[f32], uniform()],
    camera_position: Annotated[vec3[f32], uniform()],
    shininess: Annotated[f32, uniform()],
) -> Annotated[vec4[f32], location(0)]:
    unit_normal = normalize(normal)
    light_direction = normalize(light_position - world_position)
    view_direction = normalize(camera_position - world_position)
    half_direction = normalize(light_direction + view_direction)
    diffuse = max(dot(unit_normal, light_direction), 0.0)
    specular = pow(max(dot(unit_normal, half_direction), 0.0), shininess)
    diffuse_vector = vec3(diffuse, diffuse, diffuse)
    specular_vector = vec3(specular, specular, specular)
    tinted_albedo = albedo * custom_tint.xyz
    lit_color = ambient_color + tinted_albedo * diffuse_vector + specular_color * specular_vector
    return vec4(lit_color, 1.0)
