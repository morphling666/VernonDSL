from typing import Annotated

import vernon_dsl as vd

SHADOW = vd.feature("SHADOW")
ENVIRONMENT = vd.feature("ENVIRONMENT")
ROCK_TEXTURE = vd.feature("ROCK_TEXTURE")


@vd.struct
class PbrVertexOutput:
    position: Annotated[vd.Vector[vd.f32, 4], vd.builtin("position")]
    normal: vd.Vector[vd.f32, 3]
    world_position: vd.Vector[vd.f32, 3]
    base_color: vd.Vector[vd.f32, 3]
    material: vd.Vector[vd.f32, 3]
    shadow_position: vd.Vector[vd.f32, 4]


@vd.vertex
def pbr_vertex(
    position: Annotated[vd.Vector[vd.f32, 3], vd.attribute()],
    normal: Annotated[vd.Vector[vd.f32, 3], vd.attribute()],
    base_color: Annotated[vd.Vector[vd.f32, 3], vd.attribute()],
    material: Annotated[vd.Vector[vd.f32, 3], vd.attribute()],
    view_projection: Annotated[vd.Matrix[vd.f32, 4, 4], vd.uniform()],
    light_view_projection: vd.When[SHADOW, Annotated[vd.Matrix[vd.f32, 4, 4], vd.uniform()]],
) -> PbrVertexOutput:
    light_clip_position = vd.Vector([0.0, 0.0, 0.0, 1.0])
    if SHADOW:
        light_clip_position = vd.matmul(light_view_projection, vd.Vector([position, 1.0]))
    return PbrVertexOutput(
        vd.matmul(view_projection, vd.Vector([position, 1.0])),
        normal,
        position,
        base_color,
        material,
        light_clip_position,
    )


@vd.vertex
def shadow_vertex(
    position: Annotated[vd.Vector[vd.f32, 3], vd.attribute()],
    light_view_projection: Annotated[vd.Matrix[vd.f32, 4, 4], vd.uniform()],
) -> Annotated[vd.Vector[vd.f32, 4], vd.builtin("position")]:
    return vd.matmul(light_view_projection, vd.Vector([position, 1.0]))


@vd.fragment
def shadow_fragment() -> vd.Vector[vd.f32, 4]:
    return vd.Vector([1.0, 1.0, 1.0, 1.0])


@vd.func
def distribution_ggx(
    normal: vd.Vector[vd.f32, 3],
    halfway: vd.Vector[vd.f32, 3],
    roughness: vd.f32,
) -> vd.f32:
    alpha = roughness * roughness
    alpha_squared = alpha * alpha
    normal_dot_halfway = vd.max(vd.dot(normal, halfway), 0.0)
    denominator_term = normal_dot_halfway * normal_dot_halfway * (alpha_squared - 1.0) + 1.0
    return alpha_squared / (3.14159265 * denominator_term * denominator_term + 0.0001)


@vd.func
def geometry_schlick_ggx(normal_dot_direction: vd.f32, roughness: vd.f32) -> vd.f32:
    radius = roughness + 1.0
    k = radius * radius * 0.125
    return normal_dot_direction / (normal_dot_direction * (1.0 - k) + k + 0.0001)


@vd.func
def fresnel_schlick(
    cosine: vd.f32,
    reflectance: vd.Vector[vd.f32, 3],
) -> vd.Vector[vd.f32, 3]:
    one_minus_cosine = vd.clamp(1.0 - cosine, 0.0, 1.0)
    power_two = one_minus_cosine * one_minus_cosine
    power_five = power_two * power_two * one_minus_cosine
    return reflectance + (vd.Vector([1.0, 1.0, 1.0]) - reflectance) * power_five


@vd.fragment
def pbr_fragment(
    normal: Annotated[vd.Vector[vd.f32, 3], vd.varying()],
    world_position: Annotated[vd.Vector[vd.f32, 3], vd.varying()],
    base_color: Annotated[vd.Vector[vd.f32, 3], vd.varying()],
    material: Annotated[vd.Vector[vd.f32, 3], vd.varying()],
    shadow_position: Annotated[vd.Vector[vd.f32, 4], vd.varying()],
    camera_position: Annotated[vd.Vector[vd.f32, 3], vd.uniform()],
    light_position: Annotated[vd.Vector[vd.f32, 3], vd.uniform()],
    shadow_depth_scale: vd.When[SHADOW, Annotated[vd.f32, vd.uniform()]],
    shadow_depth_bias: vd.When[SHADOW, Annotated[vd.f32, vd.uniform()]],
    shadow_uv_scale: vd.When[SHADOW, Annotated[vd.Vector[vd.f32, 2], vd.uniform()]],
    shadow_texel_size: vd.When[SHADOW, Annotated[vd.Vector[vd.f32, 2], vd.uniform()]],
    shadow_map: vd.When[
        SHADOW,
        Annotated[vd.Texture["2d", vd.f32], vd.resource(set=0, binding=0)],  # pyright: ignore  # noqa: F722
    ],
    shadow_sampler: vd.When[SHADOW, Annotated[vd.Sampler, vd.resource(set=0, binding=1)]],
    environment_map: vd.When[
        ENVIRONMENT,
        Annotated[
            vd.Texture["cube", vd.f32],  # pyright: ignore  # noqa: F722, F821
            vd.resource(set=0, binding=2),
        ],
    ],
    environment_sampler: vd.When[ENVIRONMENT, Annotated[vd.Sampler, vd.resource(set=0, binding=3)]],
    rock_material: vd.When[
        ROCK_TEXTURE,
        Annotated[
            vd.Texture["2d", vd.f32],  # pyright: ignore  # noqa: F722, F821
            vd.resource(set=0, binding=2),
        ],
    ],
    rock_sampler: vd.When[ROCK_TEXTURE, Annotated[vd.Sampler, vd.resource(set=0, binding=3)]],
) -> vd.Vector[vd.f32, 4]:
    unit_normal = vd.normalize(normal)
    view_direction = vd.normalize(camera_position - world_position)
    # Thin surfaces reuse one normal buffer for both windings. Face the normal
    # toward the viewer so their back side receives the same stable PBR model.
    if vd.dot(unit_normal, view_direction) < 0.0:
        unit_normal = -unit_normal
    surface_color = base_color
    if ROCK_TEXTURE:
        weight = vd.Vector([vd.abs(unit_normal.x), vd.abs(unit_normal.y), vd.abs(unit_normal.z)])
        weight = weight * weight
        weight = weight / (weight.x + weight.y + weight.z + 0.0001)
        uv_x = world_position.zy * 0.46
        uv_y = world_position.xz * 0.46
        uv_z = world_position.xy * 0.46
        material_x = vd.texture_sample(rock_material, rock_sampler, uv_x)
        material_y = vd.texture_sample(rock_material, rock_sampler, uv_y)
        material_z = vd.texture_sample(rock_material, rock_sampler, uv_z)
        rock_luminance = material_x.w * weight.x + material_y.w * weight.y + material_z.w * weight.z
        surface_color = surface_color * (rock_luminance * 1.1 + 0.38)
        sampled_x = material_x.xyz * 2.0 - vd.Vector([1.0, 1.0, 1.0])
        sampled_y = material_y.xyz * 2.0 - vd.Vector([1.0, 1.0, 1.0])
        sampled_z = material_z.xyz * 2.0 - vd.Vector([1.0, 1.0, 1.0])
        normal_x = vd.Vector([sampled_x.z, sampled_x.y, sampled_x.x])
        normal_y = vd.Vector([sampled_y.x, sampled_y.z, sampled_y.y])
        normal_z = vd.Vector([sampled_z.x, sampled_z.y, sampled_z.z])
        detail_normal = normal_x * weight.x + normal_y * weight.y + normal_z * weight.z
        unit_normal = vd.normalize(unit_normal + detail_normal * 0.24)
    light_direction = vd.normalize(light_position - world_position)
    halfway = vd.normalize(view_direction + light_direction)

    roughness = vd.clamp(material.x, 0.06, 1.0)
    metallic = vd.clamp(material.y, 0.0, 1.0)
    object_mask = vd.clamp(material.z, 0.0, 1.0)
    dielectric = vd.Vector([0.04, 0.04, 0.04])
    reflectance = dielectric * (1.0 - metallic) + surface_color * metallic

    normal_dot_light = vd.max(vd.dot(unit_normal, light_direction), 0.0)
    normal_dot_view = vd.max(vd.dot(unit_normal, view_direction), 0.0)
    halfway_dot_view = vd.max(vd.dot(halfway, view_direction), 0.0)
    distribution = distribution_ggx(unit_normal, halfway, roughness)
    geometry = geometry_schlick_ggx(normal_dot_view, roughness) * geometry_schlick_ggx(normal_dot_light, roughness)
    fresnel = fresnel_schlick(halfway_dot_view, reflectance)
    specular = fresnel * (distribution * geometry / (4.0 * normal_dot_view * normal_dot_light + 0.0001))
    diffuse_weight = (vd.Vector([1.0, 1.0, 1.0]) - fresnel) * (1.0 - metallic)
    diffuse = diffuse_weight * surface_color * 0.318309886

    visibility = 1.0
    if SHADOW:
        projected_shadow = shadow_position.xyz / shadow_position.w
        shadow_uv = projected_shadow.xy * shadow_uv_scale + vd.Vector([0.5, 0.5])
        current_depth = projected_shadow.z * shadow_depth_scale + shadow_depth_bias
        shadow_visibility = 0.0
        sampled_depth = vd.texture_sample(shadow_map, shadow_sampler, shadow_uv - shadow_texel_size * 0.5).x
        if current_depth - 0.004 <= sampled_depth:
            shadow_visibility = shadow_visibility + 0.25
        sampled_depth = vd.texture_sample(
            shadow_map,
            shadow_sampler,
            shadow_uv + vd.Vector([shadow_texel_size.x, -shadow_texel_size.y]) * 0.5,
        ).x
        if current_depth - 0.004 <= sampled_depth:
            shadow_visibility = shadow_visibility + 0.25
        sampled_depth = vd.texture_sample(
            shadow_map,
            shadow_sampler,
            shadow_uv + vd.Vector([-shadow_texel_size.x, shadow_texel_size.y]) * 0.5,
        ).x
        if current_depth - 0.004 <= sampled_depth:
            shadow_visibility = shadow_visibility + 0.25
        sampled_depth = vd.texture_sample(shadow_map, shadow_sampler, shadow_uv + shadow_texel_size * 0.5).x
        if current_depth - 0.004 <= sampled_depth:
            shadow_visibility = shadow_visibility + 0.25
        visibility = 1.0 - (1.0 - object_mask) * (1.0 - shadow_visibility)

    direct_radiance = vd.Vector([5.4, 5.0, 4.5])
    direct = (diffuse + specular) * direct_radiance * normal_dot_light * visibility

    ambient = surface_color * 0.025
    if ENVIRONMENT:
        reflection = vd.reflect(-view_direction, unit_normal)
        environment = vd.texture_sample(environment_map, environment_sampler, reflection).xyz
        environment_weight = reflectance * (1.0 - roughness * 0.65) + surface_color * (1.0 - metallic) * 0.055
        ambient = ambient + environment * environment_weight
    emissive = vd.Vector(
        [
            vd.max(surface_color.x - 1.0, 0.0),
            vd.max(surface_color.y - 1.0, 0.0),
            vd.max(surface_color.z - 1.0, 0.0),
        ]
    )
    linear_color = direct + ambient + emissive * 0.72
    mapped = linear_color / (linear_color + vd.Vector([1.0, 1.0, 1.0]))
    gamma_corrected = vd.Vector(
        [
            vd.pow(mapped.x, 0.45454545),
            vd.pow(mapped.y, 0.45454545),
            vd.pow(mapped.z, 0.45454545),
        ]
    )
    return vd.Vector([gamma_corrected, 1.0])
