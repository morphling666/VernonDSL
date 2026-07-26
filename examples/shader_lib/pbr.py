from typing import Annotated

import vernon_dsl as vd

SHADOW = vd.feature("SHADOW")
ENVIRONMENT = vd.feature("ENVIRONMENT")


@vd.struct
class PbrVertexOutput:
    position: Annotated[vd.Vector[vd.f32, 4], vd.builtin("position")]
    normal: vd.Vector[vd.f32, 3]
    world_position: vd.Vector[vd.f32, 3]
    base_color: vd.Vector[vd.f32, 3]
    material: vd.Vector[vd.f32, 3]


@vd.vertex
def pbr_vertex(
    position: Annotated[vd.Vector[vd.f32, 3], vd.attribute()],
    normal: Annotated[vd.Vector[vd.f32, 3], vd.attribute()],
    base_color: Annotated[vd.Vector[vd.f32, 3], vd.attribute()],
    material: Annotated[vd.Vector[vd.f32, 3], vd.attribute()],
    view_projection: Annotated[vd.Matrix[vd.f32, 4, 4], vd.uniform()],
) -> PbrVertexOutput:
    return PbrVertexOutput(
        vd.matmul(view_projection, vd.Vector([position, 1.0])),
        normal,
        position,
        base_color,
        material,
    )


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
    camera_position: Annotated[vd.Vector[vd.f32, 3], vd.uniform()],
    light_position: Annotated[vd.Vector[vd.f32, 3], vd.uniform()],
) -> vd.Vector[vd.f32, 4]:
    unit_normal = vd.normalize(normal)
    view_direction = vd.normalize(camera_position - world_position)
    light_direction = vd.normalize(light_position - world_position)
    halfway = vd.normalize(view_direction + light_direction)

    roughness = vd.clamp(material.x, 0.06, 1.0)
    metallic = vd.clamp(material.y, 0.0, 1.0)
    object_mask = vd.clamp(material.z, 0.0, 1.0)
    dielectric = vd.Vector([0.04, 0.04, 0.04])
    reflectance = dielectric * (1.0 - metallic) + base_color * metallic

    normal_dot_light = vd.max(vd.dot(unit_normal, light_direction), 0.0)
    normal_dot_view = vd.max(vd.dot(unit_normal, view_direction), 0.0)
    halfway_dot_view = vd.max(vd.dot(halfway, view_direction), 0.0)
    distribution = distribution_ggx(unit_normal, halfway, roughness)
    geometry = geometry_schlick_ggx(normal_dot_view, roughness) * geometry_schlick_ggx(normal_dot_light, roughness)
    fresnel = fresnel_schlick(halfway_dot_view, reflectance)
    specular = fresnel * (distribution * geometry / (4.0 * normal_dot_view * normal_dot_light + 0.0001))
    diffuse_weight = (vd.Vector([1.0, 1.0, 1.0]) - fresnel) * (1.0 - metallic)
    diffuse = diffuse_weight * base_color * 0.318309886

    # The Runtime does not expose sampled D32 images yet. This deterministic
    # receiver shadow keeps the example runnable without pretending it is PCF.
    shadow_center = -vd.Vector([0.53, 0.71])
    shadow_delta = world_position.xz - shadow_center
    shadow_distance = vd.dot(shadow_delta, shadow_delta)
    receiver_visibility = vd.clamp((shadow_distance - 0.12) * 4.0, 0.24, 1.0)
    visibility = 1.0
    if SHADOW:
        visibility = 1.0 - (1.0 - object_mask) * (1.0 - receiver_visibility)

    direct_radiance = vd.Vector([5.4, 5.0, 4.5])
    direct = (diffuse + specular) * direct_radiance * normal_dot_light * visibility

    reflection = vd.reflect(-view_direction, unit_normal)
    sky_amount = vd.clamp(reflection.y * 0.5 + 0.5, 0.0, 1.0)
    environment = vd.Vector([0.035, 0.045, 0.055]) * (1.0 - sky_amount) + vd.Vector([0.22, 0.38, 0.62]) * sky_amount
    environment_weight = reflectance * (1.0 - roughness * 0.65) + base_color * (1.0 - metallic) * 0.055
    ambient = base_color * 0.025
    if ENVIRONMENT:
        ambient = ambient + environment * environment_weight
    linear_color = direct + ambient
    mapped = linear_color / (linear_color + vd.Vector([1.0, 1.0, 1.0]))
    return vd.Vector([mapped, 1.0])
