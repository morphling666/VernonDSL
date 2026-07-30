from typing import Annotated

import vernon_dsl as vd


@vd.func
def grid_row(index: vd.u32, width: vd.u32) -> vd.u32:
    # `/` is true division in the source language; truncation is required
    # before the row participates in neighbor addressing.
    return vd.u32(vd.f32(index) / vd.f32(width))


@vd.kernel(workgroup_size=(64, 1, 1))
def step_ocean(
    height: vd.TensorView[vd.f32, (vd.dyn,), vd.read],
    velocity: vd.TensorView[vd.f32, (vd.dyn,), vd.read],
    next_height: vd.TensorView[vd.f32, (vd.dyn,), vd.write],
    next_velocity: vd.TensorView[vd.f32, (vd.dyn,), vd.write],
    width: vd.u32,
    delta_time: vd.f32,
    phase: vd.f32,
    wave_speed: vd.f32,
    damping: vd.f32,
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    index = gid[0]
    x = index % width
    y = grid_row(index, width)
    left_index = index
    right_index = index
    upper_index = index
    lower_index = index
    if x > 0:
        left_index = index - 1
    if x + 1 < width:
        right_index = index + 1
    if y > 0:
        upper_index = index - width
    if y + 1 < width:
        lower_index = index + width

    center = height[index]
    laplacian = height[left_index] + height[right_index] + height[upper_index] + height[lower_index] - center * 4.0
    normalized_x = vd.f32(x) / vd.f32(width - 1) - 0.5
    normalized_y = vd.f32(y) / vd.f32(width - 1) - 0.5
    diagonal_wave = vd.sin(normalized_x * 19.0 + normalized_y * 13.0 - phase * 2.4)
    crossing_wave = vd.sin(normalized_x * -11.0 + normalized_y * 17.0 - phase * 1.7)
    radial_distance = vd.sqrt(normalized_x * normalized_x + normalized_y * normalized_y)
    radial_wave = vd.sin(radial_distance * 42.0 - phase * 3.2)
    forcing = (diagonal_wave * 0.52 + crossing_wave * 0.31 + radial_wave * 0.17) * 0.34
    updated_velocity = (velocity[index] + (laplacian * wave_speed + forcing) * delta_time) * damping
    updated_height = center + updated_velocity * delta_time
    boundary = x == 0 or y == 0 or x + 1 == width or y + 1 == width
    if boundary:
        updated_height = 0.0
        updated_velocity = 0.0
    next_height[index] = updated_height
    next_velocity[index] = updated_velocity


@vd.kernel(workgroup_size=(64, 1, 1))
def build_ocean_mesh(
    height: vd.TensorView[vd.f32, (vd.dyn,), vd.read],
    positions: vd.TensorView[vd.f32, (vd.dyn, vd.dyn), vd.write],
    normals: vd.TensorView[vd.f32, (vd.dyn, vd.dyn), vd.write],
    colors: vd.TensorView[vd.f32, (vd.dyn, vd.dyn), vd.write],
    materials: vd.TensorView[vd.f32, (vd.dyn, vd.dyn), vd.write],
    width: vd.u32,
    extent: vd.f32,
    phase: vd.f32,
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    index = gid[0]
    x = index % width
    y = grid_row(index, width)
    left_index = index
    right_index = index
    upper_index = index
    lower_index = index
    if x > 0:
        left_index = index - 1
    if x + 1 < width:
        right_index = index + 1
    if y > 0:
        upper_index = index - width
    if y + 1 < width:
        lower_index = index + width

    spacing = extent / vd.f32(width - 1)
    world_x = vd.f32(x) * spacing - extent * 0.5
    world_z = vd.f32(y) * spacing - extent * 0.5
    phase_one = world_x * 0.95 + world_z * 0.22 - phase * 0.9
    phase_two = world_x * -0.56 + world_z * 1.15 - phase * 1.2 + 1.3
    phase_three = world_x * 1.62 + world_z * 1.1 - phase * 1.7 + 2.1
    phase_four = world_x * -2.05 + world_z * 0.72 - phase * 2.0 + 0.6
    phase_five = world_x * 0.48 - world_z * 2.85 - phase * 2.3 + 2.8
    phase_six = world_x * -3.15 - world_z * 1.62 - phase * 2.65 + 1.7
    phase_seven = world_x * 4.35 - world_z * 0.95 - phase * 3.1 + 0.35
    phase_eight = world_x * -2.75 + world_z * 4.65 - phase * 3.65 + 2.45
    wave_height = (
        vd.sin(phase_one) * 0.075
        + vd.sin(phase_two) * 0.045
        + vd.sin(phase_three) * 0.026
        + vd.sin(phase_four) * 0.02
        + vd.sin(phase_five) * 0.014
        + vd.sin(phase_six) * 0.01
        + vd.sin(phase_seven) * 0.007
        + vd.sin(phase_eight) * 0.005
    )
    displacement_x = (
        vd.cos(phase_one) * 0.07
        + vd.cos(phase_two) * -0.025
        + vd.cos(phase_three) * 0.018
        + vd.cos(phase_four) * -0.018
        + vd.cos(phase_five) * 0.003
        + vd.cos(phase_six) * -0.008
    )
    displacement_z = (
        vd.cos(phase_one) * 0.016
        + vd.cos(phase_two) * 0.052
        + vd.cos(phase_three) * 0.012
        + vd.cos(phase_four) * 0.006
        + vd.cos(phase_five) * -0.028
        + vd.cos(phase_six) * -0.004
    )
    center_height = height[index] * 0.16 + wave_height
    state_dx = (height[right_index] - height[left_index]) * 0.08 / spacing
    state_dz = (height[lower_index] - height[upper_index]) * 0.08 / spacing
    derivative_x = (
        vd.cos(phase_one) * (0.075 * 0.95)
        + vd.cos(phase_two) * (0.045 * -0.56)
        + vd.cos(phase_three) * (0.026 * 1.62)
        + vd.cos(phase_four) * (0.02 * -2.05)
        + vd.cos(phase_five) * (0.014 * 0.48)
        + vd.cos(phase_six) * (0.01 * -3.15)
        + vd.cos(phase_seven) * (0.007 * 4.35)
        + vd.cos(phase_eight) * (0.005 * -2.75)
        + state_dx
    )
    derivative_z = (
        vd.cos(phase_one) * (0.075 * 0.22)
        + vd.cos(phase_two) * (0.045 * 1.15)
        + vd.cos(phase_three) * (0.026 * 1.1)
        + vd.cos(phase_four) * (0.02 * 0.72)
        + vd.cos(phase_five) * (0.014 * -2.85)
        + vd.cos(phase_six) * (0.01 * -1.62)
        + vd.cos(phase_seven) * (0.007 * -0.95)
        + vd.cos(phase_eight) * (0.005 * 4.65)
        + state_dz
    )
    normal = vd.normalize(vd.Vector([-derivative_x * 1.4, 1.0, -derivative_z * 1.4]))
    steepness = vd.abs(derivative_x) + vd.abs(derivative_z)
    curvature = vd.abs(
        vd.sin(phase_one) * (0.075 * (0.95 * 0.95 + 0.22 * 0.22))
        + vd.sin(phase_two) * (0.045 * (0.56 * 0.56 + 1.15 * 1.15))
        + vd.sin(phase_three) * (0.026 * (1.62 * 1.62 + 1.1 * 1.1))
        + vd.sin(phase_four) * (0.02 * (2.05 * 2.05 + 0.72 * 0.72))
        + vd.sin(phase_five) * (0.014 * (0.48 * 0.48 + 2.85 * 2.85))
        + vd.sin(phase_six) * (0.01 * (3.15 * 3.15 + 1.62 * 1.62))
        + vd.sin(phase_seven) * (0.007 * (4.35 * 4.35 + 0.95 * 0.95))
        + vd.sin(phase_eight) * (0.005 * (2.75 * 2.75 + 4.65 * 4.65))
    )
    crest_mask = vd.clamp((center_height - 0.09) * 11.0, 0.0, 1.0)
    breaking_mask = vd.clamp((curvature - 0.36) * 3.5, 0.0, 1.0)
    foam = crest_mask * breaking_mask

    positions[index, 0] = world_x + displacement_x
    positions[index, 1] = center_height
    positions[index, 2] = world_z + displacement_z
    normals[index, 0] = normal.x
    normals[index, 1] = normal.y
    normals[index, 2] = normal.z
    colors[index, 0] = center_height
    colors[index, 1] = steepness
    colors[index, 2] = foam
    materials[index, 0] = 0.08
    materials[index, 1] = 0.0
    materials[index, 2] = 1.0


@vd.struct
class OceanVertexOutput:
    position: Annotated[vd.Vector[vd.f32, 4], vd.builtin("position")]
    normal: vd.Vector[vd.f32, 3]
    world_position: vd.Vector[vd.f32, 3]
    surface_data: vd.Vector[vd.f32, 3]


@vd.vertex
def ocean_vertex(
    position: Annotated[vd.Vector[vd.f32, 3], vd.attribute()],
    normal: Annotated[vd.Vector[vd.f32, 3], vd.attribute()],
    surface_data: Annotated[vd.Vector[vd.f32, 3], vd.attribute()],
    view_projection: Annotated[vd.Matrix[vd.f32, 4, 4], vd.uniform()],
) -> OceanVertexOutput:
    return OceanVertexOutput(
        vd.matmul(view_projection, vd.Vector([position, 1.0])),
        normal,
        position,
        surface_data,
    )


@vd.func
def aces_film(color: vd.Vector[vd.f32, 3]) -> vd.Vector[vd.f32, 3]:
    mapped = (color * (color * 2.51 + vd.Vector([0.03, 0.03, 0.03]))) / (
        color * (color * 2.43 + vd.Vector([0.59, 0.59, 0.59])) + vd.Vector([0.14, 0.14, 0.14])
    )
    return vd.Vector(
        [
            vd.clamp(mapped.x, 0.0, 1.0),
            vd.clamp(mapped.y, 0.0, 1.0),
            vd.clamp(mapped.z, 0.0, 1.0),
        ]
    )


@vd.fragment
def ocean_fragment(
    normal: Annotated[vd.Vector[vd.f32, 3], vd.varying()],
    world_position: Annotated[vd.Vector[vd.f32, 3], vd.varying()],
    surface_data: Annotated[vd.Vector[vd.f32, 3], vd.varying()],
    camera_position: Annotated[vd.Vector[vd.f32, 3], vd.uniform()],
    light_position: Annotated[vd.Vector[vd.f32, 3], vd.uniform()],
    environment_map: Annotated[
        vd.Texture["cube", vd.f32],  # pyright: ignore  # noqa: F722, F821
        vd.resource(set=0, binding=0),
    ],
    environment_sampler: Annotated[
        vd.Sampler,
        vd.resource(set=0, binding=1),
    ],
    normal_map: Annotated[
        vd.Texture["2d", vd.f32],  # pyright: ignore  # noqa: F722, F821
        vd.resource(set=0, binding=2),
    ],
    normal_sampler: Annotated[
        vd.Sampler,
        vd.resource(set=0, binding=3),
    ],
    phase: Annotated[vd.f32, vd.uniform()],
) -> vd.Vector[vd.f32, 4]:
    unit_normal = vd.normalize(normal)
    uv_one = vd.Vector(
        [
            world_position.x * 0.799 - world_position.z * 0.602,
            world_position.x * 0.602 + world_position.z * 0.799,
        ]
    ) * 0.38 + vd.Vector([phase * 0.029, phase * 0.016])
    uv_two = vd.Vector(
        [
            world_position.x * -0.391 - world_position.z * 0.921,
            world_position.x * 0.921 - world_position.z * 0.391,
        ]
    ) * 0.91 + vd.Vector([-phase * 0.022, phase * 0.041])
    detail_one = vd.texture_sample(normal_map, normal_sampler, uv_one).xyz * 2.0 - vd.Vector([1.0, 1.0, 1.0])
    detail_two = vd.texture_sample(normal_map, normal_sampler, uv_two).xyz * 2.0 - vd.Vector([1.0, 1.0, 1.0])
    detail_x = detail_one.x * 0.58 + detail_two.x * 0.42
    detail_z = detail_one.y * 0.58 + detail_two.y * 0.42
    unit_normal = vd.normalize(unit_normal + vd.Vector([detail_x, 0.0, detail_z]) * 0.11)
    view_direction = vd.normalize(camera_position - world_position)
    if vd.dot(unit_normal, view_direction) < 0.0:
        unit_normal = -unit_normal
    light_direction = vd.normalize(light_position - world_position)
    halfway = vd.normalize(view_direction + light_direction)
    normal_dot_view = vd.max(vd.dot(unit_normal, view_direction), 0.0)
    fresnel_base = 1.0 - normal_dot_view
    fresnel_two = fresnel_base * fresnel_base
    fresnel = 0.035 + 0.965 * fresnel_two * fresnel_two * fresnel_base

    reflection_direction = vd.reflect(-view_direction, unit_normal)
    environment = vd.texture_sample(
        environment_map,
        environment_sampler,
        reflection_direction,
    ).xyz
    environment = environment * vd.Vector([0.38, 0.58, 0.92])
    deep_color = vd.Vector([0.002, 0.022, 0.065])
    facing_color = vd.Vector([0.003, 0.052, 0.11])
    water_color = deep_color + facing_color * normal_dot_view
    specular = vd.pow(vd.max(vd.dot(unit_normal, halfway), 0.0), 384.0) * 0.45
    foam = vd.clamp(surface_data.z, 0.0, 1.0)
    foam_color = vd.Vector([0.38, 0.66, 0.7]) * foam * 0.32
    linear_color = (
        water_color * (0.88 - fresnel * 0.24)
        + environment * (0.18 + fresnel * 0.72)
        + vd.Vector([0.68, 0.78, 0.92]) * specular
        + foam_color
    )
    distance_to_camera = vd.norm(camera_position - world_position)
    fog = vd.clamp((distance_to_camera - 5.2) * 0.14, 0.0, 0.58)
    horizon_color = vd.Vector([0.018, 0.045, 0.085])
    linear_color = linear_color * (1.0 - fog) + horizon_color * fog
    mapped = aces_film(linear_color)
    gamma_corrected = vd.Vector(
        [
            vd.pow(mapped.x, 0.45454545),
            vd.pow(mapped.y, 0.45454545),
            vd.pow(mapped.z, 0.45454545),
        ]
    )
    return vd.Vector([gamma_corrected, 1.0])
