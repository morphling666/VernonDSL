from typing import Annotated

import vernon_dsl as vd


@vd.func
def grid_row(index: vd.u32, width: vd.u32) -> vd.u32:
    # Grid addressing needs truncation because source-language division follows
    # Python true-division semantics.
    return vd.u32(vd.f32(index) / vd.f32(width))


@vd.func
def rain_source(
    normalized_x: vd.f32,
    normalized_y: vd.f32,
    terrain_height: vd.f32,
    phase: vd.f32,
    rainfall: vd.f32,
    delta_time: vd.f32,
) -> vd.f32:
    storm = vd.clamp(vd.sin(phase * 0.47) * 0.24 + 0.76, 0.35, 1.0)
    patch = vd.clamp(
        vd.sin(normalized_x * 11.7 + normalized_y * 5.3 + phase * 0.19)
        + vd.sin(normalized_x * -4.1 + normalized_y * 13.9 - phase * 0.13)
        + 1.15,
        0.12,
        1.0,
    )
    upland = vd.clamp((terrain_height + 0.35) * 0.72, 0.12, 1.0)
    return rainfall * storm * patch * upland * delta_time


@vd.kernel(workgroup_size=(64, 1, 1))
def compute_erosion_flow(
    height: vd.TensorView[vd.f32, (vd.dyn,), vd.read],
    water: vd.TensorView[vd.f32, (vd.dyn,), vd.read],
    flow: vd.TensorView[vd.f32, (vd.dyn, vd.dyn), vd.write],
    width: vd.u32,
    delta_time: vd.f32,
    phase: vd.f32,
    rainfall: vd.f32,
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

    normalized_x = vd.f32(x) / vd.f32(width - 1) - 0.5
    normalized_y = vd.f32(y) / vd.f32(width - 1) - 0.5
    rain = rain_source(normalized_x, normalized_y, height[index], phase, rainfall, delta_time)
    available_water = water[index] + rain
    center_surface = height[index] + available_water
    left_surface = height[left_index] + water[left_index]
    right_surface = height[right_index] + water[right_index]
    upper_surface = height[upper_index] + water[upper_index]
    lower_surface = height[lower_index] + water[lower_index]
    drop_left = vd.max(center_surface - left_surface, 0.0)
    drop_right = vd.max(center_surface - right_surface, 0.0)
    drop_upper = vd.max(center_surface - upper_surface, 0.0)
    drop_lower = vd.max(center_surface - lower_surface, 0.0)
    drop_sum = drop_left + drop_right + drop_upper + drop_lower
    total_flow = 0.0
    if drop_sum > 0.000001:
        total_flow = vd.clamp(drop_sum * 3.2 * delta_time, 0.0, available_water * 0.94)
    flow[index, 0] = drop_left * total_flow / (drop_sum + 0.000001)
    flow[index, 1] = drop_right * total_flow / (drop_sum + 0.000001)
    flow[index, 2] = drop_upper * total_flow / (drop_sum + 0.000001)
    flow[index, 3] = drop_lower * total_flow / (drop_sum + 0.000001)

    boundary = x == 0 or y == 0 or x + 1 == width or y + 1 == width
    if boundary:
        flow[index, 0] = available_water * 0.24
        flow[index, 1] = available_water * 0.24
        flow[index, 2] = available_water * 0.24
        flow[index, 3] = available_water * 0.24


@vd.kernel(workgroup_size=(64, 1, 1))
def apply_erosion_flow(
    height: vd.TensorView[vd.f32, (vd.dyn,), vd.read],
    water: vd.TensorView[vd.f32, (vd.dyn,), vd.read],
    sediment: vd.TensorView[vd.f32, (vd.dyn,), vd.read],
    flow: vd.TensorView[vd.f32, (vd.dyn, vd.dyn), vd.read],
    next_height: vd.TensorView[vd.f32, (vd.dyn,), vd.write],
    next_water: vd.TensorView[vd.f32, (vd.dyn,), vd.write],
    next_sediment: vd.TensorView[vd.f32, (vd.dyn,), vd.write],
    width: vd.u32,
    delta_time: vd.f32,
    phase: vd.f32,
    rainfall: vd.f32,
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

    normalized_x = vd.f32(x) / vd.f32(width - 1) - 0.5
    normalized_y = vd.f32(y) / vd.f32(width - 1) - 0.5
    rain = rain_source(normalized_x, normalized_y, height[index], phase, rainfall, delta_time)
    available_water = water[index] + rain
    outgoing = flow[index, 0] + flow[index, 1] + flow[index, 2] + flow[index, 3]
    incoming_left = flow[left_index, 1]
    incoming_right = flow[right_index, 0]
    incoming_upper = flow[upper_index, 3]
    incoming_lower = flow[lower_index, 2]
    incoming = incoming_left + incoming_right + incoming_upper + incoming_lower
    updated_water = vd.clamp((available_water - outgoing + incoming) * 0.99, 0.0, 0.5)

    current_sediment = sediment[index]
    outgoing_sediment = vd.clamp(
        current_sediment * outgoing / (available_water + 0.0005),
        0.0,
        current_sediment,
    )
    incoming_sediment = (
        vd.clamp(
            sediment[left_index] * incoming_left / (water[left_index] + 0.0005),
            0.0,
            sediment[left_index],
        )
        + vd.clamp(
            sediment[right_index] * incoming_right / (water[right_index] + 0.0005),
            0.0,
            sediment[right_index],
        )
        + vd.clamp(
            sediment[upper_index] * incoming_upper / (water[upper_index] + 0.0005),
            0.0,
            sediment[upper_index],
        )
        + vd.clamp(
            sediment[lower_index] * incoming_lower / (water[lower_index] + 0.0005),
            0.0,
            sediment[lower_index],
        )
    )
    transported_sediment = vd.max(current_sediment - outgoing_sediment + incoming_sediment, 0.0)
    height_dx = height[right_index] - height[left_index]
    height_dy = height[lower_index] - height[upper_index]
    slope = vd.sqrt(height_dx * height_dx + height_dy * height_dy)
    speed = (outgoing + incoming) / (available_water + 0.012)
    capacity = updated_water * speed * (0.18 + slope * 2.8)
    eroded = 0.0
    deposited = 0.0
    if transported_sediment < capacity:
        eroded = vd.clamp((capacity - transported_sediment) * 0.075, 0.0, 0.0028)
    else:
        deposited = vd.clamp((transported_sediment - capacity) * 0.12, 0.0, 0.0035)
    updated_height = height[index] - eroded + deposited
    updated_sediment = vd.clamp(transported_sediment + eroded - deposited, 0.0, 0.28)

    boundary = x == 0 or y == 0 or x + 1 == width or y + 1 == width
    if boundary:
        updated_water = 0.0
        updated_sediment = 0.0
    next_height[index] = updated_height
    next_water[index] = updated_water
    next_sediment[index] = updated_sediment


@vd.kernel(workgroup_size=(64, 1, 1))
def build_terrain_mesh(
    height: vd.TensorView[vd.f32, (vd.dyn,), vd.read],
    water: vd.TensorView[vd.f32, (vd.dyn,), vd.read],
    sediment: vd.TensorView[vd.f32, (vd.dyn,), vd.read],
    rock_detail: vd.TensorView[vd.f32, (vd.dyn,), vd.read],
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
    center_height = height[index]
    derivative_x = (height[right_index] - height[left_index]) / (spacing * 2.0)
    derivative_z = (height[lower_index] - height[upper_index]) / (spacing * 2.0)
    normal = vd.normalize(vd.Vector([-derivative_x, 1.0, -derivative_z]))
    steepness = vd.clamp(1.0 - normal.y, 0.0, 1.0)
    wetness = vd.clamp(water[index] * 12.0 + sediment[index] * 3.0, 0.0, 1.0)
    sediment_mask = vd.clamp(sediment[index] * 15.0, 0.0, 1.0)
    surface_variation = vd.clamp((rock_detail[index] - 0.22) * 1.5, 0.0, 1.0)
    wall = vd.clamp(steepness * 4.2, 0.0, 1.0)
    canyon_floor = vd.clamp((0.22 - center_height) * 1.35, 0.0, 1.0)
    red_rock = vd.Vector([0.42, 0.14, 0.038])
    ochre_rock = vd.Vector([0.58, 0.31, 0.095])
    broken_wall = vd.Vector([0.105, 0.038, 0.022]) * (0.72 + surface_variation * 0.28)
    base_color = red_rock * (1.0 - surface_variation * 0.58) + ochre_rock * surface_variation * 0.58
    base_color = base_color * (1.0 - wall * 0.66) + broken_wall * wall * 0.66
    floor_rock = vd.Vector([0.052, 0.028, 0.026])
    base_color = base_color * (1.0 - canyon_floor * 0.62) + floor_rock * canyon_floor * 0.62
    deposit_color = vd.Vector([0.68, 0.33, 0.075])
    base_color = base_color * (1.0 - sediment_mask * 0.5) + deposit_color * sediment_mask * 0.5
    base_color = base_color * (1.0 - wetness * 0.58)

    # The runoff is rendered as an opaque emissive mineral film. It remains
    # compatible with the current blend-free graphics pipeline.
    channel_slope = vd.clamp(steepness * 18.0 + canyon_floor * 0.35, 0.0, 1.0)
    stream = vd.clamp((water[index] - 0.018) * 35.0, 0.0, 1.0) * (0.32 + channel_slope * 0.68)
    normalized_y = vd.f32(y) / vd.f32(width - 1) - 0.5
    stream *= vd.clamp((0.49 - vd.abs(normalized_y)) * 10.0, 0.0, 1.0)
    stream_pulse = vd.sin(phase * 2.1 - world_x * 1.1 - world_z * 1.6) * 0.08 + 0.92
    stream_color = vd.Vector([0.012, 0.115, 0.18]) * stream * stream_pulse
    base_color = base_color * (1.0 - stream) + stream_color

    positions[index, 0] = world_x
    positions[index, 1] = center_height + stream * 0.012
    positions[index, 2] = world_z
    normals[index, 0] = normal.x
    normals[index, 1] = normal.y
    normals[index, 2] = normal.z
    colors[index, 0] = base_color.x
    colors[index, 1] = base_color.y
    colors[index, 2] = base_color.z
    materials[index, 0] = 0.9 - wetness * 0.58 + surface_variation * 0.06
    materials[index, 1] = stream * 0.08
    materials[index, 2] = 0.62
