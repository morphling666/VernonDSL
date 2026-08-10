from typing import Annotated

import vernon_dsl as vd


@vd.func
def bilinear_sample(field, position: vd.Vector[vd.f32, 2], width: vd.i32, height: vd.i32):
    x = vd.clamp(position.x, 0.0, vd.f32(width - 1) - 0.001)
    y = vd.clamp(position.y, 0.0, vd.f32(height - 1) - 0.001)
    x0 = vd.i32(vd.floor(x))
    y0 = vd.i32(vd.floor(y))
    x1 = x0 + 1
    y1 = y0 + 1
    tx = x - vd.f32(x0)
    ty = y - vd.f32(y0)
    lower = field[y0, x0] * (1.0 - tx) + field[y0, x1] * tx
    upper = field[y1, x0] * (1.0 - tx) + field[y1, x1] * tx
    return lower * (1.0 - ty) + upper * ty


@vd.struct(shared=True)
class SmokeFluidParameters:
    width: vd.i32
    height: vd.i32
    nozzle_count: vd.i32
    pressure_iterations: vd.i32
    delta_time: vd.f32


@vd.kernel(workgroup_size=(16, 16, 1))
def apply_forces(
    state_density: vd.TensorView[vd.f32, (vd.dyn, vd.dyn), vd.read],
    state_velocity: vd.TensorView[vd.Vector[vd.f32, 2], (vd.dyn, vd.dyn), vd.read],
    control_nozzles: vd.TensorView[vd.f32, (vd.dyn,), vd.read],
    forced_density: vd.TensorView[vd.f32, (vd.dyn, vd.dyn), vd.write],
    forced_velocity: vd.TensorView[vd.Vector[vd.f32, 2], (vd.dyn, vd.dyn), vd.write],
    width: vd.i32,
    height: vd.i32,
    nozzle_count: vd.i32,
    delta_time: vd.f32,
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    x = vd.i32(gid[0])
    y = vd.i32(gid[1])
    active = x < width and y < height
    boundary = x == 0 or y == 0 or x + 1 == width or y + 1 == height
    if active and boundary:
        forced_density[y, x] = 0.0
        forced_velocity[y, x] = vd.Vector([0.0, 0.0])
    elif active:
        nozzle = vd.i32(vd.f32(x) * vd.f32(nozzle_count) / vd.f32(width))
        nozzle_center = (vd.f32(nozzle) + 0.5) * (vd.f32(width) / vd.f32(nozzle_count))
        horizontal = vd.clamp(1.0 - vd.abs(vd.f32(x) - nozzle_center) * 0.25, 0.0, 1.0)
        source_height = vd.f32(height) * 0.125
        source_bottom = vd.f32(height) - source_height - 2.0
        vertical = vd.clamp((vd.f32(y) - source_bottom) / source_height, 0.0, 1.0)
        source = control_nozzles[nozzle] * horizontal * vertical
        density = state_density[y, x]
        velocity = state_velocity[y, x]
        forced_density[y, x] = vd.clamp(
            density * 0.995 + source * delta_time * 3.0,
            0.0,
            2.0,
        )
        forced_velocity[y, x] = vd.Vector(
            [
                velocity.x + (vd.f32(nozzle) - (vd.f32(nozzle_count) - 1.0) * 0.5) * source * delta_time * 0.08,
                velocity.y - (density * 0.9 + source * 1.6) * delta_time,
            ]
        )


@vd.kernel(workgroup_size=(16, 16, 1))
def advect_velocity(
    forced_velocity: vd.TensorView[vd.Vector[vd.f32, 2], (vd.dyn, vd.dyn), vd.read],
    advected_velocity: vd.TensorView[vd.Vector[vd.f32, 2], (vd.dyn, vd.dyn), vd.write],
    width: vd.i32,
    height: vd.i32,
    delta_time: vd.f32,
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    x = vd.i32(gid[0])
    y = vd.i32(gid[1])
    active = x < width and y < height
    boundary = x == 0 or y == 0 or x + 1 == width or y + 1 == height
    if active and boundary:
        advected_velocity[y, x] = vd.Vector([0.0, 0.0])
    elif active:
        velocity = forced_velocity[y, x]
        source = vd.Vector(
            [
                vd.f32(x) - velocity.x * delta_time,
                vd.f32(y) - velocity.y * delta_time,
            ]
        )
        advected_velocity[y, x] = (
            bilinear_sample(
                forced_velocity,
                source,
                width,
                height,
            )
            * 0.998
        )


@vd.kernel(workgroup_size=(16, 16, 1))
def initialize_pressure(
    advected_velocity: vd.TensorView[vd.Vector[vd.f32, 2], (vd.dyn, vd.dyn), vd.read],
    divergence: vd.TensorView[vd.f32, (vd.dyn, vd.dyn), vd.write],
    pressure_a: vd.TensorView[vd.f32, (vd.dyn, vd.dyn), vd.write],
    pressure_b: vd.TensorView[vd.f32, (vd.dyn, vd.dyn), vd.write],
    width: vd.i32,
    height: vd.i32,
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    x = vd.i32(gid[0])
    y = vd.i32(gid[1])
    active = x < width and y < height
    boundary = x == 0 or y == 0 or x + 1 == width or y + 1 == height
    if active and boundary:
        divergence[y, x] = 0.0
        pressure_a[y, x] = 0.0
        pressure_b[y, x] = 0.0
    elif active:
        divergence[y, x] = (
            advected_velocity[y, x + 1].x
            - advected_velocity[y, x - 1].x
            + advected_velocity[y + 1, x].y
            - advected_velocity[y - 1, x].y
        ) * 0.5
        pressure_a[y, x] = 0.0
        pressure_b[y, x] = 0.0


@vd.kernel(workgroup_size=(16, 16, 1))
def jacobi_pressure(
    divergence: vd.TensorView[vd.f32, (vd.dyn, vd.dyn), vd.read],
    pressure_input: vd.TensorView[vd.f32, (vd.dyn, vd.dyn), vd.read],
    pressure_output: vd.TensorView[vd.f32, (vd.dyn, vd.dyn), vd.write],
    width: vd.i32,
    height: vd.i32,
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    x = vd.i32(gid[0])
    y = vd.i32(gid[1])
    active = x < width and y < height
    boundary = x == 0 or y == 0 or x + 1 == width or y + 1 == height
    if active and boundary:
        pressure_output[y, x] = 0.0
    elif active:
        pressure_output[y, x] = (
            pressure_input[y, x - 1]
            + pressure_input[y, x + 1]
            + pressure_input[y - 1, x]
            + pressure_input[y + 1, x]
            - divergence[y, x]
        ) * 0.25


@vd.kernel(workgroup_size=(16, 16, 1))
def project_velocity(
    advected_velocity: vd.TensorView[vd.Vector[vd.f32, 2], (vd.dyn, vd.dyn), vd.read],
    pressure: vd.TensorView[vd.f32, (vd.dyn, vd.dyn), vd.read],
    output_velocity: vd.TensorView[vd.Vector[vd.f32, 2], (vd.dyn, vd.dyn), vd.write],
    width: vd.i32,
    height: vd.i32,
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    x = vd.i32(gid[0])
    y = vd.i32(gid[1])
    active = x < width and y < height
    boundary = x == 0 or y == 0 or x + 1 == width or y + 1 == height
    if active and boundary:
        output_velocity[y, x] = vd.Vector([0.0, 0.0])
    elif active:
        velocity = advected_velocity[y, x]
        output_velocity[y, x] = vd.Vector(
            [
                velocity.x - (pressure[y, x + 1] - pressure[y, x - 1]) * 0.5,
                velocity.y - (pressure[y + 1, x] - pressure[y - 1, x]) * 0.5,
            ]
        )


@vd.kernel(workgroup_size=(16, 16, 1))
def transport_density(
    forced_density: vd.TensorView[vd.f32, (vd.dyn, vd.dyn), vd.read],
    projected_velocity: vd.TensorView[vd.Vector[vd.f32, 2], (vd.dyn, vd.dyn), vd.read],
    output_density: vd.TensorView[vd.f32, (vd.dyn, vd.dyn), vd.write],
    width: vd.i32,
    height: vd.i32,
    delta_time: vd.f32,
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    x = vd.i32(gid[0])
    y = vd.i32(gid[1])
    active = x < width and y < height
    boundary = x == 0 or y == 0 or x + 1 == width or y + 1 == height
    if active and boundary:
        output_density[y, x] = 0.0
    elif active:
        velocity = projected_velocity[y, x]
        source = vd.Vector(
            [
                vd.f32(x) - velocity.x * delta_time,
                vd.f32(y) - velocity.y * delta_time,
            ]
        )
        transported = bilinear_sample(
            forced_density,
            source,
            width,
            height,
        )
        laplacian = (
            forced_density[y, x - 1]
            + forced_density[y, x + 1]
            + forced_density[y - 1, x]
            + forced_density[y + 1, x]
            - forced_density[y, x] * 4.0
        )
        output_density[y, x] = vd.clamp(
            (transported + laplacian * 0.0008) * 0.996,
            0.0,
            2.0,
        )


@vd.kernel(workgroup_size=(1, 1, 1))
def smoke_loss(
    output_density: vd.TensorView[vd.f32, (vd.dyn, vd.dyn), vd.read],
    objective_target_density: vd.TensorView[vd.f32, (vd.dyn, vd.dyn), vd.read],
    control_nozzles: vd.TensorView[vd.f32, (vd.dyn,), vd.read],
    output_loss: vd.TensorView[vd.f32, (1,), vd.write],
    width: vd.i32,
    height: vd.i32,
    nozzle_count: vd.i32,
) -> None:
    loss = vd.f32(0.0)
    for y in range(height):
        for x in range(width):
            difference = output_density[y, x] - objective_target_density[y, x]
            loss = loss + difference * difference
    for nozzle_index in range(nozzle_count):
        value = control_nozzles[nozzle_index]
        loss = loss + value * value * 0.002
    output_loss[0] = loss / vd.f32(width * height)
