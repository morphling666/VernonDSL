from typing import Annotated

import vernon_dsl as vd


@vd.func
def bilinear_sample(field, position: vd.Vector[vd.f32, 2], width: vd.i32, height: vd.i32):
    x = position.x - vd.floor(position.x / vd.f32(width)) * vd.f32(width)
    y = position.y - vd.floor(position.y / vd.f32(height)) * vd.f32(height)
    x0 = vd.i32(vd.floor(x))
    y0 = vd.i32(vd.floor(y))
    x1 = x0 + 1
    y1 = y0 + 1
    if x1 == width:
        x1 = 0
    if y1 == height:
        y1 = 0
    tx = x - vd.f32(x0)
    ty = y - vd.f32(y0)
    lower = field[y0, x0] * (1.0 - tx) + field[y0, x1] * tx
    upper = field[y1, x0] * (1.0 - tx) + field[y1, x1] * tx
    return lower * (1.0 - ty) + upper * ty


@vd.func
def previous_index(index: vd.i32, size: vd.i32):
    result = index - 1
    if result < 0:
        result = size - 1
    return result


@vd.func
def next_index(index: vd.i32, size: vd.i32):
    result = index + 1
    if result == size:
        result = 0
    return result


@vd.struct(shared=True)
class SmokeFluidParameters:
    width: vd.i32
    height: vd.i32
    pressure_iterations: vd.i32


@vd.kernel(workgroup_size=(16, 16, 1))
def advect_velocity(
    state_velocity: vd.TensorView[vd.Vector[vd.f32, 2], (vd.dyn, vd.dyn), vd.read],
    advected_velocity: vd.TensorView[vd.Vector[vd.f32, 2], (vd.dyn, vd.dyn), vd.write],
    width: vd.i32,
    height: vd.i32,
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    x = vd.i32(gid[0])
    y = vd.i32(gid[1])
    active = x < width and y < height
    if active:
        velocity = state_velocity[y, x]
        source = vd.Vector(
            [
                vd.f32(x) - velocity.x,
                vd.f32(y) - velocity.y,
            ]
        )
        advected_velocity[y, x] = bilinear_sample(
            state_velocity,
            source,
            width,
            height,
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
    if active:
        left = previous_index(x, width)
        right = next_index(x, width)
        bottom = previous_index(y, height)
        top = next_index(y, height)
        divergence[y, x] = (
            advected_velocity[y, right].x
            - advected_velocity[y, left].x
            + advected_velocity[top, x].y
            - advected_velocity[bottom, x].y
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
    if active:
        left = previous_index(x, width)
        right = next_index(x, width)
        bottom = previous_index(y, height)
        top = next_index(y, height)
        pressure_output[y, x] = (
            pressure_input[y, left]
            + pressure_input[y, right]
            + pressure_input[bottom, x]
            + pressure_input[top, x]
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
    if active:
        left = previous_index(x, width)
        right = next_index(x, width)
        bottom = previous_index(y, height)
        top = next_index(y, height)
        velocity = advected_velocity[y, x]
        output_velocity[y, x] = vd.Vector(
            [
                velocity.x - (pressure[y, right] - pressure[y, left]) * 0.5,
                velocity.y - (pressure[top, x] - pressure[bottom, x]) * 0.5,
            ]
        )


@vd.kernel(workgroup_size=(16, 16, 1))
def transport_density(
    state_density: vd.TensorView[vd.f32, (vd.dyn, vd.dyn), vd.read],
    projected_velocity: vd.TensorView[vd.Vector[vd.f32, 2], (vd.dyn, vd.dyn), vd.read],
    output_density: vd.TensorView[vd.f32, (vd.dyn, vd.dyn), vd.write],
    width: vd.i32,
    height: vd.i32,
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    x = vd.i32(gid[0])
    y = vd.i32(gid[1])
    active = x < width and y < height
    if active:
        velocity = projected_velocity[y, x]
        source = vd.Vector(
            [
                vd.f32(x) - velocity.x,
                vd.f32(y) - velocity.y,
            ]
        )
        output_density[y, x] = bilinear_sample(
            state_density,
            source,
            width,
            height,
        )


@vd.kernel(workgroup_size=(1, 1, 1))
def smoke_loss(
    output_density: vd.TensorView[vd.f32, (vd.dyn, vd.dyn), vd.read],
    objective_target_density: vd.TensorView[vd.f32, (vd.dyn, vd.dyn), vd.read],
    output_loss: vd.TensorView[vd.f32, (1,), vd.write],
    width: vd.i32,
    height: vd.i32,
) -> None:
    loss = vd.f32(0.0)
    for y in range(height):
        for x in range(width):
            difference = output_density[y, x] - objective_target_density[y, x]
            loss = loss + difference * difference
    output_loss[0] = loss / vd.f32(width * height)
