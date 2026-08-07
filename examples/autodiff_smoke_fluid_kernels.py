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


@vd.kernel
def smoke_fluid_step(
    state_density: vd.TensorView[vd.f32, (vd.dyn, vd.dyn), vd.read],
    state_velocity: vd.TensorView[vd.Vector[vd.f32, 2], (vd.dyn, vd.dyn), vd.read],
    control_nozzles: vd.TensorView[vd.f32, (vd.dyn,), vd.read],
    objective_target_density: vd.TensorView[vd.f32, (vd.dyn, vd.dyn), vd.read],
    scratch_forced_density: vd.TensorView[vd.f32, (vd.dyn, vd.dyn), vd.read_write],
    scratch_forced_velocity: vd.TensorView[vd.Vector[vd.f32, 2], (vd.dyn, vd.dyn), vd.read_write],
    scratch_advected_velocity: vd.TensorView[vd.Vector[vd.f32, 2], (vd.dyn, vd.dyn), vd.read_write],
    scratch_divergence: vd.TensorView[vd.f32, (vd.dyn, vd.dyn), vd.read_write],
    scratch_pressure_a: vd.TensorView[vd.f32, (vd.dyn, vd.dyn), vd.read_write],
    scratch_pressure_b: vd.TensorView[vd.f32, (vd.dyn, vd.dyn), vd.read_write],
    output_density: vd.TensorView[vd.f32, (vd.dyn, vd.dyn), vd.write],
    output_velocity: vd.TensorView[vd.Vector[vd.f32, 2], (vd.dyn, vd.dyn), vd.write],
    output_loss: vd.TensorView[vd.f32, (1,), vd.write],
    parameters: SmokeFluidParameters,
) -> None:
    loss = vd.f32(0.0)

    for y in range(parameters.height):
        for x in range(parameters.width):
            boundary = x == 0 or y == 0 or x + 1 == parameters.width or y + 1 == parameters.height
            if boundary:
                scratch_forced_density[y, x] = 0.0
                scratch_forced_velocity[y, x] = vd.Vector([0.0, 0.0])
            else:
                nozzle = vd.i32(vd.f32(x) * vd.f32(parameters.nozzle_count) / vd.f32(parameters.width))
                nozzle_center = (vd.f32(nozzle) + 0.5) * (vd.f32(parameters.width) / vd.f32(parameters.nozzle_count))
                horizontal = vd.clamp(1.0 - vd.abs(vd.f32(x) - nozzle_center) * 0.25, 0.0, 1.0)
                source_height = vd.f32(parameters.height) * 0.125
                source_bottom = vd.f32(parameters.height) - source_height - 2.0
                vertical = vd.clamp((vd.f32(y) - source_bottom) / source_height, 0.0, 1.0)
                source = control_nozzles[nozzle] * horizontal * vertical
                density = state_density[y, x]
                velocity = state_velocity[y, x]
                scratch_forced_density[y, x] = vd.clamp(
                    density * 0.995 + source * parameters.delta_time * 3.0,
                    0.0,
                    2.0,
                )
                scratch_forced_velocity[y, x] = vd.Vector(
                    [
                        velocity.x
                        + (vd.f32(nozzle) - (vd.f32(parameters.nozzle_count) - 1.0) * 0.5)
                        * source
                        * parameters.delta_time
                        * 0.08,
                        velocity.y - (density * 0.9 + source * 1.6) * parameters.delta_time,
                    ]
                )

    for y in range(1, parameters.height - 1):
        for x in range(1, parameters.width - 1):
            velocity = scratch_forced_velocity[y, x]
            source_x = vd.f32(x) - velocity.x * parameters.delta_time
            source_y = vd.f32(y) - velocity.y * parameters.delta_time
            scratch_advected_velocity[y, x] = (
                bilinear_sample(
                    scratch_forced_velocity,
                    vd.Vector([source_x, source_y]),
                    parameters.width,
                    parameters.height,
                )
                * 0.998
            )

    for y in range(parameters.height):
        for x in range(parameters.width):
            boundary = x == 0 or y == 0 or x + 1 == parameters.width or y + 1 == parameters.height
            if boundary:
                scratch_divergence[y, x] = 0.0
                scratch_pressure_a[y, x] = 0.0
                scratch_pressure_b[y, x] = 0.0
                scratch_advected_velocity[y, x] = vd.Vector([0.0, 0.0])
            else:
                scratch_divergence[y, x] = (
                    scratch_advected_velocity[y, x + 1].x
                    - scratch_advected_velocity[y, x - 1].x
                    + scratch_advected_velocity[y + 1, x].y
                    - scratch_advected_velocity[y - 1, x].y
                ) * 0.5
                scratch_pressure_a[y, x] = 0.0
                scratch_pressure_b[y, x] = 0.0

    for iteration in range(parameters.pressure_iterations):
        for y in range(1, parameters.height - 1):
            for x in range(1, parameters.width - 1):
                if iteration % 2 == 0:
                    scratch_pressure_b[y, x] = (
                        scratch_pressure_a[y, x - 1]
                        + scratch_pressure_a[y, x + 1]
                        + scratch_pressure_a[y - 1, x]
                        + scratch_pressure_a[y + 1, x]
                        - scratch_divergence[y, x]
                    ) * 0.25
                else:
                    scratch_pressure_a[y, x] = (
                        scratch_pressure_b[y, x - 1]
                        + scratch_pressure_b[y, x + 1]
                        + scratch_pressure_b[y - 1, x]
                        + scratch_pressure_b[y + 1, x]
                        - scratch_divergence[y, x]
                    ) * 0.25

    for y in range(parameters.height):
        for x in range(parameters.width):
            boundary = x == 0 or y == 0 or x + 1 == parameters.width or y + 1 == parameters.height
            if boundary:
                output_velocity[y, x] = vd.Vector([0.0, 0.0])
                output_density[y, x] = 0.0
                target = objective_target_density[y, x]
                loss = loss + target * target
            else:
                velocity = scratch_advected_velocity[y, x]
                projected_velocity = vd.Vector(
                    [
                        velocity.x - (scratch_pressure_a[y, x + 1] - scratch_pressure_a[y, x - 1]) * 0.5,
                        velocity.y - (scratch_pressure_a[y + 1, x] - scratch_pressure_a[y - 1, x]) * 0.5,
                    ]
                )
                output_velocity[y, x] = projected_velocity
                source_x = vd.f32(x) - projected_velocity.x * parameters.delta_time
                source_y = vd.f32(y) - projected_velocity.y * parameters.delta_time
                transported = bilinear_sample(
                    scratch_forced_density,
                    vd.Vector([source_x, source_y]),
                    parameters.width,
                    parameters.height,
                )
                laplacian = (
                    scratch_forced_density[y, x - 1]
                    + scratch_forced_density[y, x + 1]
                    + scratch_forced_density[y - 1, x]
                    + scratch_forced_density[y + 1, x]
                    - scratch_forced_density[y, x] * 4.0
                )
                density = vd.clamp((transported + laplacian * 0.0008) * 0.996, 0.0, 2.0)
                output_density[y, x] = density
                difference = density - objective_target_density[y, x]
                loss = loss + difference * difference

    for nozzle in range(parameters.nozzle_count):
        value = control_nozzles[nozzle]
        loss = loss + value * value * 0.002
    output_loss[0] = loss / vd.f32(parameters.width * parameters.height)


smoke_fluid_step_vjp = vd.ad.vjp(
    smoke_fluid_step,
    wrt=("control_nozzles",),
    outputs=("output_loss",),
)

smoke_fluid_cpu_asset = vd.pipeline_asset(
    id="examples/autodiff-smoke-fluid-cpu",
    program=smoke_fluid_step_vjp,
)
