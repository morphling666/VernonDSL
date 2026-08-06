import vernon_dsl as vd


@vd.struct(shared=True)
class SmokeControlHorizon:
    first: vd.Tensor[vd.f32, (8, 8)]
    second: vd.Tensor[vd.f32, (8, 8)]
    third: vd.Tensor[vd.f32, (8, 8)]


@vd.struct
class SmokeRollout:
    loss: vd.f32
    next_density: vd.Tensor[vd.f32, (8, 8)]
    final_density: vd.Tensor[vd.f32, (8, 8)]


@vd.kernel
def smoke_rollout(
    density: vd.Tensor[vd.f32, (8, 8)],
    vertical_transport: vd.Tensor[vd.f32, (8, 8)],
    horizontal_diffusion: vd.Tensor[vd.f32, (8, 8)],
    target: vd.Tensor[vd.f32, (8, 8)],
    controls: SmokeControlHorizon,
) -> SmokeRollout:
    next_density = vd.matmul(vd.matmul(vertical_transport, density), horizontal_diffusion) + controls.first
    second_density = vd.matmul(vd.matmul(vertical_transport, next_density), horizontal_diffusion) + controls.second
    final_density = vd.matmul(vd.matmul(vertical_transport, second_density), horizontal_diffusion) + controls.third
    error = final_density - target
    control_energy = (
        vd.dot(controls.first, controls.first)
        + vd.dot(controls.second, controls.second)
        + vd.dot(controls.third, controls.third)
    )
    loss = vd.dot(error, error) + control_energy * 0.002
    return SmokeRollout(loss, next_density, final_density)
