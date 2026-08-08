import vernon_dsl as vd


@vd.struct
class Particle:
    velocity: vd.Vector[vd.f16, 2]
    mass: vd.f32
    tag: vd.i32


@vd.kernel
def signed_stride_objective(
    particles: vd.TensorView[Particle, (2,), vd.read],
    output: vd.TensorView[Particle, (2,), vd.write],
) -> None:
    for index in range(2):
        particle = particles[index]
        output[index] = Particle(
            particle.velocity,
            particle.mass * particle.mass,
            particle.tag,
        )


signed_stride_asset = vd.pipeline_asset(
    id="compute/dynamic_v2_signed_stride_vjp",
    program=vd.ad.vjp(signed_stride_objective, wrt=("particles",), outputs=("output",)),
)


@vd.kernel
def strided_scatter_objective(
    values: vd.TensorView[vd.f32, (2,), vd.read],
    loss: vd.TensorView[vd.f32, (1,), vd.write],
) -> None:
    loss[0] = values[0] * values[0] + values[1]


strided_scatter_asset = vd.pipeline_asset(
    id="compute/dynamic_v2_strided_scatter_vjp",
    program=vd.ad.vjp(strided_scatter_objective, wrt=("values",), outputs=("loss",)),
)


@vd.kernel
def gather_objective(
    values: vd.TensorView[vd.f32, (4,), vd.read],
    output: vd.TensorView[vd.f32, (1,), vd.write],
) -> None:
    loss = 0.0
    for index in range(4):
        value = values[index]
        loss = loss + value * value
    output[0] = loss


gather_asset = vd.pipeline_asset(
    id="compute/dynamic_v2_gather_vjp",
    program=vd.ad.vjp(gather_objective, wrt=("values",), outputs=("output",)),
)


@vd.kernel
def multi_output_objective(
    particles: vd.TensorView[Particle, (1,), vd.read],
    first: vd.TensorView[Particle, (1,), vd.write],
    second: vd.TensorView[Particle, (1,), vd.write],
) -> None:
    particle = particles[0]
    first[0] = Particle(
        particle.velocity,
        particle.mass * particle.mass,
        particle.tag,
    )
    second[0] = Particle(
        particle.velocity,
        particle.mass * 3.0,
        particle.tag,
    )


multi_output_asset = vd.pipeline_asset(
    id="compute/dynamic_v2_multi_output_vjp",
    program=vd.ad.vjp(multi_output_objective, wrt=("particles",), outputs=("first", "second")),
)
