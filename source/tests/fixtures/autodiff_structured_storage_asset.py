import vernon_dsl as vd


@vd.kernel
def mutate(
    values: vd.TensorView[vd.f32, (3,), vd.read_write],
    source: vd.TensorView[vd.f32, (3,), vd.read],
    index: vd.i32,
    scale: vd.f32,
    loss: vd.TensorView[vd.f32, (1,), vd.write],
) -> None:
    old = source[index]
    values[index] = old * scale
    loss[0] = values[0] + values[index]


asset = vd.pipeline_asset(
    id="compute/structured_storage_vjp",
    program=vd.ad.vjp(mutate, wrt=("values", "source", "scale"), outputs=("loss",)),
)


@vd.struct
class Particle:
    velocity: vd.Vector[vd.f16, 2]
    mass: vd.f32
    tag: vd.i32


@vd.kernel
def aggregate_objective(
    particles: vd.TensorView[Particle, (1,), vd.read],
    output: vd.TensorView[Particle, (1,), vd.write],
) -> None:
    particle = particles[0]
    output[0] = Particle(
        particle.velocity,
        particle.mass * particle.mass,
        particle.tag,
    )


aggregate_asset = vd.pipeline_asset(
    id="compute/structured_storage_aggregate_vjp",
    program=vd.ad.vjp(
        aggregate_objective,
        wrt=("particles",),
        outputs=("output",),
    ),
)
