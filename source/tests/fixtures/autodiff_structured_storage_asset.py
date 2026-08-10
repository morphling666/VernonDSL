from typing import Annotated

import vernon_dsl as vd


@vd.kernel
def mutate(
    values: vd.TensorView[vd.f32, (vd.dyn, vd.dyn, vd.dyn, 3), vd.read_write],
    source: vd.TensorView[vd.f32, (vd.dyn, vd.dyn, vd.dyn, 3), vd.read],
    index: vd.i32,
    scale: vd.f32,
    loss: vd.TensorView[vd.f32, (vd.dyn, vd.dyn, vd.dyn), vd.write],
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    old = source[gid[0], gid[1], gid[2], index]
    values[gid[0], gid[1], gid[2], index] = old * scale
    loss[gid[0], gid[1], gid[2]] = values[gid[0], gid[1], gid[2], 0] + values[gid[0], gid[1], gid[2], index]


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
