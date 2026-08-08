import vernon_dsl as vd


@vd.struct
class Particle:
    velocity: vd.Vector[vd.f16, 2]
    mass: vd.f32
    tag: vd.i32


@vd.struct
class NestedParticle:
    position: vd.Vector[vd.f32, 2]
    weight: vd.f64
    tag: vd.i32


@vd.struct
class NestedRecord:
    particle: NestedParticle
    pair: vd.Tuple[vd.f32, vd.i32]
    samples: vd.Tensor[vd.f32, (2,)]


@vd.struct
class TensorProductLeaf:
    value: vd.f16
    tag: vd.i32


@vd.struct
class TensorProductRecord:
    entries: vd.Tensor[TensorProductLeaf, (2,)]


@vd.kernel
def nested_aggregate_objective(
    source: vd.TensorView[NestedRecord, (vd.dyn,), vd.read],
    output: vd.TensorView[NestedRecord, (vd.dyn,), vd.write],
) -> None:
    record = source[0]
    particle = record.particle
    position = particle.position
    pair = record.pair
    samples = record.samples
    output[0] = NestedRecord(
        NestedParticle(
            vd.Vector(
                [
                    position.x * position.x,
                    position.y + pair[0],
                ]
            ),
            particle.weight * particle.weight,
            particle.tag,
        ),
        (pair[0] * position.x, pair[1]),
        vd.Vector(
            [
                samples[0] * position.y,
                samples[1] + pair[0],
            ]
        ),
    )


nested_aggregate_objective_vjp = vd.ad.vjp(
    nested_aggregate_objective,
    wrt=("source",),
    outputs=("output",),
)


@vd.kernel
def shared_nested_inputs_objective(
    left: vd.TensorView[NestedRecord, (vd.dyn,), vd.read],
    right: vd.TensorView[NestedRecord, (vd.dyn,), vd.read],
    output: vd.TensorView[NestedRecord, (vd.dyn,), vd.write],
) -> None:
    left_record = left[0]
    right_record = right[0]
    left_particle = left_record.particle
    right_particle = right_record.particle
    output[0] = NestedRecord(
        NestedParticle(
            vd.Vector(
                [
                    left_particle.position.x * right_particle.position.x,
                    left_particle.position.y * right_particle.position.y,
                ]
            ),
            left_particle.weight * right_particle.weight,
            left_particle.tag,
        ),
        (left_record.pair[0] * right_record.pair[0], left_record.pair[1]),
        vd.Vector(
            [
                left_record.samples[0] * right_record.samples[0],
                left_record.samples[1] * right_record.samples[1],
            ]
        ),
    )


shared_nested_inputs_objective_vjp = vd.ad.vjp(
    shared_nested_inputs_objective,
    wrt=("left", "right"),
    outputs=("output",),
)


@vd.kernel
def aggregate_storage_objective(
    particles: vd.TensorView[Particle, (1,), vd.read],
    loss: vd.TensorView[vd.f32, (1,), vd.write],
) -> None:
    particle = particles[0]
    loss[0] = (
        particle.velocity.x * particle.velocity.x
        + particle.velocity.y * particle.velocity.y
        + particle.mass * particle.mass
    )


aggregate_storage_objective_vjp = vd.ad.vjp(
    aggregate_storage_objective,
    wrt=("particles",),
    outputs=("loss",),
)

aggregate_field_objective_vjp = vd.ad.vjp(
    aggregate_storage_objective,
    wrt=("particles.velocity",),
    outputs=("loss",),
)


@vd.kernel
def aggregate_output_objective(
    particles: vd.TensorView[Particle, (1,), vd.read],
    output: vd.TensorView[Particle, (1,), vd.write],
) -> None:
    particle = particles[0]
    output[0] = Particle(
        particle.velocity,
        particle.mass * particle.mass,
        particle.tag,
    )


aggregate_output_objective_vjp = vd.ad.vjp(
    aggregate_output_objective,
    wrt=("particles",),
    outputs=("output",),
)


@vd.kernel
def aggregate_output_pair_objective(
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


aggregate_output_pair_objective_vjp = vd.ad.vjp(
    aggregate_output_pair_objective,
    wrt=("particles",),
    outputs=("output",),
)


@vd.kernel
def aggregate_multi_output_objective(
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


aggregate_multi_output_objective_vjp = vd.ad.vjp(
    aggregate_multi_output_objective,
    wrt=("particles",),
    outputs=("first", "second"),
)


@vd.kernel
def storage_objective(
    values: vd.TensorView[vd.f32, (vd.dyn,), vd.read],
    loss: vd.TensorView[vd.f32, (1,), vd.write],
) -> None:
    loss[0] = values[0] * values[0] + values[1]


storage_objective_vjp = vd.ad.vjp(
    storage_objective,
    wrt=("values",),
    outputs=("loss",),
)


@vd.kernel
def aliased_inputs_objective(
    left: vd.TensorView[vd.f32, (2,), vd.read],
    right: vd.TensorView[vd.f32, (2,), vd.read],
    loss: vd.TensorView[vd.f32, (1,), vd.write],
) -> None:
    loss[0] = left[0] * left[0] + right[1] * right[1]


aliased_inputs_objective_vjp = vd.ad.vjp(
    aliased_inputs_objective,
    wrt=("left", "right"),
    outputs=("loss",),
)


@vd.kernel
def scratch_objective(
    values: vd.TensorView[vd.f32, (1,), vd.read],
    scratch: vd.TensorView[vd.f32, (1,), vd.read_write],
    loss: vd.TensorView[vd.f32, (1,), vd.write],
) -> None:
    scratch[0] = values[0] * 2.0
    loss[0] = scratch[0] * scratch[0]


scratch_objective_vjp = vd.ad.vjp(
    scratch_objective,
    wrt=("values",),
    outputs=("loss",),
)


@vd.kernel
def overwrite_objective(
    values: vd.TensorView[vd.f32, (1,), vd.read],
    scratch: vd.TensorView[vd.f32, (1,), vd.read_write],
    loss: vd.TensorView[vd.f32, (1,), vd.write],
) -> None:
    scratch[0] = values[0] * 2.0
    scratch[0] = values[0] * 3.0
    loss[0] = scratch[0] * scratch[0]


overwrite_objective_vjp = vd.ad.vjp(
    overwrite_objective,
    wrt=("values",),
    outputs=("loss",),
)


@vd.kernel
def vector_scratch_objective(
    values: vd.TensorView[vd.f32, (1,), vd.read],
    scratch: vd.TensorView[vd.Vector[vd.f32, 2], (1,), vd.read_write],
    loss: vd.TensorView[vd.f32, (1,), vd.write],
) -> None:
    scratch[0] = vd.Vector([values[0] * 2.0, values[0] * 3.0])
    value = scratch[0]
    loss[0] = value.x * value.x + value.y * value.y


vector_scratch_objective_vjp = vd.ad.vjp(
    vector_scratch_objective,
    wrt=("values",),
    outputs=("loss",),
)


@vd.kernel
def loop_scratch_objective(
    values: vd.TensorView[vd.f32, (2,), vd.read],
    scratch: vd.TensorView[vd.f32, (2,), vd.read_write],
    output: vd.TensorView[vd.f32, (1,), vd.write],
) -> None:
    for index in range(2):
        scratch[index] = values[index] * 2.0
    loss = 0.0
    for index in range(2):
        value = scratch[index]
        loss = loss + value * value
    output[0] = loss


loop_scratch_objective_vjp = vd.ad.vjp(
    loop_scratch_objective,
    wrt=("values",),
    outputs=("output",),
)


@vd.kernel
def branch_scratch_objective(
    values: vd.TensorView[vd.f32, (1,), vd.read],
    scratch: vd.TensorView[vd.f32, (1,), vd.read_write],
    output: vd.TensorView[vd.f32, (1,), vd.write],
) -> None:
    if values[0] > 0.0:
        scratch[0] = values[0] * 2.0
    else:
        scratch[0] = values[0] * 3.0
    value = scratch[0]
    output[0] = value * value


branch_scratch_objective_vjp = vd.ad.vjp(
    branch_scratch_objective,
    wrt=("values",),
    outputs=("output",),
)


@vd.kernel
def dynamic_for_objective(
    values: vd.TensorView[vd.f32, (1,), vd.read],
    count: vd.i32,
    output: vd.TensorView[vd.f32, (1,), vd.write],
) -> None:
    result = values[0]
    for _ in range(count):
        result = result + values[0] * 0.25
    output[0] = result * result


dynamic_for_objective_vjp = vd.ad.vjp(
    dynamic_for_objective,
    wrt=("values",),
    outputs=("output",),
)


@vd.kernel
def dynamic_gather_objective(
    values: vd.TensorView[vd.f32, (vd.dyn,), vd.read],
    count: vd.i32,
    output: vd.TensorView[vd.f32, (1,), vd.write],
) -> None:
    loss = 0.0
    for index in range(count):
        value = values[index]
        loss = loss + value * value
    output[0] = loss


dynamic_gather_objective_vjp = vd.ad.vjp(
    dynamic_gather_objective,
    wrt=("values",),
    outputs=("output",),
)


@vd.kernel
def dynamic_scratch_gather_objective(
    values: vd.TensorView[vd.f32, (vd.dyn,), vd.read],
    scratch: vd.TensorView[vd.f32, (vd.dyn,), vd.read_write],
    count: vd.i32,
    output: vd.TensorView[vd.f32, (1,), vd.write],
) -> None:
    for index in range(count):
        scratch[index] = values[index]
    loss = 0.0
    for index in range(count):
        value = scratch[index]
        loss = loss + value * value
    output[0] = loss


dynamic_scratch_gather_objective_vjp = vd.ad.vjp(
    dynamic_scratch_gather_objective,
    wrt=("values",),
    outputs=("output",),
)


@vd.kernel
def nested_dynamic_scratch_objective(
    values: vd.TensorView[vd.f32, (vd.dyn,), vd.read],
    scratch: vd.TensorView[vd.f32, (vd.dyn, vd.dyn), vd.read_write],
    width: vd.i32,
    height: vd.i32,
    output: vd.TensorView[vd.f32, (1,), vd.write],
) -> None:
    for y in range(height):
        for x in range(width):
            index = vd.i32(vd.f32(x) * 4.0 / vd.f32(width))
            scratch[y, x] = values[index]
    loss = 0.0
    for y in range(height):
        for x in range(width):
            value = scratch[y, x]
            loss = loss + value * value
    output[0] = loss


nested_dynamic_scratch_objective_vjp = vd.ad.vjp(
    nested_dynamic_scratch_objective,
    wrt=("values",),
    outputs=("output",),
)


@vd.kernel
def nested_branch_accumulation_objective(
    values: vd.TensorView[vd.f32, (vd.dyn,), vd.read],
    width: vd.i32,
    height: vd.i32,
    output: vd.TensorView[vd.f32, (1,), vd.write],
) -> None:
    loss = 0.0
    for y in range(height):
        for x in range(width):
            if x == 0 or y == 0 or x + 1 == width or y + 1 == height:
                loss = loss + 1.0
            else:
                value = values[x]
                loss = loss + value * value
    output[0] = loss


nested_branch_accumulation_objective_vjp = vd.ad.vjp(
    nested_branch_accumulation_objective,
    wrt=("values",),
    outputs=("output",),
)


@vd.kernel
def dynamic_while_objective(
    values: vd.TensorView[vd.f32, (1,), vd.read],
    scratch: vd.TensorView[vd.f32, (1,), vd.read_write],
    limit: vd.i32,
    start: vd.i32,
    stop: vd.i32,
    output: vd.TensorView[vd.f32, (1,), vd.write],
) -> None:
    accumulated = 0.0
    index = 0
    while index < limit:
        index = index + 1
        if index > stop:
            break
        if index < start:
            continue
        scratch[0] = values[0] * 0.5
        accumulated = accumulated + scratch[0]
    output[0] = accumulated * accumulated


dynamic_while_objective_vjp = vd.ad.vjp(
    dynamic_while_objective,
    wrt=("values",),
    outputs=("output",),
)


@vd.kernel
def partially_dynamic_objective(
    values: vd.TensorView[vd.f32, (vd.dyn, 2, 4, vd.dyn), vd.read],
    outer: vd.i32,
    inner: vd.i32,
    loss: vd.TensorView[vd.f32, (1,), vd.write],
) -> None:
    first = values[0, 1, 3, 0]
    last = values[outer - 1, 0, 0, inner - 1]
    loss[0] = first * first + last


partially_dynamic_objective_vjp = vd.ad.vjp(
    partially_dynamic_objective,
    wrt=("values",),
    outputs=("loss",),
)
