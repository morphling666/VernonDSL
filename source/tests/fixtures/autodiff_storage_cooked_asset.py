import vernon_dsl as vd


@vd.kernel
def mutate(
    values: vd.TensorView[vd.f32, (3,), vd.read_write],
    scale: vd.f32,
) -> vd.f32:
    old = values[1]
    values[1] = old * scale
    return values[0] + values[1]


asset = vd.pipeline_asset(
    id="compute/storage-vjp",
    program=vd.ad.vjp(mutate, wrt=("scale", "values"), protocol="legacy_fixed"),
)
