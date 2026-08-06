import vernon_dsl as vd


@vd.kernel
def mutate(
    values: vd.TensorView[vd.f32, (3,), vd.read_write],
    source: vd.TensorView[vd.f32, (3,), vd.read],
    index: vd.i32,
    scale: vd.f32,
) -> vd.f32:
    old = source[index]
    values[index] = old * scale
    return values[0] + values[index]


asset = vd.pipeline_asset(
    id="compute/structured_storage_vjp",
    program=vd.ad.vjp(mutate, wrt=("values", "source", "scale")),
)
