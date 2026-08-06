import vernon_dsl as vd


@vd.struct
class Inner:
    scale: vd.f32
    bias: vd.f32


@vd.struct
class Parameters:
    inner: Inner


@vd.struct
class Pair:
    first: vd.f32
    second: vd.f32


@vd.struct
class Result:
    pair: Pair
    tuple_values: vd.Tuple[vd.f32, vd.f32]
    values: vd.Tensor[vd.f32, (2,)]


@vd.kernel
def objective(value: vd.Tensor[vd.f32, (2,)], parameters: Parameters) -> Result:
    scaled = value * parameters.inner.scale + parameters.inner.bias
    return Result(Pair(scaled[0], scaled[1]), (scaled[0], scaled[1]), scaled)


asset = vd.pipeline_asset(
    id="compute/structured_aggregate_vjp",
    program=vd.ad.vjp(
        objective,
        wrt=("value", "parameters.inner.scale", "parameters.inner.bias"),
    ),
)
