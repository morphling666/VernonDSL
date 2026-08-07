import vernon_dsl as vd


@vd.struct
class Inner:
    scale: vd.f32
    bias: vd.f32


@vd.struct
class Parameters:
    inner: Inner


@vd.kernel
def objective(
    value: vd.Tensor[vd.f32, (2,)],
    parameters: Parameters,
    output: vd.TensorView[vd.f32, (2,), vd.write],
) -> None:
    scaled = value * parameters.inner.scale + parameters.inner.bias
    output[0] = scaled[0]
    output[1] = scaled[1]


asset = vd.pipeline_asset(
    id="compute/structured_aggregate_vjp",
    program=vd.ad.vjp(
        objective,
        wrt=("value", "parameters.inner.scale", "parameters.inner.bias"),
        outputs=("output",),
    ),
)
