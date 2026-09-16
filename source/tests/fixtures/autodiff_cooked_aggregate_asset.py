import vernon_dsl as vd  # pyright: ignore[reportMissingImports]


@vd.struct
class Parameters:
    factor: vd.f32
    selector: vd.f32


@vd.kernel
def objective(
    value: vd.Tensor[vd.f32, (2,)],
    parameters: Parameters,
    count: vd.i32,
    output: vd.TensorView[vd.f32, (2,), vd.write],
) -> None:
    # Repeated field reads exercise accumulation into one flattened leaf gradient.
    result = value * parameters.factor
    if parameters.selector > 0.0:
        result = result * parameters.factor
        remaining = count
        for _ in range(8):
            if remaining <= 0:
                break
            result = result + value
            remaining = remaining - 1
    else:
        result = result / parameters.factor
        remaining = count
        for _ in range(8):
            if remaining <= 0:
                break
            result = result + value
            remaining = remaining - 1
    output[0] = result[0]
    output[1] = result[1]


asset = vd.program_asset(
    id="runtime/cooked-autodiff",
    program=vd.ad.vjp(
        objective,
        wrt=("value", "parameters.factor", "parameters.selector"),
        outputs=("output",),
    ),
)
