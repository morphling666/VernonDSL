import vernon_dsl as vd


@vd.kernel
def objective(
    value: vd.Tensor[vd.f32, (2,)],
    factor: vd.f32,
    selector: vd.f32,
    count: vd.i32,
) -> vd.Tensor[vd.f32, (2,)]:
    result = value * factor
    if selector > 0.0:
        result = result * factor
        remaining = count
        for _ in range(8):
            if remaining <= 0:
                break
            result = result + value
            remaining = remaining - 1
        return result
    result = result / factor
    remaining = count
    for _ in range(8):
        if remaining <= 0:
            break
        result = result + value
        remaining = remaining - 1
    return result


asset = vd.pipeline_asset(
    id="runtime/cooked-autodiff",
    program=vd.ad.vjp(objective, wrt=("value", "factor", "selector")),
)
