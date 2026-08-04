import vernon_dsl as vd  # pyright: ignore[reportMissingImports]


@vd.kernel
def f16_objective(x: vd.f16) -> vd.f16:
    return x * x


asset = vd.pipeline_asset(
    id="runtime/native-autodiff-f16",
    program=vd.ad.vjp(f16_objective, wrt=("x",)),
)
