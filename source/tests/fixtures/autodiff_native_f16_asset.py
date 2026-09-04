import vernon_dsl as vd  # pyright: ignore[reportMissingImports]


@vd.kernel
def f16_objective(x: vd.f16, output: vd.TensorView[vd.f16, (1,), vd.write]) -> None:
    output[0] = x * x


asset = vd.program_asset(
    id="runtime/native-autodiff-f16",
    program=vd.ad.vjp(f16_objective, wrt=("x",), outputs=("output",)),
)
