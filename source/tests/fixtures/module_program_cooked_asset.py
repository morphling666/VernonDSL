import vernon_dsl as vd  # pyright: ignore[reportMissingImports]


@vd.kernel(workgroup_size=(1, 1, 1))
def square(
    source: vd.TensorView[vd.f32, (1,), vd.read],
    output: vd.TensorView[vd.f32, (1,), vd.write],
) -> None:
    output[0] = source[0] * source[0]


class Square(vd.Module):
    def forward(
        self,
        source: vd.TensorView[vd.f32, (1,), vd.read],
    ) -> vd.TensorStorage:
        output = vd.empty_like(source)
        square(source, output, grid=(1, 1, 1))
        return output


asset = vd.program_asset(
    id="runtime/module-program-cooked",
    program=vd.ad.vjp(Square(), wrt=("source",), outputs=("output",)),
)
