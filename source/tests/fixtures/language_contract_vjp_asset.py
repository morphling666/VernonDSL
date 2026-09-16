from __future__ import annotations

from dataclasses import dataclass

import vernon_dsl as vd  # pyright: ignore[reportMissingImports]


@vd.kernel(workgroup_size=(1, 1, 1))
def strided_objective(
    source: vd.TensorView[vd.f32, (vd.dyn,), vd.read],
    output: vd.TensorView[vd.f32, (1,), vd.write],
) -> None:
    output[0] = source[0] * source[0] + source[1] * source[1]


signed_stride_asset = vd.program_asset(
    id="runtime/language-contract-signed-stride-vjp",
    program=vd.ad.vjp(strided_objective, wrt=("source",), outputs=("output",)),
)


@vd.kernel(workgroup_size=(1, 1, 1))
def square(
    source: vd.TensorView[vd.f32, (1,), vd.read],
    output: vd.TensorView[vd.f32, (1,), vd.write],
) -> None:
    output[0] = source[0] * source[0]


@vd.kernel(workgroup_size=(1, 1, 1))
def cube(
    source: vd.TensorView[vd.f32, (1,), vd.read],
    output: vd.TensorView[vd.f32, (1,), vd.write],
) -> None:
    output[0] = source[0] * source[0] * source[0]


@dataclass
class Branches:
    square: vd.TensorStorage
    cube: vd.TensorStorage


class FanOut(vd.Module):
    def forward(self, source: vd.TensorStorage) -> Branches:
        squared = vd.empty_like(source)
        cubed = vd.empty_like(source)
        square(source, squared, grid=(1, 1, 1))
        cube(source, cubed, grid=(1, 1, 1))
        return Branches(squared, cubed)


fan_out_asset = vd.program_asset(
    id="runtime/language-contract-fan-out-vjp",
    program=vd.ad.vjp(FanOut(), wrt=("source",), outputs=("square", "cube")),
)
