from __future__ import annotations

import vernon_dsl as vd  # pyright: ignore[reportMissingImports]


@vd.struct
class Pair:
    left: vd.f32
    right: vd.f32


@vd.struct
class Payload:
    value: vd.f32
    index: vd.i32


@vd.func
def copy_matrix_value(
    output: vd.TensorView[vd.f32, (4,), vd.write],
    matrix: vd.TensorView[vd.f32, (2, 2), vd.read],
) -> None:
    output[0] = matrix[1, 0]


@vd.kernel(workgroup_size=(1, 1, 1))
def exercise_view_contracts(
    matrix: vd.TensorView[vd.f32, (2, 2), vd.read],
    pairs: vd.TensorView[Pair, (vd.dyn,), vd.read_write],
    payloads: vd.TensorView[Payload, (1,), vd.read_write],
    output: vd.TensorView[vd.f32, (4,), vd.write],
) -> None:
    pairs[0] = pairs[1]
    payloads[0] = Payload(payloads[0].value + 1.0, payloads[0].index + 2)
    copy_matrix_value(output, matrix)
    output[1] = pairs[0].left
    output[2] = pairs[0].right
    output[3] = payloads[0].value + vd.f32(payloads[0].index)


asset = vd.program_asset(
    id="runtime/language-contract-views",
    program=exercise_view_contracts,
)
