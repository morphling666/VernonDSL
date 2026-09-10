from __future__ import annotations

from typing import Annotated

import vernon_dsl as vd  # pyright: ignore[reportMissingImports]


@vd.struct
class Pair:
    left: vd.f32
    right: vd.Vector[vd.f32, 2]


@vd.func
def select_pair(value: Pair, enabled: bool) -> Pair:
    if enabled:
        return Pair(value.left + 1.0, value.right * 2.0)
    return value


@vd.func
def search(limit: vd.i32) -> vd.i32:
    result = 0
    for index in range(limit):
        if index == 2:
            continue
        if index > 4:
            break
        result = index
    return result


@vd.func
def normalize_shape(
    value: vd.Tensor[vd.Tensor[vd.f32, (3,)], (2,)],
) -> vd.Tensor[vd.f32, (2, 3)]:
    return value


@vd.kernel(workgroup_size=(1, 1, 1))
def exercise_language_contracts(
    output: vd.TensorView[vd.f32, (18,), vd.write],
    global_id: Annotated[vd.Vector[vd.u32, 3], vd.builtin("global_invocation_id")],
) -> None:
    vector = vd.Vector([1.0, 2.0])
    matrix = vd.Matrix([[1.0, 2.0], [3.0, 4.0]])
    product = vd.matmul(matrix, vector)
    pair = select_pair(Pair(3.0, vector), True)
    values = vd.Tensor([[1.0, 2.0], [3.0, 4.0]])
    normalized = normalize_shape(vd.Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    tuple_value = (pair.left, product[1])

    total = 0.0
    for index in range(4):
        total += vd.f32(index)

    selected = 5.0 if pair.left == 4.0 and global_id[0] == 0 else -1.0
    output[0] = vd.f32(7 // 2)
    output[1] = vd.f32(7 % 4)
    output[2] = product[0]
    output[3] = product[1]
    output[4] = pair.right[0]
    output[5] = pair.right[1]
    output[6] = values[0, 1]
    output[7] = values[1, 0]
    output[8] = tuple_value[0]
    output[9] = tuple_value[1]
    output[10] = total
    output[11] = selected
    output[12] = vd.sqrt(9.0)
    output[13] = vd.sin(0.0)
    output[14] = vd.f32(vd.i32(6))
    output[15] = vector.x + vector.y
    output[16] = vd.f32(search(7))
    output[17] = normalized[1, 2]


asset = vd.program_asset(
    id="runtime/language-contract-compute",
    program=exercise_language_contracts,
)
