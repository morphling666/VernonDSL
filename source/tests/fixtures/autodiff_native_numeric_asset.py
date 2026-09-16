from typing import Annotated

import vernon_dsl as vd  # pyright: ignore[reportMissingImports]


@vd.kernel
def native_numeric_objective(
    left: vd.Tensor[vd.f32, (2, 2, 2)],
    right: vd.Tensor[vd.f32, (1, 2, 2)],
    vector: vd.Tensor[vd.f32, (3,)],
    normal: vd.Tensor[vd.f32, (3,)],
    values: vd.TensorView[vd.f32, (vd.dyn, vd.dyn, vd.dyn, 2), vd.read_write],
    auxiliary: vd.TensorView[vd.f32, (vd.dyn, vd.dyn, vd.dyn, 2), vd.read_write],
    angle: vd.f32,
    ordinate: vd.f32,
    abscissa: vd.f32,
    signed: vd.f32,
    count: vd.i32,
    flag: vd.i32,
    output: vd.TensorView[vd.f32, (vd.dyn, vd.dyn, vd.dyn), vd.write],
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    product = vd.matmul(left, right)
    unit = vd.normalize(vector)
    bounced = vd.reflect(vector, normal)
    term = (
        product[0, 0, 0]
        + product[1, 1, 1]
        + unit[0]
        + bounced[1]
        + vd.acos(angle)
        + vd.atan2(ordinate, abscissa)
        + vd.abs(signed)
    )
    remaining = count
    if flag > 0:
        values[gid[0], gid[1], gid[2], 0] = values[gid[0], gid[1], gid[2], 0] + term
        auxiliary[gid[0], gid[1], gid[2], 0] = auxiliary[gid[0], gid[1], gid[2], 0] + unit[1]
        for _ in range(8):
            if remaining <= 0:
                break
            values[gid[0], gid[1], gid[2], 0] = values[gid[0], gid[1], gid[2], 0] + unit[2]
            remaining = remaining - 1
        output[gid[0], gid[1], gid[2]] = (
            values[gid[0], gid[1], gid[2], 0]
            + values[gid[0], gid[1], gid[2], 1]
            + auxiliary[gid[0], gid[1], gid[2], 0]
            + auxiliary[gid[0], gid[1], gid[2], 1]
            + unit[2]
            + bounced[2]
        )
        return
    else:
        values[gid[0], gid[1], gid[2], 1] = values[gid[0], gid[1], gid[2], 1] * term
        auxiliary[gid[0], gid[1], gid[2], 1] = auxiliary[gid[0], gid[1], gid[2], 1] + bounced[0]
    for _ in range(8):
        if remaining <= 0:
            break
        auxiliary[gid[0], gid[1], gid[2], 1] = auxiliary[gid[0], gid[1], gid[2], 1] + unit[2]
        remaining = remaining - 1
    output[gid[0], gid[1], gid[2]] = (
        values[gid[0], gid[1], gid[2], 0]
        + values[gid[0], gid[1], gid[2], 1]
        + auxiliary[gid[0], gid[1], gid[2], 0]
        + auxiliary[gid[0], gid[1], gid[2], 1]
        + unit[2]
        + bounced[2]
    )


asset = vd.program_asset(
    id="runtime/cooked-native-autodiff-numeric-cpu",
    program=vd.ad.vjp(
        native_numeric_objective,
        wrt=(
            "left",
            "right",
            "vector",
            "normal",
            "values",
            "auxiliary",
            "angle",
            "ordinate",
            "abscissa",
            "signed",
        ),
        outputs=("output",),
    ),
)
