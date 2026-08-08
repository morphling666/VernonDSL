import vernon_dsl as vd  # pyright: ignore[reportMissingImports]


@vd.kernel
def native_numeric_objective(
    left: vd.Tensor[vd.f32, (2, 2, 2)],
    right: vd.Tensor[vd.f32, (1, 2, 2)],
    vector: vd.Tensor[vd.f32, (3,)],
    normal: vd.Tensor[vd.f32, (3,)],
    values: vd.TensorView[vd.f32, (2,), vd.read_write],
    auxiliary: vd.TensorView[vd.f32, (2,), vd.read_write],
    angle: vd.f32,
    ordinate: vd.f32,
    abscissa: vd.f32,
    signed: vd.f32,
    count: vd.i32,
    flag: vd.i32,
) -> vd.f32:
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
        values[0] = values[0] + term
        auxiliary[0] = auxiliary[0] + unit[1]
        for _ in range(8):
            if remaining <= 0:
                break
            values[0] = values[0] + unit[2]
            remaining = remaining - 1
        return values[0] + values[1] + auxiliary[0] + auxiliary[1] + unit[2] + bounced[2]
    else:
        values[1] = values[1] * term
        auxiliary[1] = auxiliary[1] + bounced[0]
    for _ in range(8):
        if remaining <= 0:
            break
        auxiliary[1] = auxiliary[1] + unit[2]
        remaining = remaining - 1
    return values[0] + values[1] + auxiliary[0] + auxiliary[1] + unit[2] + bounced[2]


@vd.kernel
def native_numeric_cpu_objective(
    left: vd.Tensor[vd.f32, (2, 2, 2)],
    right: vd.Tensor[vd.f32, (1, 2, 2)],
    vector: vd.Tensor[vd.f32, (3,)],
    normal: vd.Tensor[vd.f32, (3,)],
    values: vd.TensorView[vd.f32, (2,), vd.read_write],
    auxiliary: vd.TensorView[vd.f32, (2,), vd.read_write],
    angle: vd.f32,
    ordinate: vd.f32,
    abscissa: vd.f32,
    signed: vd.f32,
    count: vd.i32,
    flag: vd.i32,
    output: vd.TensorView[vd.f32, (1,), vd.write],
) -> None:
    # Keep the GPU legacy entry above unchanged while exercising the same
    # numeric surface through the CPU dynamic_v2 Storage-output ABI.
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
        values[0] = values[0] + term
        auxiliary[0] = auxiliary[0] + unit[1]
        for _ in range(8):
            if remaining <= 0:
                break
            values[0] = values[0] + unit[2]
            remaining = remaining - 1
        output[0] = values[0] + values[1] + auxiliary[0] + auxiliary[1] + unit[2] + bounced[2]
        return
    else:
        values[1] = values[1] * term
        auxiliary[1] = auxiliary[1] + bounced[0]
    for _ in range(8):
        if remaining <= 0:
            break
        auxiliary[1] = auxiliary[1] + unit[2]
        remaining = remaining - 1
    output[0] = values[0] + values[1] + auxiliary[0] + auxiliary[1] + unit[2] + bounced[2]


asset = vd.pipeline_asset(
    id="runtime/cooked-native-autodiff-numeric",
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
        protocol="legacy_fixed",
    ),
)

cpu_asset = vd.pipeline_asset(
    id="runtime/cooked-native-autodiff-numeric-cpu",
    program=vd.ad.vjp(
        native_numeric_cpu_objective,
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
