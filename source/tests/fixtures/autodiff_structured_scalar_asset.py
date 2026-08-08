import vernon_dsl as vd


@vd.kernel(workgroup_size=(2, 3, 1))
def objective(
    x: vd.f32,
    y: vd.f32,
    z: vd.f32,
    output: vd.TensorView[vd.f32, (1,), vd.write],
) -> None:
    linear = x + y
    difference = x - y
    product = linear * difference
    quotient = product / y
    negated = -z
    output[0] = (
        quotient
        + negated
        + vd.sin(x)
        + vd.cos(y)
        + vd.exp(z)
        + vd.log(x)
        + vd.sqrt(y)
        + vd.acos(z)
        + vd.atan2(y, x)
        + vd.abs(y - z)
        + x**y
    )


asset = vd.pipeline_asset(
    id="compute/structured_scalar_vjp",
    program=vd.ad.vjp(objective, wrt=("x", "y", "z"), outputs=("output",)),
)


@vd.kernel
def half_objective(x: vd.f16, y: vd.f16, output: vd.TensorView[vd.f16, (1,), vd.write]) -> None:
    output[0] = x * y + x


half_program = vd.ad.vjp(half_objective, wrt=("x", "y"), outputs=("output",))


@vd.kernel
def dynamic_objective(x: vd.f32, count: vd.i32, output: vd.TensorView[vd.f32, (1,), vd.write]) -> None:
    result = x
    index = 0
    while index < count:
        inner = 0
        while inner < 2:
            result += x
            inner += 1
        index += 1
    output[0] = result


dynamic_program = vd.ad.vjp(dynamic_objective, wrt=("x",), outputs=("output",))
dynamic_asset = vd.pipeline_asset(
    id="compute/structured_dynamic_vjp",
    program=vd.ad.vjp(dynamic_objective, wrt=("x",), outputs=("output",)),
)


@vd.kernel
def double_objective(x: vd.f64, output: vd.TensorView[vd.f64, (1,), vd.write]) -> None:
    output[0] = x * x + x


double_asset = vd.pipeline_asset(
    id="compute/structured_f64_vjp",
    program=vd.ad.vjp(double_objective, wrt=("x",), outputs=("output",)),
)


@vd.kernel
def control_flow_objective(
    x: vd.f32,
    y: vd.f32,
    limit: vd.i32,
    mode: vd.i32,
    output: vd.TensorView[vd.f32, (1,), vd.write],
) -> None:
    result = x + y
    index = 0
    while index < limit:
        index += 1
        inner = 0
        while inner < 3:
            inner += 1
            if inner < 2:
                continue
            shared = result * x + y
            result = shared
            if mode == 1:
                if index == 2:
                    output[0] = result
                    return
        if mode == 2:
            if index > 2:
                break
    else:
        result += x * y
    output[0] = result


control_flow_program = vd.ad.vjp(control_flow_objective, wrt=("x", "y"), outputs=("output",))


@vd.kernel
def triple_nested_objective(
    x: vd.f32,
    y: vd.f32,
    outer_limit: vd.i32,
    middle_limit: vd.i32,
    inner_limit: vd.i32,
    output: vd.TensorView[vd.f32, (1,), vd.write],
) -> None:
    result = x + y
    outer = 0
    while outer < outer_limit:
        middle = 0
        while middle < middle_limit:
            inner = 0
            while inner < inner_limit:
                if inner < middle:
                    result = result * x + y
                else:
                    result = result + x * y
                inner += 1
            middle += 1
        outer += 1
    output[0] = result


triple_nested_program = vd.ad.vjp(triple_nested_objective, wrt=("x", "y"), outputs=("output",))


@vd.kernel
def boundary_unary_objective(
    x: vd.f32,
    mode: vd.i32,
    output: vd.TensorView[vd.f32, (1,), vd.write],
) -> None:
    if mode == 0:
        output[0] = vd.log(x)
    elif mode == 1:
        output[0] = vd.sqrt(x)
    elif mode == 2:
        output[0] = vd.acos(x)
    else:
        output[0] = vd.abs(x)


boundary_unary_program = vd.ad.vjp(boundary_unary_objective, wrt=("x",), outputs=("output",))


@vd.kernel
def boundary_division_objective(
    x: vd.f32,
    y: vd.f32,
    output: vd.TensorView[vd.f32, (1,), vd.write],
) -> None:
    output[0] = x / y


boundary_division_program = vd.ad.vjp(boundary_division_objective, wrt=("x", "y"), outputs=("output",))


@vd.kernel
def boundary_power_objective(
    base: vd.f32,
    exponent: vd.f32,
    output: vd.TensorView[vd.f32, (1,), vd.write],
) -> None:
    output[0] = base**exponent


boundary_power_program = vd.ad.vjp(boundary_power_objective, wrt=("base", "exponent"), outputs=("output",))
