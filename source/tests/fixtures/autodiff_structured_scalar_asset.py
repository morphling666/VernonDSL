import vernon_dsl as vd


@vd.kernel(workgroup_size=(2, 3, 1))
def objective(x: vd.f32, y: vd.f32, z: vd.f32) -> vd.f32:
    linear = x + y
    difference = x - y
    product = linear * difference
    quotient = product / y
    negated = -z
    return (
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
    program=vd.ad.vjp(objective, wrt=("x", "y", "z")),
)


@vd.kernel
def half_objective(x: vd.f16, y: vd.f16) -> vd.f16:
    return x * y + x


half_program = vd.ad.vjp(half_objective, wrt=("x", "y"))


@vd.kernel
def dynamic_objective(x: vd.f32, count: vd.i32) -> vd.f32:
    result = x
    index = 0
    while index < count:
        inner = 0
        while inner < 2:
            result += x
            inner += 1
        index += 1
    return result


dynamic_program = vd.ad.vjp(dynamic_objective, wrt=("x",))


@vd.kernel
def control_flow_objective(x: vd.f32, y: vd.f32, limit: vd.i32, mode: vd.i32) -> vd.f32:
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
                    return result
        if mode == 2:
            if index > 2:
                break
    else:
        result += x * y
    return result


control_flow_program = vd.ad.vjp(control_flow_objective, wrt=("x", "y"))


@vd.kernel
def triple_nested_objective(
    x: vd.f32,
    y: vd.f32,
    outer_limit: vd.i32,
    middle_limit: vd.i32,
    inner_limit: vd.i32,
) -> vd.f32:
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
    return result


triple_nested_program = vd.ad.vjp(triple_nested_objective, wrt=("x", "y"))


@vd.kernel
def boundary_unary_objective(x: vd.f32, mode: vd.i32) -> vd.f32:
    if mode == 0:
        return vd.log(x)
    if mode == 1:
        return vd.sqrt(x)
    if mode == 2:
        return vd.acos(x)
    return vd.abs(x)


boundary_unary_program = vd.ad.vjp(boundary_unary_objective, wrt=("x",))


@vd.kernel
def boundary_division_objective(x: vd.f32, y: vd.f32) -> vd.f32:
    return x / y


boundary_division_program = vd.ad.vjp(boundary_division_objective, wrt=("x", "y"))


@vd.kernel
def boundary_power_objective(base: vd.f32, exponent: vd.f32) -> vd.f32:
    return base**exponent


boundary_power_program = vd.ad.vjp(boundary_power_objective, wrt=("base", "exponent"))
