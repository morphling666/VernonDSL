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
