from typing import Annotated

import vernon_dsl as vd


@vd.kernel(workgroup_size=(4, 1, 1))
def static_objective(
    x: vd.f32,
    y: vd.f32,
    output: vd.TensorView[vd.f32, (vd.dyn,), vd.write],
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    total = x + y
    difference = x - y
    output[gid[0]] = total * difference / y


static_asset = vd.program_asset(
    id="compute/gpu_static_tape_vjp",
    program=vd.ad.vjp(
        static_objective,
        wrt=("x", "y"),
        outputs=("output",),
        planning_policy="min_runtime",
    ),
)


@vd.kernel(workgroup_size=(4, 1, 1))
def dynamic_objective(
    x: vd.f32,
    count: vd.i32,
    output: vd.TensorView[vd.f32, (vd.dyn,), vd.write],
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    total = x
    index = 0
    while index < count:
        total = total * x + 1.0
        index += 1
    output[gid[0]] = total


dynamic_asset = vd.program_asset(
    id="compute/gpu_dynamic_tape_vjp",
    program=vd.ad.vjp(
        dynamic_objective,
        wrt=("x",),
        outputs=("output",),
        planning_policy="min_runtime",
    ),
)
