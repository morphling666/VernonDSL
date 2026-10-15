from vernon_dsl import *


@struct
class SmokeVertexOutput:
    position: Annotated[Vector[f32, 4], builtin("position")]
    color: Vector[f32, 4]


@vertex
def vertex_main(
    position: Annotated[Vector[f32, 4], attribute()],
    offset: Annotated[Vector[f32, 4], uniform()],
) -> SmokeVertexOutput:
    translated = position + offset
    return SmokeVertexOutput(translated, translated)


@fragment
def fragment_main(
    color: Annotated[Vector[f32, 4], varying()],
) -> Vector[f32, 4]:
    return color


@kernel(workgroup_size=(8, 4, 1))
def compute_main(
    values: TensorView[f32, (dyn,), read_write],
    invocation: Annotated[Vector[u32, 3], builtin("global_invocation_id")],
) -> None:
    linear_invocation = invocation[0] + invocation[1] * 8
    values[linear_invocation] = values[linear_invocation] + 1.0
    for index in range(4):
        current = index
