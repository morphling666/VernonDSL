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
    values: Annotated[TensorView[f32, 1, read_write], resource(set=0, binding=0)],
    invocation: Annotated[Vector[u32, 3], builtin("global_invocation_id")],
) -> None:
    values[invocation[0]] = values[invocation[0]] + 1.0
    for index in range(4):
        current = index
