from vernon_dsl import *


@vertex
def vertex_main(
    position: Annotated[vec4[f32], location(0)],
    offset: Annotated[vec4[f32], uniform()],
) -> Annotated[vec4[f32], location(0)]:
    return position + offset


@fragment
def fragment_main(
    color: Annotated[vec4[f32], varying()],
) -> Annotated[vec4[f32], location(0)]:
    return color


@kernel(workgroup_size=(8, 4, 1))
def compute_main(
    values: Annotated[Buffer[f32], resource(set=0, binding=0)],
    invocation: Annotated[vec3[u32], builtin("global_invocation_id")],
) -> None:
    values[invocation[0]] = values[invocation[0]] + 1.0
    for index in range(4):
        current = index
