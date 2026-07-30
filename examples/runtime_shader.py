from shader_lib.shadow import shadow_visibility
from vernon_dsl import *


@vertex
def runtime_vertex(
    position: Annotated[Vector[f32, 3], attribute()],
) -> Annotated[Vector[f32, 4], builtin("position")]:
    return Vector([position, 1.0])


@fragment
def runtime_fragment() -> Vector[f32, 4]:
    visibility = shadow_visibility(1.0, 0.5, 0.0, 1.0)
    return Vector([visibility, visibility, visibility, 1.0])
