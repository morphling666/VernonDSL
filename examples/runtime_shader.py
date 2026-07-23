from shader_lib.shadow import shadow_visibility
from vernon_dsl import *


@vertex
def runtime_vertex(
    position: Annotated[vec3[f32], location(0)],
) -> Annotated[vec4[f32], builtin("position")]:
    return vec4(position, 1.0)


@fragment
def runtime_fragment() -> Annotated[vec4[f32], location(0)]:
    visibility = shadow_visibility(1.0, 0.5, 0.0, 1.0)
    return vec4(visibility, visibility, visibility, 1.0)
