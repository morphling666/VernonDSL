from shader_lib.shadow import shadow_visibility
from vernon_dsl import *


@fragment
def shadowed_fragment(
    sampled_depth: Annotated[f32, location(0)],
    fragment_depth: Annotated[f32, location(1)],
    bias: Annotated[f32, uniform(set=0, binding=0)],
    softness: Annotated[f32, uniform(set=0, binding=1)],
    albedo: Annotated[vec3[f32], uniform(set=0, binding=2)],
) -> Annotated[vec4[f32], location(0)]:
    visibility = shadow_visibility(sampled_depth, fragment_depth, bias, softness)
    shaded = albedo * vec3(visibility, visibility, visibility)
    return vec4(shaded, 1.0)
