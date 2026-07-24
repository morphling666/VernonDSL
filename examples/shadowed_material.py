from shader_lib.shadow import shadow_visibility
from vernon_dsl import *


@fragment
def shadowed_fragment(
    sampled_depth: Annotated[f32, location(0)],
    fragment_depth: Annotated[f32, location(1)],
    bias: Annotated[f32, uniform(set=0, binding=0)],
    softness: Annotated[f32, uniform(set=0, binding=1)],
    albedo: Annotated[Vector[f32, 3], uniform(set=0, binding=2)],
) -> Annotated[Vector[f32, 4], location(0)]:
    visibility = shadow_visibility(sampled_depth, fragment_depth, bias, softness)
    return Vector([albedo * visibility, 1.0])
