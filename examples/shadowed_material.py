from shader_lib.shadow import shadow_visibility
from vernon_dsl import *


@fragment
def shadowed_fragment(
    sampled_depth: Annotated[f32, varying()],
    fragment_depth: Annotated[f32, varying()],
    bias: Annotated[f32, uniform(set=0, binding=0)],
    softness: Annotated[f32, uniform(set=0, binding=1)],
    albedo: Annotated[Vector[f32, 3], uniform(set=0, binding=2)],
) -> Vector[f32, 4]:
    visibility = shadow_visibility(sampled_depth, fragment_depth, bias, softness)
    return Vector([albedo * visibility, 1.0])
