from __future__ import annotations

from vernon_dsl import Annotated, builtin, f32, location, struct, vec3, vec4


@struct
class BlinnPhongVertexData:
    position: Annotated[vec4[f32], builtin("position")]
    normal: Annotated[vec3[f32], location(0)]
    world_position: Annotated[vec3[f32], location(1)]
    custom_tint: Annotated[vec4[f32], location(2)]
