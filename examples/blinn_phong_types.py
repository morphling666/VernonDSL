from __future__ import annotations

from vernon_dsl import Annotated, Vector, builtin, f32, location, struct


@struct
class BlinnPhongVertexData:
    position: Annotated[Vector[f32, 4], builtin("position")]
    normal: Annotated[Vector[f32, 3], location(0)]
    world_position: Annotated[Vector[f32, 3], location(1)]
    custom_tint: Annotated[Vector[f32, 4], location(2)]
