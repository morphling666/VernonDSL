from __future__ import annotations

from vernon_dsl import Annotated, Vector, builtin, f32, struct


@struct
class BlinnPhongVertexData:
    position: Annotated[Vector[f32, 4], builtin("position")]
    normal: Vector[f32, 3]
    world_position: Vector[f32, 3]
    custom_tint: Vector[f32, 4]
