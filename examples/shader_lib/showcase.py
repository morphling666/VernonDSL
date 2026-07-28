from typing import Annotated

import vernon_dsl as vd


@vd.struct
class SkyVertexOutput:
    position: Annotated[vd.Vector[vd.f32, 4], vd.builtin("position")]
    direction: vd.Vector[vd.f32, 3]


@vd.vertex
def sky_vertex(
    direction: Annotated[vd.Vector[vd.f32, 3], vd.attribute()],
    view_projection: Annotated[vd.Matrix[vd.f32, 4, 4], vd.uniform()],
    camera_position: Annotated[vd.Vector[vd.f32, 3], vd.uniform()],
) -> SkyVertexOutput:
    world_position = camera_position + direction * 12.0
    return SkyVertexOutput(
        vd.matmul(view_projection, vd.Vector([world_position, 1.0])),
        direction,
    )


@vd.fragment
def sky_fragment(
    direction: Annotated[vd.Vector[vd.f32, 3], vd.varying()],
    environment_map: Annotated[
        vd.Texture["cube", vd.f32],  # pyright: ignore  # noqa: F722, F821
        vd.resource(set=0, binding=0),
    ],
    environment_sampler: Annotated[
        vd.Sampler,
        vd.resource(set=0, binding=1),
    ],
) -> vd.Vector[vd.f32, 4]:
    unit_direction = vd.normalize(direction)
    color = vd.texture_sample(
        environment_map,
        environment_sampler,
        unit_direction,
    ).xyz
    lifted = color
    gamma_corrected = vd.Vector(
        [
            vd.pow(vd.clamp(lifted.x, 0.0, 1.0), 0.45454545),
            vd.pow(vd.clamp(lifted.y, 0.0, 1.0), 0.45454545),
            vd.pow(vd.clamp(lifted.z, 0.0, 1.0), 0.45454545),
        ]
    )
    return vd.Vector([gamma_corrected, 1.0])
