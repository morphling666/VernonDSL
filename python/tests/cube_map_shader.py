# pyright: reportMissingImports=false

from typing import Annotated

import vernon_dsl as vd


@vd.struct
class CubeMapVertexOutput:
    position: Annotated[vd.Vector[vd.f32, 4], vd.builtin("position")]
    tex_coord: vd.Vector[vd.f32, 3]


@vd.struct
class CubeMapFragmentOutput:
    color: vd.Vector[vd.f32, 4]
    bloom_color: vd.Vector[vd.f32, 4]


@vd.vertex
def cube_map_vertex(
    aPos: Annotated[vd.Vector[vd.f32, 3], vd.attribute()],
    projection: Annotated[vd.Matrix[vd.f32, 4, 4], vd.uniform()],
    view: Annotated[vd.Matrix[vd.f32, 4, 4], vd.uniform()],
    model: Annotated[vd.Matrix[vd.f32, 4, 4], vd.uniform()],
) -> CubeMapVertexOutput:
    position = vd.matmul(
        projection,
        vd.matmul(view, vd.matmul(model, vd.Vector([aPos, 1.0]))),
    )
    return CubeMapVertexOutput(
        vd.Vector([position.x, position.y, position.w, position.w]),
        aPos,
    )


@vd.fragment
def cube_map_fragment(
    tex_coord: Annotated[
        vd.Vector[vd.f32, 3],
        vd.varying(),
    ],
    cubeMap: Annotated[
        vd.Texture["cube", vd.f32],
        vd.resource(set=0, binding=1),
    ],
) -> CubeMapFragmentOutput:
    color = vd.texture_sample(cubeMap, tex_coord)
    brightness = vd.dot(color.rgb, vd.Vector([0.2126, 0.7152, 0.0722]))
    bloom_color = vd.Vector([0.0, 0.0, 0.0, 1.0])
    if brightness > 1.0:
        bloom_color = color
    return CubeMapFragmentOutput(color, bloom_color)


cube_map_asset = vd.pipeline_asset(
    id="pipelines/cube_map",
    program=(cube_map_vertex, cube_map_fragment),
    variants=((),),
)
