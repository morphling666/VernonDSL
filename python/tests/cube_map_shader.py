# pyright: reportMissingImports=false

from typing import Annotated

import vernon_dsl as vd


@vd.struct
class CubeMapVertexOutput:
    position: Annotated[vd.vec4[vd.f32], vd.builtin("position")]
    tex_coord: Annotated[vd.vec3[vd.f32], vd.location(0)]


@vd.struct
class CubeMapFragmentOutput:
    color: Annotated[vd.vec4[vd.f32], vd.location(0)]
    bloom_color: Annotated[vd.vec4[vd.f32], vd.location(1)]


@vd.vertex
def cube_map_vertex(
    aPos: Annotated[vd.vec3[vd.f32], vd.location(0)],
    projection: Annotated[vd.mat4[vd.f32], vd.uniform()],
    view: Annotated[vd.mat4[vd.f32], vd.uniform()],
    model: Annotated[vd.mat4[vd.f32], vd.uniform()],
) -> CubeMapVertexOutput:
    position = vd.matmul(
        projection,
        vd.matmul(view, vd.matmul(model, vd.vec4(aPos, 1.0))),
    )
    return CubeMapVertexOutput(
        vd.vec4(position.x, position.y, position.w, position.w),
        aPos,
    )


@vd.fragment
def cube_map_fragment(
    tex_coord: Annotated[
        vd.vec3[vd.f32],
        vd.varying(),
        vd.location(0),
    ],
    cubeMap: Annotated[
        vd.Texture["cube", vd.f32],
        vd.resource(set=0, binding=0),
    ],
) -> CubeMapFragmentOutput:
    color = vd.texture_sample(cubeMap, tex_coord)
    brightness = vd.dot(color.rgb, vd.vec3(0.2126, 0.7152, 0.0722))
    bloom_color = vd.vec4(0.0, 0.0, 0.0, 1.0)
    if brightness > 1.0:
        bloom_color = color
    return CubeMapFragmentOutput(color, bloom_color)


cube_map_asset = vd.pipeline_asset(
    id="pipelines/cube_map",
    vertex=cube_map_vertex,
    fragment=cube_map_fragment,
    variants=((),),
    targets={"vulkan": {}},
)
