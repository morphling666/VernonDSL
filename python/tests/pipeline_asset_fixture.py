from typing import Annotated

import vernon_dsl as vd

OFFSET = vd.feature("OFFSET")


@vd.vertex
def triangle_vertex(
    position: Annotated[vd.vec2[vd.f32], vd.location(0)],
) -> Annotated[vd.vec4[vd.f32], vd.builtin("position")]:
    if OFFSET:
        position = position + vd.vec2(0.1, 0.0)
    return vd.vec4(position, 0.0, 1.0)


@vd.fragment
def solid_fragment() -> Annotated[vd.vec4[vd.f32], vd.location(0)]:
    return vd.vec4(1.0, 0.25, 0.0, 1.0)


@vd.kernel(workgroup_size=(1, 1, 1))
def scale(
    values: vd.Tensor[vd.f32, (4, )],
    factor: vd.f32,
    gid: Annotated[vd.Tensor[vd.u32, (3, )],
                   vd.builtin("global_invocation_id")],
) -> None:
    index = gid[0]
    values[index] = values[index] * factor


triangle_asset = vd.pipeline_asset(
    id="pipelines/triangle",
    vertex=triangle_vertex,
    fragment=solid_fragment,
    variants=((), (OFFSET, )),
    targets={"opengl": {
        "glsl_version": 330
    }},
)

scale_asset = vd.pipeline_asset(
    id="pipelines/scale",
    compute=scale,
    variants=((), ),
    targets={
        "cpu": {},
        "cuda": {},
    },
)
