from typing import Annotated

import vernon_dsl as vd

OFFSET = vd.feature("OFFSET")


@vd.vertex
def triangle_vertex(
    position: Annotated[vd.Vector[vd.f32, 2], vd.location(0)],
) -> Annotated[vd.Vector[vd.f32, 4], vd.builtin("position")]:
    if OFFSET:
        position = position + vd.Vector([0.1, 0.0])
    return vd.Vector([position, 0.0, 1.0])


@vd.fragment
def solid_fragment() -> Annotated[vd.Vector[vd.f32, 4], vd.location(0)]:
    return vd.Vector([1.0, 0.25, 0.0, 1.0])


@vd.fragment
def sampled_fragment(
    image: Annotated[vd.Texture["2d", vd.f32], vd.resource(set=0, binding=0)],
    sampler: Annotated[vd.Sampler, vd.resource(set=0, binding=1)],
) -> Annotated[vd.Vector[vd.f32, 4], vd.location(0)]:
    return vd.texture_sample(image, sampler, vd.Vector([0.5, 0.5]))


@vd.kernel(workgroup_size=(1, 1, 1))
def scale(
    values: vd.TensorView[vd.f32, 1, vd.read_write],
    factor: vd.f32,
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    index = gid[0]
    values[index] = values[index] * factor


triangle_asset = vd.pipeline_asset(
    id="pipelines/triangle",
    vertex=triangle_vertex,
    fragment=solid_fragment,
    variants=((), (OFFSET,)),
    targets={"opengl": {"glsl_version": 330}, "vulkan": {}},
)

sampled_asset = vd.pipeline_asset(
    id="pipelines/sampled_triangle",
    vertex=triangle_vertex,
    fragment=sampled_fragment,
    variants=((),),
    targets={"vulkan": {}},
)

scale_asset = vd.pipeline_asset(
    id="pipelines/scale",
    compute=scale,
    variants=((),),
    targets={
        "cpu": {},
        "cuda": {},
    },
)
