from typing import Annotated

import vernon_dsl as vd

OFFSET = vd.feature("OFFSET")


@vd.vertex
def triangle_vertex(
    position: Annotated[vd.Vector[vd.f32, 2], vd.attribute()],
) -> Annotated[vd.Vector[vd.f32, 4], vd.builtin("position")]:
    if OFFSET:
        position = position + vd.Vector([0.1, 0.0])
    return vd.Vector([position, 0.0, 1.0])


@vd.fragment
def solid_fragment() -> vd.Vector[vd.f32, 4]:
    return vd.Vector([1.0, 0.25, 0.0, 1.0])


@vd.fragment
def resolution_fragment() -> vd.Vector[vd.f32, 4]:
    size = vd.resolution() / 32.0
    return vd.Vector([size, 0.0, 1.0])


@vd.fragment
def opengl_runtime_acceptance_fragment(
    max_steps: Annotated[vd.i32, vd.uniform()],
) -> vd.Vector[vd.f32, 4]:
    size = vd.resolution()
    return vd.Vector([vd.f32(max_steps) / size.x, size.y / size.x, 0.0, 1.0])


@vd.fragment
def sampled_fragment(
    image: Annotated[vd.Texture["2d", vd.f32], vd.resource(set=0, binding=0)],
    sampler: Annotated[vd.Sampler, vd.resource(set=0, binding=1)],
) -> vd.Vector[vd.f32, 4]:
    return vd.texture_sample(image, sampler, vd.Vector([0.5, 0.5]))


@vd.kernel(workgroup_size=(1, 1, 1))
def scale(
    values: vd.TensorView[vd.f32, (vd.dyn,), vd.read_write],
    factor: vd.f32,
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    index = gid[0]
    values[index] = values[index] * factor


triangle_asset = vd.program_asset(
    id="pipelines/triangle",
    program=vd.pipeline(
        triangle_vertex,
        solid_fragment,
        targets=vd.target_formats(colors={0: vd.rgba8_unorm}),
    ),
    variants=((), (OFFSET,)),
)

sampled_asset = vd.program_asset(
    id="pipelines/sampled_triangle",
    program=vd.pipeline(
        triangle_vertex,
        sampled_fragment,
        targets=vd.target_formats(colors={0: vd.rgba8_unorm}),
    ),
    variants=((),),
)

sampled_depth_asset = vd.program_asset(
    id="pipelines/sampled_depth_triangle",
    program=vd.pipeline(
        triangle_vertex,
        sampled_fragment,
        state=vd.graphics_state(
            rasterization=vd.RasterizationState(front_face=vd.FrontFace.CLOCKWISE),
            depth_stencil=vd.DepthStencilState(depth_test=True, depth_write=True),
        ),
        targets=vd.target_formats(colors={0: vd.rgba8_unorm}, depth=vd.d32_float),
    ),
    variants=((),),
)

resolution_asset = vd.program_asset(
    id="pipelines/resolution_triangle",
    program=vd.pipeline(
        triangle_vertex,
        resolution_fragment,
        targets=vd.target_formats(colors={0: vd.rgba8_unorm}),
    ),
    variants=((),),
)

opengl_runtime_acceptance_asset = vd.program_asset(
    id="pipelines/opengl_runtime_acceptance",
    program=vd.pipeline(
        triangle_vertex,
        opengl_runtime_acceptance_fragment,
        targets=vd.target_formats(colors={0: vd.rgba8_unorm}),
    ),
    variants=((),),
)

scale_asset = vd.program_asset(
    id="pipelines/scale",
    program=scale,
    variants=((),),
)
