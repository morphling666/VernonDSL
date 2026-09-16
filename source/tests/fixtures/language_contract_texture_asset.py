from __future__ import annotations

from typing import Annotated

import vernon_dsl as vd  # pyright: ignore[reportMissingImports]


@vd.vertex
def triangle_vertex(
    vertex_index: Annotated[vd.u32, vd.builtin("vertex_index")],
) -> Annotated[vd.Vector[vd.f32, 4], vd.builtin("position")]:
    x = -0.75 if vertex_index == 0 else (0.75 if vertex_index == 1 else 0.0)
    y = -0.75 if vertex_index < 2 else 0.75
    return vd.Vector([x, y, 0.0, 1.0])


@vd.vertex
def lod_vertex(
    vertex_image: Annotated[vd.Texture["2d", vd.f32], vd.resource(set=0, binding=0)],  # noqa: F722
    vertex_sampler: Annotated[vd.Sampler, vd.resource(set=0, binding=1)],
    vertex_index: Annotated[vd.u32, vd.builtin("vertex_index")],
) -> Annotated[vd.Vector[vd.f32, 4], vd.builtin("position")]:
    coordinate = vd.Vector([0.5, 0.5])
    sampled = vd.texture_sample(vertex_image, vertex_sampler, coordinate, 0.0)
    x = -0.75 if vertex_index == 0 else (0.75 if vertex_index == 1 else 0.0)
    y = -0.75 if vertex_index < 2 else 0.75
    return vd.Vector([x, y, sampled.x * 0.0, 1.0])


@vd.fragment
def lod_fragment(
    fragment_image: Annotated[vd.Texture["2d", vd.f32], vd.resource(set=0, binding=2)],  # noqa: F722
) -> vd.Vector[vd.f32, 4]:
    extent = vd.texture_size(fragment_image, 0)
    coordinate = vd.fragment_coord().xy / vd.Vector([vd.f32(extent.x), vd.f32(extent.y)])
    return vd.texture_sample(fragment_image, coordinate, 0.0)


@vd.fragment
def multiset_fragment(
    fragment_image: Annotated[vd.Texture["2d", vd.f32], vd.resource(set=1, binding=0)],  # noqa: F722
) -> vd.Vector[vd.f32, 4]:
    extent = vd.texture_size(fragment_image, 0)
    coordinate = vd.fragment_coord().xy / vd.Vector([vd.f32(extent.x), vd.f32(extent.y)])
    return vd.texture_sample(fragment_image, coordinate, 0.0)


@vd.fragment
def implicit_fragment(
    fragment_image: Annotated[vd.Texture["2d", vd.f32], vd.resource(set=0, binding=2)],  # noqa: F722
) -> vd.Vector[vd.f32, 4]:
    extent = vd.resolution()
    coordinate = vd.fragment_coord().xy / extent
    return vd.texture_sample(fragment_image, coordinate)


@vd.fragment
def explicit_fragment(
    fragment_image: Annotated[vd.Texture["2d", vd.f32], vd.resource(set=0, binding=2)],  # noqa: F722
    fragment_sampler: Annotated[vd.Sampler, vd.resource(set=0, binding=3)],
) -> vd.Vector[vd.f32, 4]:
    return vd.texture_sample(fragment_image, fragment_sampler, vd.Vector([0.5, 0.5]))


@vd.kernel(workgroup_size=(1, 1, 1))
def storage_write(
    image: vd.Texture["2d", vd.rgba32_float, vd.write],  # noqa: F722
) -> None:
    vd.texture_store(image, vd.Vector([vd.i32(0), vd.i32(0)]), vd.Vector([1.0, 1.0, 1.0, 1.0]))


pair_asset = vd.program_asset(
    id="runtime/language-contract-texture-pair",
    program=vd.pipeline(
        lod_vertex,
        lod_fragment,
        targets=vd.target_formats(colors={0: vd.rgba8_unorm}),
    ),
)

multiset_asset = vd.program_asset(
    id="runtime/vulkan-multiset-texture-regression",
    program=vd.pipeline(
        lod_vertex,
        multiset_fragment,
        targets=vd.target_formats(colors={0: vd.rgba8_unorm}),
    ),
)

implicit_asset = vd.program_asset(
    id="runtime/language-contract-texture-implicit",
    program=vd.pipeline(
        lod_vertex,
        implicit_fragment,
        targets=vd.target_formats(colors={0: vd.rgba8_unorm}),
    ),
)

explicit_asset = vd.program_asset(
    id="runtime/language-contract-texture-explicit",
    program=vd.pipeline(
        triangle_vertex,
        explicit_fragment,
        targets=vd.target_formats(colors={0: vd.rgba8_unorm}),
    ),
)

storage_asset = vd.program_asset(
    id="runtime/language-contract-texture-storage",
    program=storage_write,
)
