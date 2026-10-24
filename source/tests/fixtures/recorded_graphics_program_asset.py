from __future__ import annotations

from typing import Annotated, Any

import vernon_dsl as vd


@vd.vertex
def surface_vertex(
    vertices: Annotated[vd.Vector[vd.f32, 2], vd.attribute()],
) -> Annotated[vd.Vector[vd.f32, 4], vd.builtin("position")]:
    return vd.Vector([vertices, 0.0, 1.0])


@vd.fragment
def tinted_fragment(
    tint: Annotated[vd.Vector[vd.f32, 4], vd.uniform()],
) -> vd.Vector[vd.f32, 4]:
    return tint


@vd.fragment
def packed_tint_fragment(
    tint: Annotated[vd.Vector[vd.u32, 4], vd.uniform()],
) -> vd.Vector[vd.f32, 4]:
    return vd.Vector([vd.f32(tint.x), vd.f32(tint.y), vd.f32(tint.z), vd.f32(tint.w)]) / 255.0


@vd.binding_group
class Tint:
    color: vd.Vector[vd.f32, 4]


@vd.fragment
def grouped_tint_fragment(tint: Tint) -> vd.Vector[vd.f32, 4]:
    return tint.color


@vd.binding_group
class Shape:
    vertices: Annotated[vd.Vector[vd.f32, 2], vd.attribute()]
    offset: vd.Vector[vd.f32, 2]

    @vd.func
    def position(self) -> vd.Vector[vd.f32, 4]:
        return vd.Vector([self.vertices + self.offset, 0.0, 1.0])


@vd.vertex
def shape_vertex(shape: Shape) -> Annotated[vd.Vector[vd.f32, 4], vd.builtin("position")]:
    return shape.position()


_targets = vd.target_formats(colors={0: vd.rgba8_unorm})

opaque_pipeline = vd.pipeline(surface_vertex, tinted_fragment, targets=_targets)

grouped_tint_pipeline = vd.pipeline(surface_vertex, grouped_tint_fragment, targets=_targets)

shape_pipeline = vd.pipeline(shape_vertex, tinted_fragment, targets=_targets)

blended_pipeline = vd.pipeline(
    surface_vertex,
    tinted_fragment,
    state=vd.graphics_state(
        color_blends={
            0: vd.ColorBlendState(
                enabled=True,
                source_color=vd.BlendFactor.SOURCE_ALPHA,
                destination_color=vd.BlendFactor.ONE_MINUS_SOURCE_ALPHA,
                source_alpha=vd.BlendFactor.ONE,
                destination_alpha=vd.BlendFactor.ONE_MINUS_SOURCE_ALPHA,
            )
        }
    ),
    targets=_targets,
)


@vd.fragment
def sampled_fragment(
    coordinate: Annotated[vd.Vector[vd.f32, 4], vd.builtin("frag_coord")],
    source: vd.Texture["2d", vd.f32],  # noqa: F722
    source_sampler: vd.Sampler,
) -> vd.Vector[vd.f32, 4]:
    return vd.texture_sample(source, source_sampler, vd.Vector([coordinate.x / 32.0, coordinate.y / 32.0]))


packed_tint_pipeline = vd.pipeline(surface_vertex, packed_tint_fragment, targets=_targets)

sampled_pipeline = vd.pipeline(surface_vertex, sampled_fragment, targets=_targets)

float_target_pipeline = vd.pipeline(
    surface_vertex, tinted_fragment, targets=vd.target_formats(colors={0: vd.rgba16_float})
)


@vd.kernel(workgroup_size=(1, 1, 1))
def mirror_vertices(
    vertices: vd.TensorView[vd.f32, (3, 2), vd.read_write],
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    index = gid[0]
    vertices[index, 0] = -vertices[index, 0]


@vd.kernel(workgroup_size=(1, 1, 1))
def clear_scratch(
    scratch: vd.TensorView[vd.f32, (3, 2), vd.write],
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    index = gid[0]
    scratch[index, 0] = 0.0
    scratch[index, 1] = 0.0


class OpaquePass(vd.Module):
    def forward(
        self,
        vertices: vd.TensorView[vd.f32, (3, 2), vd.read],
        tint: vd.Vector[vd.f32, 4],
        render_pass: vd.RenderPass,
        draw: vd.DrawCommand,
        dynamic_state: vd.DynamicState,
    ) -> Any:
        opaque_pipeline(vertices=vertices, tint=tint, render_pass=render_pass, draw=draw, dynamic_state=dynamic_state)
        return vd.color_output(render_pass)


class GroupedTintPass(vd.Module):
    """`vertices` alone fills group 0 and `tint` fills group 1, so a set binding either one covers a whole group."""

    def forward(
        self,
        vertices: vd.TensorView[vd.f32, (3, 2), vd.read],
        tint: Tint,
        render_pass: vd.RenderPass,
        draw: vd.DrawCommand,
        dynamic_state: vd.DynamicState,
    ) -> Any:
        grouped_tint_pipeline(
            vertices=vertices, tint=tint, render_pass=render_pass, draw=draw, dynamic_state=dynamic_state
        )
        return vd.color_output(render_pass)


class VertexGroupPass(vd.Module):
    """`shape` fills group 1 with its vertex buffer and offset, and `tint` alone fills group 0."""

    def forward(
        self,
        shape: Shape,
        tint: vd.Vector[vd.f32, 4],
        render_pass: vd.RenderPass,
        draw: vd.DrawCommand,
        dynamic_state: vd.DynamicState,
    ) -> Any:
        shape_pipeline(shape=shape, tint=tint, render_pass=render_pass, draw=draw, dynamic_state=dynamic_state)
        return vd.color_output(render_pass)


class BlendedSurface(vd.Module):
    def forward(
        self,
        vertices: vd.TensorView[vd.f32, (3, 2), vd.read],
        tint: vd.Vector[vd.f32, 4],
        render_pass: vd.RenderPass,
        draw: vd.DrawCommand,
        dynamic_state: vd.DynamicState,
    ) -> Any:
        blended_pipeline(vertices=vertices, tint=tint, render_pass=render_pass, draw=draw, dynamic_state=dynamic_state)
        return vd.color_output(render_pass)


class PackedTintSurface(vd.Module):
    def forward(
        self,
        vertices: vd.TensorView[vd.f32, (3, 2), vd.read],
        tint: vd.Vector[vd.u32, 4],
        render_pass: vd.RenderPass,
        draw: vd.DrawCommand,
        dynamic_state: vd.DynamicState,
    ) -> Any:
        packed_tint_pipeline(
            vertices=vertices, tint=tint, render_pass=render_pass, draw=draw, dynamic_state=dynamic_state
        )
        return vd.color_output(render_pass)


class FloatTargetSurface(vd.Module):
    def forward(
        self,
        vertices: vd.TensorView[vd.f32, (3, 2), vd.read],
        tint: vd.Vector[vd.f32, 4],
        render_pass: vd.RenderPass,
        draw: vd.DrawCommand,
        dynamic_state: vd.DynamicState,
    ) -> Any:
        float_target_pipeline(
            vertices=vertices, tint=tint, render_pass=render_pass, draw=draw, dynamic_state=dynamic_state
        )
        return vd.color_output(render_pass)


class SampledPass(vd.Module):
    """Copies `source` texel by texel into a 32 x 32 color target."""

    def forward(
        self,
        vertices: vd.TensorView[vd.f32, (3, 2), vd.read],
        source: vd.Texture["2d", vd.f32],  # noqa: F722
        source_sampler: vd.Sampler,
        render_pass: vd.RenderPass,
        draw: vd.DrawCommand,
        dynamic_state: vd.DynamicState,
    ) -> Any:
        sampled_pipeline(
            vertices=vertices,
            source=source,
            source_sampler=source_sampler,
            render_pass=render_pass,
            draw=draw,
            dynamic_state=dynamic_state,
        )
        return vd.color_output(render_pass)


class MirroredVerticesPass(vd.Module):
    """A compute Node rewrites the caller's `vertices`, so the pass reads them as a fixed endpoint."""

    def forward(
        self,
        vertices: vd.TensorView[vd.f32, (3, 2), vd.read_write],
        tint: vd.Vector[vd.f32, 4],
        render_pass: vd.RenderPass,
        draw: vd.DrawCommand,
        dynamic_state: vd.DynamicState,
    ) -> Any:
        mirror_vertices(vertices, grid=(3, 1, 1))
        opaque_pipeline(vertices=vertices, tint=tint, render_pass=render_pass, draw=draw, dynamic_state=dynamic_state)
        return vd.color_output(render_pass)


class ScratchWritingPass(vd.Module):
    """`vertices` stays per-draw while a compute Node of the same invocation writes `scratch`."""

    def forward(
        self,
        vertices: vd.TensorView[vd.f32, (3, 2), vd.read],
        scratch: vd.TensorView[vd.f32, (3, 2), vd.write],
        tint: vd.Vector[vd.f32, 4],
        render_pass: vd.RenderPass,
        draw: vd.DrawCommand,
        dynamic_state: vd.DynamicState,
    ) -> Any:
        clear_scratch(scratch, grid=(3, 1, 1))
        opaque_pipeline(vertices=vertices, tint=tint, render_pass=render_pass, draw=draw, dynamic_state=dynamic_state)
        return vd.color_output(render_pass)


opaque_pass_asset = vd.program_asset(id="runtime/recorded-graphics-opaque-pass", program=OpaquePass())

grouped_tint_pass_asset = vd.program_asset(id="runtime/recorded-graphics-grouped-tint-pass", program=GroupedTintPass())

vertex_group_pass_asset = vd.program_asset(id="runtime/recorded-graphics-vertex-group-pass", program=VertexGroupPass())

blended_surface_asset = vd.program_asset(id="runtime/recorded-graphics-blended-surface", program=BlendedSurface())

packed_tint_surface_asset = vd.program_asset(
    id="runtime/recorded-graphics-packed-tint-surface", program=PackedTintSurface()
)

float_target_surface_asset = vd.program_asset(
    id="runtime/recorded-graphics-float-target-surface", program=FloatTargetSurface()
)

sampled_pass_asset = vd.program_asset(id="runtime/recorded-graphics-sampled-pass", program=SampledPass())

mirrored_vertices_pass_asset = vd.program_asset(
    id="runtime/recorded-graphics-mirrored-vertices-pass", program=MirroredVerticesPass()
)

scratch_writing_pass_asset = vd.program_asset(
    id="runtime/recorded-graphics-scratch-writing-pass", program=ScratchWritingPass()
)
