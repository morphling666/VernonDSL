from __future__ import annotations

from typing import Annotated, Any

import vernon_dsl as vd


@vd.vertex
def module_vertex(
    vertices: Annotated[vd.Vector[vd.f32, 2], vd.attribute()],
) -> Annotated[vd.Vector[vd.f32, 4], vd.builtin("position")]:
    return vd.Vector([vertices, 0.0, 1.0])


@vd.fragment
def module_fragment() -> vd.Vector[vd.f32, 4]:
    return vd.Vector([1.0, 0.25, 0.0, 1.0])


module_pipeline = vd.pipeline(
    module_vertex,
    module_fragment,
    targets=vd.target_formats(colors={0: vd.rgba8_unorm}),
)


@vd.kernel(workgroup_size=(1, 1, 1))
def generate_vertices(
    vertices: vd.TensorView[vd.f32, (3, 2), vd.write],
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    index = gid[0]
    vertices[index, 0] = vd.f32(index) - 1.0
    vertices[index, 1] = -0.75 if index < 2 else 0.75


class ExternalVerticesGraphics(vd.Module):
    def forward(
        self,
        vertices: vd.TensorView[vd.f32, (3, 2), vd.read],
        render_pass: vd.RenderPass,
        draw: vd.DrawCommand,
        dynamic_state: vd.DynamicState,
    ) -> Any:
        module_pipeline(
            vertices=vertices,
            render_pass=render_pass,
            draw=draw,
            dynamic_state=dynamic_state,
        )
        return vd.color_output(render_pass)


class ComputeGeneratedGraphics(vd.Module):
    def forward(
        self,
        grid_x: vd.u32,
        grid_y: vd.u32,
        grid_z: vd.u32,
        render_pass: vd.RenderPass,
        draw: vd.DrawCommand,
        dynamic_state: vd.DynamicState,
    ) -> Any:
        vertices = vd.empty(dtype=vd.f32, shape=(3, 2))
        generate_vertices(vertices, grid=(grid_x, grid_y, grid_z))
        module_pipeline(
            vertices=vertices,
            render_pass=render_pass,
            draw=draw,
            dynamic_state=dynamic_state,
        )
        return vd.color_output(render_pass)


graphics_asset = vd.program_asset(
    id="runtime/module-graphics-external-vertices",
    program=ExternalVerticesGraphics(),
)

mixed_asset = vd.program_asset(
    id="runtime/module-compute-generated-graphics",
    program=ComputeGeneratedGraphics(),
)

graphics_vjp_asset = vd.program_asset(
    id="runtime/module-graphics-vjp-unsupported",
    program=vd.ad.vjp(ExternalVerticesGraphics(), wrt=("vertices",)),
)
