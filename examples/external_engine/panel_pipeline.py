from typing import Annotated

import vernon_dsl as vd


@vd.struct
class PanelVertexOutput:
    position: Annotated[vd.Vector[vd.f32, 4], vd.builtin("position")]
    uv: vd.Vector[vd.f32, 2]


# Normalized-device y = +1 is the target's row 0, and v = 0 the panel image's first row, so the image draws upright.
@vd.vertex
def panel_vertex(
    position: Annotated[vd.Vector[vd.f32, 2], vd.attribute()],
) -> PanelVertexOutput:
    return PanelVertexOutput(
        vd.Vector([position, 0.0, 1.0]),
        vd.Vector([position.x * 0.5 + 0.5, 0.5 - position.y * 0.5]),
    )


@vd.fragment
def panel_fragment(
    uv: Annotated[vd.Vector[vd.f32, 2], vd.varying()],
    panel: Annotated[vd.Texture["2d", vd.f32], vd.resource(set=0, binding=0)],  # noqa: F722, F821
    panel_sampler: Annotated[vd.Sampler, vd.resource(set=0, binding=1)],
) -> vd.Vector[vd.f32, 4]:
    return vd.texture_sample(panel, panel_sampler, uv)


asset = vd.program_asset(
    id="examples/external_engine/panel",
    program=vd.pipeline(
        panel_vertex,
        panel_fragment,
        targets=vd.target_formats(colors={0: vd.rgba8_unorm}),
    ),
    variants=({},),
)
