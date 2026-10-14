from __future__ import annotations

import vernon_dsl as vd  # pyright: ignore[reportMissingImports]

_STENCIL_REPLACE = vd.StencilFaceState(pass_operation=vd.StencilOperation.REPLACE)
_STATE = vd.graphics_state(
    depth_stencil=vd.DepthStencilState(
        depth_test=True,
        depth_write=True,
        stencil_test=True,
        front=_STENCIL_REPLACE,
        back=_STENCIL_REPLACE,
    )
)
_TARGETS = vd.target_formats(colors={0: vd.rgba8_unorm}, depth=vd.d32_float_s8_uint)


@vd.vertex
def covering_vertex() -> vd.Vector[vd.f32, 4]:
    vertex = vd.vertex_id()
    x = 3.0 if vertex == 1 else -1.0
    y = 3.0 if vertex == 2 else -1.0
    return vd.Vector([x, y, 0.0, 1.0])


@vd.func
def checker_parity(coordinate: vd.Vector[vd.f32, 4]) -> vd.f32:
    cell = vd.floor(coordinate.x) + vd.floor(coordinate.y)
    return cell - 2.0 * vd.floor(cell * 0.5)


@vd.fragment
def checker_discard_fragment() -> vd.Vector[vd.f32, 4]:
    if checker_parity(vd.fragment_coord()) > 0.5:
        vd.discard_fragment()
    return vd.Vector([1.0, 0.25, 0.0, 1.0])


@vd.func
def discard_odd(parity: vd.f32) -> None:
    if parity > 0.5:
        vd.discard_fragment()


@vd.fragment
def loop_and_helper_discard_fragment() -> vd.Vector[vd.f32, 4]:
    coordinate = vd.fragment_coord()
    parity = checker_parity(coordinate)
    if coordinate.x < vd.resolution().x * 0.5:
        for iteration in range(3):
            if parity > 0.5 and iteration == 1:
                vd.discard_fragment()
    else:
        discard_odd(parity)
    return vd.Vector([1.0, 0.25, 0.0, 1.0])


@vd.fragment
def unreachable_after_discard_fragment() -> vd.Vector[vd.f32, 4]:
    color = vd.Vector([1.0, 0.25, 0.0, 1.0])
    if checker_parity(vd.fragment_coord()) > 0.5:
        vd.discard_fragment()
        color = vd.Vector([0.0, 1.0, 1.0, 1.0])
    return color


checker_asset = vd.program_asset(
    id="runtime/fragment-discard-checker",
    program=vd.pipeline(covering_vertex, checker_discard_fragment, state=_STATE, targets=_TARGETS),
)
loop_and_helper_asset = vd.program_asset(
    id="runtime/fragment-discard-loop-and-helper",
    program=vd.pipeline(covering_vertex, loop_and_helper_discard_fragment, state=_STATE, targets=_TARGETS),
)
unreachable_asset = vd.program_asset(
    id="runtime/fragment-discard-unreachable",
    program=vd.pipeline(covering_vertex, unreachable_after_discard_fragment, state=_STATE, targets=_TARGETS),
)
