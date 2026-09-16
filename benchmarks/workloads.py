from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any

import vernon_dsl as vd


@vd.kernel(workgroup_size=(64, 1, 1))
def increment(
    output: vd.TensorView[vd.f32, (vd.dyn,), vd.write],
    source: vd.TensorView[vd.f32, (vd.dyn,), vd.read],
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    output[gid[0]] = source[gid[0]] + 1.0


@vd.kernel(workgroup_size=(64, 1, 1))
def add_parameter(
    output: vd.TensorView[vd.f32, (vd.dyn,), vd.write],
    source: vd.TensorView[vd.f32, (vd.dyn,), vd.read],
    amount: vd.f32,
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    output[gid[0]] = source[gid[0]] + amount


@vd.kernel(workgroup_size=(1, 1, 1))
def square_kernel(
    output: vd.TensorView[vd.f32, (1,), vd.write],
    source: vd.TensorView[vd.f32, (1,), vd.read],
) -> None:
    output[0] = source[0] * source[0]


@vd.kernel(workgroup_size=(1, 1, 1))
def empty_kernel() -> None:
    return


@vd.kernel(workgroup_size=(64, 1, 1))
def elementwise_kernel(
    output: vd.TensorView[vd.f32, (1048576,), vd.write],
    source: vd.TensorView[vd.f32, (1048576,), vd.read],
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    output[gid[0]] = source[gid[0]] * 2.0 + 1.0


@vd.kernel(workgroup_size=(1, 1, 1))
def reduction_kernel(
    output: vd.TensorView[vd.f32, (1,), vd.write],
    source: vd.TensorView[vd.f32, (1048576,), vd.read],
) -> None:
    total = vd.f32(0.0)
    for index in range(1048576):
        total = total + source[index]
    output[0] = total


class PersistentlyBoundAdd(vd.Module):
    def forward(
        self,
        output: vd.TensorStorage[vd.f32, (1024,), vd.write],
        source: vd.TensorStorage[vd.f32, (1024,), vd.read],
        amount: vd.f32,
        groups: vd.u32,
    ) -> vd.TensorStorage:
        add_parameter(output, source, amount, grid=(groups, 1, 1))
        return output


class DynamicIncrement(vd.Module):
    def forward(self, source: vd.TensorStorage[vd.f32], groups: vd.u32) -> vd.TensorStorage:
        output = vd.empty_like(source)
        increment(output, source, grid=(groups, 1, 1))
        return output


class Square(vd.Module):
    def forward(
        self,
        source: vd.TensorStorage[vd.f32, (1,), vd.read],
    ) -> vd.TensorStorage:
        output = vd.empty_like(source)
        square_kernel(output, source, grid=(1, 1, 1))
        return output


class IncrementChain(vd.Module):
    def __init__(self, node_count: int):
        super().__init__()
        if node_count < 1:
            raise ValueError("node_count must be positive")
        self.node_count = node_count

    def forward(self, source: vd.TensorStorage[vd.f32, (1024,), vd.read], groups: vd.u32) -> vd.TensorStorage:
        value = source
        for _ in range(self.node_count):
            output = vd.empty_like(source)
            increment(output, value, grid=(groups, 1, 1))
            value = output
        return value


@vd.vertex
def control_vertex(
    position: Annotated[vd.Vector[vd.f32, 2], vd.attribute()],
) -> Annotated[vd.Vector[vd.f32, 4], vd.builtin("position")]:
    return vd.Vector([position, 0.0, 1.0])


@vd.fragment
def control_fragment() -> vd.Vector[vd.f32, 4]:
    return vd.Vector([1.0, 0.25, 0.0, 1.0])


control_pipeline = vd.pipeline(
    control_vertex,
    control_fragment,
    targets=vd.target_formats(colors={0: vd.rgba8_unorm}),
)


class ControlDraw(vd.Module):
    def forward(
        self,
        vertices: vd.TensorStorage[vd.f32, (3, 2), vd.read],
        render_pass: vd.RenderPass,
        draw: vd.DrawCommand,
        dynamic_state: vd.DynamicState,
    ) -> Any:
        control_pipeline(
            position=vertices,
            render_pass=render_pass,
            draw=draw,
            dynamic_state=dynamic_state,
        )
        return vd.color_output(render_pass)


class RepeatedDraw(vd.Module):
    def __init__(self, draw_count: int):
        super().__init__()
        self.draw_count = draw_count

    def forward(
        self,
        vertices: vd.TensorStorage[vd.f32, (3, 2), vd.read],
        render_pass: vd.RenderPass,
        draw: vd.DrawCommand,
        dynamic_state: vd.DynamicState,
    ) -> Any:
        for _ in range(self.draw_count):
            control_pipeline(
                position=vertices,
                render_pass=render_pass,
                draw=draw,
                dynamic_state=dynamic_state,
            )
        return vd.color_output(render_pass)


@dataclass(frozen=True)
class WorkloadAsset:
    name: str
    asset: object


binding_asset = vd.program_asset(id="benchmark/binding", program=PersistentlyBoundAdd())
dynamic_binding_asset = vd.program_asset(id="benchmark/binding-dynamic", program=increment)
square_asset = vd.program_asset(id="benchmark/program-square", program=Square())
chain_1_asset = vd.program_asset(id="benchmark/program-chain-1", program=IncrementChain(1))
chain_8_asset = vd.program_asset(id="benchmark/program-chain-8", program=IncrementChain(8))
chain_64_asset = vd.program_asset(id="benchmark/program-chain-64", program=IncrementChain(64))
control_asset = vd.program_asset(id="benchmark/binding-control", program=ControlDraw())
empty_kernel_asset = vd.program_asset(id="benchmark/backend-empty", program=empty_kernel)
elementwise_asset = vd.program_asset(id="benchmark/backend-elementwise", program=elementwise_kernel)
reduction_asset = vd.program_asset(id="benchmark/backend-reduction", program=reduction_kernel)
draw_1_asset = vd.program_asset(id="benchmark/graphics-draw-1", program=RepeatedDraw(1))
draw_64_asset = vd.program_asset(id="benchmark/graphics-draw-64", program=RepeatedDraw(64))
