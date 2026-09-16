from __future__ import annotations

from typing import Annotated

import vernon_dsl as vd


@vd.kernel(workgroup_size=(1, 1, 1))
def increment(
    source: vd.TensorView[vd.f32, (4,), vd.read],
    output: vd.TensorView[vd.f32, (4,), vd.write],
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    output[gid[0]] = source[gid[0]] + 1.0


class ReusedIncrementStage(vd.Module):
    def forward(
        self,
        source: vd.TensorView[vd.f32, (4,), vd.read],
        grid_x: vd.u32,
        grid_y: vd.u32,
        grid_z: vd.u32,
    ) -> vd.TensorStorage:
        intermediate = vd.empty_like(source)
        output = vd.empty_like(source)
        increment(source, intermediate, grid=(grid_x, grid_y, grid_z))
        increment(intermediate, output, grid=(grid_x, grid_y, grid_z))
        return output


@vd.kernel(workgroup_size=(1, 1, 1))
def produce_scalar(
    source: vd.TensorView[vd.f32, (1,), vd.read],
    intermediate: vd.TensorView[vd.f32, (1,), vd.write],
) -> None:
    intermediate[0] = source[0] + 1.0


@vd.kernel(workgroup_size=(1, 1, 1))
def consume_scalar_view(
    intermediate: vd.TensorView[vd.f32, (1,), vd.read],
    output: vd.TensorView[vd.f32, (1,), vd.write],
) -> None:
    output[0] = intermediate[0] * 2.0


class TensorViewChain(vd.Module):
    def forward(self, source: vd.TensorView[vd.f32, (1,), vd.read]) -> vd.TensorStorage:
        intermediate = vd.empty(dtype=vd.f32, shape=(1,))
        output = vd.empty_like(source)
        produce_scalar(source, intermediate, grid=(1, 1, 1))
        consume_scalar_view(intermediate, output, grid=(1, 1, 1))
        return output


@vd.kernel(workgroup_size=(1, 1, 1))
def scale_dynamic(
    source: vd.TensorView[vd.f32, (vd.dyn,), vd.read],
    output: vd.TensorView[vd.f32, (vd.dyn,), vd.write],
    factor: vd.f32,
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    output[gid[0]] = source[gid[0]] * factor


reused_stage_asset = vd.program_asset(
    id="runtime/module-reused-stage",
    program=ReusedIncrementStage(),
)

tensor_view_chain_asset = vd.program_asset(
    id="runtime/module-tensor-view-chain",
    program=TensorViewChain(),
)

dynamic_shape_grid_asset = vd.program_asset(
    id="runtime/dynamic-shape-grid",
    program=scale_dynamic,
)
