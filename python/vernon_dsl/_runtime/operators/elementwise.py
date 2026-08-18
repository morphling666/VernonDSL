from __future__ import annotations

from typing import Annotated, Any

import numpy as np

import vernon_dsl as vd

from .. import session as state
from ..resources import TensorStorage, TensorView

_WORKGROUP_SIZE = 64


class OperatorLoweringUnavailable(Exception):
    pass


@vd.kernel(workgroup_size=(1, 1, 1))
def _add_f32_rank0(
    output: vd.TensorView[vd.f32, (), vd.write],
    left: vd.TensorView[vd.f32, (), vd.read],
    right: vd.TensorView[vd.f32, (), vd.read],
) -> None:
    output[()] = left[()] + right[()]


@vd.kernel(workgroup_size=(_WORKGROUP_SIZE, 1, 1))
def _add_f32_rank1(
    output: vd.TensorView[vd.f32, (vd.dyn,), vd.write],
    left: vd.TensorView[vd.f32, (vd.dyn,), vd.read],
    right: vd.TensorView[vd.f32, (vd.dyn,), vd.read],
    count: vd.u32,
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    linear = gid[0]
    if linear < count:
        output[linear] = left[linear] + right[linear]


@vd.kernel(workgroup_size=(_WORKGROUP_SIZE, 1, 1))
def _add_f32_rank2(
    output: vd.TensorView[vd.f32, (vd.dyn, vd.dyn), vd.write],
    left: vd.TensorView[vd.f32, (vd.dyn, vd.dyn), vd.read],
    right: vd.TensorView[vd.f32, (vd.dyn, vd.dyn), vd.read],
    width: vd.u32,
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    column = gid[0]
    if column < width:
        row = gid[1]
        output[row, column] = left[row, column] + right[row, column]


@vd.kernel(workgroup_size=(_WORKGROUP_SIZE, 1, 1))
def _add_f32_rank3(
    output: vd.TensorView[vd.f32, (vd.dyn, vd.dyn, vd.dyn), vd.write],
    left: vd.TensorView[vd.f32, (vd.dyn, vd.dyn, vd.dyn), vd.read],
    right: vd.TensorView[vd.f32, (vd.dyn, vd.dyn, vd.dyn), vd.read],
    width: vd.u32,
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    column = gid[0]
    if column < width:
        row = gid[1]
        depth = gid[2]
        output[depth, row, column] = left[depth, row, column] + right[depth, row, column]


def _view(value: TensorStorage | TensorView, access: str) -> TensorView:
    if isinstance(value, TensorStorage):
        return value._full_view(access)
    return value._with_access(access)


def _add_invocation(
    output: TensorStorage | TensorView,
    left: TensorStorage | TensorView,
    right: TensorStorage | TensorView,
) -> tuple[Any, tuple[Any, ...], tuple[int, int, int]]:
    shape = tuple(left.shape)
    if state._architecture == state.cpu:
        raise OperatorLoweringUnavailable("the active runtime is CPU")
    if tuple(right.shape) != shape or tuple(output.shape) != shape:
        raise OperatorLoweringUnavailable("elementwise Add shapes do not match")
    if (
        np.dtype(left.dtype) != np.dtype(np.float32)
        or np.dtype(right.dtype) != np.dtype(np.float32)
        or np.dtype(output.dtype) != np.dtype(np.float32)
    ):
        raise OperatorLoweringUnavailable("device elementwise Add currently supports f32")
    if len(shape) > 3:
        raise OperatorLoweringUnavailable("device elementwise Add currently supports ranks zero through three")
    count = int(np.prod(shape, dtype=np.uint64))
    if not 0 < count <= np.iinfo(np.uint32).max:
        raise OperatorLoweringUnavailable("elementwise Add iteration count exceeds u32")
    output_view = _view(output, "write")
    left_view = _view(left, "read")
    right_view = _view(right, "read")
    arguments: tuple[Any, ...]
    if len(shape) == 0:
        grid = (1, 1, 1)
        arguments = (output_view, left_view, right_view)
        kernel = _add_f32_rank0
    elif len(shape) == 1:
        grid = ((count + _WORKGROUP_SIZE - 1) // _WORKGROUP_SIZE, 1, 1)
        arguments = (output_view, left_view, right_view, count)
        kernel = _add_f32_rank1
    elif len(shape) == 2:
        grid = ((shape[1] + _WORKGROUP_SIZE - 1) // _WORKGROUP_SIZE, shape[0], 1)
        arguments = (output_view, left_view, right_view, shape[1])
        kernel = _add_f32_rank2
    else:
        grid = ((shape[2] + _WORKGROUP_SIZE - 1) // _WORKGROUP_SIZE, shape[1], shape[0])
        arguments = (output_view, left_view, right_view, shape[2])
        kernel = _add_f32_rank3
    return kernel, arguments, grid


def append_add(
    operator_dag: Any | None,
    output: TensorStorage | TensorView,
    left: TensorStorage | TensorView,
    right: TensorStorage | TensorView,
) -> Any:
    kernel, arguments, grid = _add_invocation(output, left, right)
    operator_kernel: Any = kernel
    return operator_kernel._append_operator(
        operator_dag,
        *arguments,
        grid=grid,
        append=lambda dag, invocation: dag.add_elementwise_add(invocation, "output", "left", "right"),
    )
