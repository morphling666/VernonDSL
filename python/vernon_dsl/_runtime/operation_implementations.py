"""Backend-neutral implementation mapping for semantic Program operations."""

from __future__ import annotations

from typing import Annotated, Any

import vernon_dsl as vd

from .resources import TensorStorage, TensorView


class ImplementationUnavailable(Exception):
    pass


@vd.kernel(workgroup_size=(1, 1, 1))
def _add_f32_rank0(
    output: vd.TensorView[vd.f32, (), vd.write],
    left: vd.TensorView[vd.f32, (), vd.read],
    right: vd.TensorView[vd.f32, (), vd.read],
) -> None:
    output[()] = left[()] + right[()]


@vd.kernel(workgroup_size=(1, 1, 1))
def _program_add_f32_rank1(
    output: vd.TensorView[vd.f32, (vd.dyn,), vd.write],
    left: vd.TensorView[vd.f32, (vd.dyn,), vd.read],
    right: vd.TensorView[vd.f32, (vd.dyn,), vd.read],
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    index = gid[0]
    output[index] = left[index] + right[index]


@vd.kernel(workgroup_size=(1, 1, 1))
def _program_add_f32_rank2(
    output: vd.TensorView[vd.f32, (vd.dyn, vd.dyn), vd.write],
    left: vd.TensorView[vd.f32, (vd.dyn, vd.dyn), vd.read],
    right: vd.TensorView[vd.f32, (vd.dyn, vd.dyn), vd.read],
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    output[gid[1], gid[0]] = left[gid[1], gid[0]] + right[gid[1], gid[0]]


@vd.kernel(workgroup_size=(1, 1, 1))
def _program_add_f32_rank3(
    output: vd.TensorView[vd.f32, (vd.dyn, vd.dyn, vd.dyn), vd.write],
    left: vd.TensorView[vd.f32, (vd.dyn, vd.dyn, vd.dyn), vd.read],
    right: vd.TensorView[vd.f32, (vd.dyn, vd.dyn, vd.dyn), vd.read],
    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],
) -> None:
    output[gid[2], gid[1], gid[0]] = left[gid[2], gid[1], gid[0]] + right[gid[2], gid[1], gid[0]]


def _view(value: TensorStorage | TensorView, access: str) -> TensorView:
    if isinstance(value, TensorStorage):
        return value._full_view(access)
    return value._with_access(access)


def program_add_invocation(
    output: TensorStorage | TensorView,
    left: TensorStorage | TensorView,
    right: TensorStorage | TensorView,
) -> tuple[Any, tuple[Any, ...], tuple[int, int, int]]:
    shape = tuple(left.shape)
    if tuple(right.shape) != shape or tuple(output.shape) != shape:
        raise ImplementationUnavailable("elementwise Add shapes do not match")
    if len(shape) == 0:
        return _add_f32_rank0, (_view(output, "write"), _view(left, "read"), _view(right, "read")), (1, 1, 1)
    if len(shape) > 3 or any(extent <= 0 for extent in shape):
        raise ImplementationUnavailable("Program Add requires a non-empty static shape of rank at most three")
    kernels = {1: _program_add_f32_rank1, 2: _program_add_f32_rank2, 3: _program_add_f32_rank3}
    grid = (*reversed(shape), *(1 for _ in range(3 - len(shape))))
    return (
        kernels[len(shape)],
        (_view(output, "write"), _view(left, "read"), _view(right, "read")),
        grid,
    )


__all__ = ["ImplementationUnavailable", "program_add_invocation"]
