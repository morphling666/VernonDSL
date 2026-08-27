"""Linearized copy / add kernels for Program builtins."""

from __future__ import annotations

import math
from typing import Any

from ..resources import TensorStorage, TensorView
from ._indexing import element_token, linear_index_prelude, linear_index_target, view_shape_annotation
from ._kernel_cache import load_generated_kernel
from ._types import ImplementationUnavailable, as_view, scalar_name


def _elementwise_kernel_source(operation: str, element: str, rank: int) -> tuple[str, str]:
    token = element_token(element)
    entry = f"_program_{operation}_{token}_rank{rank}"
    view = f"vd.TensorView[{element}, {view_shape_annotation(rank)}"
    if operation == "add":
        parameters = [
            f"    output: {view}, vd.write],",
            f"    left: {view}, vd.read],",
            f"    right: {view}, vd.read],",
        ]
        index = linear_index_target(rank)
        assignment = f"output[{index}] = left[{index}] + right[{index}]"
    elif operation == "copy":
        parameters = [
            f"    output: {view}, vd.write],",
            f"    source: {view}, vd.read],",
        ]
        index = linear_index_target(rank)
        assignment = f"output[{index}] = source[{index}]"
    else:
        raise ImplementationUnavailable(f"unsupported Program elementwise operation {operation!r}")
    if rank == 0:
        body = [f"    {assignment}"]
    else:
        parameters.append('    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],')
        body = [*linear_index_prelude(rank), f"    {assignment}"]
    source = "\n".join(
        [
            "from typing import Annotated",
            "",
            "import vernon_dsl as vd",
            "",
            "@vd.kernel(workgroup_size=(1, 1, 1))",
            f"def {entry}(",
            *parameters,
            ") -> None:",
            *body,
            "",
        ]
    )
    return entry, source


def elementwise_kernel(operation: str, element: str, rank: int) -> Any:
    return load_generated_kernel(
        (operation, element, rank),
        *_elementwise_kernel_source(operation, element, rank),
        operation,
    )


def program_add_invocation(
    output: TensorStorage | TensorView,
    left: TensorStorage | TensorView,
    right: TensorStorage | TensorView,
) -> tuple[Any, tuple[Any, ...], tuple[int, int, int]]:
    shape = tuple(left.shape)
    if tuple(right.shape) != shape or tuple(output.shape) != shape:
        raise ImplementationUnavailable("elementwise Add shapes do not match")
    dtypes = {scalar_name(output), scalar_name(left), scalar_name(right)}
    if len(dtypes) != 1 or None in dtypes:
        raise ImplementationUnavailable("elementwise Add requires a shared scalar dtype")
    dtype = next(iter(dtypes))
    assert dtype is not None
    if any(extent <= 0 for extent in shape):
        raise ImplementationUnavailable("Program Add requires a non-empty static shape")
    numel = 1 if not shape else int(math.prod(shape))
    if numel > 2**31 - 1:
        raise ImplementationUnavailable("Program Add linearized grid exceeds the portable workgroup count")
    return (
        elementwise_kernel("add", f"vd.{dtype}", len(shape)),
        (as_view(output, "write"), as_view(left, "read"), as_view(right, "read")),
        (numel, 1, 1),
    )


__all__ = [
    "elementwise_kernel",
    "program_add_invocation",
]
