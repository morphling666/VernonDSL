"""Backend-neutral implementation mapping for semantic Program operations."""

from __future__ import annotations

import importlib.util
import math
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .resources import TensorStorage, TensorView


class ImplementationUnavailable(Exception):
    pass


_SCALAR_BY_DTYPE = {
    np.dtype(np.bool_): "bool",
    np.dtype(np.int32): "i32",
    np.dtype(np.uint32): "u32",
    np.dtype(np.float16): "f16",
    np.dtype(np.float32): "f32",
    np.dtype(np.float64): "f64",
}

_generated_add: dict[tuple[str, int], Any] = {}
_generated_directories: list[tempfile.TemporaryDirectory[str]] = []


def _scalar_name(value: TensorStorage | TensorView) -> str | None:
    return _SCALAR_BY_DTYPE.get(np.dtype(value.dtype))


def _view_shape_annotation(rank: int) -> str:
    if rank == 0:
        return "()"
    if rank == 1:
        return "(vd.dyn,)"
    return "(" + ", ".join(["vd.dyn"] * rank) + ")"


def _linear_index_body(rank: int) -> list[str]:
    if rank == 1:
        return [
            "    linear = gid[0]",
            "    output[linear] = left[linear] + right[linear]",
        ]
    names = [f"index{axis}" for axis in range(rank)]
    lines = [
        "    linear = gid[0]",
        "    shape = output.shape",
    ]
    for axis in range(rank - 1, 0, -1):
        lines.append(f"    {names[axis]} = linear % shape[{axis}]")
        lines.append(f"    linear = linear // shape[{axis}]")
    lines.append(f"    {names[0]} = linear")
    index = ", ".join(names)
    lines.append(f"    output[{index}] = left[{index}] + right[{index}]")
    return lines


def _add_kernel_source(dtype: str, rank: int) -> tuple[str, str]:
    entry = f"_program_add_{dtype}_rank{rank}"
    view = f"vd.TensorView[vd.{dtype}, {_view_shape_annotation(rank)}"
    parameters = [
        f"    output: {view}, vd.write],",
        f"    left: {view}, vd.read],",
        f"    right: {view}, vd.read],",
    ]
    if rank == 0:
        body = ["    output[()] = left[()] + right[()]"]
    else:
        parameters.append('    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],')
        body = _linear_index_body(rank)
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


def _kernel_for(dtype: str, rank: int) -> Any:
    cached = _generated_add.get((dtype, rank))
    if cached is not None:
        return cached
    entry, source = _add_kernel_source(dtype, rank)
    directory = tempfile.TemporaryDirectory(prefix="vernon-add-")
    path = Path(directory.name) / f"{entry}.py"
    path.write_text(source, encoding="utf-8")
    module_name = f"vernon_dsl._runtime._generated_add_{dtype}_rank{rank}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load generated add kernel {entry}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    kernel = getattr(module, entry)
    _generated_directories.append(directory)
    _generated_add[(dtype, rank)] = kernel
    return kernel


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
    dtypes = {_scalar_name(output), _scalar_name(left), _scalar_name(right)}
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
        _kernel_for(dtype, len(shape)),
        (_view(output, "write"), _view(left, "read"), _view(right, "read")),
        (numel, 1, 1),
    )


__all__ = ["ImplementationUnavailable", "program_add_invocation"]
