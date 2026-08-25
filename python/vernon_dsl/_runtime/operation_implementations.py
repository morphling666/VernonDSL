"""Backend-neutral implementation mapping for semantic Program operations."""

from __future__ import annotations

import importlib.util
import math
import sys
import tempfile
from collections.abc import Mapping
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

_SCALAR_ANNOTATIONS = {
    "f16": "vd.f16",
    "f32": "vd.f32",
    "f64": "vd.f64",
    "i32": "vd.i32",
    "u32": "vd.u32",
    "bool": "vd.bool",
    "i1": "vd.bool",
}

_generated_kernels: dict[tuple[str, str, int], Any] = {}
_generated_directories: list[tempfile.TemporaryDirectory[str]] = []


def _scalar_name(value: TensorStorage | TensorView) -> str | None:
    return _SCALAR_BY_DTYPE.get(np.dtype(value.dtype))


def _view_shape_annotation(rank: int) -> str:
    if rank == 0:
        return "()"
    if rank == 1:
        return "(vd.dyn,)"
    return "(" + ", ".join(["vd.dyn"] * rank) + ")"


def _mlir_tensor_view_element(spelling: str) -> str | None:
    marker = "tensor_view<"
    start = spelling.find(marker)
    if start < 0:
        return None
    body = spelling[start + len(marker) :]
    depth = 0
    for index, character in enumerate(body):
        if character == "<":
            depth += 1
        elif character == ">":
            depth -= 1
        elif character == "," and depth == 0:
            return body[:index].strip()
    return None


def python_element_annotation(value: Mapping[str, Any]) -> str | None:
    element = _mlir_tensor_view_element(str(value.get("type") or ""))
    if element in _SCALAR_ANNOTATIONS:
        return _SCALAR_ANNOTATIONS[element]
    if element and element.startswith("tensor<") and element.endswith(">"):
        parts = element[len("tensor<") : -1].split("x")
        if len(parts) >= 2 and parts[-1] in _SCALAR_ANNOTATIONS and all(part.isdigit() for part in parts[:-1]):
            extents = tuple(int(part) for part in parts[:-1])
            scalar = _SCALAR_ANNOTATIONS[parts[-1]]
            if len(extents) == 1:
                return f"vd.Vector[{scalar}, {extents[0]}]"
            return f"vd.Tensor[{scalar}, ({', '.join(str(extent) for extent in extents)},)]"
    dtype = value.get("dtype")
    if isinstance(dtype, str) and dtype in _SCALAR_ANNOTATIONS:
        return _SCALAR_ANNOTATIONS[dtype]
    return None


def _linear_index_target(rank: int) -> str:
    if rank <= 1:
        return "linear" if rank == 1 else "()"
    return ", ".join(f"index{axis}" for axis in range(rank))


def _linear_index_prelude(rank: int) -> list[str]:
    if rank <= 1:
        return ["    linear = gid[0]"] if rank == 1 else []
    names = [f"index{axis}" for axis in range(rank)]
    lines = [
        "    linear = gid[0]",
        "    shape = output.shape",
    ]
    for axis in range(rank - 1, 0, -1):
        lines.append(f"    {names[axis]} = linear % shape[{axis}]")
        lines.append(f"    linear = linear // shape[{axis}]")
    lines.append(f"    {names[0]} = linear")
    return lines


def _elementwise_kernel_source(operation: str, element: str, rank: int) -> tuple[str, str]:
    if element.startswith("vd.") and "[" not in element:
        token = element[3:]
    else:
        token = element.replace("[", "_").replace("]", "_").replace(", ", "x").replace(".", "_")
    entry = f"_program_{operation}_{token}_rank{rank}"
    view = f"vd.TensorView[{element}, {_view_shape_annotation(rank)}"
    if operation == "add":
        parameters = [
            f"    output: {view}, vd.write],",
            f"    left: {view}, vd.read],",
            f"    right: {view}, vd.read],",
        ]
        index = _linear_index_target(rank)
        assignment = f"output[{index}] = left[{index}] + right[{index}]"
    elif operation == "copy":
        parameters = [
            f"    output: {view}, vd.write],",
            f"    source: {view}, vd.read],",
        ]
        index = _linear_index_target(rank)
        assignment = f"output[{index}] = source[{index}]"
    else:
        raise ImplementationUnavailable(f"unsupported Program elementwise operation {operation!r}")
    if rank == 0:
        body = [f"    {assignment}"]
    else:
        parameters.append('    gid: Annotated[vd.Tensor[vd.u32, (3,)], vd.builtin("global_invocation_id")],')
        body = [*_linear_index_prelude(rank), f"    {assignment}"]
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
    cached = _generated_kernels.get((operation, element, rank))
    if cached is not None:
        return cached
    entry, source = _elementwise_kernel_source(operation, element, rank)
    directory = tempfile.TemporaryDirectory(prefix=f"vernon-{operation}-")
    path = Path(directory.name) / f"{entry}.py"
    path.write_text(source, encoding="utf-8")
    module_name = f"vernon_dsl._runtime._generated_{operation}_{abs(hash((element, rank)))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load generated {operation} kernel {entry}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    kernel = getattr(module, entry)
    _generated_directories.append(directory)
    _generated_kernels[(operation, element, rank)] = kernel
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
        elementwise_kernel("add", f"vd.{dtype}", len(shape)),
        (_view(output, "write"), _view(left, "read"), _view(right, "read")),
        (numel, 1, 1),
    )


__all__ = [
    "ImplementationUnavailable",
    "elementwise_kernel",
    "program_add_invocation",
    "python_element_annotation",
]
