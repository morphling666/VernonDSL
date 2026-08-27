"""Dtype maps and TensorView helpers for generated Program operators."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from ..resources import TensorStorage, TensorView


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


def scalar_name(value: TensorStorage | TensorView) -> str | None:
    return _SCALAR_BY_DTYPE.get(np.dtype(value.dtype))


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


def as_view(value: TensorStorage | TensorView, access: str) -> TensorView:
    if isinstance(value, TensorStorage):
        return value._full_view(access)
    return value._with_access(access)


__all__ = [
    "ImplementationUnavailable",
    "as_view",
    "python_element_annotation",
    "scalar_name",
]
