"""Dtype maps and TensorView helpers for generated Program operators."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..._dtypes import scalar_name as _scalar_name
from ..._mlir import first_generic_type_argument, generic_type_arguments, ranked_tensor_parts
from ...language.scalar_types import SCALAR_TYPES
from ..tensor import TensorStorage, TensorView


class ImplementationUnavailable(Exception):
    pass


_SCALAR_ANNOTATIONS = {name: f"vd.{name}" for name in SCALAR_TYPES} | {"i1": "vd.bool"}


def scalar_name(value: TensorStorage | TensorView) -> str | None:
    return _scalar_name(value.dtype)


def _python_type_annotation(element: str) -> str | None:
    if element in _SCALAR_ANNOTATIONS:
        return _SCALAR_ANNOTATIONS[element]
    tensor = ranked_tensor_parts(element)
    if tensor is not None:
        shape, dtype = tensor
        if dtype in _SCALAR_ANNOTATIONS and all(extent.isdigit() for extent in shape):
            extents = tuple(int(extent) for extent in shape)
            scalar = _SCALAR_ANNOTATIONS[dtype]
            if len(extents) == 1:
                return f"vd.Vector[{scalar}, {extents[0]}]"
            return f"vd.Tensor[{scalar}, ({', '.join(str(extent) for extent in extents)},)]"
    if element.startswith("tuple<") and element.endswith(">"):
        arguments = generic_type_arguments(element, "tuple")
        members = tuple(_python_type_annotation(argument) for argument in arguments or ())
        if members and all(member is not None for member in members):
            return f"vd.Tuple[{', '.join(member for member in members if member is not None)}]"
    return None


def python_element_annotation(value: Mapping[str, Any]) -> str | None:
    element = program_element_type(value)
    if element:
        annotation = _python_type_annotation(element)
        if annotation is not None:
            return annotation
    dtype = value.get("dtype")
    if isinstance(dtype, str) and dtype in _SCALAR_ANNOTATIONS:
        return _SCALAR_ANNOTATIONS[dtype]
    return None


def program_element_type(value: Mapping[str, Any]) -> str | None:
    return first_generic_type_argument(str(value.get("type") or ""), "!vernon.tensor_view")


def as_view(value: TensorStorage | TensorView, access: str) -> TensorView:
    if isinstance(value, TensorStorage):
        return value._full_view(access)
    return value._with_access(access)


__all__ = [
    "ImplementationUnavailable",
    "as_view",
    "python_element_annotation",
    "program_element_type",
    "scalar_name",
]
