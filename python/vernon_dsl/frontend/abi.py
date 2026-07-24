from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .model import ConcreteType


@dataclass(frozen=True)
class ValueAbiLayout:
    size: int
    alignment: int
    field_offsets: tuple[int, ...] = ()
    element_stride: int | None = None


_SCALAR_LAYOUTS = {
    "bool": ValueAbiLayout(1, 1),
    "i32": ValueAbiLayout(4, 4),
    "u32": ValueAbiLayout(4, 4),
    "f16": ValueAbiLayout(2, 2),
    "f32": ValueAbiLayout(4, 4),
    "f64": ValueAbiLayout(8, 8),
}


def _align_to(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def value_abi_layout(
    value_type: ConcreteType,
    struct_fields: Callable[[str], tuple[ConcreteType, ...]],
    active_structs: frozenset[str] = frozenset(),
) -> ValueAbiLayout:
    if value_type.kind == "scalar":
        return _SCALAR_LAYOUTS[value_type.name]
    if value_type.kind == "tensor":
        element = value_type.arguments[0]
        if not isinstance(element, ConcreteType):
            raise ValueError("Tensor element type is not concrete")
        element_layout = value_abi_layout(element, struct_fields, active_structs)
        stride = _align_to(element_layout.size, element_layout.alignment)
        count = 1
        for dimension in value_type.arguments[1:]:
            if not isinstance(dimension, int) or dimension <= 0:
                raise ValueError("Tensor ABI layout requires a positive static shape")
            count *= dimension
        return ValueAbiLayout(stride * count, element_layout.alignment, element_stride=stride)
    if value_type.kind == "tuple":
        elements = tuple(element for element in value_type.arguments if isinstance(element, ConcreteType))
        if len(elements) != len(value_type.arguments):
            raise ValueError("Tuple element type is not concrete")
        return _product_layout(elements, struct_fields, active_structs)
    if value_type.kind == "struct":
        if value_type.name in active_structs:
            raise ValueError(f"recursive Struct '{value_type.name}' has no finite ABI layout")
        return _product_layout(
            struct_fields(value_type.name),
            struct_fields,
            active_structs | {value_type.name},
        )
    raise ValueError(f"{value_type.name} is not an ABI-stable Value")


def _product_layout(
    fields: tuple[ConcreteType, ...],
    struct_fields: Callable[[str], tuple[ConcreteType, ...]],
    active_structs: frozenset[str],
) -> ValueAbiLayout:
    offset = 0
    alignment = 1
    offsets: list[int] = []
    for field in fields:
        layout = value_abi_layout(field, struct_fields, active_structs)
        offset = _align_to(offset, layout.alignment)
        offsets.append(offset)
        offset += layout.size
        alignment = max(alignment, layout.alignment)
    return ValueAbiLayout(_align_to(offset, alignment), alignment, tuple(offsets))
