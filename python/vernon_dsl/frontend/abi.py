from __future__ import annotations

import importlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias

from ..language.scalar_types import SCALAR_TYPES
from .model import ConcreteType

AbiPathComponent: TypeAlias = str | int
StructFields: TypeAlias = tuple[tuple[str, ConcreteType], ...]


@dataclass(frozen=True)
class ValueLeaf:
    path: tuple[AbiPathComponent, ...]
    dtype: str
    scalar_count: int
    shape: tuple[int, ...] = ()


@dataclass(frozen=True)
class AttributeLeaf:
    location_offset: int
    path: tuple[AbiPathComponent, ...]
    dtype: str
    component_count: int


@dataclass(frozen=True)
class AttributeLayout:
    leaves: tuple[AttributeLeaf, ...]

    @property
    def location_span(self) -> int:
        return len(self.leaves)


_ATTRIBUTE_DTYPES = {"i32", "u32", "f16", "f32", "f64"}


def _scalar_bytes(dtype: str) -> int:
    scalar = SCALAR_TYPES.get(dtype)
    if scalar is None:
        raise ValueError(f"{dtype} is not an ABI-stable scalar Value")
    return max(scalar.width // 8, 1)


def _native_value_abi_nodes(
    value_type: ConcreteType,
    struct_fields: Callable[[str], StructFields],
) -> tuple[tuple[int, int, tuple[int, ...], int | None], ...]:
    declarations: dict[str, StructFields] = {}

    def collect(current: ConcreteType, active: frozenset[str] = frozenset()) -> None:
        if current.kind == "struct":
            if current.name in active:
                raise ValueError(f"recursive Struct '{current.name}' has no finite Value ABI")
            fields = struct_fields(current.name)
            previous = declarations.setdefault(current.name, fields)
            if previous != fields:
                raise ValueError(f"conflicting Struct declarations named '{current.name}'")
            for _, field in fields:
                collect(field, active | {current.name})
            return
        for argument in current.arguments:
            if isinstance(argument, ConcreteType):
                collect(argument, active)

    collect(value_type)
    declaration_lines = []
    for name, fields in sorted(declarations.items()):
        field_text = ", ".join(json.dumps(f"{field_name}:{field_type.mlir}") for field_name, field_type in fields)
        declaration_lines.append(
            f'  "vernon.struct"() {{fields = [{field_text}], sym_name = {json.dumps(name)}}} : () -> ()'
        )
    module = "\n".join(
        [
            "module {",
            *declaration_lines,
            f"  func.func private @__vernon_plan_value_abi(%value: {value_type.mlir})",
            "}",
        ]
    )
    logical_dtypes = tuple(leaf.dtype for leaf in value_leaves(value_type, struct_fields))
    native = importlib.import_module("vernon_dsl._native")
    nodes = tuple(
        (
            int(size),
            int(alignment),
            tuple(int(offset) for offset in offsets),
            None if element_stride is None else int(element_stride),
        )
        for size, alignment, offsets, element_stride in native._plan_value_abi(module, logical_dtypes)
    )
    if not nodes:
        raise ValueError(f"{value_type.mlir} has no finite canonical Value ABI")
    return nodes


def value_abi_extent(
    value_type: ConcreteType,
    struct_fields: Callable[[str], StructFields],
) -> tuple[int, int]:
    nodes = _native_value_abi_nodes(value_type, struct_fields)
    size, alignment, _, _ = nodes[0]
    return size, alignment


def workgroup_physical_bytes(
    element: ConcreteType,
    shape: tuple[int, ...],
    struct_fields: Callable[[str], StructFields],
) -> int:
    nodes = _native_value_abi_nodes(element, struct_fields)

    def scalar_lanes(index: int) -> tuple[tuple[tuple[int, int], ...], int]:
        size, _, child_offsets, element_stride = nodes[index]
        if element_stride is not None:
            lanes, next_index = scalar_lanes(index + 1)
            count = size // element_stride
            return tuple((scalar_size, scalar_count * count) for scalar_size, scalar_count in lanes), next_index
        if child_offsets:
            lanes: list[tuple[int, int]] = []
            next_index = index + 1
            for _ in child_offsets:
                child_lanes, next_index = scalar_lanes(next_index)
                lanes.extend(child_lanes)
            return tuple(lanes), next_index
        return ((size, 1),), index + 1

    lanes, next_index = scalar_lanes(0)
    if next_index != len(nodes):
        raise ValueError(f"{element.mlir} produced an invalid canonical Value ABI tree")
    records = 1
    for extent in shape:
        records *= extent
    return sum(((records * count * size + 15) // 16) * 16 for size, count in lanes)


def attribute_layout(
    value_type: ConcreteType,
    struct_fields: Callable[[str], StructFields],
) -> AttributeLayout:
    leaves: list[AttributeLeaf] = []
    for value_leaf in value_leaves(value_type, struct_fields):
        if value_leaf.dtype not in _ATTRIBUTE_DTYPES:
            path = _format_path(value_leaf.path)
            raise ValueError(f"{value_leaf.dtype} Value leaf '{path}' is not a numeric vertex attribute dtype")
        scalar_size = _scalar_bytes(value_leaf.dtype)
        component_limit = min(4, 16 // scalar_size)
        for scalar_index in range(0, value_leaf.scalar_count, component_limit):
            leaves.append(
                AttributeLeaf(
                    location_offset=len(leaves),
                    path=value_leaf.path,
                    dtype=value_leaf.dtype,
                    component_count=min(component_limit, value_leaf.scalar_count - scalar_index),
                )
            )
    return AttributeLayout(tuple(leaves))


def value_leaves(
    value_type: ConcreteType,
    struct_fields: Callable[[str], StructFields],
    active_structs: frozenset[str] = frozenset(),
) -> tuple[ValueLeaf, ...]:
    """Return logical leaves only; native code exclusively plans Value ABI bytes."""
    return _value_leaves(value_type, struct_fields, active_structs, ())


def _value_leaves(
    value_type: ConcreteType,
    struct_fields: Callable[[str], StructFields],
    active_structs: frozenset[str],
    path: tuple[AbiPathComponent, ...],
) -> tuple[ValueLeaf, ...]:
    if value_type.kind == "scalar":
        _scalar_bytes(value_type.name)
        return (ValueLeaf(path, value_type.name, 1),)
    if value_type.kind == "tensor":
        element = value_type.arguments[0]
        if not isinstance(element, ConcreteType):
            raise ValueError("Tensor element type is not concrete")
        dimensions = value_type.arguments[1:]
        if any(not isinstance(extent, int) or extent <= 0 for extent in dimensions):
            raise ValueError("Tensor ABI layout requires positive static shape dimensions")
        count = 1
        for dimension in dimensions:
            assert isinstance(dimension, int)
            count *= dimension
        if element.kind == "scalar":
            return (ValueLeaf(path, element.name, count, tuple(dimensions)),)
        element_leaves = _value_leaves(element, struct_fields, active_structs, path)
        return tuple(
            ValueLeaf(
                (
                    *path,
                    *_linear_index_path(index, dimensions),
                    *leaf.path[len(path) :],
                ),
                leaf.dtype,
                leaf.scalar_count,
                leaf.shape,
            )
            for index in range(count)
            for leaf in element_leaves
        )
    if value_type.kind == "tuple":
        elements = tuple(element for element in value_type.arguments if isinstance(element, ConcreteType))
        if len(elements) != len(value_type.arguments):
            raise ValueError("Tuple element type is not concrete")
        return tuple(
            leaf
            for index, element in enumerate(elements)
            for leaf in _value_leaves(element, struct_fields, active_structs, (*path, index))
        )
    if value_type.kind == "struct":
        if value_type.name in active_structs:
            raise ValueError(f"recursive Struct '{value_type.name}' has no finite ABI layout")
        fields = struct_fields(value_type.name)
        if any(not isinstance(name, str) or not isinstance(field, ConcreteType) for name, field in fields):
            raise ValueError(f"Struct '{value_type.name}' has unresolved fields")
        return tuple(
            leaf
            for name, field in fields
            for leaf in _value_leaves(
                field,
                struct_fields,
                active_structs | {value_type.name},
                (*path, name),
            )
        )
    raise ValueError(f"{value_type.name} is not an ABI-stable Value")


def _linear_index_path(index: int, dimensions: tuple[ConcreteType | int | str, ...]) -> tuple[int, ...]:
    coordinates = [0] * len(dimensions)
    for dimension in range(len(dimensions) - 1, -1, -1):
        extent = dimensions[dimension]
        assert isinstance(extent, int)
        coordinates[dimension] = index % extent
        index //= extent
    return tuple(coordinates)


def _format_path(path: tuple[AbiPathComponent, ...]) -> str:
    if not path:
        return "<value>"
    result = ""
    for component in path:
        if isinstance(component, int):
            result += f"[{component}]"
        else:
            result += ("." if result else "") + component
    return result
