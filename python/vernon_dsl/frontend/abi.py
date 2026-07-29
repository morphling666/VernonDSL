from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import TypeAlias

from .model import ConcreteType

AbiPathComponent: TypeAlias = str | int
StructFields: TypeAlias = tuple[tuple[str, ConcreteType], ...]


@dataclass(frozen=True)
class ValueLeaf:
    path: tuple[AbiPathComponent, ...]
    dtype: str
    byte_offset: int
    scalar_count: int
    shape: tuple[int, ...] = ()


@dataclass(frozen=True)
class ValueAbiLayout:
    size: int
    alignment: int
    field_offsets: tuple[int, ...] = ()
    element_stride: int | None = None
    leaves: tuple[ValueLeaf, ...] = dataclass_field(default=())
    _canonical: str = dataclass_field(default="", repr=False, compare=False)

    @property
    def layout_hash(self) -> str:
        dtypes = ",".join(leaf.dtype for leaf in self.leaves)
        return hashlib.sha256(f"{self._canonical}|dtypes={dtypes}".encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class AttributeLeaf:
    location_offset: int
    path: tuple[AbiPathComponent, ...]
    dtype: str
    component_count: int
    byte_offset: int


@dataclass(frozen=True)
class AttributeLayout:
    value_layout: ValueAbiLayout
    leaves: tuple[AttributeLeaf, ...]

    @property
    def location_span(self) -> int:
        return len(self.leaves)


@dataclass(frozen=True)
class _ScalarLayout:
    size: int
    alignment: int


_SCALAR_LAYOUTS = {
    "bool": _ScalarLayout(1, 1),
    "i32": _ScalarLayout(4, 4),
    "u32": _ScalarLayout(4, 4),
    "f16": _ScalarLayout(2, 2),
    "f32": _ScalarLayout(4, 4),
    "f64": _ScalarLayout(8, 8),
}

_ATTRIBUTE_DTYPES = {"i32", "u32", "f16", "f32", "f64"}


def workgroup_physical_bytes(
    element: ConcreteType,
    shape: tuple[int, ...],
    struct_fields: Callable[[str], StructFields],
) -> int:
    layout = value_abi_layout(element, struct_fields)
    records = 1
    for extent in shape:
        records *= extent
    if element.kind == "scalar":
        return records * layout.size
    total = 0
    for leaf in layout.leaves:
        total += records * leaf.scalar_count * _SCALAR_LAYOUTS[leaf.dtype].size
    return total


def attribute_layout(
    value_type: ConcreteType,
    struct_fields: Callable[[str], StructFields],
) -> AttributeLayout:
    value_layout = value_abi_layout(value_type, struct_fields)
    leaves: list[AttributeLeaf] = []
    for value_leaf in value_layout.leaves:
        if value_leaf.dtype not in _ATTRIBUTE_DTYPES:
            path = _format_path(value_leaf.path)
            raise ValueError(f"{value_leaf.dtype} Value leaf '{path}' is not a numeric vertex attribute dtype")
        scalar_size = _SCALAR_LAYOUTS[value_leaf.dtype].size
        component_limit = min(4, 16 // scalar_size)
        for scalar_index in range(0, value_leaf.scalar_count, component_limit):
            leaves.append(
                AttributeLeaf(
                    location_offset=len(leaves),
                    path=value_leaf.path,
                    dtype=value_leaf.dtype,
                    component_count=min(component_limit, value_leaf.scalar_count - scalar_index),
                    byte_offset=value_leaf.byte_offset + scalar_index * scalar_size,
                )
            )
    return AttributeLayout(value_layout, tuple(leaves))


def _align_to(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def value_abi_layout(
    value_type: ConcreteType,
    struct_fields: Callable[[str], StructFields],
    active_structs: frozenset[str] = frozenset(),
) -> ValueAbiLayout:
    return _value_abi_layout(value_type, struct_fields, active_structs, ())


def _value_abi_layout(
    value_type: ConcreteType,
    struct_fields: Callable[[str], StructFields],
    active_structs: frozenset[str],
    path: tuple[AbiPathComponent, ...],
) -> ValueAbiLayout:
    if value_type.kind == "scalar":
        scalar = _SCALAR_LAYOUTS.get(value_type.name)
        if scalar is None:
            raise ValueError(f"{value_type.name} is not an ABI-stable scalar Value")
        # i32 and u32 share signless MLIR storage. Keep the physical spelling
        # canonical here and include the logical leaf dtype separately below.
        physical_name = "i32" if value_type.name == "u32" else value_type.name
        canonical = f"scalar({physical_name},{scalar.size},{scalar.alignment})"
        return ValueAbiLayout(
            scalar.size,
            scalar.alignment,
            leaves=(ValueLeaf(path, value_type.name, 0, 1),),
            _canonical=canonical,
        )
    if value_type.kind == "tensor":
        element = value_type.arguments[0]
        if not isinstance(element, ConcreteType):
            raise ValueError("Tensor element type is not concrete")
        dimensions = value_type.arguments[1:]
        if not dimensions or any(not isinstance(extent, int) or extent <= 0 for extent in dimensions):
            raise ValueError("Tensor ABI layout requires a positive static shape")
        element_layout = _value_abi_layout(element, struct_fields, active_structs, path)
        stride = _align_to(element_layout.size, element_layout.alignment)
        count = 1
        for dimension in dimensions:
            assert isinstance(dimension, int)
            count *= dimension
        if element.kind == "scalar":
            leaves = (ValueLeaf(path, element.name, 0, count, tuple(dimensions)),)
        else:
            leaves = tuple(
                ValueLeaf(
                    (
                        *path,
                        *_linear_index_path(index, dimensions),
                        *leaf.path[len(path) :],
                    ),
                    leaf.dtype,
                    index * stride + leaf.byte_offset,
                    leaf.scalar_count,
                    leaf.shape,
                )
                for index in range(count)
                for leaf in element_layout.leaves
            )
        shape_text = ",".join(str(extent) for extent in dimensions)
        canonical = f"tensor([{shape_text}],stride={stride},size={stride * count},element={element_layout._canonical})"
        return ValueAbiLayout(
            stride * count,
            element_layout.alignment,
            element_stride=stride,
            leaves=leaves,
            _canonical=canonical,
        )
    if value_type.kind == "tuple":
        elements = tuple(element for element in value_type.arguments if isinstance(element, ConcreteType))
        if len(elements) != len(value_type.arguments):
            raise ValueError("Tuple element type is not concrete")
        return _product_layout(
            tuple((index, element) for index, element in enumerate(elements)),
            struct_fields,
            active_structs,
            path,
            "tuple",
        )
    if value_type.kind == "struct":
        if value_type.name in active_structs:
            raise ValueError(f"recursive Struct '{value_type.name}' has no finite ABI layout")
        fields = struct_fields(value_type.name)
        if any(not isinstance(name, str) or not isinstance(field, ConcreteType) for name, field in fields):
            raise ValueError(f"Struct '{value_type.name}' has unresolved fields")
        return _product_layout(
            fields,
            struct_fields,
            active_structs | {value_type.name},
            path,
            f"struct({value_type.name})",
        )
    raise ValueError(f"{value_type.name} is not an ABI-stable Value")


def _product_layout(
    fields: tuple[tuple[AbiPathComponent, ConcreteType], ...],
    struct_fields: Callable[[str], StructFields],
    active_structs: frozenset[str],
    path: tuple[AbiPathComponent, ...],
    kind: str,
) -> ValueAbiLayout:
    offset = 0
    alignment = 1
    offsets: list[int] = []
    leaves: list[ValueLeaf] = []
    canonical_fields: list[str] = []
    for component, field in fields:
        layout = _value_abi_layout(field, struct_fields, active_structs, (*path, component))
        offset = _align_to(offset, layout.alignment)
        offsets.append(offset)
        leaves.extend(
            ValueLeaf(leaf.path, leaf.dtype, offset + leaf.byte_offset, leaf.scalar_count, leaf.shape)
            for leaf in layout.leaves
        )
        canonical_fields.append(f"{component}@{offset}:{layout._canonical}")
        offset += layout.size
        alignment = max(alignment, layout.alignment)
    size = _align_to(offset, alignment)
    canonical = f"{kind}(align={alignment},size={size};{';'.join(canonical_fields)})"
    return ValueAbiLayout(
        size,
        alignment,
        tuple(offsets),
        leaves=tuple(leaves),
        _canonical=canonical,
    )


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
