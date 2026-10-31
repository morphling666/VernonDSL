"""C layouts of Host Values and class fields.

Layouts are the same on every supported target, wasm32 included: natural alignment, fields in declaration order,
bool as one byte, handles as one uint64_t, Tensors as row-major arrays, `list[T]` and `str` fields as the
`VhList` record of three uint64_t, which holds its data address as an integer, `dict[K, V]` fields as the
`VhDict` record of six, and `TensorStorage[T, (e0, ..)]` as `{uint64_t data; uint64_t shape[rank];}`.
"""

from __future__ import annotations

import struct
from collections.abc import Callable
from dataclasses import dataclass

from ..frontend.model import ConcreteType

StructFields = Callable[[str], tuple[tuple[str, ConcreteType], ...]]
# Types whose values own memory: a class field of one is released by the class's fini and moved, not copied.
OWNED_KINDS = frozenset({"list", "str", "dict", "storage"})

_SCALARS = {
    "bool": (1, "?"),
    "i32": (4, "i"),
    "u32": (4, "I"),
    "i64": (8, "q"),
    "u64": (8, "Q"),
    "f32": (4, "f"),
    "f64": (8, "d"),
}


@dataclass(frozen=True)
class Layout:
    size: int
    alignment: int
    # Byte offsets of a Struct's fields or a Tuple's elements; a Tensor's elements follow at element.size strides.
    offsets: tuple[int, ...] = ()


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def tensor_count(value_type: ConcreteType) -> int:
    count = 1
    for extent in value_type.arguments[1:]:
        count *= int(extent)  # type: ignore[arg-type]
    return count


def storage_rank(value_type: ConcreteType) -> int:
    extents = value_type.arguments[1]
    assert isinstance(extents, tuple)
    return len(extents)


def members(value_type: ConcreteType, fields: StructFields) -> tuple[ConcreteType, ...]:
    if value_type.kind == "struct":
        return tuple(field for _, field in fields(value_type.name))
    if value_type.kind == "tuple":
        return tuple(element for element in value_type.arguments if isinstance(element, ConcreteType))
    raise ValueError(f"{value_type.spelling} has no members")


def layout(value_type: ConcreteType, fields: StructFields) -> Layout:
    if value_type.kind == "scalar":
        size, _ = _SCALARS[value_type.name]
        return Layout(size, size)
    if value_type.kind == "handle":
        return Layout(8, 8)
    if value_type.kind == "dict":
        return Layout(48, 8, (0, 8, 16, 24, 32, 40))
    if value_type.kind == "storage":
        words = 1 + storage_rank(value_type)
        return Layout(8 * words, 8, tuple(8 * word for word in range(words)))
    if value_type.kind in OWNED_KINDS:
        return Layout(24, 8, (0, 8, 16))
    if value_type.kind == "tensor":
        element = value_type.arguments[0]
        assert isinstance(element, ConcreteType)
        inner = layout(element, fields)
        return Layout(inner.size * tensor_count(value_type), inner.alignment)
    if value_type.kind in {"struct", "tuple"}:
        offset = 0
        alignment = 1
        offsets: list[int] = []
        for member in members(value_type, fields):
            inner = layout(member, fields)
            offset = _align(offset, inner.alignment)
            offsets.append(offset)
            offset += inner.size
            alignment = max(alignment, inner.alignment)
        # A memberless Struct still occupies a byte, as in C++.
        return Layout(max(_align(offset, alignment), 1), alignment, tuple(offsets))
    raise ValueError(f"{value_type.spelling} has no Host layout")


def dict_entry(value_type: ConcreteType) -> ConcreteType:
    """A `dict[K, V]` entry, `{uint64_t hash; K key; V value;}`, as the Tuple with its layout."""

    key, value = value_type.arguments
    assert isinstance(key, ConcreteType) and isinstance(value, ConcreteType)
    return ConcreteType("tuple", "Tuple", (ConcreteType("scalar", "u64"), key, value))


def pack(value: object, value_type: ConcreteType, fields: StructFields) -> bytes:
    """Packs an interpreter Value: Python scalars, flat tuples for Tensors, and member tuples for Structs and
    Tuples."""

    buffer = bytearray(layout(value_type, fields).size)
    _pack_into(buffer, 0, value, value_type, fields)
    return bytes(buffer)


def _pack_into(buffer: bytearray, offset: int, value: object, value_type: ConcreteType, fields: StructFields) -> None:
    if value_type.kind == "scalar":
        struct.pack_into("<" + _SCALARS[value_type.name][1], buffer, offset, value)
    elif value_type.kind == "handle":
        struct.pack_into("<Q", buffer, offset, value)
    elif value_type.kind == "tensor":
        element = value_type.arguments[0]
        assert isinstance(element, ConcreteType) and isinstance(value, tuple)
        stride = layout(element, fields).size
        for index, item in enumerate(value):
            _pack_into(buffer, offset + index * stride, item, element, fields)
    else:
        assert isinstance(value, tuple)
        for member_offset, member, item in zip(
            layout(value_type, fields).offsets, members(value_type, fields), value, strict=True
        ):
            _pack_into(buffer, offset + member_offset, item, member, fields)


def unpack(data: bytes | bytearray | memoryview, value_type: ConcreteType, fields: StructFields) -> object:
    return _unpack_from(bytes(data), 0, value_type, fields)


def _unpack_from(data: bytes, offset: int, value_type: ConcreteType, fields: StructFields) -> object:
    from .values import scalar_value

    if value_type.kind == "scalar":
        (raw,) = struct.unpack_from("<" + _SCALARS[value_type.name][1], data, offset)
        return scalar_value(raw, value_type.name)
    if value_type.kind == "handle":
        return struct.unpack_from("<Q", data, offset)[0]
    if value_type.kind == "tensor":
        element = value_type.arguments[0]
        assert isinstance(element, ConcreteType)
        stride = layout(element, fields).size
        return tuple(
            _unpack_from(data, offset + index * stride, element, fields) for index in range(tensor_count(value_type))
        )
    return tuple(
        _unpack_from(data, offset + member_offset, member, fields)
        for member_offset, member in zip(layout(value_type, fields).offsets, members(value_type, fields), strict=True)
    )
