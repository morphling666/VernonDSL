"""The JSON form of a Host value, read and written through its reflection descriptor alone.

Works on any `VhType`, from a Python-mode table or a loaded AOT object, so editors, inspection tools and error
reports share one textual form of a value: Structs and classes are objects in field order, Tuples, Tensors and
TensorStorages (nested by shape) and lists are lists, strs are strings, dicts are objects in insertion order whose
keys are str keys or the JSON text of other keys ("7", "true"), handles are their uint64 bits, and non-finite floats
are the strings "nan", "inf" and "-inf", which JSON has no numbers for. `vh_write_json` in `vernon_host/reflect.h`
writes the same form from C.
"""

from __future__ import annotations

import ctypes
import json
import math
from collections.abc import Callable
from typing import Any

import numpy as np

from .native import (
    DictLayout,
    VhDictRecord,
    VhListRecord,
    clear_dict,
    dict_layout,
    insert_dict_entry,
    replace_storage,
    reserve_record,
)
from .reflect import ARRAY, CLASS, DICT, HANDLE, KIND_CODES, LIST, STORAGE, STR, STRUCT, TUPLE, VhType

_SCALARS = {
    KIND_CODES["bool"]: ctypes.c_bool,
    KIND_CODES["i32"]: ctypes.c_int32,
    KIND_CODES["u32"]: ctypes.c_uint32,
    KIND_CODES["i64"]: ctypes.c_int64,
    KIND_CODES["u64"]: ctypes.c_uint64,
    KIND_CODES["f32"]: ctypes.c_float,
    KIND_CODES["f64"]: ctypes.c_double,
    HANDLE: ctypes.c_uint64,
}
_FLOATS = {KIND_CODES["f32"], KIND_CODES["f64"]}
_NON_FINITE = {"nan": math.nan, "inf": math.inf, "-inf": -math.inf}


class ValueJsonError(ValueError):
    pass


def _count(descriptor: VhType) -> int:
    return math.prod(descriptor.shape[index] for index in range(descriptor.rank))


def _fields(descriptor: VhType) -> list[Any]:
    return [descriptor.fields[index] for index in range(descriptor.field_count)]


def to_json(descriptor: VhType, address: int) -> Any:
    """The JSON value of the `descriptor` value at `address`, as Python lists, dicts and scalars."""

    kind = descriptor.kind
    if kind in _SCALARS:
        value = _SCALARS[kind].from_address(address).value
        if kind not in _FLOATS:
            return value
        if not math.isfinite(value):
            return "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")
        # The shortest decimal that reads back as the same f32, not the f64 expansion of its bits.
        return float(str(np.float32(value))) if kind == KIND_CODES["f32"] else value
    if kind in {ARRAY, STORAGE}:
        element = descriptor.element.contents
        if kind == STORAGE:
            shape = list((ctypes.c_uint64 * (1 + descriptor.rank)).from_address(address)[1:])
            address = ctypes.c_uint64.from_address(address).value
        else:
            shape = [descriptor.shape[axis] for axis in range(descriptor.rank)]
        if not math.prod(shape):
            # Rows of no elements nest as numpy's tolist gives them.
            return _empty_rows(shape)
        flat = [to_json(element, address + index * element.size) for index in range(math.prod(shape))]
        for extent in reversed(shape[1:]):
            flat = [flat[start : start + extent] for start in range(0, len(flat), extent)]
        return flat
    if kind == TUPLE:
        return [to_json(field.type.contents, address + field.offset) for field in _fields(descriptor)]
    if kind == LIST:
        record = VhListRecord.from_address(address)
        element = descriptor.element.contents
        return [to_json(element, record.data + index * element.size) for index in range(record.size)]
    if kind == STR:
        record = VhListRecord.from_address(address)
        return ctypes.string_at(record.data, record.size).decode("utf-8") if record.size else ""
    if kind == DICT:
        key, element = descriptor.key.contents, descriptor.element.contents
        entry = _entry_layout(descriptor)
        record = VhDictRecord.from_address(address)
        result = {}
        for index in range(record.used):
            at = record.entries + index * entry.entry_size
            if ctypes.c_uint64.from_address(at).value != _REMOVED:
                text = to_json(key, at + entry.key_offset)
                name = text if key.kind == STR else json.dumps(text)
                result[name] = to_json(element, at + entry.value_offset)
        return result
    if kind in {STRUCT, CLASS}:
        return {
            field.name.decode("utf-8"): to_json(field.type.contents, address + field.offset)
            for field in _fields(descriptor)
        }
    raise ValueJsonError(f"type kind {kind} has no JSON form")


def from_json(
    descriptor: VhType, value: Any, address: int, *, path: str = "value", allocator: int | None = None
) -> None:
    """Writes the JSON value `value` over the `descriptor` value at `address`. An object may list only some fields,
    leaving the others as they are, so an edit names just what it changes; a list or str is replaced whole, its
    memory grown through `allocator`, the address of the table's `VhAllocator` (None for VernonRuntime's default).
    Every check runs before any write, so a rejected value leaves memory untouched."""

    writes: list[Callable[[], None]] = []
    _plan(descriptor, value, address, path, writes, allocator)
    for write in writes:
        write()


def _store(ctype: Any, address: int, item: Any) -> Callable[[], None]:
    def write() -> None:
        ctype.from_address(address).value = item

    return write


def _replace(
    record: Any, source: Any, count: int, size: int, alignment: int, allocator: int | None
) -> Callable[[], None]:
    """Replaces a list or str's contents with `count` items of `size` bytes, copied from the buffer `source` when
    the write runs."""

    def write() -> None:
        reserve_record(record, count, size, alignment, allocator)
        if count:
            ctypes.memmove(record.data, ctypes.addressof(source), count * size)
        record.size = count

    return write


_REMOVED = (1 << 64) - 1


def _entry_layout(descriptor: VhType) -> DictLayout:
    key, element = descriptor.key.contents, descriptor.element.contents
    return dict_layout((key.size, key.alignment), (element.size, element.alignment), key.kind == STR)


def _refill(
    address: int,
    entry: DictLayout,
    value_size: int,
    entries: list[tuple[Any, int, Any]],
    staged: list[Callable[[], None]],
    allocator: int | None,
) -> Callable[[], None]:
    """Replaces a dict's entries with `entries`, (alive, key pointer, value buffer), once `staged` filled them."""

    def write() -> None:
        for step in staged:
            step()
        record = VhDictRecord.from_address(address)
        clear_dict(address, entry, allocator)
        for _, key, value in entries:
            target = insert_dict_entry(record, key, entry, allocator)
            ctypes.memmove(target + entry.value_offset, ctypes.addressof(value), value_size)

    return write


def _plan(
    descriptor: VhType,
    value: Any,
    address: int,
    path: str,
    writes: list[Callable[[], None]],
    allocator: int | None,
) -> None:
    kind = descriptor.kind
    name = descriptor.name.decode("utf-8")
    if kind in _SCALARS:
        writes.append(_store(_SCALARS[kind], address, _scalar(kind, name, value, path)))
    elif kind == STR:
        if not isinstance(value, str):
            raise ValueJsonError(f"{path}: str expects a string, not {value!r}")
        data = value.encode("utf-8")
        source = ctypes.create_string_buffer(data, len(data))
        writes.append(_replace(VhListRecord.from_address(address), source, len(data), 1, 1, allocator))
    elif kind == LIST:
        if not isinstance(value, list):
            raise ValueJsonError(f"{path}: {name} expects a list")
        # Elements are built in a zeroed scratch buffer, so an element object names all its fields or leaves
        # the rest zero, and the list's memory is only grown once every element checked.
        element = descriptor.element.contents
        scratch = ctypes.create_string_buffer(len(value) * element.size)
        staged: list[Callable[[], None]] = []
        for index, item in enumerate(value):
            _plan(
                element, item, ctypes.addressof(scratch) + index * element.size, f"{path}[{index}]", staged, allocator
            )

        def build() -> None:
            for write in staged:
                write()

        writes.append(build)
        record = VhListRecord.from_address(address)
        writes.append(_replace(record, scratch, len(value), element.size, element.alignment, allocator))
    elif kind == DICT:
        if not isinstance(value, dict):
            raise ValueJsonError(f"{path}: {name} expects an object")
        key, element = descriptor.key.contents, descriptor.element.contents
        # Keys and values are staged in zeroed scratch buffers like list elements; the dict is cleared and
        # refilled only once every entry checked. A key that is not a str is its JSON text, "7" or "true".
        entries = []
        staged = []
        for text, item in value.items():
            entry_path = f"{path}[{text!r}]"
            if key.kind == STR:
                parsed: Any = text
            else:
                try:
                    parsed = json.loads(text)
                except json.JSONDecodeError:
                    raise ValueJsonError(f"{entry_path}: {name} keys are JSON text of its key type") from None
            key_scratch = ctypes.create_string_buffer(max(key.size, 1))
            value_scratch = ctypes.create_string_buffer(max(element.size, 1))
            if key.kind == STR:
                encoded = text.encode("utf-8")
                data = ctypes.create_string_buffer(encoded, max(len(encoded), 1))
                view = (ctypes.c_uint64 * 2)(ctypes.addressof(data), len(encoded))
                entries.append(((data, view), ctypes.addressof(view), value_scratch))
            else:
                _plan(key, parsed, ctypes.addressof(key_scratch), entry_path, staged, allocator)
                entries.append((key_scratch, ctypes.addressof(key_scratch), value_scratch))
            _plan(element, item, ctypes.addressof(value_scratch), entry_path, staged, allocator)
        writes.append(_refill(address, _entry_layout(descriptor), element.size, entries, staged, allocator))
    elif kind == ARRAY:
        element = descriptor.element.contents
        shape = [descriptor.shape[index] for index in range(descriptor.rank)]
        flat = _flatten(value, shape, path)
        for index, item in enumerate(flat):
            _plan(element, item, address + index * element.size, f"{path}[{index}]", writes, allocator)
    elif kind == STORAGE:
        # The shape is the nesting's, which must be rectangular and match every static extent; the storage is
        # replaced whole, like a list.
        element = descriptor.element.contents
        statics = [descriptor.shape[axis] for axis in range(descriptor.rank)]
        shape = _nested_shape(value, descriptor.rank, path)
        for axis, extent in enumerate(shape):
            if statics[axis] and extent != statics[axis]:
                raise ValueJsonError(f"{path}: {name} has {statics[axis]} items on axis {axis}, not {extent}")
        # Axes below an empty list hold no items to count: they take their static extent, a dyn one 0.
        shape += statics[len(shape) :]
        flat = _flatten(value, shape, path)
        scratch = ctypes.create_string_buffer(max(len(flat) * element.size, 1))
        for index, item in enumerate(flat):
            _plan(
                element, item, ctypes.addressof(scratch) + index * element.size, f"{path}[{index}]", writes, allocator
            )

        def replace() -> None:
            replace_storage(address, shape, element.size, ctypes.addressof(scratch), allocator)

        writes.append(replace)
    elif kind == TUPLE:
        fields = _fields(descriptor)
        if not isinstance(value, list) or len(value) != len(fields):
            raise ValueJsonError(f"{path}: {name} is a list of {len(fields)} items")
        for index, (field, item) in enumerate(zip(fields, value, strict=True)):
            _plan(field.type.contents, item, address + field.offset, f"{path}[{index}]", writes, allocator)
    elif kind in {STRUCT, CLASS}:
        if not isinstance(value, dict):
            raise ValueJsonError(f"{path}: {name} is an object of its fields")
        fields = {field.name.decode("utf-8"): field for field in _fields(descriptor)}
        for key, item in value.items():
            field = fields.get(key)
            if field is None:
                raise ValueJsonError(f"{path}: {name} has no field '{key}'; its fields are {', '.join(fields)}")
            _plan(field.type.contents, item, address + field.offset, f"{path}.{key}", writes, allocator)
    else:
        raise ValueJsonError(f"{path}: type kind {kind} has no JSON form")


def _scalar(kind: int, name: str, value: Any, path: str) -> Any:
    if kind == KIND_CODES["bool"]:
        if not isinstance(value, bool):
            raise ValueJsonError(f"{path}: bool expects true or false, not {value!r}")
        return value
    if kind in _FLOATS:
        if isinstance(value, str) and value in _NON_FINITE:
            return _NON_FINITE[value]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueJsonError(f'{path}: {name} expects a number or "nan", "inf", "-inf", not {value!r}')
        return float(value)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueJsonError(f"{path}: {name} expects an integer, not {value!r}")
    bits = ctypes.sizeof(_SCALARS[kind]) * 8
    signed = kind in {KIND_CODES["i32"], KIND_CODES["i64"]}
    low, high = (-(1 << (bits - 1)), (1 << (bits - 1)) - 1) if signed else (0, (1 << bits) - 1)
    if not low <= value <= high:
        raise ValueJsonError(f"{path}: {value} is out of range for {name}")
    return value


def _empty_rows(shape: list[int]) -> list[Any]:
    if not shape or shape[0] == 0:
        return []
    return [_empty_rows(shape[1:]) for _ in range(shape[0])]


def _nested_shape(value: Any, rank: int, path: str) -> list[int]:
    """The extents of up to `rank` nested lists, read along their first items, stopping below an empty one;
    `_flatten` checks the other items match."""

    shape: list[int] = []
    item = value
    while len(shape) < rank:
        if not isinstance(item, list):
            raise ValueJsonError(f"{path}: expects lists nested {rank} deep")
        shape.append(len(item))
        if not item:
            break
        item = item[0]
    return shape


def _flatten(value: Any, shape: list[int], path: str) -> list[Any]:
    if not shape:
        return [value]
    if not isinstance(value, list) or len(value) != shape[0]:
        raise ValueJsonError(f"{path}: expects a list of {shape[0]} items")
    flat: list[Any] = []
    for index, item in enumerate(value):
        flat.extend(_flatten(item, shape[1:], f"{path}[{index}]"))
    return flat


__all__ = ["ValueJsonError", "from_json", "to_json"]
