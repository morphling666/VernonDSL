"""Host code in development: the source runs as ordinary CPython, with no rewriting.

Scalars are numpy scalars (`f32(x)` is `numpy.float32(x)`), Vectors and Matrices are the immutable numpy Tensors of
`vernon_dsl.types`, Structs are `@struct(shared=True)` values, and an `@extern` call dispatches to the binding of the
registry the outermost Host call runs with. Arithmetic follows numpy, not the compiled code's rules, so results may
differ where the two disagree (`specs/host/design.md` lists where); the type checker still runs before Host code
first runs, so code the compiler would reject fails in development too.

A Host class instance keeps its state in a ctypes Structure with the class's C layout. The reflection table, C
callers, and state migration therefore see the same bytes in development as in packaged builds.
"""

from __future__ import annotations

import ctypes
import functools
import inspect
import sys
import weakref
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, TypeVar, get_args, get_origin, get_type_hints

import numpy as np

from .._markers import dsl_kind
from ..frontend.model import ConcreteType
from ..types import Matrix, Tensor, TypeExpr, Vector, _Scalar
from .externs import ExternRegistry, HostStr
from .layout import layout, storage_rank
from .model import ExternAbi, HostProgram, HostStatus
from .reflect import BuiltTable, VhType, describe
from .values import HostExecutionError, element_of, key_of

if TYPE_CHECKING:
    from .function import ExternFunction

_ACTIVE: ContextVar[HostRuntime | None] = ContextVar("vernon_host_runtime", default=None)

_SCALARS: dict[str, tuple[Any, Any]] = {
    "bool": (np.bool_, ctypes.c_bool),
    "i32": (np.int32, ctypes.c_int32),
    "u32": (np.uint32, ctypes.c_uint32),
    "i64": (np.int64, ctypes.c_int64),
    "u64": (np.uint64, ctypes.c_uint64),
    "f32": (np.float32, ctypes.c_float),
    "f64": (np.float64, ctypes.c_double),
}
_ALIASES: dict[Any, str] = {bool: "bool", int: "i32", float: "f32"}
_TENSORS: dict[str, type[Tensor]] = {"Vector": Vector, "Matrix": Matrix, "Tensor": Tensor}


def _structure(name: str, fields: list[tuple[str, Any]]) -> Any:
    # A memberless Struct still occupies a byte, as `layout` gives it.
    return type(name, (ctypes.Structure,), {"_fields_": fields or [("_vh_empty", ctypes.c_uint8)]})


def _nested(array: np.ndarray) -> Any:
    if array.ndim <= 1:
        return tuple(array)
    return tuple(_nested(row) for row in array)


# Codecs: how one Host type moves between Python values, Host code's values, and its C layout.


class Codec:
    """`value` accepts what a Python caller or binding may give and returns what Host code holds; `boundary` is what
    Python extern bindings receive; `read` and `write` convert a ctypes field's value, which is a Python scalar when
    `simple` and a ctypes object sharing the field's memory otherwise."""

    ctype: Any
    simple = False

    def value(self, item: Any) -> Any:
        raise NotImplementedError

    def boundary(self, item: Any) -> Any:
        return item

    def read(self, raw: Any) -> Any:
        raise NotImplementedError

    def write(self, item: Any) -> Any:
        raise NotImplementedError

    def get(self, state: Any, name: str, owner: Any) -> Any:
        return self.read(getattr(state, name))

    def set(self, state: Any, name: str, item: Any, owner: Any = None) -> None:
        setattr(state, name, self.write(item))

    def release(self, address: int, allocator: int | None) -> None:
        """Frees the memory a value at `address` owns; only containers, and classes holding them, own any."""

    def read_at(self, address: int) -> Any:
        raw = self.ctype.from_address(address)
        return self.read(raw.value if self.simple else raw)

    def write_at(self, address: int, item: Any) -> None:
        if self.simple:
            self.ctype.from_address(address).value = self.write(self.value(item))
        else:
            converted = self.write(item)
            ctypes.memmove(address, ctypes.addressof(converted), ctypes.sizeof(converted))


class _ScalarCodec(Codec):
    simple = True

    def __init__(self, name: str):
        self.name = name
        self.type, self.ctype = _SCALARS[name]
        self.convert: Callable[[Any], Any] = (
            float if name in {"f32", "f64"} else (lambda item: bool(item)) if name == "bool" else int
        )

    def value(self, item: Any) -> Any:
        return self.type(item)

    def boundary(self, item: Any) -> Any:
        return item if self.name in {"f32", "f64"} else self.convert(item)

    def read(self, raw: Any) -> Any:
        return self.type(raw)

    def write(self, item: Any) -> Any:
        return self.convert(item)


class _HandleCodec(Codec):
    """Handles are their 64 bits as a Python int; `Handle()` is the null handle, 0."""

    simple = True
    ctype = ctypes.c_uint64

    def value(self, item: Any) -> Any:
        return int(getattr(item, "bits", item))

    boundary = read = write = value


class _StrCodec(Codec):
    ctype = HostStr

    def value(self, item: Any) -> Any:
        return str(item)


class _TensorCodec(Codec):
    def __init__(self, kind: type[Tensor], scalar: str, shape: tuple[int, ...]):
        self.kind = kind
        self.shape = shape
        self.dtype = np.dtype(_SCALARS[scalar][0])
        self.ctype = _SCALARS[scalar][1] * int(np.prod(shape))

    def value(self, item: Any) -> Any:
        if (
            isinstance(item, self.kind)
            and item.dtype == self.dtype
            and item.shape == self.shape
            and not item.flags.writeable
        ):
            return item
        array = np.array(item, dtype=self.dtype)
        if array.shape != self.shape:
            if array.size != int(np.prod(self.shape)):
                raise TypeError(f"{self.kind.__name__} of shape {self.shape} cannot hold shape {array.shape}")
            array = array.reshape(self.shape)
        return self.kind._result(array)

    def boundary(self, item: Any) -> Any:
        return _nested(np.asarray(item))

    def read(self, raw: Any) -> Any:
        return self.kind._result(np.ctypeslib.as_array(raw).reshape(self.shape).copy())

    def write(self, item: Any) -> Any:
        return self.ctype.from_buffer_copy(np.ascontiguousarray(self.value(item)).tobytes())


class _StructCodec(Codec):
    def __init__(self, cls: type[Any]):
        from ..host_values import declared_fields

        self.cls = cls
        self.fields = [(name, codec_of(annotation)) for name, annotation in declared_fields(cls).items()]
        self.ctype = _structure(f"{cls.__name__}_C", [(name, codec.ctype) for name, codec in self.fields])

    def value(self, item: Any) -> Any:
        if isinstance(item, self.cls):
            return item
        if isinstance(item, Mapping):
            items: Sequence[Any] = [item[name] for name, _ in self.fields]
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            items = item
        else:
            items = [getattr(item, name) for name, _ in self.fields]
        return self.cls(*(codec.value(part) for (_, codec), part in zip(self.fields, items, strict=True)))

    def boundary(self, item: Any) -> Any:
        item = self.value(item)
        return {name: codec.boundary(getattr(item, name)) for name, codec in self.fields}

    def read(self, raw: Any) -> Any:
        return self.cls(*(codec.read(getattr(raw, name)) for name, codec in self.fields))

    def write(self, item: Any) -> Any:
        item = self.value(item)
        return self.ctype(*(codec.write(getattr(item, name)) for name, codec in self.fields))


class _TupleCodec(Codec):
    def __init__(self, items: list[Codec]):
        self.items = items
        self.ctype = _structure("Tuple_C", [(f"item{index}", codec.ctype) for index, codec in enumerate(items)])

    def value(self, item: Any) -> Any:
        return tuple(codec.value(part) for codec, part in zip(self.items, item, strict=True))

    def boundary(self, item: Any) -> Any:
        return tuple(codec.boundary(part) for codec, part in zip(self.items, item, strict=True))

    def read(self, raw: Any) -> Any:
        return tuple(codec.read(getattr(raw, f"item{index}")) for index, codec in enumerate(self.items))

    def write(self, item: Any) -> Any:
        return self.ctype(*(codec.write(part) for codec, part in zip(self.items, item, strict=True)))


# Containers: a list or str field is a VhList record, {data, size, capacity}, and a dict field a VhDict record. Their
# memory VernonRuntime allocates through the allocator of the module's table, as compiled code does, so either may
# free what the other grew, and dicts change only through VernonRuntime's functions, so both share one hash table.


class VhListRecord(ctypes.Structure):
    _fields_ = [("data", ctypes.c_uint64), ("size", ctypes.c_uint64), ("capacity", ctypes.c_uint64)]


_POINTER, _U64 = ctypes.c_void_p, ctypes.c_uint64
_RUNTIME_SIGNATURES: dict[str, tuple[Any, ...]] = {
    "vernonRuntimeHostListReserve": (ctypes.c_int32, _POINTER, _U64, _U64, _U64, _POINTER),
    "vernonRuntimeHostListRelease": (None, _POINTER, _U64, _U64, _POINTER),
    "vernonRuntimeHostDictFind": (ctypes.c_int64, _POINTER, _POINTER, _POINTER),
    "vernonRuntimeHostDictInsert": (ctypes.c_int32, _POINTER, _POINTER, _POINTER, _POINTER, _POINTER),
    "vernonRuntimeHostDictRemove": (ctypes.c_int32, _POINTER, _POINTER, _POINTER, _POINTER),
    "vernonRuntimeHostDictClear": (None, _POINTER, _POINTER, _POINTER),
    "vernonRuntimeHostDictRelease": (None, _POINTER, _POINTER, _POINTER),
    "vernonRuntimeHostStorageAllocate": (ctypes.c_int32, _POINTER, ctypes.c_uint32, _POINTER, _U64, _U64, _POINTER),
    "vernonRuntimeHostStorageRelease": (None, _POINTER, ctypes.c_uint32, _U64, _U64, _POINTER),
}


@functools.cache
def _container_runtime() -> dict[str, Any]:
    from .. import _native

    symbols = _native._host_runtime_symbols()
    return {
        name[len("vernonRuntimeHost") :]: ctypes.CFUNCTYPE(*signature)(symbols[name])
        for name, signature in _RUNTIME_SIGNATURES.items()
    }


def reserve_record(record: VhListRecord, capacity: int, size: int, alignment: int, allocator: int | None) -> None:
    status = _container_runtime()["ListReserve"](ctypes.addressof(record), capacity, size, alignment, allocator)
    if status != 0:
        raise HostExecutionError(status, "the allocator could not grow a container")


def release_record(address: int, size: int, alignment: int, allocator: int | None) -> None:
    _container_runtime()["ListRelease"](address, size, alignment, allocator)


class VhDictRecord(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint64) for name in ("entries", "size", "used", "capacity", "slots", "slot_count")]


class DictLayout(ctypes.Structure):
    """VernonRuntime's `VernonHostDictLayout`."""

    _fields_ = [
        (name, ctypes.c_uint64)
        for name in ("entry_size", "entry_alignment", "key_offset", "key_size", "value_offset", "key_kind")
    ]


_REMOVED = (1 << 64) - 1


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def dict_layout(key: tuple[int, int], value: tuple[int, int], str_key: bool) -> DictLayout:
    """The entry layout of a dict from its key's and value's (size, alignment), as `vh_dict_layout` gives it."""

    key_offset = _align(8, key[1])
    value_offset = _align(key_offset + key[0], value[1])
    entry_size = _align(value_offset + value[0], max(8, key[1], value[1]))
    return DictLayout(entry_size, max(8, key[1], value[1]), key_offset, key[0], value_offset, int(str_key))


def release_dict(address: int, entry: DictLayout, allocator: int | None) -> None:
    _container_runtime()["DictRelease"](address, ctypes.addressof(entry), allocator)


def clear_dict(address: int, entry: DictLayout, allocator: int | None) -> None:
    _container_runtime()["DictClear"](address, ctypes.addressof(entry), allocator)


def insert_dict_entry(record: VhDictRecord, key: int, entry: DictLayout, allocator: int | None) -> int:
    """Finds or inserts the key at address `key`, returning its entry's address."""

    index = ctypes.c_uint64()
    status = _container_runtime()["DictInsert"](
        ctypes.addressof(record), key, ctypes.addressof(entry), allocator, ctypes.addressof(index)
    )
    if status != 0:
        raise HostExecutionError(status, "the allocator could not grow a dict")
    return record.entries + index.value * entry.entry_size


def _missing(key: Any) -> HostExecutionError:
    return HostExecutionError(HostStatus.KEY_NOT_FOUND, f"key {key!r} is not in the dict")


class HostDict:
    """A dict field of a Host class instance, over the instance's memory; it is never copied. Its order is insertion
    order, and like Python's dict iterators, a step fails once the dict's size changed since the loop began."""

    __slots__ = ("_record", "_codec", "_owner")

    def __init__(self, record: VhDictRecord, codec: _DictCodec, owner: type[Any]):
        self._record = record
        self._codec = codec
        self._owner = owner

    def _call(self, function: str, key: int, *rest: Any) -> Any:
        """A runtime call taking a key: the record, the key, the layout, then `rest`."""

        return _container_runtime()[function](
            ctypes.addressof(self._record), key, ctypes.addressof(self._codec.layout), *rest
        )

    def _find(self, key: Any) -> int:
        pointer, alive = self._codec.key_argument(key)
        index = int(self._call("DictFind", pointer))
        del alive
        return index

    def _entry(self, index: int) -> int:
        return self._record.entries + index * self._codec.layout.entry_size

    def _read_key(self, entry: int) -> Any:
        return self._codec.key.read_at(entry + self._codec.layout.key_offset)

    def _read_value(self, entry: int) -> Any:
        return self._codec.item.read_at(entry + self._codec.layout.value_offset)

    def _entries(self) -> Iterator[int]:
        size, position = self._record.size, 0
        while True:
            if self._record.size != size:
                raise HostExecutionError(HostStatus.CONTAINER_CHANGED, "dict changed size during iteration")
            if position >= self._record.used:
                return
            entry = self._entry(position)
            position += 1
            if ctypes.c_uint64.from_address(entry).value != _REMOVED:
                yield entry

    def __len__(self) -> int:
        return self._record.size

    def __contains__(self, key: Any) -> bool:
        return self._find(key) >= 0

    def __getitem__(self, key: Any) -> Any:
        index = self._find(key)
        if index < 0:
            raise _missing(key)
        return self._read_value(self._entry(index))

    def __setitem__(self, key: Any, item: Any) -> None:
        value = self._codec.item.value(item)
        pointer, alive = self._codec.key_argument(key)
        entry = insert_dict_entry(self._record, pointer, self._codec.layout, _allocator(self._owner))
        del alive
        self._codec.item.write_at(entry + self._codec.layout.value_offset, value)

    def __delitem__(self, key: Any) -> None:
        pointer, alive = self._codec.key_argument(key)
        removed = self._call("DictRemove", pointer, _allocator(self._owner))
        del alive
        if not removed:
            raise _missing(key)

    def __iter__(self) -> Iterator[Any]:
        return self.keys()

    def keys(self) -> Iterator[Any]:
        for entry in self._entries():
            yield self._read_key(entry)

    def values(self) -> Iterator[Any]:
        for entry in self._entries():
            yield self._read_value(entry)

    def items(self) -> Iterator[tuple[Any, Any]]:
        for entry in self._entries():
            yield self._read_key(entry), self._read_value(entry)

    def get(self, key: Any, default: Any) -> Any:
        index = self._find(key)
        return self._read_value(self._entry(index)) if index >= 0 else self._codec.item.value(default)

    def pop(self, key: Any, *default: Any) -> Any:
        index = self._find(key)
        if index < 0:
            if default:
                return self._codec.item.value(default[0])
            raise _missing(key)
        value = self._read_value(self._entry(index))
        del self[key]
        return value

    def clear(self) -> None:
        clear_dict(ctypes.addressof(self._record), self._codec.layout, _allocator(self._owner))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, (Mapping, HostDict)):
            return NotImplemented
        return dict(self.items()) == dict(other.items())

    def __repr__(self) -> str:
        return repr(dict(self.items()))


def _out_of_range(index: int, size: int) -> HostExecutionError:
    return HostExecutionError(HostStatus.INDEX_OUT_OF_RANGE, f"list index {index} is out of range for size {size}")


class HostList:
    """A list field of a Host class instance, over the instance's memory; it is never copied. Like Python's list
    iterator, iteration reads the current size at every step, so the loop body may change the list."""

    __slots__ = ("_record", "_codec", "_owner")

    def __init__(self, record: VhListRecord, codec: Codec, owner: type[Any]):
        self._record = record
        self._codec = codec
        self._owner = owner

    def _address(self, index: int) -> int:
        size = self._record.size
        position = index + size if index < 0 else index
        if not 0 <= position < size:
            raise _out_of_range(index, size)
        return self._record.data + position * ctypes.sizeof(self._codec.ctype)

    def _read(self, address: int) -> Any:
        return self._codec.read_at(address)

    def _write(self, address: int, item: Any) -> None:
        self._codec.write_at(address, item)

    def __len__(self) -> int:
        return self._record.size

    def __getitem__(self, index: int) -> Any:
        return self._read(self._address(int(index)))

    def __setitem__(self, index: int, item: Any) -> None:
        self._write(self._address(int(index)), item)

    def __iter__(self) -> Iterator[Any]:
        position = 0
        while position < self._record.size:
            yield self[position]
            position += 1

    def __eq__(self, other: object) -> bool:
        try:
            return len(self) == len(other) and all(a == b for a, b in zip(self, other, strict=False))  # type: ignore[arg-type]
        except TypeError:
            return NotImplemented

    def __repr__(self) -> str:
        return repr(list(self))

    def append(self, item: Any) -> None:
        codec = self._codec
        value = codec.value(item)
        size = self._record.size
        reserve_record(
            self._record, size + 1, ctypes.sizeof(codec.ctype), ctypes.alignment(codec.ctype), _allocator(self._owner)
        )
        self._write(self._record.data + size * ctypes.sizeof(codec.ctype), value)
        self._record.size = size + 1

    def pop(self) -> Any:
        size = self._record.size
        if size == 0:
            raise HostExecutionError(HostStatus.INDEX_OUT_OF_RANGE, "pop from an empty list")
        value = self._read(self._record.data + (size - 1) * ctypes.sizeof(self._codec.ctype))
        self._record.size = size - 1
        return value

    def clear(self) -> None:
        self._record.size = 0

    def extend(self, items: Any) -> None:
        for item in list(items):
            self.append(item)


def _allocator(owner: type[Any]) -> int | None:
    return class_runtime(owner).allocator.value


class _ListCodec(Codec):
    ctype = VhListRecord

    def __init__(self, element: Codec):
        self.element = element

    def value(self, item: Any) -> Any:
        return [self.element.value(part) for part in item]

    def get(self, state: Any, name: str, owner: Any) -> Any:
        return HostList(getattr(state, name), self.element, type(owner))

    def set(self, state: Any, name: str, item: Any, owner: Any = None) -> None:
        items = self.value(item)
        placed = HostList(getattr(state, name), self.element, type(owner))
        placed.clear()
        placed.extend(items)

    def release(self, address: int, allocator: int | None) -> None:
        release_record(address, ctypes.sizeof(self.element.ctype), ctypes.alignment(self.element.ctype), allocator)


class _OwnedStrCodec(Codec):
    """A str field: UTF-8 bytes in a VhList record. A read is a Python str, a copy."""

    ctype = VhListRecord

    def value(self, item: Any) -> Any:
        return str(item)

    def get(self, state: Any, name: str, owner: Any) -> Any:
        return self.read(getattr(state, name))

    def read(self, raw: Any) -> Any:
        return ctypes.string_at(raw.data, raw.size).decode("utf-8") if raw.size else ""

    def set(self, state: Any, name: str, item: Any, owner: Any = None) -> None:
        data = str(item).encode("utf-8")
        record = getattr(state, name)
        reserve_record(record, len(data), 1, 1, _allocator(type(owner)))
        if data:
            ctypes.memmove(record.data, data, len(data))
        record.size = len(data)

    def release(self, address: int, allocator: int | None) -> None:
        release_record(address, 1, 1, allocator)


class _DictCodec(Codec):
    """A dict field: a VhDict record whose entries hold the key as `key` lays it out, a str key as an owned
    VhList."""

    ctype = VhDictRecord

    def __init__(self, key: Codec, item: Codec):
        self.key = key
        self.item = item
        self.str_key = isinstance(key, _OwnedStrCodec)
        self.layout = dict_layout(
            (ctypes.sizeof(key.ctype), ctypes.alignment(key.ctype)),
            (ctypes.sizeof(item.ctype), ctypes.alignment(item.ctype)),
            self.str_key,
        )

    def key_argument(self, key: Any) -> tuple[int, Any]:
        """The pointer the runtime takes for `key`, and what must stay alive while it is used."""

        if self.str_key:
            data = str(key).encode("utf-8")
            buffer = ctypes.create_string_buffer(data, len(data))
            view = (ctypes.c_uint64 * 2)(ctypes.addressof(buffer) if data else 0, len(data))
            return ctypes.addressof(view), (buffer, view)
        raw = self.key.write(self.key.value(key))
        held = self.key.ctype(raw) if self.key.simple else raw
        return ctypes.addressof(held), held

    def value(self, item: Any) -> Any:
        return {self.key.value(key): self.item.value(part) for key, part in dict(item.items()).items()}

    def get(self, state: Any, name: str, owner: Any) -> Any:
        return HostDict(getattr(state, name), self, type(owner))

    def set(self, state: Any, name: str, item: Any, owner: Any = None) -> None:
        items = self.value(item)
        placed = HostDict(getattr(state, name), self, type(owner))
        placed.clear()
        for key, part in items.items():
            placed[key] = part

    def release(self, address: int, allocator: int | None) -> None:
        release_dict(address, self.layout, allocator)


@functools.cache
def _storage_record(rank: int) -> type[ctypes.Structure]:
    return _structure(f"VhStorage{rank}", [("data", ctypes.c_uint64), ("shape", ctypes.c_uint64 * rank)])


# The TensorStorage a storage field last read as, by record address: (data, shape, storage). A read returns it while
# the record still holds that allocation, so Python code holding it sees Host code's writes and its residency state
# carries over between reads.
_STORAGE_VIEWS: dict[int, tuple[int, tuple[int, ...], Any]] = {}


class _StorageCodec(Codec):
    """A TensorStorage field: a `{data, shape[rank]}` record whose dense elements the module's allocator holds. A
    read is a `vd.TensorStorage` over those elements, never a copy."""

    def __init__(self, scalar: str, extents: tuple[int | None, ...]):
        self.scalar = scalar
        self.extents = extents
        self.dtype = np.dtype(_SCALARS[scalar][0])
        self.ctype = _storage_record(len(extents))

    @property
    def static(self) -> bool:
        return all(extent is not None for extent in self.extents)

    def get(self, state: Any, name: str, owner: Any) -> Any:
        from .._runtime.tensor import TensorStorage

        record = getattr(state, name)
        address = ctypes.addressof(record)
        shape = tuple(int(extent) for extent in record.shape)
        cached = _STORAGE_VIEWS.get(address)
        if cached is not None and cached[:2] == (record.data, shape):
            return cached[2]
        count = int(np.prod(shape))
        if count:
            buffer = (ctypes.c_char * (count * self.dtype.itemsize)).from_address(record.data)
            array = np.frombuffer(buffer, dtype=self.dtype).reshape(shape)
        else:
            array = np.zeros(shape, dtype=self.dtype)
        storage = TensorStorage(array)
        _STORAGE_VIEWS[address] = (record.data, shape, storage)
        return storage

    def set(self, state: Any, name: str, item: Any, owner: Any = None) -> None:
        from .._runtime.tensor import TensorStorage

        if not isinstance(item, TensorStorage) or item.dtype != self.dtype or len(item.shape) != len(self.extents):
            raise TypeError(f"'{name}' holds a {self.dtype} TensorStorage of rank {len(self.extents)}")
        if any(extent is not None and extent != size for extent, size in zip(self.extents, item.shape, strict=True)):
            raise ValueError(f"'{name}' has extents {self.extents}; a storage of shape {item.shape} does not fit")
        self.store(ctypes.addressof(getattr(state, name)), item.to_numpy(), _allocator(type(owner)))

    def store(self, address: int, elements: np.ndarray, allocator: int | None) -> None:
        elements = np.ascontiguousarray(elements, dtype=self.dtype)
        replace_storage(address, elements.shape, self.dtype.itemsize, elements.ctypes.data, allocator)

    def allocate_zeros(self, address: int, allocator: int | None) -> None:
        self.store(address, np.zeros(tuple(self.extents), dtype=self.dtype), allocator)  # type: ignore[arg-type]

    def release(self, address: int, allocator: int | None) -> None:
        release_storage(address, len(self.extents), self.dtype.itemsize, allocator)


def replace_storage(
    address: int, shape: Sequence[int], element_size: int, source: int | None, allocator: int | None
) -> None:
    """Replaces the storage record at `address` with one of `shape` holding the elements at `source`, as compiled
    code does: the new allocation is made first, so a failed one leaves the record as it was."""

    rank = len(shape)
    created = _storage_record(rank)()
    extents = (ctypes.c_int64 * rank)(*shape)
    status = _container_runtime()["StorageAllocate"](
        ctypes.addressof(created), rank, ctypes.addressof(extents), element_size, element_size, allocator
    )
    if status != 0:
        raise HostExecutionError(status, f"a TensorStorage of shape {tuple(shape)} cannot be allocated")
    count = int(np.prod(shape))
    if count and source:
        ctypes.memmove(created.data, source, count * element_size)
    release_storage(address, rank, element_size, allocator)
    ctypes.memmove(address, ctypes.addressof(created), ctypes.sizeof(created))


def release_storage(address: int, rank: int, element_size: int, allocator: int | None) -> None:
    """Frees the storage record at `address`. A `vd.TensorStorage` Python mode read it as first copies the
    elements, so Python code holding it keeps them."""

    cached = _STORAGE_VIEWS.pop(address, None)
    if cached is not None:
        cached[2]._detach_host_array()
    _container_runtime()["StorageRelease"](address, rank, element_size, element_size, allocator)


def _storage_codec(annotation: Any) -> _StorageCodec | None:
    if not isinstance(annotation, TypeExpr) or annotation.name != "TensorStorage" or len(annotation.arguments) != 2:
        return None
    element, extents = annotation.arguments
    if not isinstance(extents, tuple):
        return None
    return _StorageCodec(element.name, tuple(extent if isinstance(extent, int) else None for extent in extents))


def field_codec_of(annotation: Any) -> Codec:
    """The codec of a Host class field, which, unlike a parameter or local, may own a list, str, dict or
    TensorStorage."""

    annotation = _base(annotation)
    storage = _storage_codec(annotation)
    if storage is not None:
        return storage
    if annotation is str:
        return _OwnedStrCodec()
    if get_origin(annotation) is list:
        (element,) = get_args(annotation)
        return _ListCodec(codec_of(element))
    if get_origin(annotation) is dict:
        key, item = get_args(annotation)
        return _DictCodec(field_codec_of(key) if _base(key) is str else codec_of(key), codec_of(item))
    return codec_of(annotation)


class _PlaceCodec(Codec):
    """A Host class: never copied. A field of this type reads as a view of the parent's memory."""

    def __init__(self, cls: type[Any]):
        self.cls = cls
        self.layout = class_layout(cls)
        self.ctype = self.layout.ctype

    def value(self, item: Any) -> Any:
        if isinstance(item, self.cls):
            return item
        return self.at(int(getattr(item, "address", item)))

    def at(self, address: int) -> Any:
        return self.layout.view(self.ctype.from_address(address))

    def get(self, state: Any, name: str, owner: Any) -> Any:
        return self.layout.view(getattr(state, name))

    def set(self, state: Any, name: str, item: Any, owner: Any = None) -> None:
        raise AttributeError(f"'{name}' is a class instance; assign its fields instead")

    def release(self, address: int, allocator: int | None) -> None:
        self.layout.release(address, allocator)


_CODECS: dict[Any, Codec] = {}


def _base(annotation: Any) -> Any:
    return get_args(annotation)[0] if get_origin(annotation) is Annotated else annotation


def codec_of(annotation: Any) -> Codec:
    """The codec of a Host type annotation as Python evaluates it."""

    annotation = _base(annotation)
    codec = _CODECS.get(annotation)
    if codec is None:
        codec = _CODECS[annotation] = _make_codec(annotation)
    return codec


def _make_codec(annotation: Any) -> Codec:
    if annotation is str:
        return _StrCodec()
    if annotation in _ALIASES:
        return _ScalarCodec(_ALIASES[annotation])
    if isinstance(annotation, _Scalar):
        return _ScalarCodec(annotation.name)
    if isinstance(annotation, TypeExpr):
        arguments = annotation.arguments
        if annotation.name == "Tuple":
            return _TupleCodec([codec_of(item) for item in arguments])
        if annotation.name in {"Vector", "Matrix"} and isinstance(arguments[0], _Scalar):
            return _TensorCodec(
                _TENSORS[annotation.name], arguments[0].name, tuple(int(item) for item in arguments[1:])
            )
        if annotation.name == "Tensor" and len(arguments) >= 2 and isinstance(arguments[0], _Scalar):
            # Extents are written either flat, Tensor[f32, 2, 3], or as one tuple, Tensor[f32, (2, 3)].
            shape = arguments[1] if len(arguments) == 2 and isinstance(arguments[1], tuple) else arguments[1:]
            return _TensorCodec(Tensor, arguments[0].name, tuple(int(item) for item in shape))
    if isinstance(annotation, type):
        kind = dsl_kind(annotation)
        if kind == "handle":
            return _HandleCodec()
        if kind == "struct":
            return _StructCodec(annotation)
        if kind == "host":
            return _PlaceCodec(annotation)
    raise TypeError(f"{annotation!r} is not a Host type")


# Host classes.


class _Field:
    """A Host class field. Accessors are installed when the class's layout is first built, which happens before any
    instance exists."""

    __slots__ = ("name", "codec")

    def __init__(self, name: str):
        self.name = name
        self.codec: Codec | None = None

    def __get__(self, instance: Any, owner: type[Any]) -> Any:
        if instance is None:
            return self
        return self.codec.get(instance._vh_state, self.name, instance)  # type: ignore[union-attr]

    def __set__(self, instance: Any, item: Any) -> None:
        self.codec.set(instance._vh_state, self.name, item, instance)  # type: ignore[union-attr]


class ClassLayout:
    """The C layout of a Host class and how its instances initialize."""

    def __init__(self, cls: type[Any]):
        self.cls = cls
        hints = get_type_hints(cls, include_extras=True)
        names = list(cls.__dict__.get("__annotations__", {}))
        self.fields = [(name, field_codec_of(hints[name])) for name in names]
        self.ctype = _structure(f"{cls.__name__}State", [(name, codec.ctype) for name, codec in self.fields])
        # (offset, codec) of each field owning memory, directly or through a class field.
        self.owned = [
            (getattr(self.ctype, name).offset, codec)
            for name, codec in self.fields
            if isinstance(codec, (_ListCodec, _OwnedStrCodec, _DictCodec, _StorageCodec))
            or (isinstance(codec, _PlaceCodec) and codec.layout.owned)
        ]
        defaults = cls.__dict__.get("__vernon_defaults__", {})
        # Field order is initialization order, as the compiled initializer writes it: zeroed memory, then each field's
        # default or, for a class field, that class's own initialization, and for a storage of static extents with
        # no default, zeroed elements.
        self.steps: list[tuple[str, Codec, Any]] = [
            (name, codec, defaults[name] if isinstance(codec, _StorageCodec) else codec.value(defaults[name]))
            if name in defaults
            else (name, codec, None)
            for name, codec in self.fields
            if name in defaults or isinstance(codec, _PlaceCodec) or (isinstance(codec, _StorageCodec) and codec.static)
        ]
        for name, codec in self.fields:
            cls.__dict__[name].codec = codec

    def view(self, state: Any) -> Any:
        instance = self.cls.__new__(self.cls)
        instance._vh_state = state
        return instance

    def initialize(self, instance: Any) -> None:
        state = instance._vh_state
        ctypes.memset(ctypes.addressof(state), 0, ctypes.sizeof(state))
        for name, codec, default in self.steps:
            if isinstance(codec, _PlaceCodec):
                codec.layout.initialize(getattr(instance, name))
            elif isinstance(codec, _StorageCodec) and default is None:
                codec.allocate_zeros(ctypes.addressof(getattr(state, name)), _allocator(self.cls))
            else:
                setattr(instance, name, default)

    def release(self, address: int, allocator: int | None) -> None:
        """The compiled fini: frees what the instance at `address` owns and zeroes those records."""

        for offset, codec in self.owned:
            codec.release(address + offset, allocator)


def _finalize(state: Any, built: ClassLayout, runtime: HostRuntime) -> None:
    built.release(ctypes.addressof(state), runtime.allocator.value)


_BUILDING: set[type[Any]] = set()


def class_layout(cls: type[Any]) -> ClassLayout:
    built = cls.__dict__.get("__vernon_layout__")
    if built is None:
        if cls in _BUILDING:
            raise TypeError(f"Host class '{cls.__name__}' contains itself")
        _BUILDING.add(cls)
        try:
            built = ClassLayout(cls)
        finally:
            _BUILDING.discard(cls)
        cls.__vernon_layout__ = built
    return built


def address_of(instance: Any) -> int:
    """The address of a Host class instance's state, which C callers and the reflection table's methods take."""

    return ctypes.addressof(instance._vh_state)


def _initialize_new(self: Any) -> None:
    built = class_layout(type(self))
    self._vh_state = built.ctype()
    if built.owned:
        # An instance Python creates owns its containers; one C creates is freed by C through the table's fini.
        # The finalizer holds the state, not the instance, so views of the state outliving the instance read
        # zeroed, empty records.
        weakref.finalize(self, _finalize, self._vh_state, built, class_runtime(type(self)))
    built.initialize(self)


def _repr(self: Any) -> str:
    names = [name for name, _ in class_layout(type(self)).fields]
    return f"{type(self).__name__}(" + ", ".join(f"{name}={getattr(self, name)!r}" for name in names) + ")"


def _method(cls: type[Any], function: Callable[..., Any]) -> Callable[..., Any]:
    entry: list[Entry] = []

    @functools.wraps(function)
    def method(self: Any, *args: Any) -> Any:
        if _ACTIVE.get() is not None:
            return function(self, *args)
        if not entry:
            entry.append(Entry.of(function, cls))
        return class_runtime(cls).invoke(entry[0], (self, *args))

    method.__vernon_function__ = function  # type: ignore[attr-defined]
    return method


def make_host_class(cls: type[Any]) -> type[Any]:
    """Turns a `@host` class into one whose instances hold their fields in C memory. Field defaults are evaluated
    once, when the class body runs, and copied into each new instance."""

    from .function import HostFunction

    if "__init__" in cls.__dict__:
        raise TypeError(f"Host class '{cls.__name__}' defines no '__init__'; fields initialize from defaults")
    names = list(cls.__dict__.get("__annotations__", {}))
    defaults: dict[str, Any] = {}
    for name in names:
        if name in cls.__dict__:
            defaults[name] = cls.__dict__[name]
        setattr(cls, name, _Field(name))
    for name, member in list(cls.__dict__.items()):
        function = member.function if isinstance(member, HostFunction) else member
        if inspect.isfunction(function) and name not in names:
            setattr(cls, name, _method(cls, function))
    cls.__vernon_defaults__ = defaults
    cls.__init__ = _initialize_new  # type: ignore[misc]
    cls.__repr__ = _repr  # type: ignore[method-assign]
    return cls


def class_runtime(cls: type[Any]) -> HostRuntime:
    """The runtime a class's methods run with when called from Python: its HostModule's, or, for a class imported
    as an ordinary module, a HostModule attached to that module with the default registry. That one binds each
    extern at its first call, so methods calling no extern run without bindings for the module's other externs."""

    runtime = cls.__dict__.get("__vernon_runtime__")
    if runtime is None:
        from .module import HostModule

        HostModule.attach(sys.modules[cls.__module__], eager=False)
        runtime = cls.__dict__["__vernon_runtime__"]
    return runtime


def check_layout(cls: type[Any], program: HostProgram) -> None:
    """Fails when ctypes lays a class out differently from the compiler, which would make C callers and packaged
    builds read other bytes than development writes."""

    built = class_layout(cls)
    expected = layout(ConcreteType("struct", cls.__name__), program.struct_fields)
    offsets = tuple(getattr(built.ctype, name).offset for name, _ in built.fields)
    if ctypes.sizeof(built.ctype) != expected.size or offsets != expected.offsets:
        raise RuntimeError(
            f"Host class '{cls.__name__}' has ctypes layout size {ctypes.sizeof(built.ctype)} offsets {offsets}, "
            f"but the compiler's is size {expected.size} offsets {expected.offsets}"
        )


# Calls.


@dataclass(frozen=True)
class Entry:
    """A Python function Host code runs, with the codecs of its parameters and result."""

    function: Callable[..., Any]
    parameters: tuple[Codec, ...]
    result: Codec | None

    @classmethod
    def of(cls, function: Callable[..., Any], owner: type[Any] | None = None) -> Entry:
        hints = get_type_hints(function, include_extras=True)
        names = list(inspect.signature(function).parameters)
        parameters = tuple(
            _PlaceCodec(owner) if owner is not None and index == 0 else codec_of(hints[name])
            for index, name in enumerate(names)
        )
        returned = hints.get("return", type(None))
        return cls(function, parameters, None if returned is type(None) else codec_of(returned))


@dataclass(frozen=True)
class _ExternShape:
    """An extern declaration's annotations. A generic extern's type parameter is where Host code passes the Struct
    class, and its TypeVar stands for that class in the other annotations."""

    type_index: int | None
    variable: Any
    parameters: tuple[Any, ...]
    result: Any

    @classmethod
    def of(cls, declaration: ExternFunction) -> _ExternShape:
        hints = get_type_hints(declaration.function, include_extras=True)
        names = list(inspect.signature(declaration.function).parameters)
        type_index, variable, parameters = None, None, []
        for index, name in enumerate(names):
            annotation = hints[name]
            if get_origin(annotation) is type:
                type_index, variable = index, get_args(annotation)[0]
            else:
                parameters.append(annotation)
        returned = hints.get("return", type(None))
        return cls(type_index, variable, tuple(parameters), None if returned is type(None) else returned)

    def codecs(self, bound: type[Any] | None) -> tuple[tuple[Codec, ...], Codec | None]:
        def resolve(annotation: Any) -> Any:
            return bound if isinstance(annotation, TypeVar) and annotation is self.variable else annotation

        parameters = tuple(codec_of(resolve(annotation)) for annotation in self.parameters)
        return parameters, None if self.result is None else codec_of(resolve(self.result))


def _ctype_argument(codec: Codec) -> Any:
    return codec.ctype if codec.simple or isinstance(codec, _StrCodec) else ctypes.c_void_p


class HostRuntime:
    """Runs the Host code of `programs` in this process: the registry its extern calls dispatch to, and the
    reflection table through which C callers reach it. `entries` maps a function symbol to the Python function that
    runs it. With `eager`, every extern the programs can call binds before any code runs, so a missing symbol fails
    before any effect happens; otherwise each binds at its first call."""

    def __init__(
        self,
        programs: list[HostProgram],
        registry: ExternRegistry,
        entries: Callable[[str], Entry],
        prefix: str = "vernon_host_",
        *,
        eager: bool = True,
        module_name: str | None = None,
    ):
        self.programs = programs
        self.eager = eager
        self.registry = registry
        self.prefix = prefix
        self.module_name = module_name
        self.errors: list[BaseException] = []
        self._entries = entries
        self._shapes: dict[int, _ExternShape] = {}
        self._externs: dict[tuple[int, Any], Callable[..., Any]] = {}
        self._table: BuiltTable | None = None
        self._trampolines: dict[str, Any] = {}
        self._finis: dict[ConcreteType, Any] = {}
        # The allocator slot of the module's table, which `vh_set_allocator` writes; null is VernonRuntime's default.
        self.allocator = ctypes.c_void_p()
        self._revision = -1
        self._bind()

    def _bind(self) -> None:
        for program in self.programs if self.eager else ():
            for extern in program.reachable_externs():
                self.registry.resolve(extern.symbol, extern.schema)
        self._externs.clear()
        self._revision = self.registry.revision

    @contextmanager
    def active(self) -> Iterator[None]:
        if self.registry.revision != self._revision:
            self._bind()
        token = _ACTIVE.set(self)
        try:
            yield
        finally:
            _ACTIVE.reset(token)

    def invoke(self, entry: Entry, args: Sequence[Any]) -> Any:
        """Runs `entry` for a Python caller: arguments become Host values, and so does the result."""

        if len(args) != len(entry.parameters):
            raise TypeError(f"'{entry.function.__name__}' takes {len(entry.parameters)} arguments, got {len(args)}")
        values = [codec.value(item) for codec, item in zip(entry.parameters, args, strict=True)]
        with self.active():
            returned = entry.function(*values)
        return None if entry.result is None else entry.result.value(returned)

    # Externs.

    def call_extern(self, declaration: ExternFunction, args: tuple[Any, ...]) -> Any:
        shape = self._shapes.get(id(declaration))
        if shape is None:
            shape = self._shapes[id(declaration)] = _ExternShape.of(declaration)
        bound = args[shape.type_index] if shape.type_index is not None else None
        key = (id(declaration), bound)
        call = self._externs.get(key)
        if call is None:
            call = self._externs[key] = self._extern(declaration, shape, bound)
        return call(*args)

    def _extern(self, declaration: ExternFunction, shape: _ExternShape, bound: Any) -> Callable[..., Any]:
        binding = self.registry.resolve(declaration.symbol, declaration.schema)
        parameters, result = shape.codecs(bound)
        index = shape.type_index
        descriptor = None if bound is None else self.descriptor(bound.__name__)

        if binding.python is not None:
            python = binding.python

            def call_python(*args: Any) -> Any:
                values = list(args)
                if index is not None:
                    del values[index]
                converted = [codec.boundary(codec.value(item)) for codec, item in zip(parameters, values, strict=True)]
                if index is not None:
                    converted.insert(index, descriptor)
                returned = python(*converted)
                return None if result is None else result.value(returned)

            return call_python

        status = declaration.abi == ExternAbi.STATUS
        argument_types = [_ctype_argument(codec) for codec in parameters]
        if index is not None:
            argument_types.insert(index, ctypes.c_void_p)
        if status:
            if result is not None:
                argument_types.append(ctypes.c_void_p)
            result_type: Any = ctypes.c_int32
        else:
            result_type = None if result is None else result.ctype
        native = ctypes.CFUNCTYPE(result_type, *argument_types)(binding.native)
        symbol = declaration.symbol

        def call_native(*args: Any) -> Any:
            values = list(args)
            if index is not None:
                del values[index]
            alive: list[Any] = []
            arguments: list[Any] = []
            for codec, item in zip(parameters, values, strict=True):
                if isinstance(codec, _StrCodec):
                    data = str(item).encode("utf-8")
                    alive.append(data)
                    arguments.append(HostStr(data, len(data)))
                elif codec.simple:
                    arguments.append(codec.write(codec.value(item)))
                else:
                    converted = codec.write(item)
                    alive.append(converted)
                    arguments.append(ctypes.addressof(converted))
            if index is not None:
                arguments.insert(index, ctypes.addressof(descriptor))
            if not status:
                returned = native(*arguments)
                return None if result is None else result.value(returned)
            out = None if result is None else result.ctype()
            if out is not None:
                arguments.append(ctypes.addressof(out))
            code = native(*arguments)
            if code != 0:
                raise HostExecutionError(code, f"extern '{symbol}' failed")
            if out is None or result is None:
                return None
            return result.read(out.value if result.simple else out)

        return call_native

    # The C face: the reflection table and trampolines.

    @property
    def built_table(self) -> BuiltTable:
        if self._table is None:
            self._table = BuiltTable(
                describe(self.programs, self.prefix, self.module_name),
                self._trampoline_address,
                self._fini_address,
                self.allocator,
            )
        return self._table

    def _struct_fields(self, name: str) -> tuple[tuple[str, ConcreteType], ...]:
        return next(program.structs[name].fields for program in self.programs if name in program.structs)

    def release(self, address: int, value_type: ConcreteType) -> None:
        """What the table's fini of `value_type` does: frees the memory the value at `address` owns."""

        if value_type.kind == "str":
            release_record(address, 1, 1, self.allocator.value)
        elif value_type.kind == "dict":
            key, item = (layout(part, self._struct_fields) for part in value_type.arguments)  # type: ignore[arg-type]
            str_key = key_of(value_type).kind == "str"
            entry = dict_layout((key.size, key.alignment), (item.size, item.alignment), str_key)
            release_dict(address, entry, self.allocator.value)
        elif value_type.kind == "list":
            element = layout(element_of(value_type), self._struct_fields)
            release_record(address, element.size, element.alignment, self.allocator.value)
        elif value_type.kind == "storage":
            element = layout(element_of(value_type), self._struct_fields)
            release_storage(address, storage_rank(value_type), element.size, self.allocator.value)
        elif value_type.kind == "struct":
            fields = self._struct_fields(value_type.name)
            offsets = layout(value_type, self._struct_fields).offsets
            for (_, member), offset in zip(fields, offsets, strict=True):
                self.release(address + offset, member)

    def _fini_address(self, value_type: ConcreteType) -> int:
        trampoline = self._finis.get(value_type)
        if trampoline is None:

            def fini(address: int) -> None:
                try:
                    self.release(address, value_type)
                except BaseException as error:  # noqa: BLE001 - fini returns nothing; reported through `errors`.
                    self.errors.append(error)

            trampoline = self._finis[value_type] = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(fini)
        return ctypes.cast(trampoline, ctypes.c_void_p).value or 0

    @property
    def table(self) -> Any:
        return self.built_table.module

    def descriptor(self, struct_name: str) -> VhType:
        return self.built_table.descriptor(ConcreteType("struct", struct_name))

    def address(self, symbol: str) -> int:
        """The address of an exported symbol, its prefix included, as the packaged object would export it."""

        if symbol == self.prefix + "module":
            return self.built_table.address
        for program in self.programs:
            for name, function in program.exports:
                if self.prefix + name == symbol:
                    return self._trampoline_address(function)
        raise KeyError(f"the module exports no symbol '{symbol}'")

    def _trampoline_address(self, symbol: str) -> int:
        trampoline = self._trampolines.get(symbol)
        if trampoline is None:
            trampoline = self._trampolines[symbol] = self._trampoline(self._entries(symbol))
        return ctypes.cast(trampoline, ctypes.c_void_p).value or 0

    def _trampoline(self, entry: Entry) -> Any:
        """A C function with the Host convention, `int32_t (params..., Result *out)`, that runs `entry`. An exception
        other than a Host status cannot unwind into C, so it is kept in `errors` and the call returns 1."""

        argument_types = [_ctype_argument(codec) for codec in entry.parameters]
        if entry.result is not None:
            argument_types.append(ctypes.c_void_p)

        def run(*raw: Any) -> int:
            try:
                values = []
                for codec, item in zip(entry.parameters, raw, strict=False):
                    if isinstance(codec, _PlaceCodec):
                        values.append(codec.at(item))
                    elif codec.simple:
                        values.append(codec.read(item))
                    else:
                        values.append(codec.read(codec.ctype.from_address(item)))
                with self.active():
                    returned = entry.function(*values)
                result = entry.result
                if result is not None:
                    if result.simple:
                        result.ctype.from_address(raw[-1]).value = result.write(result.value(returned))
                    else:
                        converted = result.write(returned)
                        ctypes.memmove(raw[-1], ctypes.addressof(converted), ctypes.sizeof(converted))
                return 0
            except HostExecutionError as error:
                return error.status
            except BaseException as error:  # noqa: BLE001 - reported to the caller through `errors`.
                self.errors.append(error)
                return 1

        return ctypes.CFUNCTYPE(ctypes.c_int32, *argument_types)(run)


def initializer(cls: type[Any]) -> Entry:
    """The entry the reflection table's `init` runs: the compiled initializer's convention over zeroed memory."""

    built = class_layout(cls)

    def initialize(instance: Any) -> None:
        built.initialize(instance)

    return Entry(initialize, (_PlaceCodec(cls),), None)


def active_runtime() -> HostRuntime | None:
    return _ACTIVE.get()


__all__ = [
    "ClassLayout",
    "Codec",
    "Entry",
    "HostRuntime",
    "active_runtime",
    "address_of",
    "check_layout",
    "class_layout",
    "class_runtime",
    "codec_of",
    "initializer",
    "make_host_class",
]
