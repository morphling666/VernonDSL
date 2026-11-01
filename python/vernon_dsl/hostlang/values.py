"""Host Values at the Python boundary: what Host functions accept from and return to Python callers and bindings."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ..frontend.model import ConcreteType
from .layout import StructFields, members, tensor_count

_INTEGER_RANGES = {
    "i32": (-(1 << 31), (1 << 31) - 1),
    "u32": (0, (1 << 32) - 1),
    "i64": (-(1 << 63), (1 << 63) - 1),
    "u64": (0, (1 << 64) - 1),
}
_FLOATS = {"f32": np.float32, "f64": np.float64}


class HostExecutionError(RuntimeError):
    """A Host function returned a nonzero status: an extern's status, or a negative HostStatus of its own."""

    def __init__(self, status: int, message: str):
        super().__init__(f"{message} (status {status})")
        self.status = status


def scalar_value(raw: Any, name: str) -> Any:
    """A scalar read from native memory or a ctypes result, as Python holds it."""

    if name == "bool":
        return bool(raw)
    if name in _FLOATS:
        return _FLOATS[name](raw)
    low, high = _INTEGER_RANGES[name]
    return (int(raw) - low) % (high - low + 1) + low


def literal_value(value: bool | int | float, name: str) -> Any:
    """A source literal at the type inference gave it; the emitter prints exactly this value."""

    if name == "bool":
        return bool(value)
    if name in _FLOATS:
        return _FLOATS[name](value)
    low, high = _INTEGER_RANGES[name]
    if isinstance(value, bool):
        return int(value)
    if not isinstance(value, int) or not low <= value <= high:
        raise OverflowError(f"literal {value!r} does not fit in {name}")
    return value


def element_of(value_type: ConcreteType) -> ConcreteType:
    """The element of a Tensor, list or TensorStorage, the value type of a dict, or a scalar itself."""

    if value_type.kind in {"tensor", "list", "dict", "storage"}:
        element = value_type.arguments[-1 if value_type.kind == "dict" else 0]
        assert isinstance(element, ConcreteType)
        return element
    return value_type


def key_of(value_type: ConcreteType) -> ConcreteType:
    key = value_type.arguments[0]
    assert value_type.kind == "dict" and isinstance(key, ConcreteType)
    return key


def from_python(value: Any, value_type: ConcreteType, fields: StructFields) -> Any:
    """Checks and flattens a Python argument: scalars stay scalars, Tensors become flat tuples, and Structs and
    Tuples become member tuples. Structs may be given as mappings, sequences, or objects with the field names. An
    enum is its int32_t value: an IntEnum member is an int, and values are not checked against the members, as in
    C."""

    if value_type.kind == "enum":
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{value_type.spelling} expects an IntEnum member or int, got {type(value).__name__}")
        value_type = ConcreteType("scalar", "i32")
    if value_type.kind == "scalar":
        if value_type.name == "bool":
            return bool(value)
        if value_type.name in _FLOATS:
            return _FLOATS[value_type.name](value)
        integer = int(value)
        low, high = _INTEGER_RANGES[value_type.name]
        if not low <= integer <= high:
            raise OverflowError(f"{integer} is out of range for {value_type.name}")
        return integer
    if value_type.kind == "handle":
        integer = int(getattr(value, "bits", value))
        if not 0 <= integer < 1 << 64:
            raise OverflowError(f"handle bits {integer} do not fit in 64 bits")
        return integer
    if value_type.kind == "tensor":
        element = element_of(value_type)
        flat = np.asarray(value).reshape(-1).tolist()
        if len(flat) != tensor_count(value_type):
            raise ValueError(f"{value_type.spelling} requires {tensor_count(value_type)} elements")
        return tuple(from_python(item, element, fields) for item in flat)
    if value_type.kind == "struct":
        names = [name for name, _ in fields(value_type.name)]
        if isinstance(value, Mapping):
            items: Sequence[Any] = [value[name] for name in names]
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            items = value
        else:
            items = [getattr(value, name) for name in names]
        return tuple(
            from_python(item, member, fields) for item, member in zip(items, members(value_type, fields), strict=True)
        )
    if value_type.kind == "tuple":
        return tuple(
            from_python(item, member, fields) for item, member in zip(value, members(value_type, fields), strict=True)
        )
    raise TypeError(f"{value_type.spelling} cannot cross the Python boundary")


def to_python(value: Any, value_type: ConcreteType, fields: StructFields) -> Any:
    """Structs become dicts in field order, Vectors tuples, Matrices tuples of rows; f32 keeps numpy.float32 so its
    rounding stays visible."""

    if value_type.kind in {"scalar", "handle", "enum"}:
        return value
    if value_type.kind == "tensor":
        shape = tuple(int(extent) for extent in value_type.arguments[1:])  # type: ignore[arg-type]
        return _nested(tuple(value), shape)
    if value_type.kind == "struct":
        return {
            name: to_python(item, member, fields)
            for (name, member), item in zip(fields(value_type.name), value, strict=True)
        }
    if value_type.kind == "tuple":
        return tuple(
            to_python(item, member, fields) for item, member in zip(value, members(value_type, fields), strict=True)
        )
    raise TypeError(f"{value_type.spelling} cannot cross the Python boundary")


def _nested(flat: tuple[Any, ...], shape: tuple[int, ...]) -> tuple[Any, ...]:
    if len(shape) <= 1:
        return flat
    step = len(flat) // shape[0]
    return tuple(_nested(flat[row * step : (row + 1) * step], shape[1:]) for row in range(shape[0]))
