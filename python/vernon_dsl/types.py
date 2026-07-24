from __future__ import annotations

import builtins
from dataclasses import dataclass
from typing import Any

import numpy as np


def _host_dtype(values: tuple[Any, ...]) -> np.dtype[Any]:
    dtypes: list[np.dtype[Any]] = []
    for value in values:
        array = np.asarray(value)
        if isinstance(value, float) and not isinstance(value, np.generic):
            dtypes.append(np.dtype(np.float32))
        elif isinstance(value, int) and not isinstance(value, np.generic):
            dtypes.append(np.dtype(np.int32))
        else:
            dtypes.append(array.dtype)
    return np.result_type(*dtypes) if dtypes else np.dtype(np.float32)


def _host_tensor(name: str, values: tuple[Any, ...], shape: tuple[int, ...]) -> np.ndarray[Any, Any]:
    if not values:
        raise TypeError(f"{name} requires component arguments")
    dtype = _host_dtype(values)
    components = [np.asarray(value, dtype=dtype).reshape(-1) for value in values]
    result = np.concatenate(components)
    expected = int(np.prod(shape))
    if result.size != expected:
        raise TypeError(f"{name} requires exactly {expected} scalar components")
    result = result.reshape(shape)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class TypeExpr:
    name: str
    arguments: tuple[Any, ...] = ()


class _TypeConstructor:
    def __init__(self, name: str):
        self.name = name

    def __class_getitem__(cls, arguments: Any) -> TypeExpr:
        if not isinstance(arguments, tuple):
            arguments = (arguments,)
        return TypeExpr(cls.__name__, arguments)


@dataclass(frozen=True)
class _Access:
    name: str


read = _Access("read")
write = _Access("write")
read_write = _Access("read_write")


class When(_TypeConstructor):
    pass


class Tuple(_TypeConstructor):
    def __new__(cls, *values: Any) -> builtins.tuple[Any, ...]:
        return builtins.tuple(values)


class Tensor:
    """Construct an immutable rectangular host Tensor Value."""

    @classmethod
    def __class_getitem__(cls, arguments: Any) -> TypeExpr:
        if not isinstance(arguments, tuple):
            arguments = (arguments,)
        return TypeExpr("Tensor", arguments)

    def __new__(cls, values: Any) -> np.ndarray[Any, Any]:
        def shape_and_values(current: Any) -> tuple[tuple[int, ...], tuple[Any, ...]]:
            if isinstance(current, np.ndarray):
                if current.ndim == 0:
                    return (), (current[()],)
                return current.shape, tuple(current.reshape(-1))
            if not isinstance(current, (list, builtins.tuple)):
                return (), (current,)
            if not current:
                raise TypeError("Tensor requires a non-empty rectangular sequence")
            children = tuple(shape_and_values(value) for value in current)
            child_shape = children[0][0]
            if any(shape != child_shape for shape, _ in children[1:]):
                raise TypeError("Tensor requires a non-empty rectangular sequence")
            return (len(current), *child_shape), tuple(
                component for _, components in children for component in components
            )

        shape, components = shape_and_values(values)
        if not shape:
            raise TypeError("Tensor requires a non-empty rectangular sequence")
        return _host_tensor("Tensor", components, shape)


class Vector:
    """Construct an immutable rank-one host value from an iterable."""

    @classmethod
    def __class_getitem__(cls, arguments: Any) -> TypeExpr:
        if not isinstance(arguments, tuple):
            arguments = (arguments,)
        return TypeExpr("Vector", arguments)

    def __new__(cls, values: Any) -> np.ndarray[Any, Any]:
        components = tuple(values)
        size = sum(np.asarray(component).size for component in components)
        return _host_tensor("Vector", components, (size,))


class Matrix:
    """Construct an immutable rank-two host value from nested iterables."""

    @classmethod
    def __class_getitem__(cls, arguments: Any) -> TypeExpr:
        if not isinstance(arguments, tuple):
            arguments = (arguments,)
        return TypeExpr("Matrix", arguments)

    def __new__(cls, values: Any) -> np.ndarray[Any, Any]:
        rows = tuple(tuple(row) for row in values)
        if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
            raise TypeError("Matrix requires a non-empty rectangular sequence")
        return _host_tensor("Matrix", tuple(value for row in rows for value in row), (len(rows), len(rows[0])))


@dataclass(frozen=True)
class _Scalar:
    name: str

    def __call__(self, value: Any) -> Any:
        dtypes = {
            "bool": np.bool_,
            "i32": np.int32,
            "u32": np.uint32,
            "f16": np.float16,
            "f32": np.float32,
            "f64": np.float64,
        }
        return dtypes[self.name](value)


bool = _Scalar("bool")
i32 = _Scalar("i32")
u32 = _Scalar("u32")
f16 = _Scalar("f16")
f32 = _Scalar("f32")
f64 = _Scalar("f64")
Sampler = TypeExpr("Sampler")


@dataclass(frozen=True)
class Feature:
    name: str

    def __bool__(self) -> bool:
        raise TypeError("Vernon features are compile-time-only values")


def feature(name: str) -> Feature:
    return Feature(name)


@dataclass(frozen=True)
class Annotation:
    kind: str
    arguments: tuple[Any, ...]


def _annotation(kind: str, *arguments: Any) -> Annotation:
    return Annotation(kind, arguments)


def location(index: int) -> Annotation:
    return _annotation("location", index)


def builtin(name: str) -> Annotation:
    return _annotation("builtin", name)


def uniform(set: int | None = None, binding: int | None = None) -> Annotation:
    if (set is None) != (binding is None):
        raise ValueError("uniform set and binding must be provided together")
    return _annotation("uniform", *((set, binding) if set is not None else ()))


def varying() -> Annotation:
    return _annotation("varying")


def resource(set: int, binding: int) -> Annotation:
    return _annotation("resource", set, binding)


def instance(location: int | None = None, divisor: int = 1) -> Annotation:
    return _annotation("instance", *((location, divisor) if location is not None else ()))
