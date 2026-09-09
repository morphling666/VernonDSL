from __future__ import annotations

import builtins
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Self, TypeAlias

import numpy as np

from ._dtypes import NUMPY_DTYPE_BY_SCALAR


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


@dataclass(frozen=True)
class _DynamicExtent:
    def __repr__(self) -> str:
        return "dyn"


dyn = _DynamicExtent()


class When(_TypeConstructor):
    pass


class Tuple(_TypeConstructor):
    def __new__(cls, *values: Any) -> builtins.tuple[Any, ...]:
        return builtins.tuple(values)


class Tensor(np.ndarray[Any, Any]):
    """Construct an immutable rectangular host Tensor Value."""

    @classmethod
    def __class_getitem__(cls, arguments: Any) -> Any:
        if not isinstance(arguments, tuple):
            arguments = (arguments,)
        return TypeExpr(cls.__name__, arguments)

    def __new__(cls, values: Any) -> Tensor:
        shape, components = cls._shape_and_values(values)
        return cls._from_components(components, shape)

    def __array_finalize__(self, source: np.ndarray[Any, Any] | None) -> None:
        del source

    @classmethod
    def _shape_and_values(cls, current: Any) -> tuple[tuple[int, ...], tuple[Any, ...]]:
        if isinstance(current, np.ndarray):
            if current.ndim == 0:
                return (), (current[()],)
            return current.shape, tuple(current.reshape(-1))
        if not isinstance(current, (list, builtins.tuple)):
            return (), (current,)
        if not current:
            raise TypeError(f"{cls.__name__} requires a non-empty rectangular sequence")
        children = tuple(cls._shape_and_values(value) for value in current)
        child_shape = children[0][0]
        if any(shape != child_shape for shape, _ in children[1:]):
            raise TypeError(f"{cls.__name__} requires a non-empty rectangular sequence")
        return (len(current), *child_shape), tuple(component for _, components in children for component in components)

    @classmethod
    def _from_components(cls, components: tuple[Any, ...], shape: tuple[int, ...]) -> Self:
        return _host_tensor(cls.__name__, components, shape).view(cls)

    @classmethod
    def _result(cls, value: Any) -> Self:
        result = np.asarray(value).view(cls)
        result.setflags(write=False)
        return result

    def __getitem__(self, index: Any) -> Any:
        return super().__getitem__(index)

    def __add__(self, other: Any) -> Any:
        return self._result(np.asarray(self) + other)

    def __radd__(self, other: Any) -> Any:
        return self._result(other + np.asarray(self))

    def __sub__(self, other: Any) -> Any:
        return self._result(np.asarray(self) - other)

    def __rsub__(self, other: Any) -> Any:
        return self._result(other - np.asarray(self))

    def __mul__(self, other: Any) -> Any:
        return self._result(np.asarray(self) * other)

    def __rmul__(self, other: Any) -> Any:
        return self._result(other * np.asarray(self))

    def __truediv__(self, other: Any) -> Any:
        return self._result(np.asarray(self) / other)

    def __neg__(self) -> Any:
        return self._result(-np.asarray(self))


class Vector(Tensor):
    """Construct an immutable rank-one host value from an iterable."""

    def __new__(cls, values: Any) -> Vector:
        components = tuple(values)
        size = sum(np.asarray(component).size for component in components)
        return cls._from_components(components, (size,))

    @property
    def x(self) -> Any:
        return self[0]

    @property
    def y(self) -> Any:
        return self[1]

    @property
    def z(self) -> Any:
        return self[2]

    @property
    def w(self) -> Any:
        return self[3]

    @property
    def xy(self) -> Vector:
        return self._result(self[:2])

    @property
    def xyz(self) -> Vector:
        return self._result(self[:3])

    @property
    def xyzw(self) -> Vector:
        return self._result(self[:4])


class Matrix(Tensor):
    """Construct an immutable rank-two host value from nested iterables."""

    def __new__(cls, values: Any) -> Matrix:
        rows = tuple(tuple(row) for row in values)
        if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
            raise TypeError("Matrix requires a non-empty rectangular sequence")
        return cls._from_components(
            tuple(value for row in rows for value in row),
            (len(rows), len(rows[0])),
        )


class _Scalar(type):
    name: str

    def __new__(
        cls,
        class_name: str,
        bases: tuple[type[Any], ...] = (),
        namespace: dict[str, Any] | None = None,
    ) -> _Scalar:
        resolved_namespace = {"name": class_name} if namespace is None else namespace
        return super().__new__(cls, class_name, bases, resolved_namespace)

    def __init__(
        cls,
        class_name: str,
        bases: tuple[type[Any], ...] = (),
        namespace: dict[str, Any] | None = None,
    ) -> None:
        resolved_namespace = {"name": class_name} if namespace is None else namespace
        super().__init__(class_name, bases, resolved_namespace)

    def __call__(cls, value: Any) -> Any:
        return NUMPY_DTYPE_BY_SCALAR[cls.name].type(value)


class bool(np.bool_, metaclass=_Scalar):
    name = "bool"


class i32(np.int32, metaclass=_Scalar):
    name = "i32"


class u32(np.uint32, metaclass=_Scalar):
    name = "u32"


class f16(np.float16, metaclass=_Scalar):
    name = "f16"


class f32(np.float32, metaclass=_Scalar):
    name = "f32"


class f64(np.float64, metaclass=_Scalar):
    name = "f64"


Sampler = TypeExpr("Sampler")


@dataclass(frozen=True)
class Specialization:
    name: str
    type: _Scalar

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("specialization name must be a non-empty string")
        if self.type not in {bool, i32, u32, f32, f64}:
            raise TypeError("specialization type must be bool, i32, u32, f32, or f64")

    def __bool__(self) -> builtins.bool:
        raise TypeError("Vernon specializations are compile-time-only values")


Feature = Specialization
SpecializationScalar: TypeAlias = builtins.bool | int | float


@dataclass(frozen=True, order=True)
class SpecializationAssignment:
    name: str
    type: str
    value: SpecializationScalar

    @property
    def manifest(self) -> dict[str, object]:
        return {"name": self.name, "value": {"tag": self.type, "value": self.value}}


def specialization_key_data(key: tuple[SpecializationAssignment, ...]) -> list[dict[str, object]]:
    return [assignment.manifest for assignment in key]


def enabled_features(key: tuple[SpecializationAssignment, ...]) -> tuple[str, ...]:
    return tuple(assignment.name for assignment in key if assignment.type == "bool" and assignment.value is True)


def specialization_constants(
    key: tuple[SpecializationAssignment, ...],
) -> tuple[tuple[str, int | float | builtins.bool], ...]:
    return tuple((assignment.name, assignment.value) for assignment in key if assignment.type != "bool")


def specialization(name: str, type: _Scalar) -> Specialization:
    return Specialization(name, type)


def feature(name: str) -> Feature:
    return Specialization(name, bool)


def specialization_assignment(parameter: Specialization, value: object) -> SpecializationAssignment:
    if parameter.type is bool:
        if type(value) is not builtins.bool:
            raise TypeError(f"specialization {parameter.name!r} requires a bool value")
        return SpecializationAssignment(parameter.name, "bool", value)
    if parameter.type is i32:
        if type(value) is not int or not -(1 << 31) <= value < (1 << 31):
            raise TypeError(f"specialization {parameter.name!r} requires an i32 value")
        return SpecializationAssignment(parameter.name, "i32", value)
    if parameter.type is u32:
        if type(value) is not int or not 0 <= value < (1 << 32):
            raise TypeError(f"specialization {parameter.name!r} requires a u32 value")
        return SpecializationAssignment(parameter.name, "u32", value)
    if parameter.type is f32:
        if type(value) not in {int, float} or not math.isfinite(float(value)):
            raise TypeError(f"specialization {parameter.name!r} requires a finite f32 value")
        with np.errstate(over="ignore"):
            canonical = float(np.float32(value))
        if not math.isfinite(canonical):
            raise TypeError(f"specialization {parameter.name!r} is outside the finite f32 range")
        if canonical == 0.0:
            canonical = 0.0
        return SpecializationAssignment(parameter.name, "f32", canonical)
    if parameter.type is f64:
        if type(value) not in {int, float} or not math.isfinite(float(value)):
            raise TypeError(f"specialization {parameter.name!r} requires a finite f64 value")
        canonical = float(value)
        return SpecializationAssignment(parameter.name, "f64", 0.0 if canonical == 0.0 else canonical)
    raise TypeError(f"specialization {parameter.name!r} has an unsupported type")


def specialization_key(
    supplied: Mapping[Specialization, object] | None,
) -> tuple[SpecializationAssignment, ...]:
    values = {} if supplied is None else supplied
    if not isinstance(values, Mapping) or any(not isinstance(parameter, Specialization) for parameter in values):
        raise TypeError("specializations must map vd.specialization(...) values to scalar values")
    assignments = tuple(sorted(specialization_assignment(parameter, value) for parameter, value in values.items()))
    if len({assignment.name for assignment in assignments}) != len(assignments):
        raise ValueError("specializations contain duplicate names")
    return assignments


@dataclass(frozen=True)
class Annotation:
    kind: str
    arguments: tuple[Any, ...]


def _annotation(kind: str, *arguments: Any) -> Annotation:
    return Annotation(kind, arguments)


def attribute(location: int | None = None, divisor: int = 0) -> Annotation:
    return _annotation("attribute", -1 if location is None else location, divisor)


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
