from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TypeExpr:
    name: str
    arguments: tuple[Any, ...] = ()


class _TypeConstructor:

    def __init__(self, name: str):
        self.name = name

    def __class_getitem__(cls, arguments: Any) -> TypeExpr:
        if not isinstance(arguments, tuple):
            arguments = (arguments, )
        return TypeExpr(cls.__name__, arguments)


class Tensor(_TypeConstructor):
    pass


class Array(_TypeConstructor):
    pass


class Buffer(_TypeConstructor):
    pass


class Texture(_TypeConstructor):
    pass


class When(_TypeConstructor):
    pass


class vec(_TypeConstructor):
    pass


class mat(_TypeConstructor):
    pass


class vec2(_TypeConstructor):
    pass


class vec3(_TypeConstructor):
    pass


class vec4(_TypeConstructor):
    pass


class mat2(_TypeConstructor):
    pass


class mat3(_TypeConstructor):
    pass


class mat4(_TypeConstructor):
    pass


@dataclass(frozen=True)
class _Scalar:
    name: str


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
    return _annotation("instance",
                       *((location, divisor) if location is not None else ()))
