from __future__ import annotations

from types import MappingProxyType
from typing import Annotated, Any, get_args, get_origin, get_type_hints

import numpy as np

from .types import TypeExpr, _Scalar

_DTYPES = {
    "bool": np.bool_,
    "i32": np.int32,
    "u32": np.uint32,
    "f16": np.float16,
    "f32": np.float32,
    "f64": np.float64,
}


def _base_annotation(annotation: Any) -> Any:
    if get_origin(annotation) is Annotated:
        return get_args(annotation)[0]
    return annotation


def _shape_and_dtype(
        annotation: TypeExpr) -> tuple[tuple[int, ...], Any] | None:
    name = annotation.name
    arguments = annotation.arguments
    if name in {"vec2", "vec3", "vec4"} and len(arguments) == 1:
        scalar = arguments[0]
        if isinstance(scalar, _Scalar):
            return ((int(name[-1]), ), _DTYPES[scalar.name])
    if name in {"mat2", "mat3", "mat4"} and len(arguments) == 1:
        scalar = arguments[0]
        if isinstance(scalar, _Scalar):
            size = int(name[-1])
            return ((size, size), _DTYPES[scalar.name])
    if name == "vec" and len(arguments) == 2:
        size, scalar = arguments
        if isinstance(size, int) and isinstance(scalar, _Scalar):
            return ((size, ), _DTYPES[scalar.name])
    if name == "mat" and len(arguments) == 3:
        rows, columns, scalar = arguments
        if isinstance(rows, int) and isinstance(columns, int) and isinstance(
                scalar, _Scalar):
            return ((rows, columns), _DTYPES[scalar.name])
    return None


def coerce_host_value(annotation: Any, value: Any, field_name: str) -> Any:
    annotation = _base_annotation(annotation)
    if isinstance(annotation, _Scalar):
        try:
            return _DTYPES[annotation.name](value)
        except (TypeError, ValueError, OverflowError) as error:
            raise TypeError(
                f"field '{field_name}' requires {annotation.name}") from error
    if isinstance(annotation, TypeExpr):
        layout = _shape_and_dtype(annotation)
        if layout is None:
            raise TypeError(
                f"field '{field_name}' has no host representation for {annotation.name}"
            )
        shape, dtype = layout
        result = np.array(value, dtype=dtype, copy=True)
        if result.shape != shape:
            raise TypeError(
                f"field '{field_name}' requires shape {shape}, got {result.shape}"
            )
        result.setflags(write=False)
        return result
    if isinstance(annotation, type) and getattr(annotation, "__vernon_dsl__",
                                                (None, {}))[0] == "struct":
        if not isinstance(value, annotation):
            raise TypeError(
                f"field '{field_name}' requires {annotation.__name__}")
        return value
    raise TypeError(
        f"field '{field_name}' has unsupported host annotation {annotation!r}")


def make_shared_struct(cls: type[Any]) -> type[Any]:
    if cls.__bases__ != (object, ):
        raise TypeError("@struct classes do not support inheritance")
    annotations = get_type_hints(
        cls,
        globalns=vars(__import__(cls.__module__, fromlist=["*"])),
        localns={cls.__name__: cls},
        include_extras=True,
    )
    field_names = tuple(annotations)
    cls.__vernon_fields__ = MappingProxyType(dict(annotations))

    def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
        if len(args) > len(field_names):
            raise TypeError(
                f"{cls.__name__} expects {len(field_names)} field values")
        values = dict(zip(field_names, args, strict=False))
        for name, value in kwargs.items():
            if name not in annotations:
                raise TypeError(f"{cls.__name__} has no field named '{name}'")
            if name in values:
                raise TypeError(
                    f"{cls.__name__} got multiple values for '{name}'")
            values[name] = value
        missing = [name for name in field_names if name not in values]
        if missing:
            raise TypeError(
                f"{cls.__name__} missing field value(s): {', '.join(missing)}")
        for name in field_names:
            object.__setattr__(
                self, name,
                coerce_host_value(annotations[name], values[name], name))
        object.__setattr__(self, "_vernon_frozen", True)

    def __setattr__(self: Any, name: str, value: Any) -> None:
        del value
        if getattr(self, "_vernon_frozen", False):
            raise AttributeError(
                f"{cls.__name__} values are immutable; cannot assign '{name}'")
        raise AttributeError(
            f"{cls.__name__} fields are initialized by its constructor")

    cls.__init__ = __init__  # type: ignore[method-assign]
    cls.__setattr__ = __setattr__  # type: ignore[method-assign]
    return cls


def forbid_struct_construction(cls: type[Any]) -> type[Any]:

    def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
        del self, args, kwargs
        raise TypeError(
            f"@struct class '{cls.__name__}' is shader-only and cannot be "
            "instantiated from host Python")

    cls.__init__ = __init__  # type: ignore[method-assign]
    return cls
