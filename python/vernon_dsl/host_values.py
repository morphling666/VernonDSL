from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class HostAbiLayout:
    dtype: np.dtype[Any]
    size: int
    alignment: int
    field_names: tuple[str, ...] = ()
    field_offsets: tuple[int, ...] = ()


def _align_to(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _struct_fields(cls: type[Any]) -> MappingProxyType[str, Any]:
    existing = getattr(cls, "__vernon_fields__", None)
    if existing is not None:
        return existing
    annotations = get_type_hints(
        cls,
        globalns=vars(__import__(cls.__module__, fromlist=["*"])),
        localns={cls.__name__: cls},
        include_extras=True,
    )
    fields = MappingProxyType(dict(annotations))
    cls.__vernon_fields__ = fields
    return fields


def _product_abi_layout(
    fields: tuple[tuple[str, Any], ...],
    active_structs: frozenset[type[Any]],
) -> HostAbiLayout:
    layouts = tuple(host_abi_layout(annotation, active_structs) for _, annotation in fields)
    offset = 0
    alignment = 1
    offsets: list[int] = []
    for layout in layouts:
        offset = _align_to(offset, layout.alignment)
        offsets.append(offset)
        offset += layout.size
        alignment = max(alignment, layout.alignment)
    size = _align_to(offset, alignment)
    names = tuple(name for name, _ in fields)
    dtype = np.dtype(
        {
            "names": names,
            "formats": [layout.dtype for layout in layouts],
            "offsets": offsets,
            "itemsize": size,
        }
    )
    return HostAbiLayout(dtype, size, alignment, names, tuple(offsets))


def host_abi_layout(annotation: Any, active_structs: frozenset[type[Any]] = frozenset()) -> HostAbiLayout:
    """Resolve the canonical Value ABI into an exact NumPy storage dtype."""
    annotation = _base_annotation(annotation)
    if annotation is int:
        annotation = _Scalar("i32")
    elif annotation is float:
        annotation = _Scalar("f32")
    if isinstance(annotation, _Scalar):
        dtype = np.dtype(_DTYPES[annotation.name])
        return HostAbiLayout(dtype, dtype.itemsize, dtype.itemsize)
    if isinstance(annotation, TypeExpr):
        if annotation.name == "Tuple":
            return _product_abi_layout(
                tuple((str(index), element) for index, element in enumerate(annotation.arguments)),
                active_structs,
            )
        if annotation.name == "Tensor" and len(annotation.arguments) == 2:
            element, shape = annotation.arguments
            if not isinstance(shape, tuple):
                raise TypeError("Tensor host ABI shape must be a tuple")
        elif annotation.name in {"Vector", "Matrix"}:
            rank = 1 if annotation.name == "Vector" else 2
            element = annotation.arguments[0]
            shape = tuple(annotation.arguments[1 : rank + 1])
        else:
            raise TypeError(f"{annotation.name} has no canonical host ABI")
        if not shape or any(not isinstance(extent, int) or extent <= 0 for extent in shape):
            raise TypeError("Tensor host ABI requires a positive static shape")
        element_layout = host_abi_layout(element, active_structs)
        count = int(np.prod(shape, dtype=np.int64))
        dtype = np.dtype((element_layout.dtype, shape))
        return HostAbiLayout(dtype, element_layout.size * count, element_layout.alignment)
    if isinstance(annotation, type) and getattr(annotation, "__vernon_dsl__", (None, {}))[0] == "struct":
        if annotation in active_structs:
            raise TypeError(f"recursive Struct '{annotation.__name__}' has no finite host ABI")
        return _product_abi_layout(
            tuple(_struct_fields(annotation).items()),
            active_structs | {annotation},
        )
    raise TypeError(f"{annotation!r} is not an ABI-stable host Value")


def host_scalar_shape(annotation: Any) -> tuple[_Scalar, tuple[int, ...]]:
    """Return the scalar leaf and logical shape of a projectable field."""
    annotation = _base_annotation(annotation)
    if annotation is int:
        return _Scalar("i32"), ()
    if annotation is float:
        return _Scalar("f32"), ()
    if isinstance(annotation, _Scalar):
        return annotation, ()
    if not isinstance(annotation, TypeExpr):
        raise TypeError("Struct field projection requires a Scalar or Tensor field")
    if annotation.name == "Tensor" and len(annotation.arguments) == 2:
        element, shape = annotation.arguments
        if not isinstance(shape, tuple):
            raise TypeError("Tensor field shape must be a tuple")
    elif annotation.name in {"Vector", "Matrix"}:
        rank = 1 if annotation.name == "Vector" else 2
        element = annotation.arguments[0]
        shape = tuple(annotation.arguments[1 : rank + 1])
    else:
        raise TypeError("Struct field projection requires a Scalar or Tensor field")
    scalar, inner_shape = host_scalar_shape(element)
    return scalar, (*shape, *inner_shape)


def pack_host_value(annotation: Any, value: Any, field_name: str = "value") -> Any:
    """Pack an immutable logical Value into its canonical NumPy ABI value."""
    annotation = _base_annotation(annotation)
    if annotation is int:
        annotation = _Scalar("i32")
    elif annotation is float:
        annotation = _Scalar("f32")
    if isinstance(annotation, _Scalar):
        return coerce_host_value(annotation, value, field_name)
    if isinstance(annotation, TypeExpr):
        if annotation.name == "Tuple":
            if not isinstance(value, tuple) or len(value) != len(annotation.arguments):
                raise TypeError(f"field '{field_name}' requires a {len(annotation.arguments)}-element Tuple")
            return tuple(
                pack_host_value(element_type, element, f"{field_name}[{index}]")
                for index, (element_type, element) in enumerate(zip(annotation.arguments, value, strict=True))
            )
        return coerce_host_value(annotation, value, field_name)
    if isinstance(annotation, type) and getattr(annotation, "__vernon_dsl__", (None, {}))[0] == "struct":
        if not isinstance(value, annotation):
            raise TypeError(f"field '{field_name}' requires {annotation.__name__}")
        return tuple(
            pack_host_value(field_type, getattr(value, name), f"{field_name}.{name}")
            for name, field_type in _struct_fields(annotation).items()
        )
    raise TypeError(f"field '{field_name}' has unsupported host annotation {annotation!r}")


def unpack_host_value(annotation: Any, value: Any) -> Any:
    """Materialize an immutable logical Value from canonical NumPy ABI bytes."""
    annotation = _base_annotation(annotation)
    if annotation is int:
        annotation = _Scalar("i32")
    elif annotation is float:
        annotation = _Scalar("f32")
    if isinstance(annotation, _Scalar):
        return _DTYPES[annotation.name](value)
    if isinstance(annotation, TypeExpr):
        if annotation.name == "Tuple":
            return tuple(
                unpack_host_value(element_type, value[str(index)])
                for index, element_type in enumerate(annotation.arguments)
            )
        result = np.array(value, copy=True)
        result.setflags(write=False)
        return result
    if isinstance(annotation, type) and getattr(annotation, "__vernon_dsl__", (None, {}))[0] == "struct":
        if not getattr(annotation, "__vernon_dsl__", (None, {}))[1].get("shared", False):
            raise TypeError(f"device-only Struct '{annotation.__name__}' cannot materialize as a host Value")
        return annotation(
            *(unpack_host_value(field_type, value[name]) for name, field_type in _struct_fields(annotation).items())
        )
    raise TypeError(f"{annotation!r} has no host Value representation")


def _base_annotation(annotation: Any) -> Any:
    if get_origin(annotation) is Annotated:
        return get_args(annotation)[0]
    return annotation


def _shape_and_dtype(annotation: TypeExpr) -> tuple[tuple[int, ...], Any] | None:
    name = annotation.name
    arguments = annotation.arguments
    if name == "Tensor" and len(arguments) == 2:
        scalar, shape = arguments
        if (
            isinstance(scalar, _Scalar)
            and isinstance(shape, tuple)
            and shape
            and all(isinstance(dimension, int) and dimension > 0 for dimension in shape)
        ):
            return (shape, _DTYPES[scalar.name])
    if name in {"Vector", "Matrix"}:
        rank = 1 if name == "Vector" else 2
        if (
            len(arguments) == rank + 1
            and isinstance(arguments[0], _Scalar)
            and all(isinstance(dimension, int) and dimension > 0 for dimension in arguments[1:])
        ):
            return (tuple(arguments[1:]), _DTYPES[arguments[0].name])
    return None


def coerce_host_value(annotation: Any, value: Any, field_name: str) -> Any:
    annotation = _base_annotation(annotation)
    if isinstance(annotation, _Scalar):
        try:
            return _DTYPES[annotation.name](value)
        except (TypeError, ValueError, OverflowError) as error:
            raise TypeError(f"field '{field_name}' requires {annotation.name}") from error
    if isinstance(annotation, TypeExpr):
        if annotation.name == "Tuple":
            if not isinstance(value, tuple) or len(value) != len(annotation.arguments):
                raise TypeError(f"field '{field_name}' requires a {len(annotation.arguments)}-element Tuple")
            return tuple(
                coerce_host_value(element_type, element, f"{field_name}[{index}]")
                for index, (element_type, element) in enumerate(zip(annotation.arguments, value, strict=True))
            )
        layout = _shape_and_dtype(annotation)
        if layout is None:
            raise TypeError(f"field '{field_name}' has no host representation for {annotation.name}")
        shape, dtype = layout
        result = np.array(value, dtype=dtype, copy=True)
        if result.shape != shape:
            raise TypeError(f"field '{field_name}' requires shape {shape}, got {result.shape}")
        result.setflags(write=False)
        return result
    if isinstance(annotation, type) and getattr(annotation, "__vernon_dsl__", (None, {}))[0] == "struct":
        if not isinstance(value, annotation):
            raise TypeError(f"field '{field_name}' requires {annotation.__name__}")
        return value
    raise TypeError(f"field '{field_name}' has unsupported host annotation {annotation!r}")


def make_shared_struct(cls: type[Any]) -> type[Any]:
    if cls.__bases__ != (object,):
        raise TypeError("@struct classes do not support inheritance")
    annotations = _struct_fields(cls)
    field_names = tuple(annotations)
    cls.__vernon_fields__ = MappingProxyType(dict(annotations))

    def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
        if len(args) > len(field_names):
            raise TypeError(f"{cls.__name__} expects {len(field_names)} field values")
        values = dict(zip(field_names, args, strict=False))
        for name, value in kwargs.items():
            if name not in annotations:
                raise TypeError(f"{cls.__name__} has no field named '{name}'")
            if name in values:
                raise TypeError(f"{cls.__name__} got multiple values for '{name}'")
            values[name] = value
        missing = [name for name in field_names if name not in values]
        if missing:
            raise TypeError(f"{cls.__name__} missing field value(s): {', '.join(missing)}")
        for name in field_names:
            object.__setattr__(self, name, coerce_host_value(annotations[name], values[name], name))
        object.__setattr__(self, "_vernon_frozen", True)

    def __setattr__(self: Any, name: str, value: Any) -> None:
        del value
        if getattr(self, "_vernon_frozen", False):
            raise AttributeError(f"{cls.__name__} values are immutable; cannot assign '{name}'")
        raise AttributeError(f"{cls.__name__} fields are initialized by its constructor")

    cls.__init__ = __init__  # type: ignore[method-assign]
    cls.__setattr__ = __setattr__  # type: ignore[method-assign]
    return cls


def forbid_struct_construction(cls: type[Any]) -> type[Any]:
    _struct_fields(cls)

    def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
        del self, args, kwargs
        raise TypeError(f"@struct class '{cls.__name__}' is shader-only and cannot be instantiated from host Python")

    cls.__init__ = __init__  # type: ignore[method-assign]
    return cls
