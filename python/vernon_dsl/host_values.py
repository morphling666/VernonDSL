from __future__ import annotations

import importlib
import json
from collections.abc import Iterator
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


@dataclass(frozen=True)
class _NativeLayoutNode:
    size: int
    alignment: int
    field_offsets: tuple[int, ...]
    element_stride: int | None


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
    node: _NativeLayoutNode,
    fields: tuple[tuple[str, Any], ...],
    active_structs: frozenset[type[Any]],
    nodes: Iterator[_NativeLayoutNode],
) -> HostAbiLayout:
    layouts: list[HostAbiLayout] = []
    for _, field in fields:
        layouts.append(_consume_native_layout(field, nodes, active_structs))
    names = tuple(name for name, _ in fields)
    if len(node.field_offsets) != len(fields):
        raise RuntimeError("native Value ABI product field count does not match its host type")
    dtype = np.dtype(
        {
            "names": names,
            "formats": [layout.dtype for layout in layouts],
            "offsets": node.field_offsets,
            "itemsize": node.size,
        }
    )
    return HostAbiLayout(dtype, node.size, node.alignment, names, node.field_offsets)


def host_abi_layout(annotation: Any) -> HostAbiLayout:
    """Build an exact NumPy dtype from the native canonical Value ABI plan."""
    nodes = iter(_native_value_abi_plan(annotation))
    layout = _consume_native_layout(annotation, nodes, frozenset())
    if next(nodes, None) is not None:
        raise RuntimeError("native Value ABI plan contains unconsumed layout nodes")
    return layout


def _consume_native_layout(
    annotation: Any,
    nodes: Iterator[_NativeLayoutNode],
    active_structs: frozenset[type[Any]],
) -> HostAbiLayout:
    try:
        node = next(nodes)
    except StopIteration:
        raise RuntimeError("native Value ABI plan ended before its host type") from None
    annotation = _base_annotation(annotation)
    if annotation is int:
        annotation = _Scalar("i32")
    elif annotation is float:
        annotation = _Scalar("f32")
    if isinstance(annotation, _Scalar):
        dtype = np.dtype(_DTYPES[annotation.name])
        if dtype.itemsize != node.size:
            raise RuntimeError("NumPy scalar size does not match the native canonical Value ABI")
        return HostAbiLayout(dtype, node.size, node.alignment)
    if isinstance(annotation, TypeExpr):
        if annotation.name == "Tuple":
            return _product_abi_layout(
                node,
                tuple((str(index), element) for index, element in enumerate(annotation.arguments)),
                active_structs,
                nodes,
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
        element_layout = _consume_native_layout(element, nodes, active_structs)
        if node.element_stride != element_layout.size:
            raise RuntimeError("NumPy cannot represent the native canonical Tensor element stride")
        dtype = np.dtype((element_layout.dtype, shape))
        if dtype.itemsize != node.size:
            raise RuntimeError("NumPy cannot represent the native canonical Tensor Value ABI")
        return HostAbiLayout(dtype, node.size, node.alignment)
    if isinstance(annotation, type) and getattr(annotation, "__vernon_dsl__", (None, {}))[0] == "struct":
        if annotation in active_structs:
            raise TypeError(f"recursive Struct '{annotation.__name__}' has no finite host ABI")
        return _product_abi_layout(
            node,
            tuple(_struct_fields(annotation).items()),
            active_structs | {annotation},
            nodes,
        )
    raise TypeError(f"{annotation!r} is not an ABI-stable host Value")


def _native_value_abi_plan(annotation: Any) -> tuple[_NativeLayoutNode, ...]:
    declarations: dict[str, tuple[tuple[str, str], ...]] = {}

    def describe(value: Any, active: frozenset[type[Any]] = frozenset()) -> tuple[str, tuple[str, ...]]:
        value = _base_annotation(value)
        if value is int:
            value = _Scalar("i32")
        elif value is float:
            value = _Scalar("f32")
        if isinstance(value, _Scalar):
            spelling = "i1" if value.name == "bool" else "i32" if value.name in {"i32", "u32"} else value.name
            return spelling, (value.name,)
        if isinstance(value, TypeExpr):
            if value.name == "Tuple":
                elements = tuple(describe(element, active) for element in value.arguments)
                spelling = f"tuple<{', '.join(element[0] for element in elements)}>"
                return spelling, tuple(dtype for _, dtypes in elements for dtype in dtypes)
            if value.name == "Tensor" and len(value.arguments) == 2:
                element, shape = value.arguments
                if not isinstance(shape, tuple):
                    raise TypeError("Tensor host ABI shape must be a tuple")
            elif value.name in {"Vector", "Matrix"}:
                rank = 1 if value.name == "Vector" else 2
                element = value.arguments[0]
                shape = tuple(value.arguments[1 : rank + 1])
            else:
                raise TypeError(f"{value.name} has no canonical host ABI")
            if not shape or any(not isinstance(extent, int) or extent <= 0 for extent in shape):
                raise TypeError("Tensor host ABI requires a positive static shape")
            element_spelling, element_dtypes = describe(element, active)
            dimensions = "x".join(str(extent) for extent in shape)
            element_base = _base_annotation(element)
            if element_base is int or element_base is float or isinstance(element_base, _Scalar):
                return f"tensor<{dimensions}x{element_spelling}>", element_dtypes
            shape_text = ", ".join(str(extent) for extent in shape)
            return (
                f"!vernon.tensor<{element_spelling}, [{shape_text}]>",
                element_dtypes * int(np.prod(shape, dtype=np.int64)),
            )
        if isinstance(value, type) and getattr(value, "__vernon_dsl__", (None, {}))[0] == "struct":
            if value in active:
                raise TypeError(f"recursive Struct '{value.__name__}' has no finite host ABI")
            field_descriptions = tuple(
                (name, *describe(field, active | {value})) for name, field in _struct_fields(value).items()
            )
            fields = tuple((name, field_spelling) for name, field_spelling, _ in field_descriptions)
            previous = declarations.setdefault(value.__name__, fields)
            if previous != fields:
                raise TypeError(f"conflicting Struct declarations named '{value.__name__}'")
            dtypes = tuple(dtype for _, _, field_dtypes in field_descriptions for dtype in field_dtypes)
            return f'!vernon.struct<"{value.__name__}">', dtypes
        raise TypeError(f"{value!r} is not an ABI-stable host Value")

    type_spelling, logical_dtypes = describe(annotation)
    declaration_lines = []
    for name, fields in sorted(declarations.items()):
        field_text = ", ".join(json.dumps(f"{field_name}:{field_type}") for field_name, field_type in fields)
        declaration_lines.append(
            f'  "vernon.struct"() {{fields = [{field_text}], sym_name = {json.dumps(name)}}} : () -> ()'
        )
    module = "\n".join(
        [
            "module {",
            *declaration_lines,
            f"  func.func private @__vernon_plan_value_abi(%value: {type_spelling})",
            "}",
        ]
    )
    native = importlib.import_module("vernon_dsl._native")
    raw_nodes = native._plan_value_abi(module, logical_dtypes)
    nodes = tuple(
        _NativeLayoutNode(
            int(size),
            int(alignment),
            tuple(int(offset) for offset in offsets),
            None if element_stride is None else int(element_stride),
        )
        for size, alignment, offsets, element_stride in raw_nodes
    )
    if not nodes:
        raise RuntimeError("native Value ABI planner returned no layout nodes")
    return nodes


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
