from __future__ import annotations

import hashlib
import importlib
import json
from collections.abc import Iterator, Mapping
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
class TangentScalar:
    dtype: _Scalar


@dataclass(frozen=True)
class TangentTensor:
    shape: tuple[int, ...]
    child: TangentSchema


@dataclass(frozen=True)
class TangentProduct:
    children: tuple[tuple[str, TangentSchema], ...]


@dataclass(frozen=True)
class TangentZero:
    pass


TangentSchema = TangentScalar | TangentTensor | TangentProduct | TangentZero


@dataclass(frozen=True)
class TangentValue:
    _fields: MappingProxyType[str, Any]

    def __getitem__(self, key: str) -> Any:
        return self._fields[key]

    def __getattr__(self, name: str) -> Any:
        try:
            return self._fields[name]
        except KeyError:
            raise AttributeError(name) from None


@dataclass(frozen=True)
class TangentLeaf:
    path: str
    dtype: np.dtype[Any]
    shape: tuple[int, ...]
    byte_offset: int
    inner_byte_strides: tuple[int, ...]


@dataclass(frozen=True)
class TangentLayout:
    primal_element_type: Any
    tangent_schema: TangentSchema
    physical_abi: HostAbiLayout
    leaves: tuple[TangentLeaf, ...]
    layout_hash: str

    @property
    def dtype(self) -> np.dtype[Any]:
        return self.physical_abi.dtype

    @property
    def size(self) -> int:
        return self.physical_abi.size

    @property
    def alignment(self) -> int:
        return self.physical_abi.alignment

    def project(self, path: str) -> TangentLeaf:
        if path == "" and len(self.leaves) == 1 and self.leaves[0].path == "":
            return self.leaves[0]
        if not isinstance(path, str) or not path or any(not component for component in path.split(".")):
            raise ValueError("tangent projection requires a canonical non-empty path")
        for leaf in self.leaves:
            if leaf.path == path:
                return leaf
        schema: TangentSchema = self.tangent_schema
        components = path.split(".")
        component_index = 0
        while component_index < len(components):
            if isinstance(schema, TangentProduct):
                component = components[component_index]
                children = dict(schema.children)
                if component not in children:
                    raise ValueError(f"tangent path '{path}' does not exist")
                schema = children[component]
                component_index += 1
                continue
            if isinstance(schema, TangentTensor) and not isinstance(schema.child, TangentScalar):
                rank = len(schema.shape)
                coordinates = components[component_index : component_index + rank]
                if len(coordinates) != rank:
                    raise ValueError(f"tangent path '{path}' has an incomplete Tensor coordinate")
                for coordinate, extent in zip(coordinates, schema.shape, strict=True):
                    try:
                        index = int(coordinate)
                    except ValueError:
                        raise ValueError(f"tangent path '{path}' has a non-integer Tensor coordinate") from None
                    if str(index) != coordinate or not 0 <= index < extent:
                        raise ValueError(f"tangent path '{path}' has a Tensor coordinate outside its shape")
                schema = schema.child
                component_index += rank
                continue
            raise ValueError(f"tangent path '{path}' does not resolve to a differentiable leaf")
        if isinstance(schema, TangentZero):
            raise ValueError(f"tangent path '{path}' is non-differentiable")
        if isinstance(schema, TangentProduct):
            raise ValueError(f"tangent path '{path}' resolves to a product; select a leaf")
        raise ValueError(f"tangent path '{path}' does not resolve to one physical leaf")


@dataclass(frozen=True)
class _NativeLayoutNode:
    size: int
    alignment: int
    field_offsets: tuple[int, ...]
    element_stride: int | None


def tangent_schema(annotation: Any, active_structs: frozenset[type[Any]] = frozenset()) -> TangentSchema:
    annotation = _base_annotation(annotation)
    if annotation is int:
        annotation = _Scalar("i32")
    elif annotation is float:
        annotation = _Scalar("f32")
    if isinstance(annotation, _Scalar):
        return (
            TangentScalar(_Scalar("f64" if annotation.name == "f64" else "f32"))
            if annotation.name
            in {
                "f16",
                "f32",
                "f64",
            }
            else TangentZero()
        )
    if isinstance(annotation, TypeExpr):
        if annotation.name == "Tuple":
            return TangentProduct(
                tuple(
                    (str(index), tangent_schema(element, active_structs))
                    for index, element in enumerate(annotation.arguments)
                )
            )
        if annotation.name == "Tensor" and len(annotation.arguments) == 2:
            element, shape = annotation.arguments
            if not isinstance(shape, tuple):
                raise TypeError("Tensor tangent shape must be a tuple")
        elif annotation.name in {"Vector", "Matrix"}:
            rank = 1 if annotation.name == "Vector" else 2
            element = annotation.arguments[0]
            shape = tuple(annotation.arguments[1 : rank + 1])
        else:
            raise TypeError(f"{annotation.name} has no tangent schema")
        if any(not isinstance(extent, int) or extent <= 0 for extent in shape):
            raise TypeError("Tensor tangent requires static positive extents")
        return TangentTensor(shape, tangent_schema(element, active_structs))
    if isinstance(annotation, type) and getattr(annotation, "__vernon_dsl__", (None, {}))[0] == "struct":
        if annotation in active_structs:
            raise TypeError(f"recursive Struct '{annotation.__name__}' has no finite tangent")
        return TangentProduct(
            tuple(
                (name, tangent_schema(field, active_structs | {annotation}))
                for name, field in _struct_fields(annotation).items()
            )
        )
    raise TypeError(f"{annotation!r} has no tangent schema")


def _physical_tangent_annotation(schema: TangentSchema) -> Any | None:
    if isinstance(schema, TangentZero):
        return None
    if isinstance(schema, TangentScalar):
        return schema.dtype
    if isinstance(schema, TangentTensor):
        child = _physical_tangent_annotation(schema.child)
        return None if child is None else TypeExpr("Tensor", (child, schema.shape))
    children = tuple(
        physical for _, child in schema.children if (physical := _physical_tangent_annotation(child)) is not None
    )
    return None if not children else TypeExpr("Tuple", children)


def _tangent_dtype(schema: TangentSchema, physical: HostAbiLayout) -> np.dtype[Any]:
    if isinstance(schema, TangentZero):
        raise TypeError("Zero tangent nodes have no physical dtype")
    return _rename_tangent_dtype(schema, physical.dtype)


def _rename_tangent_dtype(schema: TangentSchema, dtype: np.dtype[Any]) -> np.dtype[Any]:
    if isinstance(schema, TangentScalar):
        return dtype
    if isinstance(schema, TangentTensor):
        if dtype.subdtype is None:
            raise RuntimeError("canonical tangent Tensor dtype has no subarray layout")
        child_dtype, shape = dtype.subdtype
        return np.dtype((_rename_tangent_dtype(schema.child, child_dtype), shape))
    if not isinstance(schema, TangentProduct) or dtype.fields is None:
        raise RuntimeError("canonical tangent dtype does not match its logical schema")
    active = tuple((name, child) for name, child in schema.children if _physical_tangent_annotation(child) is not None)
    offsets = tuple(dtype.fields[str(index)][1] for index in range(len(active)))
    formats = tuple(
        _rename_tangent_dtype(child, dtype.fields[str(index)][0]) for index, (_, child) in enumerate(active)
    )
    return np.dtype(
        {"names": tuple(name for name, _ in active), "formats": formats, "offsets": offsets, "itemsize": dtype.itemsize}
    )


def tangent_layout(annotation: Any) -> TangentLayout:
    schema = tangent_schema(annotation)
    physical_annotation = _physical_tangent_annotation(schema)
    if physical_annotation is None:
        raise TypeError(f"{annotation!r} has no differentiable tangent leaves")
    planned = host_abi_layout(physical_annotation)
    physical = HostAbiLayout(
        _tangent_dtype(schema, planned),
        planned.size,
        planned.alignment,
        planned.field_names,
        planned.field_offsets,
    )
    leaves: list[TangentLeaf] = []

    def collect(
        current: TangentSchema,
        dtype: np.dtype[Any],
        path: tuple[str, ...],
        offset: int,
        inner_shape: tuple[int, ...] = (),
        inner_strides: tuple[int, ...] = (),
    ) -> None:
        if isinstance(current, TangentZero):
            return
        if isinstance(current, TangentScalar):
            leaves.append(TangentLeaf(".".join(path), dtype, inner_shape, offset, inner_strides))
            return
        if isinstance(current, TangentTensor):
            if dtype.subdtype is None:
                raise RuntimeError("canonical tangent Tensor dtype has no subarray layout")
            child_dtype, shape = dtype.subdtype
            terminal = current.child
            while isinstance(terminal, TangentTensor):
                terminal = terminal.child
            if not isinstance(terminal, TangentScalar):
                for linear, coordinate in enumerate(np.ndindex(shape)):
                    collect(
                        current.child,
                        child_dtype,
                        (*path, *(str(index) for index in coordinate)),
                        offset + linear * child_dtype.itemsize,
                        inner_shape,
                        inner_strides,
                    )
                return
            strides: list[int] = []
            stride = child_dtype.itemsize
            for extent in reversed(shape):
                strides.append(stride)
                stride *= extent
            collect(
                current.child,
                child_dtype,
                path,
                offset,
                (*inner_shape, *shape),
                (*inner_strides, *reversed(strides)),
            )
            return
        assert dtype.fields is not None
        for name, child in current.children:
            if _physical_tangent_annotation(child) is None:
                continue
            child_dtype, child_offset = dtype.fields[name][:2]
            collect(child, child_dtype, (*path, name), offset + child_offset, inner_shape, inner_strides)

    collect(schema, physical.dtype, (), 0)
    canonical = repr((annotation, schema, physical.dtype.descr, physical.size, physical.alignment, leaves))
    return TangentLayout(
        annotation,
        schema,
        physical,
        tuple(leaves),
        hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    )


def unpack_tangent_value(schema: TangentSchema, value: Any) -> Any:
    if isinstance(schema, TangentZero):
        return None
    if isinstance(schema, TangentScalar):
        return _DTYPES[schema.dtype.name](value)
    if isinstance(schema, TangentTensor):
        terminal = schema.child
        while isinstance(terminal, TangentTensor):
            terminal = terminal.child
        if isinstance(terminal, TangentScalar):
            result = np.array(value, copy=True)
            result.setflags(write=False)
            return result

        def unpack_tensor(current: Any, dimensions: tuple[int, ...]) -> Any:
            if not dimensions:
                return unpack_tangent_value(schema.child, current)
            return tuple(unpack_tensor(current[index], dimensions[1:]) for index in range(dimensions[0]))

        return unpack_tensor(value, schema.shape)
    fields = {
        name: unpack_tangent_value(child, value[name]) if not isinstance(child, TangentZero) else None
        for name, child in schema.children
    }
    if all(name == str(index) for index, (name, _) in enumerate(schema.children)):
        return tuple(fields[str(index)] for index in range(len(fields)))
    return TangentValue(MappingProxyType(fields))


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
        if any(not isinstance(extent, int) or extent <= 0 for extent in shape):
            raise TypeError("Tensor host ABI requires static positive extents")
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
            if any(not isinstance(extent, int) or extent <= 0 for extent in shape):
                raise TypeError("Tensor host ABI requires static positive extents")
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
        fields = _struct_fields(annotation)
        mapping = value if isinstance(value, Mapping) else None
        if mapping is not None:
            unknown = set(value) - set(fields)
            if unknown:
                unknown_field = min(unknown, key=repr)
                raise TypeError(f"field '{field_name}' has unknown field {unknown_field!r}")
            missing = [name for name in fields if name not in value]
            if missing:
                raise TypeError(f"field '{field_name}' is missing field '{missing[0]}'")
        elif not isinstance(value, annotation):
            raise TypeError(f"field '{field_name}' requires {annotation.__name__}")
        return tuple(
            pack_host_value(
                field_type,
                mapping[name] if mapping is not None else getattr(value, name),
                f"{field_name}.{name}",
            )
            for name, field_type in fields.items()
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
