from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Protocol

from ..language.ast_utils import dotted_name, subscript_items
from ..language.scalar_types import SCALAR_ALIASES, SCALAR_TYPES
from .model import ConcreteType, InterfaceMetadata, is_abi_stable_value, semantic_category


class TypeContext(Protocol):
    structs: dict[str, object]

    def error(self, node: ast.AST, message: str) -> Exception: ...


@dataclass(frozen=True)
class AnnotatedType:
    type: ConcreteType
    metadata: tuple[InterfaceMetadata, ...] = ()


class TypeParser:
    def __init__(self, context: TypeContext):
        self.context = context

    def parse(self, node: ast.AST) -> AnnotatedType:
        if isinstance(node, ast.Constant) and node.value is None:
            raise self.context.error(node, "None is only valid as a function return annotation")
        if isinstance(node, ast.Subscript) and (dotted_name(node.value) or "").split(".")[-1] == "Annotated":
            items = subscript_items(node)
            if len(items) < 2:
                raise self.context.error(node, "Annotated requires a type and at least one metadata item")
            return AnnotatedType(
                self.parse_type(items[0]),
                tuple(self._metadata(item) for item in items[1:]),
            )
        return AnnotatedType(self.parse_type(node))

    def parse_type(self, node: ast.AST) -> ConcreteType:
        type_name = (dotted_name(node) or "").split(".")[-1]
        type_name = SCALAR_ALIASES.get(type_name, type_name)
        if type_name in SCALAR_TYPES:
            return ConcreteType("scalar", type_name)
        if type_name in self.context.structs:
            return ConcreteType("struct", type_name)
        if type_name == "Sampler":
            return ConcreteType("sampler", "Sampler")
        if not isinstance(node, ast.Subscript):
            raise self.context.error(node, f"unknown DSL type '{type_name or ast.dump(node)}'")

        constructor = (dotted_name(node.value) or "").split(".")[-1]
        items = subscript_items(node)
        if constructor == "Tensor":
            if len(items) < 2:
                raise self.context.error(node, "Tensor requires an element type and shape")
            element = self.parse_type(items[0])
            shape_nodes = items[1].elts if len(items) == 2 and isinstance(items[1], ast.Tuple) else items[1:]
            shape = tuple(self._positive_int(item, "tensor dimension") for item in shape_nodes)
            return self._tensor_type(items[0], element, shape)
        if constructor in {"Vector", "Matrix"}:
            rank = 1 if constructor == "Vector" else 2
            if len(items) != rank + 1:
                raise self.context.error(
                    node,
                    f"{constructor} requires an element type followed by {rank} dimension(s)",
                )
            element = self.parse_type(items[0])
            shape = tuple(self._positive_int(item, f"{constructor} dimension") for item in items[1:])
            return self._tensor_type(items[0], element, shape)
        if constructor == "Tuple":
            elements = tuple(self.parse_type(item) for item in items)
            if any(not is_abi_stable_value(element) for element in elements):
                raise self.context.error(node, "Tuple elements must be ABI-stable Values")
            return ConcreteType("tuple", "Tuple", elements)
        if constructor == "TensorStorage":
            if len(items) != 1:
                raise self.context.error(node, "TensorStorage requires one ABI-stable Value element type")
            element = self.parse_type(items[0])
            self._require_storage_element(items[0], element, "TensorStorage")
            return ConcreteType("tensor_storage", "TensorStorage", (element,))
        if constructor == "TensorView":
            if len(items) != 3:
                raise self.context.error(node, "TensorView requires an element type, shape, and access mode")
            element = self.parse_type(items[0])
            self._require_storage_element(items[0], element, "TensorView")
            if not isinstance(items[1], ast.Tuple) or not items[1].elts:
                raise self.context.error(items[1], "TensorView shape must be a non-empty tuple")
            shape = tuple(self._tensor_view_extent(item) for item in items[1].elts)
            access = self._string_or_name(items[2], "TensorView access")
            if access not in {"read", "write", "read_write"}:
                raise self.context.error(items[2], "TensorView access must be read, write, or read_write")
            return ConcreteType("tensor_view", "TensorView", (element, shape, access, "device"))
        if constructor == "Texture":
            if len(items) not in {2, 3}:
                raise self.context.error(
                    node,
                    "Texture requires dimension and sample type, or dimension, storage format, and access",
                )
            dimension = self._string_or_name(items[0], "texture dimension")
            if dimension not in {"2d", "3d", "cube"}:
                raise self.context.error(
                    items[0],
                    "texture dimension must be one of '2d', '3d', or 'cube'",
                )
            if len(items) == 2:
                element = self.parse_type(items[1])
                if element.kind != "scalar" or element.name not in {"f32", "i32", "u32"}:
                    raise self.context.error(items[1], "sampled Texture type must be f32, i32, or u32")
                return ConcreteType("texture", "Texture", (dimension, element, "unknown", "sampled"))
            if dimension == "cube":
                raise self.context.error(items[0], "storage Texture dimension must be '2d' or '3d'")
            format_name = self._string_or_name(items[1], "storage texture format")
            if format_name not in {
                "r8_unorm",
                "r16_float",
                "r32_float",
                "rg8_unorm",
                "rgba8_unorm",
                "rgba16_float",
                "rgba32_float",
            }:
                raise self.context.error(items[1], "unsupported storage texture format")
            access = self._string_or_name(items[2], "storage Texture access")
            if access not in {"read", "write", "read_write"}:
                raise self.context.error(items[2], "storage Texture access must be read, write, or read_write")
            element = ConcreteType("scalar", "f32")
            return ConcreteType("texture", "Texture", (dimension, element, format_name, access))
        raise self.context.error(node, f"unknown DSL type constructor '{constructor}'")

    def _require_storage_element(self, node: ast.AST, element: ConcreteType, constructor: str) -> None:
        if not is_abi_stable_value(element):
            category = semantic_category(element)
            description = category.value.capitalize() if category is not None else f"type kind '{element.kind}'"
            raise self.context.error(
                node,
                f"{constructor} element type must be an ABI-stable Value, not {description} '{element.name}'",
            )

    def _tensor_type(
        self,
        element_node: ast.AST,
        element: ConcreteType,
        shape: tuple[int | str, ...],
    ) -> ConcreteType:
        if not is_abi_stable_value(element):
            category = semantic_category(element)
            description = category.value.capitalize() if category is not None else f"type kind '{element.kind}'"
            raise self.context.error(
                element_node,
                f"Tensor element type must be an ABI-stable Value, not {description} '{element.name}'",
            )
        return ConcreteType("tensor", "Tensor", (element, *shape))

    def _positive_int(self, node: ast.AST, description: str) -> int:
        if (
            not isinstance(node, ast.Constant)
            or not isinstance(node.value, int)
            or isinstance(node.value, bool)
            or node.value <= 0
        ):
            raise self.context.error(node, f"{description} must be a positive integer literal")
        return node.value

    def _tensor_view_extent(self, node: ast.AST) -> int | str:
        if (dotted_name(node) or "").split(".")[-1] == "dyn":
            return "?"
        return self._positive_int(node, "TensorView dimension")

    def _string_or_name(self, node: ast.AST, description: str) -> str:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        value = dotted_name(node)
        if value:
            return value.split(".")[-1]
        raise self.context.error(node, f"{description} must be a string literal or name")

    def _metadata(self, node: ast.AST) -> InterfaceMetadata:
        if not isinstance(node, ast.Call):
            raise self.context.error(node, "DSL annotation metadata must be a call")
        kind = (dotted_name(node.func) or "").split(".")[-1]
        arities: dict[str, int | tuple[int, ...]] = {
            "attribute": (0, 1, 2),
            "builtin": 1,
            "uniform": (0, 2),
            "varying": 0,
            "resource": 2,
        }
        if kind not in arities:
            raise self.context.error(node, f"unknown annotation metadata '{kind}'")
        if node.keywords:
            values: dict[str, ast.AST] = {}
            for keyword in node.keywords:
                if keyword.arg is None:
                    raise self.context.error(keyword, "metadata does not support **kwargs")
                values[keyword.arg] = keyword.value
            if kind == "attribute" and not node.args:
                args = [
                    values.get("location", ast.Constant(value=-1)),
                    values.get("divisor", ast.Constant(value=0)),
                ]
            elif kind in {"resource", "uniform"} and not node.args:
                args = [values[key] for key in ("set", "binding") if key in values]
            else:
                raise self.context.error(node, f"{kind} metadata does not accept keyword arguments here")
        else:
            args = list(node.args)
        expected = arities[kind]
        valid = len(args) in expected if isinstance(expected, tuple) else len(args) == expected
        if not valid:
            raise self.context.error(node, f"{kind} metadata has the wrong number of arguments")
        parsed: list[int | str] = []
        for argument in args:
            if (
                not isinstance(argument, ast.Constant)
                or not isinstance(argument.value, (int, str))
                or isinstance(argument.value, bool)
            ):
                raise self.context.error(argument, "metadata arguments must be integer or string literals")
            parsed.append(argument.value)
        if kind == "attribute":
            if not parsed:
                parsed.extend((-1, 0))
            elif len(parsed) == 1:
                parsed.append(0)
            location, divisor = parsed
            if not isinstance(location, int) or location < -1:
                raise self.context.error(node, "attribute location must be non-negative")
            if not isinstance(divisor, int) or divisor < 0:
                raise self.context.error(node, "attribute divisor must be non-negative")
        return InterfaceMetadata(kind, tuple(parsed))
