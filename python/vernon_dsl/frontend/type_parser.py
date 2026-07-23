from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Protocol

from ..language.ast_utils import dotted_name, subscript_items
from ..language.scalar_types import SCALAR_ALIASES, SCALAR_TYPES
from .model import ConcreteType


class TypeContext(Protocol):
    structs: dict[str, object]

    def error(self, node: ast.AST, message: str) -> Exception: ...


@dataclass(frozen=True)
class Metadata:
    kind: str
    arguments: tuple[int | str, ...]


@dataclass(frozen=True)
class AnnotatedType:
    type: ConcreteType
    metadata: tuple[Metadata, ...] = ()


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
            return ConcreteType("tensor", "Tensor", (element, *shape))
        if constructor in {"vec", "mat"}:
            dimensions = 1 if constructor == "vec" else 2
            if len(items) != dimensions + 1:
                raise self.context.error(
                    node,
                    f"{constructor} requires {dimensions} dimension(s) followed by an element type",
                )
            shape = tuple(self._positive_int(item, f"{constructor} dimension") for item in items[:dimensions])
            return ConcreteType("tensor", "Tensor", (self.parse_type(items[-1]), *shape))
        if constructor in {"vec2", "vec3", "vec4", "mat2", "mat3", "mat4"}:
            if len(items) != 1:
                raise self.context.error(node, f"{constructor} requires one element type")
            size = int(constructor[-1])
            shape = (size,) if constructor.startswith("vec") else (size, size)
            return ConcreteType("tensor", "Tensor", (self.parse_type(items[0]), *shape))
        if constructor == "Buffer":
            if not 1 <= len(items) <= 2:
                raise self.context.error(node, "Buffer requires an element type and optional access string")
            arguments: tuple[ConcreteType | int | str, ...] = (self.parse_type(items[0]),)
            if len(items) == 2:
                arguments += (self._string_or_name(items[1], "buffer access"),)
            return ConcreteType("buffer", "Buffer", arguments)
        if constructor == "Texture":
            if len(items) != 2:
                raise self.context.error(node, "Texture requires a dimension and element type")
            dimension = self._string_or_name(items[0], "texture dimension")
            if dimension not in {"2d", "3d", "cube"}:
                raise self.context.error(
                    items[0],
                    "texture dimension must be one of '2d', '3d', or 'cube'",
                )
            return ConcreteType(
                "texture",
                "Texture",
                (dimension, self.parse_type(items[1])),
            )
        raise self.context.error(node, f"unknown DSL type constructor '{constructor}'")

    def _positive_int(self, node: ast.AST, description: str) -> int | str:
        if description == "tensor dimension" and isinstance(node, ast.Constant) and node.value is None:
            return "?"
        if (
            not isinstance(node, ast.Constant)
            or not isinstance(node.value, int)
            or isinstance(node.value, bool)
            or node.value <= 0
        ):
            raise self.context.error(node, f"{description} must be a positive integer literal")
        return node.value

    def _string_or_name(self, node: ast.AST, description: str) -> str:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        value = dotted_name(node)
        if value:
            return value.split(".")[-1]
        raise self.context.error(node, f"{description} must be a string literal or name")

    def _metadata(self, node: ast.AST) -> Metadata:
        if not isinstance(node, ast.Call):
            raise self.context.error(node, "DSL annotation metadata must be a call")
        kind = (dotted_name(node.func) or "").split(".")[-1]
        arities: dict[str, int | tuple[int, ...]] = {
            "location": 1,
            "builtin": 1,
            "uniform": (0, 2),
            "varying": 0,
            "resource": 2,
            "instance": (1, 2),
        }
        if kind not in arities:
            raise self.context.error(node, f"unknown annotation metadata '{kind}'")
        if node.keywords:
            values: dict[str, ast.AST] = {}
            for keyword in node.keywords:
                if keyword.arg is None:
                    raise self.context.error(keyword, "metadata does not support **kwargs")
                values[keyword.arg] = keyword.value
            if kind in {"resource", "uniform"} and not node.args:
                args = [values[key] for key in ("set", "binding") if key in values]
            elif kind == "instance" and not node.args:
                args = [values["location"]] if "location" in values else []
                args.append(values.get("divisor", ast.Constant(value=1)))
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
        if kind == "instance" and len(parsed) == 1:
            parsed.append(1)
        return Metadata(kind, tuple(parsed))
