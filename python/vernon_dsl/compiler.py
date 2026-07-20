from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .diagnostics import CompileError, SourceLocation
from .module_graph import load_project

_SCALARS = {
    "bool": "i1",
    "i32": "i32",
    "u32": "i32",
    "f16": "f16",
    "f32": "f32",
    "f64": "f64"
}
_SWIZZLES = set("xyzwrgba")


@dataclass(frozen=True)
class DslType:
    kind: str
    name: str
    arguments: tuple["DslType | int | str", ...] = ()

    @property
    def mlir(self) -> str:
        if self.kind == "scalar":
            return _SCALARS[self.name]
        if self.kind == "index":
            return "index"
        if self.kind == "void":
            return "none"
        if self.kind == "tensor":
            element = self.arguments[0]
            shape = self.arguments[1:]
            assert isinstance(element, DslType)
            dimensions = "x".join(str(value) for value in shape)
            return f"tensor<{dimensions}x{element.mlir}>"
        if self.kind == "array":
            element, size = self.arguments
            assert isinstance(element, DslType)
            return f"!vernon.array<{size} x {element.mlir}>"
        if self.kind == "struct":
            return f'!vernon.struct<"{self.name}">'
        if self.kind == "buffer":
            element = self.arguments[0]
            access = self.arguments[1] if len(
                self.arguments) > 1 else "read_write"
            assert isinstance(element, DslType)
            return f'!vernon.buffer<{element.mlir}, "{access}">'
        if self.kind == "addressable_tensor":
            element = self.arguments[0]
            assert isinstance(element, DslType)
            return f'!vernon.buffer<{element.mlir}, "read_write">'
        if self.kind == "texture":
            dimension, element = self.arguments
            assert isinstance(element, DslType)
            return f'!vernon.texture<"{dimension}", {element.mlir}>'
        if self.kind == "sampler":
            return "!vernon.sampler"
        raise AssertionError(f"unknown type kind {self.kind}")

    @property
    def is_float(self) -> bool:
        if self.kind == "scalar":
            return self.name.startswith("f")
        return self.kind == "tensor" and isinstance(
            self.arguments[0], DslType) and self.arguments[0].is_float

    @property
    def is_integer(self) -> bool:
        if self.kind == "index":
            return True
        if self.kind == "scalar":
            return self.name in {"i32", "u32"}
        return self.kind == "tensor" and isinstance(
            self.arguments[0], DslType) and self.arguments[0].is_integer


@dataclass(frozen=True)
class Metadata:
    kind: str
    arguments: tuple[int | str, ...]


@dataclass(frozen=True)
class AnnotatedType:
    type: DslType
    metadata: tuple[Metadata, ...] = ()


@dataclass(frozen=True)
class FunctionSignature:
    arguments: tuple[DslType, ...]
    result: DslType | None


@dataclass(frozen=True)
class Value:
    name: str
    type: DslType


@dataclass
class _ModuleContext:
    filename: str
    structs: dict[str, tuple[tuple[str, AnnotatedType],
                             ...]] = field(default_factory=dict)
    signatures: dict[str, FunctionSignature] = field(default_factory=dict)
    result_annotations: dict[str, AnnotatedType
                             | None] = field(default_factory=dict)

    def error(self, node: ast.AST, message: str) -> CompileError:
        return CompileError(
            message,
            SourceLocation(
                self.filename,
                getattr(node, "lineno", 1),
                getattr(node, "col_offset", 0) + 1,
            ),
        )


def _name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return None


class _TypeParser:

    def __init__(self, context: _ModuleContext):
        self.context = context

    def parse(self, node: ast.AST) -> AnnotatedType:
        if isinstance(node, ast.Constant) and node.value is None:
            raise self.context.error(
                node, "None is only valid as a function return annotation")
        if isinstance(node, ast.Subscript) and (_name(
                node.value) or "").split(".")[-1] == "Annotated":
            items = self._subscript_items(node)
            if len(items) < 2:
                raise self.context.error(
                    node,
                    "Annotated requires a type and at least one metadata item")
            base = self.parse_type(items[0])
            return AnnotatedType(
                base, tuple(self._metadata(item) for item in items[1:]))
        return AnnotatedType(self.parse_type(node))

    def parse_type(self, node: ast.AST) -> DslType:
        type_name = (_name(node) or "").split(".")[-1]
        if type_name in _SCALARS:
            return DslType("scalar", type_name)
        if type_name in self.context.structs:
            return DslType("struct", type_name)
        if type_name == "Sampler":
            return DslType("sampler", "Sampler")
        if not isinstance(node, ast.Subscript):
            raise self.context.error(
                node, f"unknown DSL type '{type_name or ast.dump(node)}'")

        constructor = (_name(node.value) or "").split(".")[-1]
        items = self._subscript_items(node)
        if constructor == "Tensor":
            if len(items) < 2:
                raise self.context.error(
                    node, "Tensor requires an element type and shape")
            element = self.parse_type(items[0])
            shape_nodes = items[1].elts if len(items) == 2 and isinstance(
                items[1], ast.Tuple) else items[1:]
            shape = tuple(
                self._positive_int(item, "tensor dimension")
                for item in shape_nodes)
            return DslType("tensor", "Tensor", (element, *shape))
        if constructor in {"vec", "mat"}:
            dimensions = 1 if constructor == "vec" else 2
            if len(items) != dimensions + 1:
                raise self.context.error(
                    node,
                    f"{constructor} requires {dimensions} dimension(s) followed by an element type"
                )
            shape = tuple(
                self._positive_int(item, f"{constructor} dimension")
                for item in items[:dimensions])
            return DslType("tensor", "Tensor",
                           (self.parse_type(items[-1]), *shape))
        if constructor in {"vec2", "vec3", "vec4", "mat2", "mat3", "mat4"}:
            if len(items) != 1:
                raise self.context.error(
                    node, f"{constructor} requires one element type")
            size = int(constructor[-1])
            shape = (size, ) if constructor.startswith("vec") else (size, size)
            return DslType("tensor", "Tensor",
                           (self.parse_type(items[0]), *shape))
        if constructor == "Array":
            if len(items) != 2:
                raise self.context.error(
                    node, "Array requires an element type and fixed size")
            return DslType("array", "Array", (self.parse_type(
                items[0]), self._positive_int(items[1], "array size")))
        if constructor == "Buffer":
            if not 1 <= len(items) <= 2:
                raise self.context.error(
                    node,
                    "Buffer requires an element type and optional access string"
                )
            arguments: tuple[DslType | int | str,
                             ...] = (self.parse_type(items[0]), )
            if len(items) == 2:
                arguments += (self._string_or_name(items[1],
                                                   "buffer access"), )
            return DslType("buffer", "Buffer", arguments)
        if constructor == "Texture":
            if len(items) != 2:
                raise self.context.error(
                    node, "Texture requires a dimension and element type")
            return DslType(
                "texture",
                "Texture",
                (self._string_or_name(
                    items[0], "texture dimension"), self.parse_type(items[1])),
            )
        raise self.context.error(
            node, f"unknown DSL type constructor '{constructor}'")

    @staticmethod
    def _subscript_items(node: ast.Subscript) -> list[ast.AST]:
        return list(node.slice.elts) if isinstance(
            node.slice, ast.Tuple) else [node.slice]

    def _positive_int(self, node: ast.AST, description: str) -> int | str:
        if (description == "tensor dimension"
                and isinstance(node, ast.Constant) and node.value is None):
            return "?"
        if not isinstance(node, ast.Constant) or not isinstance(
                node.value, int) or isinstance(node.value,
                                               bool) or node.value <= 0:
            raise self.context.error(
                node, f"{description} must be a positive integer literal")
        return node.value

    def _string_or_name(self, node: ast.AST, description: str) -> str:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        value = _name(node)
        if value:
            return value.split(".")[-1]
        raise self.context.error(
            node, f"{description} must be a string literal or name")

    def _metadata(self, node: ast.AST) -> Metadata:
        if not isinstance(node, ast.Call):
            raise self.context.error(node,
                                     "DSL annotation metadata must be a call")
        kind = (_name(node.func) or "").split(".")[-1]
        arities = {
            "location": 1,
            "builtin": 1,
            "uniform": (0, 2),
            "varying": 0,
            "resource": 2,
            "instance": (1, 2),
        }
        if kind not in arities:
            raise self.context.error(node,
                                     f"unknown annotation metadata '{kind}'")
        if node.keywords:
            values: dict[str, ast.AST] = {}
            for keyword in node.keywords:
                if keyword.arg is None:
                    raise self.context.error(
                        keyword, "metadata does not support **kwargs")
                values[keyword.arg] = keyword.value
            if kind in {"resource", "uniform"} and not node.args:
                args = [
                    values[key] for key in ("set", "binding") if key in values
                ]
            elif kind == "instance" and not node.args:
                args = [values["location"]] if "location" in values else []
                args.append(values.get("divisor", ast.Constant(value=1)))
            else:
                raise self.context.error(
                    node,
                    f"{kind} metadata does not accept keyword arguments here")
        else:
            args = list(node.args)
        expected = arities[kind]
        valid = len(args) in expected if isinstance(
            expected, tuple) else len(args) == expected
        if not valid:
            raise self.context.error(
                node, f"{kind} metadata has the wrong number of arguments")
        parsed: list[int | str] = []
        for argument in args:
            if not isinstance(argument, ast.Constant) or not isinstance(
                    argument.value,
                (int, str)) or isinstance(argument.value, bool):
                raise self.context.error(
                    argument,
                    "metadata arguments must be integer or string literals")
            parsed.append(argument.value)
        if kind == "instance" and len(parsed) == 1:
            parsed.append(1)
        return Metadata(kind, tuple(parsed))


class _FunctionEmitter:

    def __init__(
        self,
        context: _ModuleContext,
        node: ast.FunctionDef,
        signature: FunctionSignature,
        argument_annotations: list[AnnotatedType],
        result_annotation: AnnotatedType | None,
        stage: str | None,
        workgroup_size: tuple[int, int, int] | None,
    ):
        self.context = context
        self.node = node
        self.signature = signature
        self.argument_annotations = argument_annotations
        self.result_annotation = result_annotation
        self.stage = stage
        self.workgroup_size = workgroup_size
        self.lines: list[str] = []
        self.indent = 1
        self.next_value = 0
        self.environment: dict[str, Value] = {}
        self.returned = False

    def emit(self) -> list[str]:
        arguments: list[str] = []
        for index, (argument, annotation) in enumerate(
                zip(self.node.args.args,
                    self.argument_annotations,
                    strict=True)):
            value_type = annotation.type
            has_builtin = any(item.kind == "builtin"
                              for item in annotation.metadata)
            if (self.stage == "compute" and value_type.kind == "tensor"
                    and not has_builtin):
                value_type = DslType(
                    "addressable_tensor",
                    "Tensor",
                    value_type.arguments,
                )
            value = Value(f"%arg{index}", value_type)
            self.environment[argument.arg] = value
            attributes = self._metadata_attributes(annotation.metadata,
                                                   stage=self.stage,
                                                   is_result=False,
                                                   default_location=index)
            attributes.append(f'vernon.source_name = "{argument.arg}"')
            if annotation.type.kind == "scalar":
                attributes.append(f'vernon.dtype = "{annotation.type.name}"')
            elif annotation.type.kind == "tensor":
                element_type = annotation.type.arguments[0]
                assert isinstance(element_type, DslType)
                attributes.append(f'vernon.dtype = "{element_type.name}"')
            if value_type.kind == "addressable_tensor":
                attributes = [
                    attribute for attribute in attributes
                    if not attribute.startswith(("vernon.interface",
                                                 "vernon.location"))
                ]
                attributes.extend((
                    'vernon.interface = "resource"',
                    "vernon.set = 0 : i64",
                    f"vernon.binding = {index} : i64",
                ))
                shape = ", ".join(
                    str(value) for value in value_type.arguments[1:])
                attributes.append(f"vernon.tensor_shape = array<i64: {shape}>")
            suffix = f" {{{', '.join(attributes)}}}" if attributes else ""
            arguments.append(f"{value.name}: {value.type.mlir}{suffix}")
        result = ""
        if self.signature.result:
            result_metadata = self.result_annotation.metadata if self.result_annotation else (
            )
            result_attributes = self._metadata_attributes(result_metadata,
                                                          stage=self.stage,
                                                          is_result=True,
                                                          default_location=0)
            suffix = f" {{{', '.join(result_attributes)}}}" if result_attributes else ""
            result = f" -> ({self.signature.result.mlir}{suffix})"
        function_attributes: list[str] = []
        if self.stage:
            function_attributes.append("vernon.entry")
            function_attributes.append(f'vernon.stage = "{self.stage}"')
        if self.workgroup_size:
            values = ", ".join(str(value) for value in self.workgroup_size)
            function_attributes.append(
                f"vernon.workgroup_size = array<i32: {values}>")
        attributes = f" attributes {{{', '.join(function_attributes)}}}" if function_attributes else ""
        visibility = "" if self.stage else " private"
        self.lines.append(
            f"  func.func{visibility} @{self.node.name}({', '.join(arguments)}){result}{attributes} {{"
        )
        for statement in self.node.body:
            self._statement(statement)
        if not self.returned:
            if self.signature.result is not None:
                raise self.context.error(
                    self.node,
                    f"function '{self.node.name}' may exit without returning a value"
                )
            self._line("func.return")
        self.lines.append("  }")
        return self.lines

    def _line(self, text: str) -> None:
        self.lines.append("  " * self.indent + text)

    def _fresh(self) -> str:
        value = f"%{self.next_value}"
        self.next_value += 1
        return value

    @staticmethod
    def _metadata_attributes(
        metadata: Iterable[Metadata],
        *,
        stage: str | None,
        is_result: bool,
        default_location: int,
    ) -> list[str]:
        attributes: list[str] = []
        items = tuple(metadata)
        interface: str | None = None
        if any(item.kind == "resource" for item in items):
            interface = "resource"
        elif any(item.kind == "uniform" for item in items):
            interface = "uniform"
        elif stage and (is_result or items):
            interface = "output" if is_result else "input"
        elif stage:
            interface = "output" if is_result else "input"

        has_slot = any(item.kind in {"location", "builtin", "instance"}
                       for item in items)
        if interface:
            attributes.append(f'vernon.interface = "{interface}"')
            if interface in {"input", "output"} and not has_slot:
                if is_result and stage == "vertex":
                    attributes.append('vernon.builtin = "position"')
                else:
                    attributes.append(
                        f"vernon.location = {default_location} : i64")

        for item in items:
            if item.kind in {"location", "instance"}:
                attributes.append(
                    f"vernon.location = {item.arguments[0]} : i64")
                if item.kind == "instance":
                    attributes.append(
                        f"vernon.instance_divisor = {item.arguments[1]} : i64")
            elif item.kind == "resource":
                attributes.extend((
                    f"vernon.set = {item.arguments[0]} : i64",
                    f"vernon.binding = {item.arguments[1]} : i64",
                ))
            elif item.kind == "uniform":
                if item.arguments:
                    attributes.extend((
                        f"vernon.set = {item.arguments[0]} : i64",
                        f"vernon.binding = {item.arguments[1]} : i64",
                    ))
            elif item.kind == "varying":
                continue
            else:
                attributes.append(
                    f'vernon.{item.kind} = "{item.arguments[0]}"')
        return attributes

    def _statement(self, node: ast.stmt) -> None:
        if isinstance(node, ast.Pass):
            return
        if isinstance(node, ast.Expr):
            self._expression(node.value)
            return
        if isinstance(node, ast.Assign):
            if len(node.targets) == 1 and isinstance(node.targets[0],
                                                     ast.Subscript):
                self._store_index(node.targets[0],
                                  self._expression(node.value))
                return
            if len(node.targets) != 1 or not isinstance(
                    node.targets[0], ast.Name):
                raise self.context.error(
                    node,
                    "assignment target must be a local name or buffer element")
            value = self._expression(node.value)
            if value.type.kind == "void":
                raise self.context.error(
                    node.value, "a void function call cannot be assigned")
            self.environment[node.targets[0].id] = value
            return
        if isinstance(node, ast.AugAssign):
            operation = ast.BinOp(left=node.target,
                                  op=node.op,
                                  right=node.value)
            ast.copy_location(operation, node)
            value = self._expression(operation)
            if isinstance(node.target, ast.Name):
                if node.target.id not in self.environment:
                    raise self.context.error(
                        node.target, f"unknown local value '{node.target.id}'")
                self._require_same_type(node,
                                        self.environment[node.target.id].type,
                                        value.type)
                self.environment[node.target.id] = value
                return
            if isinstance(node.target, ast.Subscript):
                self._store_index(node.target, value)
                return
            raise self.context.error(
                node,
                "augmented assignment requires a local name or indexed value")
        if isinstance(node, ast.AnnAssign):
            if not isinstance(node.target, ast.Name) or node.value is None:
                raise self.context.error(
                    node,
                    "annotated assignment requires a local name and value")
            expected = _TypeParser(self.context).parse(node.annotation).type
            value = self._expression(node.value, expected)
            self._require_same_type(node.value, expected, value.type)
            self.environment[node.target.id] = value
            return
        if isinstance(node, ast.Return):
            if node.value is None:
                if self.signature.result is not None:
                    raise self.context.error(node, "return value is required")
                self._line("func.return")
            else:
                if self.signature.result is None:
                    raise self.context.error(
                        node, "void function cannot return a value")
                value = self._expression(node.value, self.signature.result)
                self._require_same_type(node.value, self.signature.result,
                                        value.type)
                self._line(f"func.return {value.name} : {value.type.mlir}")
            self.returned = True
            return
        if isinstance(node, ast.If):
            self._if(node)
            return
        if isinstance(node, ast.For):
            self._for(node)
            return
        if isinstance(node, ast.While):
            self._while(node)
            return
        raise self.context.error(
            node, f"unsupported statement syntax: {type(node).__name__}")

    def _if(self, node: ast.If) -> None:
        condition = self._expression(node.test)
        self._require_same_type(node.test, DslType("scalar", "bool"),
                                condition.type)
        assigned = self._assigned_names(
            (*node.body, *node.orelse)) & self.environment.keys()
        merged_names = sorted(assigned)
        merged_types = [self.environment[name].type for name in merged_names]
        results = [self._fresh() for _ in merged_names]
        lhs = f"{', '.join(results)} = " if results else ""
        result_types = f" -> ({', '.join(value.mlir for value in merged_types)})" if results else ""
        self._line(f"{lhs}scf.if {condition.name}{result_types} {{")
        outer = self.environment.copy()
        self.indent += 1
        self.environment = outer.copy()
        for statement in node.body:
            if isinstance(statement, ast.Return):
                raise self.context.error(
                    statement,
                    "return inside if is not supported in the first frontend stage"
                )
            self._statement(statement)
        self._yield_merged(node, merged_names, merged_types)
        self.indent -= 1
        self._line("} else {")
        self.indent += 1
        self.environment = outer.copy()
        for statement in node.orelse:
            if isinstance(statement, ast.Return):
                raise self.context.error(
                    statement,
                    "return inside if is not supported in the first frontend stage"
                )
            self._statement(statement)
        self._yield_merged(node, merged_names, merged_types)
        self.indent -= 1
        self._line("}")
        self.environment = outer
        for name, result, result_type in zip(merged_names,
                                             results,
                                             merged_types,
                                             strict=True):
            self.environment[name] = Value(result, result_type)

    def _yield_merged(self, node: ast.If, names: list[str],
                      types: list[DslType]) -> None:
        values: list[Value] = []
        for name, expected in zip(names, types, strict=True):
            value = self.environment[name]
            self._require_same_type(node, expected, value.type)
            values.append(value)
        if values:
            self._line(
                f"scf.yield {', '.join(value.name for value in values)} : "
                f"{', '.join(value.type.mlir for value in values)}")
        else:
            self._line("scf.yield")

    def _for(self, node: ast.For) -> None:
        if not isinstance(node.target, ast.Name) or not isinstance(
                node.iter, ast.Call) or _name(node.iter.func) != "range":
            raise self.context.error(
                node, "for loops must have the form 'for name in range(...)'")
        if node.orelse:
            raise self.context.error(node, "for-else is not supported")
        if not 1 <= len(node.iter.args) <= 3 or node.iter.keywords:
            raise self.context.error(
                node.iter,
                "range requires one to three positional integer literals")
        integer_arguments: list[int] = []
        for argument in node.iter.args:
            if not isinstance(argument, ast.Constant) or not isinstance(
                    argument.value, int) or isinstance(argument.value, bool):
                raise self.context.error(
                    argument,
                    "range bounds must be integer literals in the first frontend stage"
                )
            integer_arguments.append(argument.value)
        if len(integer_arguments) == 1:
            start, stop, step = 0, integer_arguments[0], 1
        elif len(integer_arguments) == 2:
            start, stop = integer_arguments
            step = 1
        else:
            start, stop, step = integer_arguments
        if step <= 0:
            raise self.context.error(node.iter, "range step must be positive")
        constants = []
        index_type = DslType("index", "index")
        for value in (start, stop, step):
            result = self._fresh()
            self._line(f"{result} = arith.constant {value} : index")
            constants.append(result)
        outer = self.environment.copy()
        mutated = self._assigned_names(node.body) & outer.keys()
        if mutated:
            raise self.context.error(
                node,
                f"loop-carried assignment is not supported: {', '.join(sorted(mutated))}"
            )
        induction = self._fresh()
        self._line(
            f"scf.for {induction} = {constants[0]} to {constants[1]} step {constants[2]} {{"
        )
        self.indent += 1
        self.environment = outer.copy()
        self.environment[node.target.id] = Value(induction, index_type)
        for statement in node.body:
            if isinstance(statement, ast.Return):
                raise self.context.error(
                    statement,
                    "return inside for is not supported in the first frontend stage"
                )
            self._statement(statement)
        self._line("scf.yield")
        self.indent -= 1
        self._line("}")
        self.environment = outer

    def _while(self, node: ast.While) -> None:
        if node.orelse:
            raise self.context.error(node, "while-else is not supported")
        outer = self.environment.copy()
        carried_names = sorted(self._assigned_names(node.body) & outer.keys())
        carried_types = [outer[name].type for name in carried_names]
        results = [self._fresh() for _ in carried_names]
        operand_types = ", ".join(value.mlir for value in carried_types)
        before_arguments = [self._fresh() for _ in carried_names]
        assignments = ", ".join(
            f"{argument} = {outer[name].name}" for argument, name in zip(
                before_arguments, carried_names, strict=True))
        result_prefix = f"{', '.join(results)} = " if results else ""
        signature = (
            f" ({assignments}) : ({operand_types}) -> ({operand_types})"
            if carried_names else "")
        self._line(f"{result_prefix}scf.while{signature} {{")
        self.indent += 1
        self.environment = outer.copy()
        for name, value_type, argument in zip(carried_names,
                                              carried_types,
                                              before_arguments,
                                              strict=True):
            self.environment[name] = Value(argument, value_type)
        condition = self._expression(node.test)
        self._require_same_type(node.test, DslType("scalar", "bool"),
                                condition.type)
        forwarded = ", ".join(self.environment[name].name
                              for name in carried_names)
        suffix = f" : {operand_types}" if carried_names else ""
        self._line(f"scf.condition({condition.name}) {forwarded}{suffix}")
        self.indent -= 1
        self._line("} do {")
        self.indent += 1
        self.environment = outer.copy()
        after_arguments = []
        for name, value_type in zip(carried_names, carried_types, strict=True):
            argument = self._fresh()
            after_arguments.append(argument)
            self.environment[name] = Value(argument, value_type)
        if after_arguments:
            self._line(
                f"^bb0({', '.join(f'{name}: {value_type.mlir}' for name, value_type in zip(after_arguments, carried_types, strict=True))}):"
            )
        for statement in node.body:
            self._statement(statement)
        yielded = ", ".join(self.environment[name].name
                            for name in carried_names)
        self._line(f"scf.yield {yielded}{suffix}")
        self.indent -= 1
        self._line("}")
        self.environment = outer
        for name, result, result_type in zip(carried_names,
                                             results,
                                             carried_types,
                                             strict=True):
            self.environment[name] = Value(result, result_type)

    @staticmethod
    def _assigned_names(statements: Iterable[ast.stmt]) -> set[str]:
        result: set[str] = set()
        for statement in statements:
            if isinstance(statement, ast.Assign):
                result.update(target.id for target in statement.targets
                              if isinstance(target, ast.Name))
            elif isinstance(statement, ast.AnnAssign) and isinstance(
                    statement.target, ast.Name):
                result.add(statement.target.id)
            elif isinstance(statement, ast.AugAssign) and isinstance(
                    statement.target, ast.Name):
                result.add(statement.target.id)
        return result

    def _expression(self,
                    node: ast.expr,
                    expected: DslType | None = None) -> Value:
        if isinstance(node, ast.Name):
            if node.id not in self.environment:
                raise self.context.error(node,
                                         f"unknown local value '{node.id}'")
            return self.environment[node.id]
        if isinstance(node, ast.Constant):
            return self._constant(node, expected)
        if isinstance(node, ast.BinOp):
            return self._binary(node)
        if isinstance(node, ast.UnaryOp):
            return self._unary(node)
        if isinstance(node, ast.BoolOp):
            return self._boolean(node)
        if isinstance(node, ast.Compare):
            return self._compare(node)
        if isinstance(node, ast.Call):
            return self._call(node)
        if isinstance(node, ast.Attribute):
            return self._attribute(node)
        if isinstance(node, ast.Subscript):
            return self._index(node)
        if isinstance(node, ast.IfExp):
            raise self.context.error(
                node,
                "conditional expressions are not supported; use an if statement"
            )
        raise self.context.error(
            node, f"unsupported expression syntax: {type(node).__name__}")

    def _constant(self, node: ast.Constant, expected: DslType | None) -> Value:
        if isinstance(node.value, bool):
            value_type = DslType("scalar", "bool")
            literal = "true" if node.value else "false"
        elif isinstance(node.value, int):
            value_type = expected if expected and expected.kind == "scalar" and expected.name in {
                "i32", "u32"
            } else DslType("scalar", "i32")
            literal = str(node.value)
        elif isinstance(node.value, float):
            value_type = expected if expected and expected.kind == "scalar" and expected.is_float else DslType(
                "scalar", "f32")
            literal = f"{node.value:.17g}"
            if "." not in literal and "e" not in literal.lower():
                literal += ".0"
        else:
            raise self.context.error(
                node, "only bool, integer, and float constants are supported")
        result = self._fresh()
        self._line(f"{result} = arith.constant {literal} : {value_type.mlir}")
        return Value(result, value_type)

    def _binary(self, node: ast.BinOp) -> Value:
        left = self._expression(node.left)
        right = self._expression(
            node.right, left.type if left.type.kind == "scalar" else None)
        if left.type.kind == "tensor" and right.type.kind == "scalar":
            right = self._splat(node.right, right, left.type)
        elif left.type.kind == "scalar" and right.type.kind == "tensor":
            left = self._splat(node.left, left, right.type)
        self._require_same_type(node, left.type, right.type)
        floating = left.type.is_float
        operations = {
            ast.Add:
            "arith.addf" if floating else "arith.addi",
            ast.Sub:
            "arith.subf" if floating else "arith.subi",
            ast.Mult:
            "arith.mulf" if floating else "arith.muli",
            ast.Div:
            "arith.divf" if floating else
            ("arith.divui" if left.type.name == "u32" else "arith.divsi"),
            ast.Mod:
            "arith.remf" if floating else
            ("arith.remui" if left.type.name == "u32" else "arith.remsi"),
        }
        operation = operations.get(type(node.op))
        if operation is None or not (floating or left.type.is_integer):
            raise self.context.error(
                node, f"unsupported binary operation for {left.type.name}")
        result = self._fresh()
        self._line(
            f"{result} = {operation} {left.name}, {right.name} : {left.type.mlir}"
        )
        return Value(result, left.type)

    def _splat(self, node: ast.AST, value: Value,
               tensor_type: DslType) -> Value:
        element = tensor_type.arguments[0]
        assert isinstance(element, DslType)
        self._require_same_type(node, element, value.type)
        result = self._fresh()
        self._line(
            f"{result} = tensor.splat {value.name} : {tensor_type.mlir}")
        return Value(result, tensor_type)

    def _unary(self, node: ast.UnaryOp) -> Value:
        operand = self._expression(node.operand)
        result = self._fresh()
        if isinstance(node.op, ast.Not):
            self._require_same_type(node, DslType("scalar", "bool"),
                                    operand.type)
            self._line(f"{result} = arith.xori {operand.name}, true : i1")
            return Value(result, operand.type)
        if isinstance(node.op, ast.USub) and (operand.type.is_float
                                              or operand.type.is_integer):
            operation = "arith.negf" if operand.type.is_float else "arith.subi"
            if operand.type.is_float:
                self._line(
                    f"{result} = {operation} {operand.name} : {operand.type.mlir}"
                )
            else:
                zero = self._fresh()
                self._line(f"{zero} = arith.constant 0 : {operand.type.mlir}")
                self._line(
                    f"{result} = {operation} {zero}, {operand.name} : {operand.type.mlir}"
                )
            return Value(result, operand.type)
        if isinstance(node.op, ast.UAdd):
            return operand
        raise self.context.error(node, "unsupported unary operation")

    def _boolean(self, node: ast.BoolOp) -> Value:
        values = [self._expression(value) for value in node.values]
        bool_type = DslType("scalar", "bool")
        for value in values:
            self._require_same_type(node, bool_type, value.type)
        operation = "arith.andi" if isinstance(node.op,
                                               ast.And) else "arith.ori"
        current = values[0]
        for value in values[1:]:
            result = self._fresh()
            self._line(
                f"{result} = {operation} {current.name}, {value.name} : i1")
            current = Value(result, bool_type)
        return current

    def _compare(self, node: ast.Compare) -> Value:
        if len(node.ops) != 1:
            raise self.context.error(node,
                                     "chained comparisons are not supported")
        left = self._expression(node.left)
        right = self._expression(node.comparators[0], left.type)
        self._require_same_type(node, left.type, right.type)
        predicates = {
            ast.Eq: ("oeq", "eq"),
            ast.NotEq: ("one", "ne"),
            ast.Lt: ("olt", "slt"),
            ast.LtE: ("ole", "sle"),
            ast.Gt: ("ogt", "sgt"),
            ast.GtE: ("oge", "sge"),
        }
        predicates_for_operation = predicates.get(type(node.ops[0]))
        if predicates_for_operation is None:
            raise self.context.error(
                node, f"unsupported comparison: {type(node.ops[0]).__name__}")
        floating_predicate, integer_predicate = predicates_for_operation
        if left.type.is_float:
            operation, predicate = "arith.cmpf", floating_predicate
        elif left.type.is_integer or left.type.name == "bool":
            operation, predicate = "arith.cmpi", integer_predicate
        else:
            raise self.context.error(
                node, "comparison requires scalar or tensor numeric operands")
        result = self._fresh()
        self._line(
            f"{result} = {operation} {predicate}, {left.name}, {right.name} : {left.type.mlir}"
        )
        result_type = DslType("scalar", "bool")
        if left.type.kind == "tensor":
            result_type = DslType("tensor", "Tensor",
                                  (result_type, *left.type.arguments[1:]))
        return Value(result, result_type)

    def _call(self, node: ast.Call) -> Value:
        if node.keywords:
            raise self.context.error(
                node, "function calls do not support keyword arguments")
        name = (_name(node.func) or "").split(".")[-1]
        arguments = [self._expression(argument) for argument in node.args]
        scalar_casts = {
            "int": DslType("scalar", "i32"),
            "i32": DslType("scalar", "i32"),
            "u32": DslType("scalar", "u32"),
            "float": DslType("scalar", "f32"),
            "f32": DslType("scalar", "f32"),
            "f64": DslType("scalar", "f64"),
        }
        if name in scalar_casts:
            if len(arguments) != 1 or arguments[0].type.kind not in {
                    "scalar",
                    "index",
            }:
                raise self.context.error(
                    node, f"{name} requires one scalar argument")
            return self._cast(node, arguments[0], scalar_casts[name])
        math_operations = {
            "sin": "math.sin",
            "cos": "math.cos",
            "exp": "math.exp",
            "log": "math.log",
            "sqrt": "math.sqrt",
            "abs": "math.absf",
        }
        if name in math_operations:
            if len(arguments) != 1 or not arguments[0].type.is_float:
                raise self.context.error(
                    node, f"{name} requires one floating-point argument")
            result = self._fresh()
            self._line(
                f"{result} = {math_operations[name]} {arguments[0].name} : {arguments[0].type.mlir}"
            )
            return Value(result, arguments[0].type)
        if name in {"vec2", "vec3", "vec4"}:
            size = int(name[-1])
            if not arguments:
                raise self.context.error(
                    node, f"{name} requires component arguments")
            element: DslType | None = None
            component_count = 0
            for argument in arguments:
                if argument.type.kind == "scalar":
                    argument_element = argument.type
                    component_count += 1
                elif argument.type.kind == "tensor" and len(
                        argument.type.arguments) == 2:
                    argument_element = argument.type.arguments[0]
                    assert isinstance(argument_element, DslType)
                    component_count += int(argument.type.arguments[1])
                else:
                    raise self.context.error(
                        node, f"{name} arguments must be scalars or vectors")
                if element is None:
                    element = argument_element
                else:
                    self._require_same_type(node, element, argument_element)
            if component_count != size or element is None:
                raise self.context.error(
                    node, f"{name} requires exactly {size} scalar components")
            result_type = DslType("tensor", "Tensor", (element, size))
            return self._intrinsic(node, "construct", arguments, result_type)
        if name in {"normalize", "reflect"}:
            expected = 1 if name == "normalize" else 2
            if len(arguments) != expected or any(argument.type.kind != "tensor"
                                                 or not argument.type.is_float
                                                 for argument in arguments):
                raise self.context.error(
                    node,
                    f"{name} requires {expected} floating-point vector argument(s)"
                )
            for argument in arguments[1:]:
                self._require_same_type(node, arguments[0].type, argument.type)
            return self._intrinsic(node, name, arguments, arguments[0].type)
        if name in {"dot", "cross"}:
            if len(arguments) != 2:
                raise self.context.error(
                    node, f"{name} requires two vector arguments")
            self._require_same_type(node, arguments[0].type, arguments[1].type)
            vector = arguments[0].type
            if vector.kind != "tensor" or len(
                    vector.arguments) != 2 or not vector.is_float:
                raise self.context.error(
                    node, f"{name} requires floating-point vectors")
            if name == "cross" and vector.arguments[1] != 3:
                raise self.context.error(
                    node, "cross requires three-component vectors")
            element = vector.arguments[0]
            assert isinstance(element, DslType)
            result_type = element if name == "dot" else vector
            return self._intrinsic(node, name, arguments, result_type)
        if name == "norm":
            if (len(arguments) != 1 or arguments[0].type.kind != "tensor"
                    or not arguments[0].type.is_float):
                raise self.context.error(
                    node, "norm requires one floating-point Tensor argument")
            element = arguments[0].type.arguments[0]
            assert isinstance(element, DslType)
            squared = self._intrinsic(node, "dot",
                                      [arguments[0], arguments[0]], element)
            result = self._fresh()
            self._line(f"{result} = math.sqrt {squared.name} : {element.mlir}")
            return Value(result, element)
        if name in {"min", "max", "pow"}:
            if len(arguments) != 2:
                raise self.context.error(node,
                                         f"{name} requires two arguments")
            self._require_same_type(node, arguments[0].type, arguments[1].type)
            if not arguments[0].type.is_float:
                raise self.context.error(
                    node,
                    f"{name} currently requires floating-point arguments")
            return self._intrinsic(node, name, arguments, arguments[0].type)
        if name == "clamp":
            if len(arguments) != 3:
                raise self.context.error(
                    node, "clamp requires value, minimum, and maximum")
            for argument in arguments[1:]:
                self._require_same_type(node, arguments[0].type, argument.type)
            return self._intrinsic(node, name, arguments, arguments[0].type)
        if name == "matmul":
            if len(arguments) != 2:
                raise self.context.error(node, "matmul requires two arguments")
            left, right = arguments
            if left.type.kind != "tensor" or len(left.type.arguments) != 3:
                raise self.context.error(
                    node, "matmul left operand must be a matrix")
            rows, columns = left.type.arguments[1:]
            element = left.type.arguments[0]
            assert isinstance(element, DslType)
            if right.type == DslType("tensor", "Tensor", (element, columns)):
                result_type = DslType("tensor", "Tensor", (element, rows))
            elif right.type.kind == "tensor" and len(
                    right.type.arguments
            ) == 3 and right.type.arguments[1] == columns:
                result_type = DslType("tensor", "Tensor",
                                      (element, rows, right.type.arguments[2]))
            else:
                raise self.context.error(
                    node, "matmul operands have incompatible shapes")
            return self._intrinsic(node, name, arguments, result_type)
        if name == "texture_sample":
            if len(arguments
                   ) != 3 or arguments[0].type.kind != "texture" or arguments[
                       1].type.kind != "sampler":
                raise self.context.error(
                    node,
                    "texture_sample requires texture, sampler, and coordinates"
                )
            element = arguments[0].type.arguments[1]
            assert isinstance(element, DslType)
            return self._intrinsic(node, name, arguments,
                                   DslType("tensor", "Tensor", (element, 4)))
        if name in self.context.structs:
            fields = self.context.structs[name]
            if len(arguments) != len(fields):
                raise self.context.error(
                    node,
                    f"{name} constructor requires {len(fields)} arguments")
            for argument, (_, annotation) in zip(arguments,
                                                 fields,
                                                 strict=True):
                self._require_same_type(node, annotation.type, argument.type)
            result_type = DslType("struct", name)
            result = self._fresh()
            operand_types = ", ".join(argument.type.mlir
                                      for argument in arguments)
            self._line(
                f'{result} = "vernon.struct_create"({", ".join(argument.name for argument in arguments)}) '
                f'{{type_name = "{name}"}} : ({operand_types}) -> {result_type.mlir}'
            )
            return Value(result, result_type)
        if name not in self.context.signatures:
            raise self.context.error(node, f"unknown DSL function '{name}'")
        signature = self.context.signatures[name]
        if len(arguments) != len(signature.arguments):
            raise self.context.error(
                node,
                f"function '{name}' expects {len(signature.arguments)} arguments"
            )
        for argument, expected in zip(arguments,
                                      signature.arguments,
                                      strict=True):
            self._require_same_type(node, expected, argument.type)
        if signature.result is None:
            self._line(
                f"func.call @{name}({', '.join(argument.name for argument in arguments)}) : "
                f"({', '.join(argument.type.mlir for argument in arguments)}) -> ()"
            )
            return Value("", DslType("void", "void"))
        result = self._fresh()
        self._line(
            f"{result} = func.call @{name}({', '.join(argument.name for argument in arguments)}) : "
            f"({', '.join(argument.type.mlir for argument in arguments)}) -> {signature.result.mlir}"
        )
        return Value(result, signature.result)

    def _cast(self, node: ast.AST, value: Value, target: DslType) -> Value:
        if value.type == target:
            return value
        if value.type.kind == "index":
            operation = "arith.index_cast"
        elif value.type.is_integer and target.is_float:
            operation = ("arith.uitofp"
                         if value.type.name == "u32" else "arith.sitofp")
        elif value.type.is_float and target.is_integer:
            operation = ("arith.fptoui"
                         if target.name == "u32" else "arith.fptosi")
        elif value.type.is_float and target.is_float:
            source_width = int(value.type.name[1:])
            target_width = int(target.name[1:])
            operation = ("arith.extf"
                         if source_width < target_width else "arith.truncf")
        elif value.type.is_integer and target.is_integer:
            operation = "arith.bitcast"
        else:
            raise self.context.error(
                node, f"cannot convert {value.type.mlir} to {target.mlir}")
        result = self._fresh()
        self._line(
            f"{result} = {operation} {value.name} : {value.type.mlir} to {target.mlir}"
        )
        return Value(result, target)

    def _intrinsic(self, node: ast.AST, name: str, arguments: list[Value],
                   result_type: DslType) -> Value:
        result = self._fresh()
        operand_types = ", ".join(argument.type.mlir for argument in arguments)
        self._line(
            f'{result} = "vernon.intrinsic"({", ".join(argument.name for argument in arguments)}) '
            f'{{name = "{name}"}} : ({operand_types}) -> {result_type.mlir}')
        return Value(result, result_type)

    def _attribute(self, node: ast.Attribute) -> Value:
        value = self._expression(node.value)
        if value.type.kind == "struct":
            fields = dict(self.context.structs[value.type.name])
            if node.attr not in fields:
                raise self.context.error(
                    node,
                    f"struct '{value.type.name}' has no field '{node.attr}'")
            result_type = fields[node.attr].type
            result = self._fresh()
            self._line(
                f'{result} = "vernon.struct_get"({value.name}) {{field = "{node.attr}"}} : '
                f"({value.type.mlir}) -> {result_type.mlir}")
            return Value(result, result_type)
        if value.type.kind == "tensor" and node.attr and set(
                node.attr) <= _SWIZZLES:
            shape = value.type.arguments[1:]
            if len(shape) != 1:
                raise self.context.error(node,
                                         "swizzle requires a rank-1 tensor")
            indices = [
                "xyzw".find(character)
                if character in "xyzw" else "rgba".find(character)
                for character in node.attr
            ]
            if any(index >= int(shape[0]) for index in indices):
                raise self.context.error(
                    node, f"swizzle '{node.attr}' is out of bounds")
            element = value.type.arguments[0]
            assert isinstance(element, DslType)
            result_type = element if len(indices) == 1 else DslType(
                "tensor", "Tensor", (element, len(indices)))
            result = self._fresh()
            self._line(
                f'{result} = "vernon.swizzle"({value.name}) {{mask = "{node.attr}"}} : '
                f"({value.type.mlir}) -> {result_type.mlir}")
            return Value(result, result_type)
        raise self.context.error(
            node,
            f"type '{value.type.name}' has no supported attribute '{node.attr}'"
        )

    def _index(self, node: ast.Subscript) -> Value:
        value = self._expression(node.value)
        if value.type.kind in {"buffer", "addressable_tensor"}:
            index = (self._tensor_buffer_index(node, value.type)
                     if value.type.kind == "addressable_tensor" else
                     self._buffer_index(node.slice))
            element = value.type.arguments[0]
            assert isinstance(element, DslType)
            result = self._fresh()
            self._line(
                f'{result} = "vernon.intrinsic"({value.name}, {index.name}) '
                f'{{name = "buffer_load"}} : ({value.type.mlir}, index) -> {element.mlir}'
            )
            return Value(result, element)
        if value.type.kind != "tensor":
            raise self.context.error(
                node, "indexing requires a Tensor or Buffer value")
        indices = list(node.slice.elts) if isinstance(
            node.slice, ast.Tuple) else [node.slice]
        if len(indices) != len(value.type.arguments) - 1:
            raise self.context.error(
                node, "tensor.extract requires one index per tensor dimension")
        index_values: list[str] = []
        for index_node in indices:
            index = self._expression(index_node)
            if not index.type.is_integer:
                raise self.context.error(index_node,
                                         "tensor index must be an integer")
            if index.type.mlir == "index":
                index_values.append(index.name)
            else:
                cast = self._fresh()
                self._line(
                    f"{cast} = arith.index_cast {index.name} : {index.type.mlir} to index"
                )
                index_values.append(cast)
        element = value.type.arguments[0]
        assert isinstance(element, DslType)
        result_type = element
        result = self._fresh()
        self._line(
            f"{result} = tensor.extract {value.name}[{', '.join(index_values)}] : {value.type.mlir}"
        )
        return Value(result, result_type)

    def _tensor_buffer_index(self, node: ast.Subscript,
                             value_type: DslType) -> Value:
        index_nodes = (list(node.slice.elts) if isinstance(
            node.slice, ast.Tuple) else [node.slice])
        shape = value_type.arguments[1:]
        if len(index_nodes) != len(shape):
            raise self.context.error(
                node, "Tensor indexing requires one index per dimension")
        indices = [self._buffer_index(index) for index in index_nodes]
        current = indices[0]
        for dimension, index in zip(shape[1:], indices[1:], strict=True):
            if not isinstance(dimension, int):
                raise self.context.error(
                    node,
                    "runtime Tensor dimensions must be specialized before lowering"
                )
            extent = self._fresh()
            self._line(f"{extent} = arith.constant {dimension} : index")
            multiplied = self._fresh()
            self._line(
                f"{multiplied} = arith.muli {current.name}, {extent} : index")
            added = self._fresh()
            self._line(
                f"{added} = arith.addi {multiplied}, {index.name} : index")
            current = Value(added, DslType("index", "index"))
        return current

    def _buffer_index(self, node: ast.AST) -> Value:
        if isinstance(node, ast.Tuple):
            raise self.context.error(node, "buffers require exactly one index")
        index = self._expression(node)
        if not index.type.is_integer:
            raise self.context.error(node, "buffer index must be an integer")
        if index.type.mlir == "index":
            return index
        cast = self._fresh()
        self._line(
            f"{cast} = arith.index_cast {index.name} : {index.type.mlir} to index"
        )
        return Value(cast, DslType("scalar", "index"))

    def _store_index(self, target: ast.Subscript, value: Value) -> None:
        buffer = self._expression(target.value)
        if buffer.type.kind not in {"buffer", "addressable_tensor"}:
            raise self.context.error(
                target,
                "indexed assignment is supported only for addressable Tensor or Buffer values"
            )
        element = buffer.type.arguments[0]
        assert isinstance(element, DslType)
        self._require_same_type(target, element, value.type)
        index = (self._tensor_buffer_index(target, buffer.type)
                 if buffer.type.kind == "addressable_tensor" else
                 self._buffer_index(target.slice))
        self._line(
            f'"vernon.intrinsic"({buffer.name}, {index.name}, {value.name}) '
            f'{{name = "buffer_store"}} : ({buffer.type.mlir}, index, {element.mlir}) -> ()'
        )

    def _require_same_type(self, node: ast.AST, expected: DslType,
                           actual: DslType) -> None:
        if expected != actual:
            raise self.context.error(
                node,
                f"type mismatch: expected {expected.mlir}, got {actual.mlir}")


class Compiler:
    """Compiles a restricted Python source string without importing or executing it."""

    def compile(
            self,
            source: str,
            filename: str = "<string>",
            dependencies: tuple[tuple[str, str], ...] = (),
            declared_features: tuple[str, ...] = (),
            enabled_features: tuple[str, ...] = (),
    ) -> str:
        try:
            module = ast.parse(source, filename=filename, type_comments=False)
        except SyntaxError as error:
            raise CompileError(
                error.msg,
                SourceLocation(filename, error.lineno or 1, error.offset or 1),
            ) from None
        context = _ModuleContext(filename)
        self._collect_struct_names(module, context)
        type_parser = _TypeParser(context)
        self._collect_structs(module, context, type_parser)
        self._collect_signatures(module, context, type_parser)

        module_attributes = [
            'vernon.frontend = "python"',
            "vernon.frontend_version = 2 : i64",
        ]
        if dependencies:
            encoded = ", ".join(
                json.dumps(f"{path}={digest}")
                for path, digest in dependencies)
            module_attributes.append(
                f"vernon.source_dependencies = [{encoded}]")
        if declared_features:
            declarations = ", ".join(
                json.dumps(name) for name in sorted(declared_features))
            module_attributes.append(
                f"vernon.feature_declarations = [{declarations}]")
        if enabled_features:
            variant = ", ".join(
                json.dumps(name) for name in sorted(enabled_features))
            module_attributes.append(f"vernon.variant_key = [{variant}]")
        body: list[str] = [
            f"module attributes {{{', '.join(module_attributes)}}} {{"
        ]
        for name in sorted(context.structs):
            fields = context.structs[name]
            field_text = ", ".join(f'"{field_name}:{annotation.type.mlir}"'
                                   for field_name, annotation in fields)
            body.append(
                f'  "vernon.struct"() {{fields = [{field_text}], sym_name = "{name}"}} : () -> ()'
            )
        for node in module.body:
            if isinstance(node, ast.FunctionDef):
                stage, workgroup_size = self._decorator(node, context)
                annotations = [
                    type_parser.parse(argument.annotation)
                    for argument in node.args.args
                ]
                body.extend(
                    _FunctionEmitter(
                        context,
                        node,
                        context.signatures[node.name],
                        annotations,
                        context.result_annotations[node.name],
                        stage,
                        workgroup_size,
                    ).emit())
            elif isinstance(node, (ast.Import, ast.ImportFrom, ast.ClassDef)):
                continue
            elif isinstance(node, ast.Expr) and isinstance(
                    node.value, ast.Constant) and isinstance(
                        node.value.value, str):
                continue
            else:
                raise context.error(
                    node,
                    f"unsupported module-level syntax: {type(node).__name__}")
        body.append("}")
        return "\n".join(body) + "\n"

    def compile_file(self,
                     input_path: str | Path,
                     *,
                     features: Iterable[str] = (),
                     entry: str | None = None) -> str:
        path = Path(input_path)
        enabled_features = tuple(sorted(set(features)))
        project = load_project(path, enabled_features, entry)
        return self.compile(project.source, str(path), project.dependencies,
                            project.features, enabled_features)

    @staticmethod
    def _collect_struct_names(module: ast.Module,
                              context: _ModuleContext) -> None:
        for node in module.body:
            if isinstance(node, ast.ClassDef):
                decorators = {
                    (_name(decorator.func) if isinstance(decorator, ast.Call)
                     else _name(decorator) or "").split(".")[-1]
                    for decorator in node.decorator_list
                }
                if "struct" not in decorators:
                    raise context.error(node, "DSL classes must use @struct")
                context.structs[node.name] = ()

    @staticmethod
    def _collect_structs(module: ast.Module, context: _ModuleContext,
                         parser: _TypeParser) -> None:
        for node in module.body:
            if not isinstance(node, ast.ClassDef):
                continue
            fields: list[tuple[str, AnnotatedType]] = []
            for statement in node.body:
                if isinstance(statement, ast.Pass):
                    continue
                if not isinstance(statement, ast.AnnAssign) or not isinstance(
                        statement.target,
                        ast.Name) or statement.value is not None:
                    raise context.error(
                        statement,
                        "@struct bodies may contain only annotation-only fields"
                    )
                fields.append(
                    (statement.target.id, parser.parse(statement.annotation)))
            context.structs[node.name] = tuple(fields)

    @staticmethod
    def _collect_signatures(module: ast.Module, context: _ModuleContext,
                            parser: _TypeParser) -> None:
        for node in module.body:
            if not isinstance(node, ast.FunctionDef):
                continue
            if node.args.posonlyargs or node.args.kwonlyargs or node.args.vararg or node.args.kwarg or node.args.defaults:
                raise context.error(
                    node,
                    "DSL functions support only required positional arguments")
            arguments: list[DslType] = []
            for argument in node.args.args:
                if argument.annotation is None:
                    raise context.error(
                        argument,
                        f"argument '{argument.arg}' requires a type annotation"
                    )
                arguments.append(parser.parse(argument.annotation).type)
            result = None
            result_annotation = None
            if node.returns is not None and not (
                    isinstance(node.returns, ast.Constant)
                    and node.returns.value is None):
                result_annotation = parser.parse(node.returns)
                result = result_annotation.type
            context.signatures[node.name] = FunctionSignature(
                tuple(arguments), result)
            context.result_annotations[node.name] = result_annotation

    @staticmethod
    def _decorator(
        node: ast.FunctionDef, context: _ModuleContext
    ) -> tuple[str | None, tuple[int, int, int] | None]:
        stage = None
        workgroup_size = None
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name) or isinstance(
                    decorator, ast.Attribute):
                name = (_name(decorator) or "").split(".")[-1]
                if name in {"vertex", "fragment"}:
                    stage = name
                elif name in {"compute", "kernel"}:
                    stage = "compute"
                    workgroup_size = (1, 1, 1)
                else:
                    raise context.error(decorator,
                                        f"unknown DSL decorator '{name}'")
            elif isinstance(decorator, ast.Call) and (
                    _name(decorator.func)
                    or "").split(".")[-1] in {"compute", "kernel"}:
                stage = "compute"
                values = None
                for keyword in decorator.keywords:
                    if keyword.arg == "workgroup_size":
                        values = keyword.value
                    else:
                        raise context.error(
                            keyword, f"unknown kernel option '{keyword.arg}'")
                if decorator.args or not isinstance(values, ast.Tuple) or len(
                        values.elts) != 3:
                    raise context.error(
                        decorator, "kernel requires workgroup_size=(x, y, z)")
                parsed: list[int] = []
                for value in values.elts:
                    if not isinstance(value, ast.Constant) or not isinstance(
                            value.value, int) or value.value <= 0:
                        raise context.error(
                            value,
                            "workgroup dimensions must be positive integer literals"
                        )
                    parsed.append(value.value)
                workgroup_size = tuple(parsed)  # type: ignore[assignment]
            else:
                raise context.error(decorator, "unsupported decorator syntax")
        return stage, workgroup_size


def compile_source(source: str, filename: str = "<string>") -> str:
    return Compiler().compile(source, filename)


def compile_file(input_path: str | Path,
                 *,
                 features: Iterable[str] = (),
                 entry: str | None = None) -> str:
    return Compiler().compile_file(input_path, features=features, entry=entry)
