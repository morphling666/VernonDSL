from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from ..diagnostics import CompileError, SourceLocation
from ..language.ast_utils import dotted_name as _name
from ..language.ast_utils import rectangular_literal
from ..language.syntax import FRONTEND_VERSION, INTRINSIC_METHODS
from ..module_graph import load_project
from ..shader_contracts import BUILTIN_CONTRACTS, GENERATED_INTERFACE_CONTRACTS, TypeContract, texture_sampling_contract
from ..struct_methods import normalize_struct_methods
from .abi import value_abi_layout
from .interfaces import plan_generated_interface
from .model import (
    AccessMode,
    ConcreteType,
    StorageEffect,
    StorageRegionKind,
    Termination,
    TypedExpression,
    TypedFunctionInstance,
    TypedStatement,
    is_abi_stable_value,
)
from .monomorphize import infer_and_monomorphize_helpers
from .request import FrontendCompileRequest, FrontendCompileResult
from .type_parser import AnnotatedType, Metadata, TypeParser
from .type_solver import can_convert

DslType = ConcreteType

_SWIZZLES = set("xyzwrgba")


def _dsl_type_from_contract(contract: TypeContract) -> DslType:
    if contract.kind == "tensor":
        element = DslType("scalar", contract.name)
        return DslType("tensor", "Tensor", (element, *contract.shape))
    return DslType(contract.kind, contract.name)


@dataclass(frozen=True)
class FunctionSignature:
    arguments: tuple[DslType, ...]
    result: DslType | None


@dataclass(frozen=True)
class _ViewLayout:
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    offset: int


@dataclass(frozen=True)
class Value:
    name: str
    type: DslType
    fields: tuple["Value", ...] | None = None
    access: AccessMode = AccessMode.READ
    view_layout: _ViewLayout | None = None

    @property
    def abi_type(self) -> DslType:
        if self.type.kind == "tensor_view":
            element, rank, _ = self.type.arguments
            assert isinstance(element, DslType)
            return DslType("tensor_view_abi", "TensorViewAbi", (element, rank, self.access.value))
        return self.type


@dataclass
class _ModuleContext:
    filename: str
    runtime_entry: str | None = None
    tensor_view_layouts: dict[str, _ViewLayout] = field(default_factory=dict)
    structs: dict[str, tuple[tuple[str, AnnotatedType], ...]] = field(default_factory=dict)
    signatures: dict[str, FunctionSignature] = field(default_factory=dict)
    result_annotations: dict[str, AnnotatedType | None] = field(default_factory=dict)
    shared_functions: set[str] = field(default_factory=set)
    typed_functions: dict[str, TypedFunctionInstance] = field(default_factory=dict)

    def error(self, node: ast.AST, message: str) -> CompileError:
        return CompileError(
            message,
            SourceLocation(
                self.filename,
                getattr(node, "lineno", 1),
                getattr(node, "col_offset", 0) + 1,
            ),
        )


class _FunctionEmitter:
    def __init__(
        self,
        context: _ModuleContext,
        signature: FunctionSignature,
        argument_annotations: list[AnnotatedType],
        result_annotation: AnnotatedType | None,
        typed_function: TypedFunctionInstance,
        stage: str | None,
        workgroup_size: tuple[int, int, int] | None,
    ):
        self.context = context
        self.node = typed_function.source
        self.signature = signature
        self.argument_annotations = argument_annotations
        self.result_annotation = result_annotation
        self.typed_function = typed_function
        self.typed_statements: dict[int, TypedStatement] = {}
        self.typed_expressions: dict[int, TypedExpression] = {}
        self._index_typed_statements(typed_function.body)
        self.stage = stage
        self.workgroup_size = workgroup_size
        self.lines: list[str] = []
        self.indent = 1
        self.next_value = 0
        self.environment: dict[str, Value] = {}
        self.generated_values: dict[str, Value] = {}
        self.implicit_samplers: dict[str, Value] = {}
        self.interface_plan = plan_generated_interface(context, self.node, argument_annotations, stage)
        self.returned = False
        self.loop_controls: list[str] = []
        self.next_hidden = 0
        direct_returns = {id(statement) for statement in self.node.body if isinstance(statement, ast.Return)}
        self.use_return_state = any(
            isinstance(value, ast.Return) and id(value) not in direct_returns for value in ast.walk(self.node)
        )
        self.return_flag_name = self._hidden_name("returned") if self.use_return_state else None
        self.return_value_name = (
            self._hidden_name("return_value") if self.use_return_state and self.signature.result is not None else None
        )

    def _index_typed_statements(self, statements: tuple[TypedStatement, ...]) -> None:
        for statement in statements:
            self.typed_statements[id(statement.source)] = statement
            for expression in statement.expressions:
                self.typed_expressions[id(expression.source)] = expression
            self._index_typed_statements(statement.children)

    def _typed_expression(self, node: ast.expr) -> TypedExpression:
        typed = self.typed_expressions.get(id(node))
        if typed is None:
            raise self.context.error(node, "internal error: expression is missing from the typed semantic model")
        return typed

    def _validate_builtin_contract(self, builtin: str, value_type: DslType, direction: str, label: str) -> None:
        contract = BUILTIN_CONTRACTS.get(builtin)
        if contract is None:
            raise self.context.error(self.node, f"{label} uses unknown builtin '{builtin}'")
        expected_type = _dsl_type_from_contract(contract.type)
        if self.stage != contract.stage or direction != contract.direction:
            raise self.context.error(
                self.node,
                f"builtin '{builtin}' requires a {contract.stage} "
                f"{contract.direction}, but {label} is a "
                f"{self.stage or 'non-entry'} {direction}",
            )
        if value_type != expected_type:
            raise self.context.error(
                self.node,
                f"builtin '{builtin}' requires type {expected_type.mlir}, but {label} has type {value_type.mlir}",
            )

    def _validate_builtin_annotation(self, annotation: AnnotatedType, direction: str, label: str) -> None:
        builtins = [str(item.arguments[0]) for item in annotation.metadata if item.kind == "builtin"]
        if len(builtins) > 1:
            raise self.context.error(self.node, f"{label} has more than one builtin annotation")
        if builtins:
            self._validate_builtin_contract(builtins[0], annotation.type, direction, label)

    def _entry_result_fields(self) -> tuple[tuple[str, AnnotatedType], ...] | None:
        if (
            self.stage not in {"vertex", "fragment"}
            or self.signature.result is None
            or self.signature.result.kind != "struct"
        ):
            return None
        fields = self.context.structs[self.signature.result.name]
        occupied: set[int] = set()
        builtins: set[str] = set()
        for field_name, annotation in fields:
            self._validate_builtin_annotation(annotation, "output", f"output field '{field_name}'")
            locations = [int(item.arguments[0]) for item in annotation.metadata if item.kind == "location"]
            builtin_values = [str(item.arguments[0]) for item in annotation.metadata if item.kind == "builtin"]
            if self.stage == "fragment" and (len(locations) != 1 or builtin_values):
                raise self.context.error(
                    self.node, f"fragment output field '{field_name}' requires exactly one location"
                )
            if self.stage == "vertex" and (len(locations) + len(builtin_values) != 1):
                raise self.context.error(
                    self.node, f"vertex output field '{field_name}' requires exactly one location or builtin"
                )
            if builtin_values:
                builtin = builtin_values[0]
                if builtin in builtins:
                    raise self.context.error(self.node, f"vertex output builtin '{builtin}' is used more than once")
                builtins.add(builtin)
                continue
            location = locations[0]
            if location in occupied:
                raise self.context.error(self.node, f"fragment output location {location} is used more than once")
            occupied.add(location)
        return fields

    def emit(self) -> list[str]:
        emit_value_abi_metadata = self.stage is not None or self.node.name in self.context.shared_functions
        arguments: list[str] = []
        for index, (argument, annotation) in enumerate(
            zip(self.node.args.args, self.argument_annotations, strict=True)
        ):
            self._validate_builtin_annotation(annotation, "input", f"argument '{argument.arg}'")
            value_type = annotation.type
            typed_parameter = self.typed_function.parameters[index]
            access = typed_parameter.access
            view_layout = (
                self.context.tensor_view_layouts.get(argument.arg)
                if self.node.name == self.context.runtime_entry and value_type.kind == "tensor_view"
                else None
            )
            value = Value(
                f"%arg{index}",
                value_type,
                access=access,
                view_layout=view_layout,
            )
            self.environment[argument.arg] = value
            attributes = self._metadata_attributes(
                annotation.metadata, stage=self.stage, is_result=False, default_location=index
            )
            attributes.append(f'vernon.source_name = "{argument.arg}"')
            if annotation.type.kind == "scalar":
                attributes.append(f'vernon.dtype = "{annotation.type.name}"')
            elif annotation.type.kind == "tensor":
                element_type = annotation.type.arguments[0]
                assert isinstance(element_type, DslType)
                attributes.append(f'vernon.dtype = "{element_type.name}"')
            elif annotation.type.kind == "tensor_view":
                element_type = annotation.type.arguments[0]
                assert isinstance(element_type, DslType)
                if element_type.kind == "scalar":
                    attributes.append(f'vernon.dtype = "{element_type.name}"')
                attributes.extend(
                    attribute.replace("vernon.abi_", "vernon.element_abi_")
                    for attribute in self._abi_attributes(element_type)
                )
            if emit_value_abi_metadata and value_type.kind in {"scalar", "tensor", "tuple", "struct"}:
                attributes.extend(self._abi_attributes(value_type))
            if value_type.kind == "tensor_view":
                has_explicit_binding = any(attribute.startswith("vernon.binding") for attribute in attributes)
                attributes = [
                    attribute
                    for attribute in attributes
                    if not attribute.startswith(("vernon.interface", "vernon.location"))
                ]
                attributes.append('vernon.interface = "resource"')
                if not has_explicit_binding:
                    attributes.extend(("vernon.set = 0 : i64", f"vernon.binding = {index} : i64"))
            suffix = f" {{{', '.join(attributes)}}}" if attributes else ""
            arguments.append(f"{value.name}: {value.abi_type.mlir}{suffix}")
        self._append_generated_arguments(arguments)
        result = ""
        entry_fields = self._entry_result_fields()
        if entry_fields is not None:
            flattened_results: list[str] = []
            for field_name, annotation in entry_fields:
                result_attributes = self._metadata_attributes(
                    annotation.metadata,
                    stage=self.stage,
                    is_result=True,
                    default_location=0,
                )
                result_attributes.append(f'vernon.source_name = "{field_name}"')
                if emit_value_abi_metadata:
                    result_attributes.extend(self._abi_attributes(annotation.type))
                flattened_results.append(f"{annotation.type.mlir} {{{', '.join(result_attributes)}}}")
            result = f" -> ({', '.join(flattened_results)})"
        elif self.signature.result:
            result_metadata = self.result_annotation.metadata if self.result_annotation else ()
            if self.result_annotation is not None:
                self._validate_builtin_annotation(self.result_annotation, "output", "result")
            has_result_slot = any(item.kind in {"location", "builtin", "instance"} for item in result_metadata)
            if self.stage == "vertex" and not has_result_slot:
                self._validate_builtin_contract("position", self.signature.result, "output", "result")
            result_attributes = self._metadata_attributes(
                result_metadata, stage=self.stage, is_result=True, default_location=0
            )
            if emit_value_abi_metadata:
                result_attributes.extend(self._abi_attributes(self.signature.result))
            suffix = f" {{{', '.join(result_attributes)}}}" if result_attributes else ""
            result = f" -> ({self.signature.result.mlir}{suffix})"
        function_attributes: list[str] = []
        if self.stage:
            function_attributes.append("vernon.entry")
            function_attributes.append(f'vernon.stage = "{self.stage}"')
            reflected_effects: list[str] = []
            for effect in self.typed_function.effects:
                if not isinstance(effect, StorageEffect):
                    continue
                fields = [
                    f'kind = "{effect.kind.value}"',
                    f'owner = "{effect.owner.parameter}"',
                    f'region = "{effect.region.kind.value}"',
                ]
                if effect.region.kind is StorageRegionKind.ELEMENT:
                    indices = ", ".join(str(index) for index in effect.region.indices)
                    fields.append(f"indices = array<i64: {indices}>")
                reflected_effects.append("{" + ", ".join(fields) + "}")
            function_attributes.append(f"vernon.storage_effects = [{', '.join(reflected_effects)}]")
        elif emit_value_abi_metadata:
            function_attributes.append("vernon.shared")
        if self.workgroup_size:
            values = ", ".join(str(value) for value in self.workgroup_size)
            function_attributes.append(f"vernon.workgroup_size = array<i32: {values}>")
        attributes = f" attributes {{{', '.join(function_attributes)}}}" if function_attributes else ""
        visibility = "" if self.stage else " private"
        self.lines.append(f"  func.func{visibility} @{self.node.name}({', '.join(arguments)}){result}{attributes} {{")
        if self.use_return_state:
            assert self.return_flag_name is not None
            self.environment[self.return_flag_name] = self._bool_constant(False)
            if self.return_value_name is not None:
                assert self.signature.result is not None
                self.environment[self.return_value_name] = self._default_value(self.node, self.signature.result)
        self._emit_source_block(self.node.body)
        if self.use_return_state:
            value = self.environment[self.return_value_name] if self.return_value_name is not None else None
            self._emit_function_return(self.node, value)
            self.returned = True
        if not self.returned:
            if self.signature.result is not None:
                raise self.context.error(self.node, f"function '{self.node.name}' may exit without returning a value")
            self._line("func.return")
        self.lines.append("  }")
        return self.lines

    def _abi_attributes(self, value_type: DslType) -> list[str]:
        if value_type.kind not in {"scalar", "tensor", "tuple", "struct"}:
            return []

        def fields(name: str) -> tuple[DslType, ...]:
            return tuple(annotation.type for _, annotation in self.context.structs[name])

        layout = value_abi_layout(value_type, fields)
        attributes = [
            f"vernon.abi_alignment = {layout.alignment} : i64",
            f"vernon.abi_size = {layout.size} : i64",
        ]
        if layout.field_offsets:
            offsets = ", ".join(str(offset) for offset in layout.field_offsets)
            attributes.append(f"vernon.abi_field_offsets = array<i64: {offsets}>")
        if layout.element_stride is not None:
            attributes.append(f"vernon.abi_element_stride = {layout.element_stride} : i64")
        return attributes

    def _append_generated_arguments(self, arguments: list[str]) -> None:
        for sampler_plan in self.interface_plan.implicit_samplers:
            texture_name = sampler_plan.texture_name
            index = len(arguments)
            value = Value(f"%arg{index}", DslType("sampler", "Sampler"))
            self.implicit_samplers[self.environment[texture_name].name] = value
            source_name = f"__vernon_implicit_sampler_{texture_name}"
            binding_attributes = (
                f", vernon.set = {sampler_plan.descriptor_set} : i64, vernon.binding = {sampler_plan.binding} : i64"
            )
            arguments.append(
                f'{value.name}: !vernon.sampler {{vernon.interface = "resource", '
                f'vernon.source_name = "{source_name}", '
                f'vernon.implicit = "sampler", '
                f'vernon.implicit_texture = "{texture_name}"'
                f"{binding_attributes}}}"
            )

        for api in self.interface_plan.generated_apis:
            contract = GENERATED_INTERFACE_CONTRACTS[api]
            index = len(arguments)
            value_type = _dsl_type_from_contract(contract.type)
            value = Value(f"%arg{index}", value_type)
            self.generated_values[api] = value
            attributes = [
                f'vernon.interface = "{contract.interface}"',
                f'vernon.source_name = "__vernon_{api}"',
                f'vernon.implicit = "{api}"',
            ]
            if contract.builtin is not None:
                attributes.append(f'vernon.builtin = "{contract.builtin}"')
            else:
                attributes.append('vernon.dtype = "f32"')
            arguments.append(f"{value.name}: {value.type.mlir} {{{', '.join(attributes)}}}")

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

        has_slot = any(item.kind in {"location", "builtin", "instance"} for item in items)
        if interface:
            attributes.append(f'vernon.interface = "{interface}"')
            if interface in {"input", "output"} and not has_slot:
                if is_result and stage == "vertex":
                    attributes.append('vernon.builtin = "position"')
                else:
                    attributes.append(f"vernon.location = {default_location} : i64")

        for item in items:
            if item.kind in {"location", "instance"}:
                attributes.append(f"vernon.location = {item.arguments[0]} : i64")
                if item.kind == "instance":
                    attributes.append(f"vernon.instance_divisor = {item.arguments[1]} : i64")
            elif item.kind == "resource":
                attributes.extend(
                    (
                        f"vernon.set = {item.arguments[0]} : i64",
                        f"vernon.binding = {item.arguments[1]} : i64",
                    )
                )
            elif item.kind == "uniform":
                if item.arguments:
                    attributes.extend(
                        (
                            f"vernon.set = {item.arguments[0]} : i64",
                            f"vernon.binding = {item.arguments[1]} : i64",
                        )
                    )
            elif item.kind == "varying":
                attributes.append("vernon.varying = true")
            else:
                attributes.append(f'vernon.{item.kind} = "{item.arguments[0]}"')
        return attributes

    def _statement(self, typed_statement: TypedStatement) -> None:
        node = typed_statement.source
        if isinstance(node, ast.Pass):
            return
        if isinstance(node, ast.Expr):
            self._expression(node.value)
            return
        if isinstance(node, ast.Assign):
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Subscript):
                self._store_index(node.targets[0], self._expression(node.value))
                return
            if len(node.targets) == 1 and isinstance(node.targets[0], (ast.Tuple, ast.List)):
                self._destructure(node.targets[0], self._expression(node.value))
                return
            if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                raise self.context.error(node, "assignment target must be a local name or TensorView element")
            value = self._expression(node.value)
            if value.type.kind == "void":
                raise self.context.error(node.value, "a void function call cannot be assigned")
            target_name = node.targets[0].id
            typed_statement = self.typed_statements[id(node)]
            if not typed_statement.lvalues:
                raise self.context.error(node, "internal error: assignment has no typed lvalue")
            value = self._coerce_implicit(node.value, value, typed_statement.lvalues[0].type)
            self.environment[target_name] = value
            return
        if isinstance(node, ast.AugAssign):
            operation = ast.BinOp(left=node.target, op=node.op, right=node.value)
            ast.copy_location(operation, node)
            typed_statement = self.typed_statements[id(node)]
            if not typed_statement.lvalues:
                raise self.context.error(node, "internal error: augmented assignment has no typed lvalue")
            self.typed_expressions[id(operation)] = TypedExpression(
                operation,
                typed_statement.lvalues[0].type,
                {
                    ast.Add: "add",
                    ast.Sub: "sub",
                    ast.Mult: "mul",
                    ast.Div: "div",
                    ast.Mod: "mod",
                    ast.Pow: "pow",
                }.get(type(node.op)),
                (typed_statement.lvalues[0].type, typed_statement.lvalues[0].type),
            )
            value = self._expression(operation)
            if isinstance(node.target, ast.Name):
                if node.target.id not in self.environment:
                    raise self.context.error(node.target, f"unknown local value '{node.target.id}'")
                self._require_same_type(node, self.environment[node.target.id].type, value.type)
                self.environment[node.target.id] = value
                return
            if isinstance(node.target, ast.Subscript):
                self._store_index(node.target, value)
                return
            raise self.context.error(node, "augmented assignment requires a local name or indexed value")
        if isinstance(node, ast.AnnAssign):
            if not isinstance(node.target, ast.Name) or node.value is None:
                raise self.context.error(node, "annotated assignment requires a local name and value")
            typed_statement = self.typed_statements[id(node)]
            expected = typed_statement.lvalues[0].type
            value = self._coerce_implicit(node.value, self._expression(node.value, expected), expected)
            self.environment[node.target.id] = value
            return
        if isinstance(node, ast.Return):
            if node.value is None:
                if self.signature.result is not None:
                    raise self.context.error(node, "return value is required")
                value = None
            else:
                if self.signature.result is None:
                    raise self.context.error(node, "void function cannot return a value")
                value = self._expression(node.value, self.signature.result)
                value = self._coerce_implicit(node.value, value, self.signature.result)
            if self.use_return_state:
                assert self.return_flag_name is not None
                self.environment[self.return_flag_name] = self._bool_constant(True)
                if self.return_value_name is not None:
                    assert value is not None
                    self.environment[self.return_value_name] = value
                return
            self._emit_function_return(node, value)
            self.returned = True
            return
        if isinstance(node, ast.Break):
            if not self.loop_controls:
                raise self.context.error(node, "break is only valid inside a loop")
            self.environment[self.loop_controls[-1]] = self._control_constant(1)
            return
        if isinstance(node, ast.Continue):
            if not self.loop_controls:
                raise self.context.error(node, "continue is only valid inside a loop")
            self.environment[self.loop_controls[-1]] = self._control_constant(2)
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
        raise self.context.error(node, f"unsupported statement syntax: {type(node).__name__}")

    def _hidden_name(self, prefix: str) -> str:
        name = f"${prefix}_{self.next_hidden}"
        self.next_hidden += 1
        return name

    def _control_constant(self, value: int) -> Value:
        result = self._fresh()
        self._line(f"{result} = arith.constant {value} : i32")
        return Value(result, DslType("scalar", "i32"))

    def _bool_constant(self, value: bool) -> Value:
        result = self._fresh()
        self._line(f"{result} = arith.constant {'true' if value else 'false'}")
        return Value(result, DslType("scalar", "bool"))

    def _emit_function_return(self, node: ast.AST, value: Value | None) -> None:
        if value is None:
            self._line("func.return")
            return
        entry_fields = self._entry_result_fields()
        if entry_fields is None:
            self._line(f"func.return {value.name} : {value.type.mlir}")
            return
        fields = self._struct_value_fields(node, value)
        if len(fields) != len(entry_fields):
            raise self.context.error(node, "entry struct result field count does not match its annotation")
        self._line(
            "func.return "
            + ", ".join(field.name for field in fields)
            + " : "
            + ", ".join(field.type.mlir for field in fields)
        )

    def _struct_value_fields(self, node: ast.AST, value: Value) -> tuple[Value, ...]:
        if value.fields is not None:
            return value.fields
        if value.type.kind != "struct":
            raise self.context.error(node, "entry struct result requires a Struct value")
        fields: list[Value] = []
        for index, (name, annotation) in enumerate(self.context.structs[value.type.name]):
            result = self._fresh()
            self._line(
                f'{result} = "vernon.struct_get"({value.name}) '
                f'{{field = "{name}", index = {index} : i64}} : '
                f"({value.type.mlir}) -> {annotation.type.mlir}"
            )
            fields.append(Value(result, annotation.type))
        return tuple(fields)

    def _destructure(self, target: ast.Tuple | ast.List, value: Value) -> None:
        fields = self._tuple_fields(target, value)
        if len(target.elts) != len(fields):
            raise self.context.error(target, "Tuple destructuring arity does not match the value")
        for item, tuple_field in zip(target.elts, fields, strict=True):
            if isinstance(item, ast.Name):
                self.environment[item.id] = tuple_field
            elif isinstance(item, (ast.Tuple, ast.List)):
                self._destructure(item, tuple_field)
            else:
                raise self.context.error(item, "Tuple destructuring targets must be local names")

    def _statement_from_source(self, statement: ast.stmt) -> None:
        typed = self.typed_statements.get(id(statement))
        if typed is None:
            raise self.context.error(statement, "internal error: statement is missing from the typed semantic model")
        self._statement(typed)

    def _if(self, node: ast.If) -> None:
        condition = self._expression(node.test)
        self._require_same_type(node.test, DslType("scalar", "bool"), condition.type)
        typed_statement = self.typed_statements[id(node)]
        merged_names = [merge.name for merge in typed_statement.branch_merges]
        merged_types = [merge.type for merge in typed_statement.branch_merges]
        for control_name in self.loop_controls:
            if control_name not in merged_names:
                merged_names.append(control_name)
                merged_types.append(DslType("scalar", "i32"))
        for return_name, return_type in self._return_state_items():
            if return_name not in merged_names:
                merged_names.append(return_name)
                merged_types.append(return_type)
        outer = self.environment.copy()
        outer_lines = self.lines
        outer_indent = self.indent

        self.lines = []
        self.indent = outer_indent + 1
        self.environment = outer.copy()
        self._emit_source_block(node.body)
        then_lines = self.lines
        then_environment = self.environment.copy()

        self.lines = []
        self.environment = outer.copy()
        self._emit_source_block(node.orelse)
        else_lines = self.lines
        else_environment = self.environment.copy()

        self.lines = then_lines
        self.environment = then_environment
        self._yield_merged(node, merged_names, merged_types)
        self.lines = else_lines
        self.environment = else_environment
        self._yield_merged(node, merged_names, merged_types)

        results = [self._fresh() for _ in merged_names]
        lhs = f"{', '.join(results)} = " if results else ""
        result_types = f" -> ({', '.join(value.mlir for value in merged_types)})" if results else ""
        self.lines = outer_lines
        self.indent = outer_indent
        self.environment = outer
        self._line(f"{lhs}scf.if {condition.name}{result_types} {{")
        self.lines.extend(then_lines)
        self._line("} else {")
        self.lines.extend(else_lines)
        self._line("}")
        for name, result, result_type in zip(merged_names, results, merged_types, strict=True):
            self.environment[name] = Value(result, result_type)

    def _emit_source_block(self, statements: list[ast.stmt]) -> None:
        needs_guard = False
        for statement in statements:
            typed = self.typed_statements.get(id(statement))
            if typed is None:
                break
            if needs_guard:
                control_name = self.loop_controls[-1] if self.loop_controls else None
                self._emit_guarded_statement(typed, control_name)
            else:
                self._statement(typed)
            if self._contains_return(typed) or (
                self.loop_controls and self._contains_loop_exit(typed, typed.loop_depth)
            ):
                needs_guard = True
            if typed.termination is not Termination.FALLTHROUGH:
                break

    def _yield_merged(self, node: ast.If, names: list[str], types: list[DslType]) -> None:
        values: list[Value] = []
        for name, expected in zip(names, types, strict=True):
            value = self._coerce_implicit(node, self.environment[name], expected)
            values.append(value)
        if values:
            self._line(
                f"scf.yield {', '.join(value.name for value in values)} : "
                f"{', '.join(value.type.mlir for value in values)}"
            )
        else:
            self._line("scf.yield")

    def _default_value(self, node: ast.AST, value_type: DslType) -> Value:
        if value_type.kind in {"scalar", "index"}:
            result = self._fresh()
            literal = "0.0" if value_type.is_float else "false" if value_type.name == "bool" else "0"
            self._line(f"{result} = arith.constant {literal} : {value_type.mlir}")
            return Value(result, value_type)
        if value_type.kind == "tensor":
            element = value_type.arguments[0]
            assert isinstance(element, DslType)
            if element.kind == "scalar":
                result = self._fresh()
                literal = "0.0" if element.is_float else "false" if element.name == "bool" else "0"
                self._line(f"{result} = arith.constant dense<{literal}> : {value_type.mlir}")
                return Value(result, value_type)
            count = 1
            for extent in value_type.arguments[1:]:
                assert isinstance(extent, int)
                count *= extent
            values = [self._default_value(node, element) for _ in range(count)]
            result = self._fresh()
            self._line(
                f'{result} = "vernon.intrinsic"({", ".join(value.name for value in values)}) '
                f'{{name = "construct"}} : '
                f"({', '.join(value.type.mlir for value in values)}) -> {value_type.mlir}"
            )
            return Value(result, value_type)
        if value_type.kind == "tuple":
            elements = [element for element in value_type.arguments if isinstance(element, DslType)]
            values = [self._default_value(node, element) for element in elements]
            result = self._fresh()
            self._line(
                f'{result} = "vernon.tuple_create"({", ".join(value.name for value in values)}) : '
                f"({', '.join(value.type.mlir for value in values)}) -> {value_type.mlir}"
            )
            return Value(result, value_type, tuple(values))
        if value_type.kind == "struct":
            fields = self.context.structs[value_type.name]
            values = [self._default_value(node, annotation.type) for _, annotation in fields]
            result = self._fresh()
            self._line(
                f'{result} = "vernon.struct_create"({", ".join(value.name for value in values)}) '
                f'{{type_name = "{value_type.name}"}} : '
                f"({', '.join(value.type.mlir for value in values)}) -> {value_type.mlir}"
            )
            return Value(result, value_type, tuple(values))
        raise self.context.error(node, f"cannot create a control-flow payload for {value_type.mlir}")

    def _contains_loop_exit(self, statement: TypedStatement, target_depth: int) -> bool:
        if statement.termination in {Termination.BREAK, Termination.CONTINUE} and statement.loop_depth == target_depth:
            return True
        if isinstance(statement.source, (ast.For, ast.While)):
            return False
        return any(self._contains_loop_exit(child, target_depth) for child in statement.children)

    @classmethod
    def _contains_return(cls, statement: TypedStatement) -> bool:
        if statement.termination is Termination.RETURN:
            return True
        return any(cls._contains_return(child) for child in statement.children)

    def _return_state_items(self) -> list[tuple[str, DslType]]:
        items: list[tuple[str, DslType]] = []
        if self.return_flag_name is not None:
            items.append((self.return_flag_name, DslType("scalar", "bool")))
        if self.return_value_name is not None:
            assert self.signature.result is not None
            items.append((self.return_value_name, self.signature.result))
        return items

    def _emit_guarded_statement(self, typed: TypedStatement, control_name: str | None) -> None:
        outer = self.environment.copy()
        assigned = self._assigned_names([typed.source])
        lvalue_types = {value.name: value.type for value in typed.lvalues if value.kind == "local"}
        for name in sorted(assigned - outer.keys()):
            value_type = lvalue_types.get(name)
            if value_type is not None:
                outer[name] = self._default_value(typed.source, value_type)
        carried_names = [
            name
            for name in sorted(assigned & outer.keys())
            if outer[name].type.kind not in {"sampler", "tensor_storage", "tensor_view", "texture"}
        ]
        for name in self.loop_controls:
            if name in outer and name not in carried_names:
                carried_names.append(name)
        for name, _ in self._return_state_items():
            if name in outer and name not in carried_names:
                carried_names.append(name)
        carried_types = [outer[name].type for name in carried_names]
        active: Value | None = None
        if control_name is not None:
            zero = self._control_constant(0)
            control_active = self._fresh()
            self._line(f"{control_active} = arith.cmpi eq, {outer[control_name].name}, {zero.name} : i32")
            active = Value(control_active, DslType("scalar", "bool"))
        if self.return_flag_name is not None:
            false_value = self._bool_constant(False)
            not_returned = self._fresh()
            self._line(f"{not_returned} = arith.cmpi eq, {outer[self.return_flag_name].name}, {false_value.name} : i1")
            if active is None:
                active = Value(not_returned, DslType("scalar", "bool"))
            else:
                combined = self._fresh()
                self._line(f"{combined} = arith.andi {active.name}, {not_returned} : i1")
                active = Value(combined, DslType("scalar", "bool"))
        if active is None:
            self._statement(typed)
            return

        outer_lines = self.lines
        outer_indent = self.indent
        self.lines = []
        self.indent = outer_indent + 1
        self.environment = outer.copy()
        self._statement(typed)
        self._yield_values(typed.source, carried_names, carried_types)
        then_lines = self.lines

        self.lines = []
        self.environment = outer.copy()
        self._yield_values(typed.source, carried_names, carried_types)
        else_lines = self.lines

        results = [self._fresh() for _ in carried_names]
        result_prefix = f"{', '.join(results)} = " if results else ""
        result_types = f" -> ({', '.join(value.mlir for value in carried_types)})" if results else ""
        self.lines = outer_lines
        self.indent = outer_indent
        self.environment = outer
        self._line(f"{result_prefix}scf.if {active.name}{result_types} {{")
        self.lines.extend(then_lines)
        self._line("} else {")
        self.lines.extend(else_lines)
        self._line("}")
        for name, result, value_type in zip(carried_names, results, carried_types, strict=True):
            self.environment[name] = Value(result, value_type)

    def _yield_values(self, node: ast.AST, names: list[str], types: list[DslType]) -> None:
        values = [
            self._coerce_implicit(node, self.environment[name], value_type)
            for name, value_type in zip(names, types, strict=True)
        ]
        if values:
            self._line(
                f"scf.yield {', '.join(value.name for value in values)} : "
                f"{', '.join(value.type.mlir for value in values)}"
            )
        else:
            self._line("scf.yield")

    def _emit_loop_body(self, statements: list[ast.stmt], control_name: str, target_depth: int) -> None:
        needs_guard = False
        for source in statements:
            typed = self.typed_statements.get(id(source))
            if typed is None:
                break
            if needs_guard:
                self._emit_guarded_statement(typed, control_name)
            else:
                self._statement(typed)
            if self._contains_loop_exit(typed, target_depth) or self._contains_return(typed):
                needs_guard = True
            if typed.termination is not Termination.FALLTHROUGH:
                break

    def _for(self, node: ast.For) -> None:
        if (
            not isinstance(node.target, ast.Name)
            or not isinstance(node.iter, ast.Call)
            or _name(node.iter.func) != "range"
        ):
            raise self.context.error(node, "for loops must have the form 'for name in range(...)'")
        if node.orelse:
            raise self.context.error(node, "for-else is not supported")
        if not 1 <= len(node.iter.args) <= 3 or node.iter.keywords:
            raise self.context.error(node.iter, "range requires one to three positional i32 arguments")
        integer_type = DslType("scalar", "i32")
        arguments = [
            self._coerce_implicit(argument, self._expression(argument, integer_type), integer_type)
            for argument in node.iter.args
        ]
        zero = self._control_constant(0)
        one = self._control_constant(1)
        if len(arguments) == 1:
            start, stop, step = zero, arguments[0], one
        elif len(arguments) == 2:
            start, stop = arguments
            step = one
        else:
            start, stop, step = arguments
        if len(node.iter.args) == 3 and self._literal_integer(node.iter.args[2]) == 0:
            raise self.context.error(node.iter.args[2], "range step must not be zero")
        if len(node.iter.args) == 3 and self._literal_integer(node.iter.args[2]) is None:
            nonzero = self._fresh()
            self._line(f"{nonzero} = arith.cmpi ne, {step.name}, {zero.name} : i32")
            self._line(f'cf.assert {nonzero}, "range step must not be zero"')

        outer = self.environment.copy()
        typed_statement = self.typed_statements[id(node)]
        control_name = self._hidden_name("loop_control")
        current_name = self._hidden_name("range_current")
        active_name = self._hidden_name("range_active")
        outer[control_name] = self._control_constant(0)
        outer[current_name] = start
        outer[active_name] = self._bool_constant(True)
        carried_names = [merge.name for merge in typed_statement.branch_merges]
        carried_types = [merge.type for merge in typed_statement.branch_merges]
        carried_names.extend((current_name, active_name, control_name))
        carried_types.extend((integer_type, DslType("scalar", "bool"), integer_type))
        for return_name, return_type in self._return_state_items():
            if return_name not in carried_names:
                carried_names.append(return_name)
                carried_types.append(return_type)
        for name, value_type in zip(carried_names, carried_types, strict=True):
            outer[name] = self._coerce_implicit(node, outer[name], value_type)

        results = [self._fresh() for _ in carried_names]
        before_arguments = [self._fresh() for _ in carried_names]
        operand_types = ", ".join(value_type.mlir for value_type in carried_types)
        assignments = ", ".join(
            f"{argument} = {outer[name].name}" for argument, name in zip(before_arguments, carried_names, strict=True)
        )
        self._line(f"{', '.join(results)} = scf.while ({assignments}) : ({operand_types}) -> ({operand_types}) {{")
        self.indent += 1
        self.environment = outer.copy()
        for name, value_type, argument in zip(carried_names, carried_types, before_arguments, strict=True):
            self.environment[name] = Value(argument, value_type)
        step_positive = self._fresh()
        self._line(f"{step_positive} = arith.cmpi sgt, {step.name}, {zero.name} : i32")
        forward = self._fresh()
        self._line(f"{forward} = arith.cmpi slt, {self.environment[current_name].name}, {stop.name} : i32")
        backward = self._fresh()
        self._line(f"{backward} = arith.cmpi sgt, {self.environment[current_name].name}, {stop.name} : i32")
        range_condition = self._fresh()
        self._line(f"{range_condition} = arith.select {step_positive}, {forward}, {backward} : i1")
        break_value = self._control_constant(1)
        control_active = self._fresh()
        self._line(f"{control_active} = arith.cmpi ne, {self.environment[control_name].name}, {break_value.name} : i32")
        condition = self._fresh()
        self._line(f"{condition} = arith.andi {range_condition}, {self.environment[active_name].name} : i1")
        combined = self._fresh()
        self._line(f"{combined} = arith.andi {condition}, {control_active} : i1")
        condition = combined
        if self.return_flag_name is not None:
            false_value = self._bool_constant(False)
            not_returned = self._fresh()
            self._line(
                f"{not_returned} = arith.cmpi eq, {self.environment[self.return_flag_name].name}, "
                f"{false_value.name} : i1"
            )
            combined = self._fresh()
            self._line(f"{combined} = arith.andi {condition}, {not_returned} : i1")
            condition = combined
        forwarded = ", ".join(self.environment[name].name for name in carried_names)
        self._line(f"scf.condition({condition}) {forwarded} : {operand_types}")
        self.indent -= 1
        self._line("} do {")
        self.indent += 1
        self.environment = outer.copy()
        after_arguments = [self._fresh() for _ in carried_names]
        block_arguments = ", ".join(
            f"{argument}: {value_type.mlir}"
            for argument, value_type in zip(after_arguments, carried_types, strict=True)
        )
        self._line(f"^bb0({block_arguments}):")
        for name, value_type, argument in zip(carried_names, carried_types, after_arguments, strict=True):
            self.environment[name] = Value(argument, value_type)
        induction = self._fresh()
        self._line(f"{induction} = arith.index_cast {self.environment[current_name].name} : i32 to index")
        self.environment[node.target.id] = Value(induction, DslType("index", "index"))
        self.loop_controls.append(control_name)
        self._emit_loop_body(node.body, control_name, typed_statement.loop_depth + 1)
        self.loop_controls.pop()

        next_value = self._fresh()
        self._line(f"{next_value} = arith.addi {self.environment[current_name].name}, {step.name} : i32")
        positive_overflow = self._fresh()
        self._line(f"{positive_overflow} = arith.cmpi slt, {next_value}, {self.environment[current_name].name} : i32")
        negative_overflow = self._fresh()
        self._line(f"{negative_overflow} = arith.cmpi sgt, {next_value}, {self.environment[current_name].name} : i32")
        body_step_positive = self._fresh()
        self._line(f"{body_step_positive} = arith.cmpi sgt, {step.name}, {zero.name} : i32")
        overflow = self._fresh()
        self._line(f"{overflow} = arith.select {body_step_positive}, {positive_overflow}, {negative_overflow} : i1")
        false_value = self._bool_constant(False)
        no_overflow = self._fresh()
        self._line(f"{no_overflow} = arith.cmpi eq, {overflow}, {false_value.name} : i1")
        continue_value = self._control_constant(2)
        normal_value = self._control_constant(0)
        is_continue = self._fresh()
        self._line(f"{is_continue} = arith.cmpi eq, {self.environment[control_name].name}, {continue_value.name} : i32")
        normalized_control = self._fresh()
        self._line(
            f"{normalized_control} = arith.select {is_continue}, {normal_value.name}, "
            f"{self.environment[control_name].name} : i32"
        )
        self.environment[control_name] = Value(normalized_control, integer_type)
        self.environment[current_name] = Value(next_value, integer_type)
        self.environment[active_name] = Value(no_overflow, DslType("scalar", "bool"))
        yielded = ", ".join(self.environment[name].name for name in carried_names)
        self._line(f"scf.yield {yielded} : {operand_types}")
        self.indent -= 1
        self._line("}")
        self.environment = outer
        internal_names = {current_name, active_name, control_name}
        for name, result, value_type in zip(carried_names, results, carried_types, strict=True):
            if name not in internal_names:
                self.environment[name] = Value(result, value_type)
        for name in internal_names:
            self.environment.pop(name, None)

    @staticmethod
    def _literal_integer(node: ast.expr) -> int | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
            return node.value
        if (
            isinstance(node, ast.UnaryOp)
            and isinstance(node.op, (ast.UAdd, ast.USub))
            and isinstance(node.operand, ast.Constant)
            and isinstance(node.operand.value, int)
            and not isinstance(node.operand.value, bool)
        ):
            return node.operand.value if isinstance(node.op, ast.UAdd) else -node.operand.value
        return None

    def _while(self, node: ast.While) -> None:
        if node.orelse:
            raise self.context.error(node, "while-else is not supported")
        outer = self.environment.copy()
        typed_statement = self.typed_statements[id(node)]
        control_name = self._hidden_name("loop_control")
        outer[control_name] = self._control_constant(0)
        carried_names = [merge.name for merge in typed_statement.branch_merges]
        carried_types = [merge.type for merge in typed_statement.branch_merges]
        carried_names.append(control_name)
        carried_types.append(DslType("scalar", "i32"))
        for return_name, return_type in self._return_state_items():
            if return_name not in carried_names:
                carried_names.append(return_name)
                carried_types.append(return_type)
        for name, value_type in zip(carried_names, carried_types, strict=True):
            outer[name] = self._coerce_implicit(node, outer[name], value_type)
        results = [self._fresh() for _ in carried_names]
        operand_types = ", ".join(value.mlir for value in carried_types)
        before_arguments = [self._fresh() for _ in carried_names]
        assignments = ", ".join(
            f"{argument} = {outer[name].name}" for argument, name in zip(before_arguments, carried_names, strict=True)
        )
        result_prefix = f"{', '.join(results)} = " if results else ""
        signature = f" ({assignments}) : ({operand_types}) -> ({operand_types})" if carried_names else ""
        self._line(f"{result_prefix}scf.while{signature} {{")
        self.indent += 1
        self.environment = outer.copy()
        for name, value_type, argument in zip(carried_names, carried_types, before_arguments, strict=True):
            self.environment[name] = Value(argument, value_type)
        break_value = self._control_constant(1)
        active = self._fresh()
        self._line(f"{active} = arith.cmpi ne, {self.environment[control_name].name}, {break_value.name} : i32")
        if self.return_flag_name is not None:
            false_value = self._bool_constant(False)
            not_returned = self._fresh()
            self._line(
                f"{not_returned} = arith.cmpi eq, {self.environment[self.return_flag_name].name}, "
                f"{false_value.name} : i1"
            )
            combined = self._fresh()
            self._line(f"{combined} = arith.andi {active}, {not_returned} : i1")
            active = combined
        condition_result = self._fresh()
        self._line(f"{condition_result} = scf.if {active} -> (i1) {{")
        self.indent += 1
        condition = self._expression(node.test)
        self._require_same_type(node.test, DslType("scalar", "bool"), condition.type)
        self._line(f"scf.yield {condition.name} : i1")
        self.indent -= 1
        self._line("} else {")
        self.indent += 1
        false_value = self._fresh()
        self._line(f"{false_value} = arith.constant false")
        self._line(f"scf.yield {false_value} : i1")
        self.indent -= 1
        self._line("}")
        forwarded = ", ".join(self.environment[name].name for name in carried_names)
        suffix = f" : {operand_types}" if carried_names else ""
        self._line(f"scf.condition({condition_result}) {forwarded}{suffix}")
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
            block_arguments = ", ".join(
                f"{name}: {value_type.mlir}" for name, value_type in zip(after_arguments, carried_types, strict=True)
            )
            self._line(f"^bb0({block_arguments}):")
        self.loop_controls.append(control_name)
        self._emit_loop_body(node.body, control_name, typed_statement.loop_depth + 1)
        self.loop_controls.pop()
        continue_value = self._control_constant(2)
        normal_value = self._control_constant(0)
        is_continue = self._fresh()
        self._line(f"{is_continue} = arith.cmpi eq, {self.environment[control_name].name}, {continue_value.name} : i32")
        normalized_control = self._fresh()
        self._line(
            f"{normalized_control} = arith.select {is_continue}, {normal_value.name}, "
            f"{self.environment[control_name].name} : i32"
        )
        self.environment[control_name] = Value(normalized_control, DslType("scalar", "i32"))
        yielded = ", ".join(self.environment[name].name for name in carried_names)
        self._line(f"scf.yield {yielded}{suffix}")
        self.indent -= 1
        self._line("}")
        self.environment = outer
        for name, result, result_type in zip(carried_names, results, carried_types, strict=True):
            if name != control_name:
                self.environment[name] = Value(result, result_type)
        self.environment.pop(control_name, None)

    def _reject_nested_returns(self, node: ast.If | ast.For | ast.While) -> None:
        nested = next((value for value in ast.walk(node) if isinstance(value, ast.Return)), None)
        if nested is not None:
            raise self.context.error(nested, "nested return is not supported")

    @staticmethod
    def _assigned_names(statements: Iterable[ast.stmt]) -> set[str]:
        result: set[str] = set()

        def target_names(target: ast.expr) -> set[str]:
            if isinstance(target, ast.Name):
                return {target.id}
            if isinstance(target, (ast.Tuple, ast.List)):
                return set().union(*(target_names(item) for item in target.elts))
            return set()

        for statement in statements:
            if isinstance(statement, ast.Assign):
                for target in statement.targets:
                    result.update(target_names(target))
            elif isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
                result.add(statement.target.id)
            elif isinstance(statement, ast.AugAssign) and isinstance(statement.target, ast.Name):
                result.add(statement.target.id)
        return result

    def _expression(self, node: ast.expr, expected: DslType | None = None) -> Value:
        typed = self._typed_expression(node)
        if expected is not None and not can_convert(typed.type, expected):
            raise self.context.error(
                node,
                f"typed model mismatch: cannot use {typed.type.mlir} as {expected.mlir}",
            )
        if isinstance(node, ast.Name):
            if node.id not in self.environment:
                raise self.context.error(node, f"unknown local value '{node.id}'")
            return self.environment[node.id]
        if isinstance(node, ast.Constant):
            return self._constant(node, typed.type)
        if isinstance(node, ast.BinOp):
            return self._binary(node)
        if isinstance(node, ast.UnaryOp):
            return self._unary(node)
        if isinstance(node, ast.BoolOp):
            return self._bool_op(node)
        if isinstance(node, ast.Compare):
            return self._compare(node)
        if isinstance(node, ast.Tuple):
            return self._tuple(node)
        if isinstance(node, ast.Call):
            return self._call(node)
        if isinstance(node, ast.Attribute):
            return self._attribute(node)
        if isinstance(node, ast.Subscript):
            return self._index(node)
        if isinstance(node, ast.IfExp):
            return self._conditional_expression(node)
        raise self.context.error(node, f"unsupported expression syntax: {type(node).__name__}")

    def _conditional_expression(self, node: ast.IfExp) -> Value:
        typed = self._typed_expression(node)
        condition = self._expression(node.test, DslType("scalar", "bool"))
        self._require_same_type(node.test, DslType("scalar", "bool"), condition.type)
        result = self._fresh()
        self._line(f"{result} = scf.if {condition.name} -> ({typed.type.mlir}) {{")
        self.indent += 1
        then_value = self._coerce_implicit(node.body, self._expression(node.body, typed.type), typed.type)
        self._line(f"scf.yield {then_value.name} : {typed.type.mlir}")
        self.indent -= 1
        self._line("} else {")
        self.indent += 1
        else_value = self._coerce_implicit(node.orelse, self._expression(node.orelse, typed.type), typed.type)
        self._line(f"scf.yield {else_value.name} : {typed.type.mlir}")
        self.indent -= 1
        self._line("}")
        return Value(result, typed.type)

    def _bool_op(self, node: ast.BoolOp) -> Value:
        result_type = self._typed_expression(node).type
        self._require_same_type(node, DslType("scalar", "bool"), result_type)
        current = self._expression(node.values[0], result_type)
        self._require_same_type(node.values[0], result_type, current.type)
        for source in node.values[1:]:
            result = self._fresh()
            self._line(f"{result} = scf.if {current.name} -> ({result_type.mlir}) {{")
            self.indent += 1
            if isinstance(node.op, ast.And):
                alternative = self._expression(source, result_type)
                self._line(f"scf.yield {alternative.name} : {result_type.mlir}")
            else:
                self._line(f"scf.yield {current.name} : {result_type.mlir}")
            self.indent -= 1
            self._line("} else {")
            self.indent += 1
            if isinstance(node.op, ast.And):
                self._line(f"scf.yield {current.name} : {result_type.mlir}")
            else:
                alternative = self._expression(source, result_type)
                self._line(f"scf.yield {alternative.name} : {result_type.mlir}")
            self.indent -= 1
            self._line("}")
            current = Value(result, result_type)
        return current

    def _tuple(self, node: ast.Tuple) -> Value:
        return self._tuple_values(node, node.elts)

    def _tuple_values(self, node: ast.expr, sources: list[ast.expr]) -> Value:
        result_type = self._typed_expression(node).type
        elements = [element for element in result_type.arguments if isinstance(element, DslType)]
        values = [
            self._coerce_implicit(source, self._expression(source, expected), expected)
            for source, expected in zip(sources, elements, strict=True)
        ]
        result = self._fresh()
        self._line(
            f'{result} = "vernon.tuple_create"({", ".join(value.name for value in values)}) : '
            f"({', '.join(value.type.mlir for value in values)}) -> {result_type.mlir}"
        )
        return Value(result, result_type, tuple(values))

    def _constant(self, node: ast.Constant, expected: DslType | None) -> Value:
        if isinstance(node.value, bool):
            value_type = DslType("scalar", "bool")
            literal = "1" if node.value else "0"
        elif isinstance(node.value, int):
            value_type = (
                expected
                if expected and expected.kind == "scalar" and (expected.is_float or expected.name in {"i32", "u32"})
                else DslType("scalar", "i32")
            )
            literal = f"{node.value}.0" if value_type.is_float else str(node.value)
        elif isinstance(node.value, float):
            value_type = (
                expected if expected and expected.kind == "scalar" and expected.is_float else DslType("scalar", "f32")
            )
            literal = f"{node.value:.17g}"
            if "." not in literal and "e" not in literal.lower():
                literal += ".0"
        else:
            raise self.context.error(node, "only bool, integer, and float constants are supported")
        result = self._fresh()
        self._line(f"{result} = arith.constant {literal} : {value_type.mlir}")
        return Value(result, value_type)

    def _binary(self, node: ast.BinOp) -> Value:
        result_type = self._typed_expression(node).type
        left = self._coerce_numeric(node.left, self._expression(node.left), result_type)
        right = self._coerce_numeric(node.right, self._expression(node.right), result_type)
        floating = left.type.is_float
        if isinstance(node.op, ast.Pow):
            if not floating:
                raise self.context.error(node, "power requires floating-point operands")
            return self._intrinsic(node, "pow", [left, right], result_type)
        operations = {
            ast.Add: "arith.addf" if floating else "arith.addi",
            ast.Sub: "arith.subf" if floating else "arith.subi",
            ast.Mult: "arith.mulf" if floating else "arith.muli",
            ast.Div: "arith.divf" if floating else ("arith.divui" if left.type.name == "u32" else "arith.divsi"),
            ast.Mod: "arith.remf" if floating else ("arith.remui" if left.type.name == "u32" else "arith.remsi"),
        }
        operation = operations.get(type(node.op))
        if operation is None or not (floating or left.type.is_integer):
            raise self.context.error(node, f"unsupported binary operation for {left.type.name}")
        result = self._fresh()
        self._line(f"{result} = {operation} {left.name}, {right.name} : {left.type.mlir}")
        return Value(result, result_type)

    @staticmethod
    def _element_type(value_type: DslType) -> DslType | None:
        if value_type.kind == "scalar":
            return value_type
        if value_type.kind == "tensor":
            element = value_type.arguments[0]
            assert isinstance(element, DslType)
            return element
        return None

    def _splat(self, node: ast.AST, value: Value, tensor_type: DslType) -> Value:
        element = tensor_type.arguments[0]
        assert isinstance(element, DslType)
        self._require_same_type(node, element, value.type)
        result = self._fresh()
        self._line(f"{result} = tensor.splat {value.name} : {tensor_type.mlir}")
        return Value(result, tensor_type)

    def _unary(self, node: ast.UnaryOp) -> Value:
        operand = self._expression(node.operand)
        result_type = self._typed_expression(node).type
        result = self._fresh()
        if isinstance(node.op, ast.Not):
            self._require_same_type(node, DslType("scalar", "bool"), operand.type)
            self._line(f"{result} = arith.xori {operand.name}, true : i1")
            return Value(result, result_type)
        if isinstance(node.op, ast.USub) and (operand.type.is_float or operand.type.is_integer):
            operation = "arith.negf" if operand.type.is_float else "arith.subi"
            if operand.type.is_float:
                self._line(f"{result} = {operation} {operand.name} : {operand.type.mlir}")
            else:
                zero = self._fresh()
                self._line(f"{zero} = arith.constant 0 : {operand.type.mlir}")
                self._line(f"{result} = {operation} {zero}, {operand.name} : {operand.type.mlir}")
            return Value(result, result_type)
        if isinstance(node.op, ast.UAdd):
            return Value(operand.name, result_type, operand.fields, operand.access)
        raise self.context.error(node, "unsupported unary operation")

    def _compare(self, node: ast.Compare) -> Value:
        if len(node.ops) != 1:
            raise self.context.error(node, "chained comparisons are not supported")
        typed = self._typed_expression(node)
        if len(typed.operand_types) != 2:
            raise self.context.error(node, "internal error: comparison has no typed operands")
        common = typed.operand_types[0]
        left = self._expression(node.left)
        right = self._expression(node.comparators[0])
        left = self._coerce_numeric(node.left, left, common)
        right = self._coerce_numeric(node.comparators[0], right, common)
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
            raise self.context.error(node, f"unsupported comparison: {type(node.ops[0]).__name__}")
        floating_predicate, integer_predicate = predicates_for_operation
        if left.type.is_float:
            operation, predicate = "arith.cmpf", floating_predicate
        elif left.type.is_integer or left.type.name == "bool":
            operation, predicate = "arith.cmpi", integer_predicate
        else:
            raise self.context.error(node, "comparison requires scalar or tensor numeric operands")
        result = self._fresh()
        self._line(f"{result} = {operation} {predicate}, {left.name}, {right.name} : {left.type.mlir}")
        return Value(result, typed.type)

    def _call(self, node: ast.Call) -> Value:
        if node.keywords:
            raise self.context.error(node, "function calls do not support keyword arguments")
        typed_call = self._typed_expression(node)
        name = typed_call.operation or ""
        if name in {"Tensor", "Vector", "Matrix"}:
            return self._aggregate_constructor(node, name)
        if name == "Tuple":
            return self._tuple_values(node, node.args)
        known_signature = self.context.signatures.get(name)
        struct_fields = self.context.structs.get(name)
        if struct_fields is not None and len(node.args) == len(struct_fields):
            arguments = [
                self._expression(argument, field.type)
                for argument, (_, field) in zip(node.args, struct_fields, strict=True)
            ]
        elif known_signature is not None and len(node.args) == len(known_signature.arguments):
            arguments = [
                self._expression(argument, expected)
                for argument, expected in zip(node.args, known_signature.arguments, strict=True)
            ]
        else:
            arguments = [self._expression(argument) for argument in node.args]
        if (
            isinstance(node.func, ast.Attribute)
            and name in INTRINSIC_METHODS
            and (not isinstance(node.func.value, ast.Name) or node.func.value.id in self.environment)
        ):
            arguments.insert(0, self._expression(node.func.value))
        scalar_casts = {
            "int": DslType("scalar", "i32"),
            "i32": DslType("scalar", "i32"),
            "u32": DslType("scalar", "u32"),
            "float": DslType("scalar", "f32"),
            "f16": DslType("scalar", "f16"),
            "f32": DslType("scalar", "f32"),
            "f64": DslType("scalar", "f64"),
        }
        if name in scalar_casts:
            if len(arguments) != 1 or arguments[0].type.kind not in {
                "scalar",
                "index",
            }:
                raise self.context.error(node, f"{name} requires one scalar argument")
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
                raise self.context.error(node, f"{name} requires one floating-point argument")
            result = self._fresh()
            self._line(f"{result} = {math_operations[name]} {arguments[0].name} : {arguments[0].type.mlir}")
            return Value(result, typed_call.type)
        if name in {
            "resolution",
            "fragment_coord",
            "front_facing",
            "vertex_id",
            "instance_id",
        }:
            if arguments:
                raise self.context.error(node, f"{name} does not accept arguments")
            value = self.generated_values.get(name)
            if value is None:
                raise self.context.error(node, f"{name}() is not available in this shader stage")
            return value
        if name in {"normalize", "reflect"}:
            expected = 1 if name == "normalize" else 2
            if len(arguments) != expected or any(
                argument.type.kind != "tensor" or not argument.type.is_float for argument in arguments
            ):
                raise self.context.error(node, f"{name} requires {expected} floating-point vector argument(s)")
            for argument in arguments[1:]:
                self._require_same_type(node, arguments[0].type, argument.type)
            return self._intrinsic(node, name, arguments, typed_call.type)
        if name in {"dot", "cross"}:
            if len(arguments) != 2:
                raise self.context.error(node, f"{name} requires two vector arguments")
            self._require_same_type(node, arguments[0].type, arguments[1].type)
            vector = arguments[0].type
            if vector.kind != "tensor" or len(vector.arguments) != 2 or not vector.is_float:
                raise self.context.error(node, f"{name} requires floating-point vectors")
            if name == "cross" and vector.arguments[1] != 3:
                raise self.context.error(node, "cross requires three-component vectors")
            element = vector.arguments[0]
            assert isinstance(element, DslType)
            return self._intrinsic(node, name, arguments, typed_call.type)
        if name == "norm":
            if len(arguments) != 1 or arguments[0].type.kind != "tensor" or not arguments[0].type.is_float:
                raise self.context.error(node, "norm requires one floating-point Tensor argument")
            element = arguments[0].type.arguments[0]
            assert isinstance(element, DslType)
            squared = self._intrinsic(node, "dot", [arguments[0], arguments[0]], element)
            result = self._fresh()
            self._line(f"{result} = math.sqrt {squared.name} : {element.mlir}")
            return Value(result, typed_call.type)
        if name in {"min", "max", "pow"}:
            if len(arguments) != 2:
                raise self.context.error(node, f"{name} requires two arguments")
            result_type = self._typed_expression(node).type
            arguments = [
                self._coerce_numeric(source, argument, result_type)
                for source, argument in zip(node.args, arguments, strict=True)
            ]
            if not arguments[0].type.is_float:
                raise self.context.error(node, f"{name} currently requires floating-point arguments")
            return self._intrinsic(node, name, arguments, result_type)
        if name == "clamp":
            if len(arguments) != 3:
                raise self.context.error(node, "clamp requires value, minimum, and maximum")
            result_type = self._typed_expression(node).type
            arguments = [
                self._coerce_numeric(source, argument, result_type)
                for source, argument in zip(node.args, arguments, strict=True)
            ]
            return self._intrinsic(node, name, arguments, result_type)
        if name == "matmul":
            if len(arguments) != 2:
                raise self.context.error(node, "matmul requires two arguments")
            return self._intrinsic(node, name, arguments, self._typed_expression(node).type)
        if name == "texture_sample":
            if self.node.name in self.context.shared_functions:
                raise self.context.error(
                    node, f"shared function '{self.node.name}' uses device-only operation 'texture_sample'"
                )
            sampling = texture_sampling_contract(tuple(argument.type.kind for argument in arguments))
            if sampling is None:
                if len(arguments) == 4:
                    raise self.context.error(node, "the four-argument texture_sample form requires an explicit sampler")
                raise self.context.error(
                    node, "texture_sample requires texture, optional sampler, coordinates, and optional lod"
                )
            texture = arguments[0]
            texture_source = node.args[0].id if isinstance(node.args[0], ast.Name) else None
            planned_mode = (
                self.interface_plan.texture_sampler_modes.get(texture_source) if texture_source is not None else None
            )
            if planned_mode is not None and planned_mode != sampling.sampler_mode:
                raise self.context.error(node, "texture sampling overload differs from its interface plan")
            if self.stage not in sampling.stages:
                requirement = "fragment shaders" if not sampling.has_lod else "graphics stages"
                raise self.context.error(
                    node,
                    f"texture_sample {'without' if not sampling.has_lod else 'with'} "
                    f"lod is supported only in {requirement}",
                )
            if sampling.explicit_sampler:
                sampler = arguments[1]
                coordinates = arguments[2]
                lod = arguments[3] if sampling.has_lod else None
            else:
                sampler = self.implicit_samplers.get(texture.name)
                if sampler is None:
                    raise self.context.error(
                        node.args[0], "implicitly sampled texture must be a texture entry parameter"
                    )
                coordinates = arguments[1]
                lod = arguments[2] if sampling.has_lod else None
            if lod is not None:
                if lod.type.kind != "scalar" or not lod.type.is_float:
                    raise self.context.error(node.args[-1], "texture_sample lod must be a floating-point scalar")
            element = texture.type.arguments[1]
            assert isinstance(element, DslType)
            dimension = texture.type.arguments[0]
            coordinate_rank = {"2d": 2, "3d": 3, "cube": 3}[dimension]
            if (
                coordinates.type.kind != "tensor"
                or coordinates.type.arguments != (element, coordinate_rank)
                or not coordinates.type.is_float
            ):
                raise self.context.error(
                    node.args[2] if sampling.explicit_sampler else node.args[1],
                    f"texture_sample coordinates for a {dimension} texture "
                    f"must be a {coordinate_rank}-component floating-point vector",
                )
            operands = [texture, sampler, coordinates]
            if lod is not None:
                operands.append(lod)
            return self._intrinsic(node, name, operands, typed_call.type)
        if name == "texture_size":
            if self.node.name in self.context.shared_functions:
                raise self.context.error(
                    node, f"shared function '{self.node.name}' uses device-only operation 'texture_size'"
                )
            if len(arguments) not in {1, 2} or arguments[0].type.kind != "texture":
                raise self.context.error(node, "texture_size requires texture and optional lod")
            if self.stage not in {"vertex", "fragment"}:
                raise self.context.error(node, "texture_size is supported only in graphics stages")
            if len(arguments) == 2 and (arguments[1].type.kind != "scalar" or not arguments[1].type.is_integer):
                raise self.context.error(node.args[1], "texture_size lod must be an integer scalar")
            return self._intrinsic(node, name, arguments, typed_call.type)
        if name in self.context.structs:
            fields = self.context.structs[name]
            if len(arguments) != len(fields):
                raise self.context.error(node, f"{name} constructor requires {len(fields)} arguments")
            for argument, (_, annotation) in zip(arguments, fields, strict=True):
                self._require_same_type(node, annotation.type, argument.type)
            result_type = typed_call.type
            # Entry-point structs are interface aggregates flattened by the
            # frontend. Materializing them would leave an otherwise dead
            # custom struct operation for SPIR-V lowering.
            if self.stage is not None and self.signature.result == result_type:
                return Value("", result_type, tuple(arguments))
            result = self._fresh()
            self._line(
                f'{result} = "vernon.struct_create"({", ".join(argument.name for argument in arguments)}) '
                f'{{type_name = "{name}"}} : '
                f"({', '.join(argument.type.mlir for argument in arguments)}) -> {result_type.mlir}"
            )
            # Retain the aggregate fields so entry-point struct results can
            # still be flattened without struct extraction operations.
            return Value(result, result_type, tuple(arguments))
        if name not in self.context.signatures:
            raise self.context.error(node, f"unknown DSL function '{name}'")
        if self.node.name in self.context.shared_functions and name not in self.context.shared_functions:
            raise self.context.error(
                node, f"shared function '{self.node.name}' cannot call device-only function '{name}'"
            )
        signature = self.context.signatures[name]
        if len(arguments) != len(signature.arguments):
            raise self.context.error(node, f"function '{name}' expects {len(signature.arguments)} arguments")
        arguments = [
            self._coerce_implicit(source, argument, expected)
            for source, argument, expected in zip(node.args, arguments, signature.arguments, strict=True)
        ]
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

    def _aggregate_constructor(self, node: ast.Call, name: str) -> Value:
        if len(node.args) != 1:
            raise self.context.error(node, f"{name} requires one sequence literal")
        if name == "Vector":
            sequence = node.args[0]
            if not isinstance(sequence, (ast.List, ast.Tuple)) or not sequence.elts:
                raise self.context.error(node, "Vector requires a non-empty sequence literal")
            result_type = self._typed_expression(node).type
            element_type = self._element_type(result_type)
            values = [
                self._coerce_constructor_argument(value, self._expression(value), element_type)
                for value in sequence.elts
            ]
            return self._intrinsic(node, "construct", values, result_type)
        literal = rectangular_literal(node.args[0])
        if literal is None:
            raise self.context.error(node, f"{name} requires a non-empty rectangular sequence literal")
        elements, shape = literal
        expected_rank = {"Vector": 1, "Matrix": 2}.get(name)
        if expected_rank is not None and len(shape) != expected_rank:
            raise self.context.error(node, f"{name} requires a rank-{expected_rank} sequence literal")
        values = [self._expression(value) for value in elements]
        result_type = self._typed_expression(node).type
        element_type = self._element_type(result_type)
        values = [
            self._coerce_implicit(source, value, element_type) for source, value in zip(elements, values, strict=True)
        ]
        return self._intrinsic(node, "construct", values, result_type)

    def _coerce_constructor_argument(self, node: ast.AST, value: Value, element: DslType) -> Value:
        if value.type.kind == "tensor":
            return self._coerce_implicit(
                node,
                value,
                DslType("tensor", "Tensor", (element, *value.type.arguments[1:])),
            )
        return self._coerce_implicit(node, value, element)

    def _cast(self, node: ast.AST, value: Value, target: DslType) -> Value:
        if value.type == target:
            return value
        source_scalar = self._element_type(value.type)
        target_scalar = self._element_type(target)
        if value.type.kind == "index":
            operation = "arith.index_cast"
        elif source_scalar.is_integer and target_scalar.is_float:
            operation = "arith.uitofp" if source_scalar.name == "u32" else "arith.sitofp"
        elif source_scalar.is_float and target_scalar.is_integer:
            operation = "arith.fptoui" if target_scalar.name == "u32" else "arith.fptosi"
        elif source_scalar.is_float and target_scalar.is_float:
            source_width = int(source_scalar.name[1:])
            target_width = int(target_scalar.name[1:])
            operation = "arith.extf" if source_width < target_width else "arith.truncf"
        elif source_scalar.is_integer and target_scalar.is_integer:
            operation = "arith.bitcast"
        else:
            raise self.context.error(node, f"cannot convert {value.type.mlir} to {target.mlir}")
        result = self._fresh()
        self._line(f"{result} = {operation} {value.name} : {value.type.mlir} to {target.mlir}")
        return Value(result, target)

    @staticmethod
    def _element_type(value_type: DslType) -> DslType:
        if value_type.kind == "tensor":
            element = value_type.arguments[0]
            assert isinstance(element, DslType)
            return element
        return value_type

    def _coerce_numeric(self, node: ast.AST, value: Value, target: DslType) -> Value:
        if value.type == target:
            return value
        target_element = self._element_type(target)
        if value.type.kind == "scalar" and target.kind == "tensor":
            value = self._coerce_implicit(node, value, target_element)
            return self._splat(node, value, target)
        return self._coerce_implicit(node, value, target)

    def _coerce_implicit(self, node: ast.AST, value: Value, target: DslType) -> Value:
        if value.type == target:
            return value
        if can_convert(value.type, target):
            return self._cast(node, value, target)
        raise self.context.error(node, f"unsafe implicit conversion from {value.type.mlir} to {target.mlir}")

    def _intrinsic(self, node: ast.AST, name: str, arguments: list[Value], result_type: DslType) -> Value:
        result = self._fresh()
        operand_types = ", ".join(argument.type.mlir for argument in arguments)
        self._line(
            f'{result} = "vernon.intrinsic"({", ".join(argument.name for argument in arguments)}) '
            f'{{name = "{name}"}} : ({operand_types}) -> {result_type.mlir}'
        )
        return Value(result, result_type)

    def _attribute(self, node: ast.Attribute) -> Value:
        value = self._expression(node.value)
        if value.type.kind == "struct":
            fields = self.context.structs[value.type.name]
            field_names = [name for name, _ in fields]
            if node.attr not in field_names:
                raise self.context.error(node, f"struct '{value.type.name}' has no field '{node.attr}'")
            field_index = field_names.index(node.attr)
            if value.fields is not None:
                return value.fields[field_index]
            result_type = fields[field_index][1].type
            result = self._fresh()
            self._line(
                f'{result} = "vernon.struct_get"({value.name}) '
                f'{{field = "{node.attr}", index = {field_index} : i64}} : '
                f"({value.type.mlir}) -> {result_type.mlir}"
            )
            return Value(result, result_type)
        if value.type.kind == "tensor" and node.attr and set(node.attr) <= _SWIZZLES:
            shape = value.type.arguments[1:]
            if len(shape) != 1:
                raise self.context.error(node, "swizzle requires a rank-1 tensor")
            indices = [
                "xyzw".find(character) if character in "xyzw" else "rgba".find(character) for character in node.attr
            ]
            if any(index >= int(shape[0]) for index in indices):
                raise self.context.error(node, f"swizzle '{node.attr}' is out of bounds")
            canonical_mask = "".join("xyzw"[index] for index in indices)
            element = value.type.arguments[0]
            assert isinstance(element, DslType)
            result_type = element if len(indices) == 1 else DslType("tensor", "Tensor", (element, len(indices)))
            result = self._fresh()
            self._line(
                f'{result} = "vernon.swizzle"({value.name}) {{mask = "{canonical_mask}"}} : '
                f"({value.type.mlir}) -> {result_type.mlir}"
            )
            return Value(result, result_type)
        raise self.context.error(node, f"type '{value.type.name}' has no supported attribute '{node.attr}'")

    def _index(self, node: ast.Subscript) -> Value:
        value = self._expression(node.value)
        if value.type.kind == "tuple":
            assert isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, int)
            index = node.slice.value
            if index < 0:
                index += len(value.type.arguments)
            fields = self._tuple_fields(node, value)
            return fields[index]
        if value.type.kind == "tensor_view":
            index = self._tensor_view_index(node, value)
            element = value.type.arguments[0]
            assert isinstance(element, DslType)
            result = self._fresh()
            self._line(
                f'{result} = "vernon.intrinsic"({value.name}, {index.name}) '
                f'{{name = "tensor_view_load"}} : ({value.abi_type.mlir}, index) -> {element.mlir}'
            )
            return Value(result, element)
        if value.type.kind != "tensor":
            raise self.context.error(node, "indexing requires a Tensor or TensorView value")
        indices = list(node.slice.elts) if isinstance(node.slice, ast.Tuple) else [node.slice]
        if len(indices) != len(value.type.arguments) - 1:
            raise self.context.error(node, "tensor.extract requires one index per tensor dimension")
        index_values: list[str] = []
        for index_node in indices:
            index = self._expression(index_node)
            if not index.type.is_integer:
                raise self.context.error(index_node, "tensor index must be an integer")
            if index.type.mlir == "index":
                index_values.append(index.name)
            else:
                cast = self._fresh()
                self._line(f"{cast} = arith.index_cast {index.name} : {index.type.mlir} to index")
                index_values.append(cast)
        element = value.type.arguments[0]
        assert isinstance(element, DslType)
        result_type = element
        result = self._fresh()
        if element.kind == "scalar":
            self._line(f"{result} = tensor.extract {value.name}[{', '.join(index_values)}] : {value.type.mlir}")
        else:
            operands = ", ".join((value.name, *index_values))
            operand_types = ", ".join((value.type.mlir, *(["index"] * len(index_values))))
            self._line(f'{result} = "vernon.tensor_get"({operands}) : ({operand_types}) -> {result_type.mlir}')
        return Value(result, result_type)

    def _tuple_fields(self, node: ast.AST, value: Value) -> tuple[Value, ...]:
        if value.fields is not None:
            return value.fields
        if value.type.kind != "tuple":
            raise self.context.error(node, "internal error: expected a Tuple value")
        element_types = [element for element in value.type.arguments if isinstance(element, DslType)]
        fields: list[Value] = []
        for index, element in enumerate(element_types):
            result = self._fresh()
            self._line(
                f'{result} = "vernon.tuple_get"({value.name}) {{index = {index} : i64}} : '
                f"({value.type.mlir}) -> {element.mlir}"
            )
            fields.append(Value(result, element))
        return tuple(fields)

    def _buffer_index(self, node: ast.AST) -> Value:
        if isinstance(node, ast.Tuple):
            raise self.context.error(node, "buffers require exactly one index")
        index = self._expression(node)
        if not index.type.is_integer:
            raise self.context.error(node, "buffer index must be an integer")
        if index.type.mlir == "index":
            return index
        cast = self._fresh()
        self._line(f"{cast} = arith.index_cast {index.name} : {index.type.mlir} to index")
        return Value(cast, DslType("index", "index"))

    def _tensor_view_index(self, node: ast.Subscript, value: Value) -> Value:
        rank = value.type.arguments[1]
        index_nodes = list(node.slice.elts) if isinstance(node.slice, ast.Tuple) else [node.slice]
        if len(index_nodes) != rank:
            raise self.context.error(node, "TensorView indexing requires one index per dimension")
        if value.view_layout is None:
            if rank == 1:
                return self._buffer_index(index_nodes[0])
            raise self.context.error(
                node,
                "multi-dimensional TensorView lowering requires runtime stride descriptors and is not implemented",
            )
        layout = value.view_layout
        if len(layout.shape) != rank or len(layout.strides) != rank:
            raise self.context.error(node, "runtime TensorView layout rank does not match its annotation")

        physical: Value | None = None
        if layout.offset:
            offset = self._fresh()
            self._line(f"{offset} = arith.constant {layout.offset} : index")
            physical = Value(offset, DslType("index", "index"))
        for index_node, stride in zip(index_nodes, layout.strides, strict=True):
            term = self._buffer_index(index_node)
            if stride != 1:
                stride_value = self._fresh()
                self._line(f"{stride_value} = arith.constant {stride} : index")
                multiplied = self._fresh()
                self._line(f"{multiplied} = arith.muli {term.name}, {stride_value} : index")
                term = Value(multiplied, DslType("index", "index"))
            if physical is None:
                physical = term
            else:
                added = self._fresh()
                self._line(f"{added} = arith.addi {physical.name}, {term.name} : index")
                physical = Value(added, DslType("index", "index"))
        assert physical is not None
        return physical

    def _store_index(self, target: ast.Subscript, value: Value) -> None:
        buffer = self._expression(target.value)
        if buffer.type.kind != "tensor_view":
            raise self.context.error(target, "indexed assignment is supported only for TensorView values")
        element = buffer.type.arguments[0]
        assert isinstance(element, DslType)
        if buffer.access is AccessMode.READ:
            raise self.context.error(target, "cannot assign through a read-only TensorView")
        value = self._coerce_implicit(target, value, element)
        index = self._tensor_view_index(target, buffer)
        self._line(
            f'"vernon.intrinsic"({buffer.name}, {index.name}, {value.name}) '
            f'{{name = "tensor_view_store"}} : ({buffer.abi_type.mlir}, index, {element.mlir}) -> ()'
        )

    def _require_same_type(self, node: ast.AST, expected: DslType, actual: DslType) -> None:
        if expected != actual:
            raise self.context.error(node, f"type mismatch: expected {expected.mlir}, got {actual.mlir}")


def _specialize_frontend_source(source: str, request: FrontendCompileRequest) -> str:
    tree = ast.parse(source, filename=str(request.source_path))
    function = next(
        (node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == request.entry),
        None,
    )
    if function is None:
        raise CompileError(
            f"entry function '{request.entry}' was not found",
            SourceLocation(str(request.source_path), 1, 1),
        )
    shapes = {name: shape for name, _, shape in request.tensor_shapes}
    for argument in function.args.args:
        shape = shapes.get(argument.arg)
        if shape is None or argument.annotation is None:
            continue
        annotation = argument.annotation
        if isinstance(annotation, ast.Subscript) and (_name(annotation.value) or "").split(".")[-1] == "Annotated":
            values = list(annotation.slice.elts) if isinstance(annotation.slice, ast.Tuple) else [annotation.slice]
            annotation = values[0]
        if not (isinstance(annotation, ast.Subscript) and (_name(annotation.value) or "").split(".")[-1] == "Tensor"):
            continue
        items = list(annotation.slice.elts) if isinstance(annotation.slice, ast.Tuple) else [annotation.slice]
        if len(items) < 2:
            continue
        shape_node = items[1]
        shape_nodes = list(shape_node.elts) if isinstance(shape_node, ast.Tuple) else items[1:]
        if len(shape_nodes) != len(shape):
            raise CompileError(
                f"Tensor argument '{argument.arg}' rank does not match annotation",
                SourceLocation(str(request.source_path), argument.lineno, argument.col_offset + 1),
            )
        for index, (declared, concrete) in enumerate(zip(shape_nodes, shape, strict=True)):
            if isinstance(declared, ast.Constant) and declared.value is None:
                shape_nodes[index] = ast.copy_location(ast.Constant(value=concrete), declared)
            elif not (isinstance(declared, ast.Constant) and declared.value == concrete):
                raise CompileError(
                    f"Tensor argument '{argument.arg}' shape does not match annotation",
                    SourceLocation(str(request.source_path), argument.lineno, argument.col_offset + 1),
                )
        items[1] = ast.Tuple(elts=shape_nodes, ctx=ast.Load())
        annotation.slice = ast.Tuple(elts=items, ctx=ast.Load())

    constants = dict(request.captured_constants)

    class ConstantSpecializer(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.expr:
            if isinstance(node.ctx, ast.Load) and node.id in constants:
                return ast.copy_location(ast.Constant(constants[node.id]), node)
            return node

    specialized = ConstantSpecializer().visit(tree)
    assert isinstance(specialized, ast.Module)
    ast.fix_missing_locations(specialized)
    return ast.unparse(specialized)


class Compiler:
    """Compiles a restricted Python source string without importing or executing it."""

    def compile(
        self,
        source: str,
        filename: str = "<string>",
        dependencies: tuple[tuple[str, str], ...] = (),
        declared_features: tuple[str, ...] = (),
        enabled_features: tuple[str, ...] = (),
        runtime_entry: str | None = None,
        tensor_view_layouts: dict[str, _ViewLayout] | None = None,
    ) -> str:
        try:
            module = ast.parse(source, filename=filename, type_comments=False)
        except SyntaxError as error:
            raise CompileError(
                error.msg,
                SourceLocation(filename, error.lineno or 1, error.offset or 1),
            ) from None
        module = normalize_struct_methods(module, filename)
        context = _ModuleContext(
            filename,
            runtime_entry=runtime_entry,
            tensor_view_layouts=tensor_view_layouts or {},
        )
        self._collect_struct_names(module, context)
        type_parser = TypeParser(context)
        self._collect_structs(module, context, type_parser)
        self._validate_entry_annotations(module, context)
        module = infer_and_monomorphize_helpers(
            module,
            lambda annotation: type_parser.parse(annotation).type,
            context.error,
            tuple(sorted(enabled_features)),
        )
        self._helper_specializations = tuple(getattr(module, "_vernon_helper_specializations", ()))
        self._typed_functions = tuple(getattr(module, "_vernon_typed_functions", ()))
        context.typed_functions = {function.symbol: function for function in self._typed_functions}
        self._collect_signatures(module, context, type_parser)

        module_attributes = [
            'vernon.frontend = "python"',
            f"vernon.frontend_version = {FRONTEND_VERSION} : i64",
            "vernon.value_abi_version = 1 : i64",
        ]
        if dependencies:
            encoded = ", ".join(json.dumps(f"{path}={digest}") for path, digest in dependencies)
            module_attributes.append(f"vernon.source_dependencies = [{encoded}]")
        if declared_features:
            declarations = ", ".join(json.dumps(name) for name in sorted(declared_features))
            module_attributes.append(f"vernon.feature_declarations = [{declarations}]")
        if enabled_features:
            variant = ", ".join(json.dumps(name) for name in sorted(enabled_features))
            module_attributes.append(f"vernon.variant_key = [{variant}]")
        body: list[str] = [f"module attributes {{{', '.join(module_attributes)}}} {{"]

        def struct_field_types(name: str) -> tuple[ConcreteType, ...]:
            return tuple(annotation.type for _, annotation in context.structs[name])

        for name in sorted(context.structs):
            fields = context.structs[name]
            field_text = ", ".join(f'"{field_name}:{annotation.type.mlir}"' for field_name, annotation in fields)
            layout = value_abi_layout(ConcreteType("struct", name), struct_field_types)
            offsets = ", ".join(str(offset) for offset in layout.field_offsets)
            body.append(
                f'  "vernon.struct"() {{abi_alignment = {layout.alignment} : i64, '
                f"abi_field_offsets = array<i64: {offsets}>, abi_size = {layout.size} : i64, "
                f'fields = [{field_text}], sym_name = "{name}"}} : () -> ()'
            )
        for node in module.body:
            if isinstance(node, ast.FunctionDef):
                stage, workgroup_size = self._decorator(node, context)
                annotations = [type_parser.parse(argument.annotation) for argument in node.args.args]
                body.extend(
                    _FunctionEmitter(
                        context,
                        context.signatures[node.name],
                        annotations,
                        context.result_annotations[node.name],
                        context.typed_functions[node.name],
                        stage,
                        workgroup_size,
                    ).emit()
                )
            elif isinstance(node, (ast.Import, ast.ImportFrom, ast.ClassDef)):
                continue
            elif (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                continue
            else:
                raise context.error(node, f"unsupported module-level syntax: {type(node).__name__}")
        body.append("}")
        return "\n".join(body) + "\n"

    def compile_file(self, input_path: str | Path, *, features: Iterable[str] = (), entry: str | None = None) -> str:
        path = Path(input_path)
        enabled_features = tuple(sorted(set(features)))
        if entry is None:
            project = load_project(path, enabled_features, entry)
            return self.compile(project.source, str(path), project.dependencies, project.features, enabled_features)
        return self.compile_request(FrontendCompileRequest(path, entry, enabled_features)).mlir

    def compile_request(self, request: FrontendCompileRequest) -> FrontendCompileResult:
        project = load_project(request.source_path, request.enabled_features, request.entry)
        specialized_source = _specialize_frontend_source(project.source, request)
        mlir = self.compile(
            specialized_source,
            str(request.source_path),
            project.dependencies,
            project.features,
            request.enabled_features,
            request.entry,
            {
                name: _ViewLayout(shape, strides, offset)
                for name, _, shape, strides, offset in request.tensor_view_layouts
            },
        )
        return FrontendCompileResult(
            mlir,
            specialized_source,
            project.dependencies,
            project.features,
            request,
            self._helper_specializations,
            self._typed_functions,
        )

    @staticmethod
    def _validate_entry_annotations(module: ast.Module, context: _ModuleContext) -> None:
        for node in module.body:
            if not isinstance(node, ast.FunctionDef):
                continue
            decorators = {
                (_name(item.func if isinstance(item, ast.Call) else item) or "").split(".")[-1]
                for item in node.decorator_list
            }
            if not decorators & {"kernel", "vertex", "fragment"}:
                continue
            if node.returns is None:
                raise context.error(
                    node,
                    f"entry function '{node.name}' requires a result annotation",
                )
            for argument in node.args.args:
                if argument.annotation is None:
                    raise context.error(
                        argument,
                        f"entry argument '{argument.arg}' requires a type annotation",
                    )

    @staticmethod
    def _collect_struct_names(module: ast.Module, context: _ModuleContext) -> None:
        for node in module.body:
            if isinstance(node, ast.ClassDef):
                decorators = {
                    (_name(decorator.func) if isinstance(decorator, ast.Call) else _name(decorator) or "").split(".")[
                        -1
                    ]
                    for decorator in node.decorator_list
                }
                if "struct" not in decorators:
                    raise context.error(node, "DSL classes must use @struct")
                context.structs[node.name] = ()

    @staticmethod
    def _collect_structs(module: ast.Module, context: _ModuleContext, parser: TypeParser) -> None:
        for node in module.body:
            if not isinstance(node, ast.ClassDef):
                continue
            fields: list[tuple[str, AnnotatedType]] = []
            for statement in node.body:
                if isinstance(statement, ast.Pass) or (
                    isinstance(statement, ast.Expr)
                    and isinstance(statement.value, ast.Constant)
                    and isinstance(statement.value.value, str)
                ):
                    continue
                if (
                    not isinstance(statement, ast.AnnAssign)
                    or not isinstance(statement.target, ast.Name)
                    or statement.value is not None
                ):
                    raise context.error(statement, "@struct bodies may contain only annotation-only fields")
                fields.append((statement.target.id, parser.parse(statement.annotation)))
            context.structs[node.name] = tuple(fields)

        def field_types(name: str) -> tuple[ConcreteType, ...]:
            return tuple(annotation.type for _, annotation in context.structs[name])

        for node in module.body:
            if not isinstance(node, ast.ClassDef):
                continue
            for statement, (field_name, annotation) in zip(
                (item for item in node.body if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)),
                context.structs[node.name],
                strict=True,
            ):
                if not is_abi_stable_value(annotation.type, field_types):
                    raise context.error(
                        statement.annotation,
                        f"Struct field '{node.name}.{field_name}' must be an ABI-stable Value",
                    )

    @staticmethod
    def _collect_signatures(module: ast.Module, context: _ModuleContext, parser: TypeParser) -> None:
        for node in module.body:
            if not isinstance(node, ast.FunctionDef):
                continue
            shared = Compiler._is_shared_function(node)
            if shared:
                context.shared_functions.add(node.name)
            if (
                node.args.posonlyargs
                or node.args.kwonlyargs
                or node.args.vararg
                or node.args.kwarg
                or node.args.defaults
            ):
                raise context.error(node, "DSL functions support only required positional arguments")
            arguments: list[DslType] = []
            for argument in node.args.args:
                if argument.annotation is None:
                    raise context.error(argument, f"argument '{argument.arg}' requires a type annotation")
                annotation = parser.parse(argument.annotation)
                if annotation.type.kind == "tensor_storage":
                    raise context.error(
                        argument,
                        "TensorStorage is a host-runtime owner; device parameters must use TensorView",
                    )
                if shared and (
                    annotation.metadata
                    or annotation.type.kind in {"sampler", "tensor_storage", "tensor_view", "texture"}
                ):
                    raise context.error(argument, f"shared function '{node.name}' uses a device-only argument")
                arguments.append(annotation.type)
            result = None
            result_annotation = None
            if node.returns is not None and not (isinstance(node.returns, ast.Constant) and node.returns.value is None):
                result_annotation = parser.parse(node.returns)
                result = result_annotation.type
                if shared and (
                    result_annotation.metadata or result.kind in {"sampler", "tensor_storage", "tensor_view", "texture"}
                ):
                    raise context.error(node.returns, f"shared function '{node.name}' uses a device-only result")
            context.signatures[node.name] = FunctionSignature(tuple(arguments), result)
            context.result_annotations[node.name] = result_annotation

    @staticmethod
    def _is_shared_function(node: ast.FunctionDef) -> bool:
        if len(node.decorator_list) != 1:
            return False
        decorator = node.decorator_list[0]
        if not isinstance(decorator, ast.Call) or (_name(decorator.func) or "").split(".")[-1] != "func":
            return False
        return any(
            keyword.arg == "shared" and isinstance(keyword.value, ast.Constant) and keyword.value.value is True
            for keyword in decorator.keywords
        )

    @staticmethod
    def _decorator(node: ast.FunctionDef, context: _ModuleContext) -> tuple[str | None, tuple[int, int, int] | None]:
        stage = None
        workgroup_size = None
        function_kind = None
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name) or isinstance(decorator, ast.Attribute):
                name = (_name(decorator) or "").split(".")[-1]
                if name in {"vertex", "fragment"}:
                    function_kind = name
                    stage = name
                elif name == "kernel":
                    function_kind = "compute"
                    stage = "compute"
                    workgroup_size = (1, 1, 1)
                elif name == "func":
                    function_kind = "func"
                else:
                    raise context.error(decorator, f"unknown DSL decorator '{name}'")
            elif isinstance(decorator, ast.Call) and (_name(decorator.func) or "").split(".")[-1] == "kernel":
                if function_kind is not None:
                    raise context.error(decorator, "DSL functions require exactly one function decorator")
                function_kind = "compute"
                stage = "compute"
                values = None
                for keyword in decorator.keywords:
                    if keyword.arg == "workgroup_size":
                        values = keyword.value
                    else:
                        raise context.error(keyword, f"unknown kernel option '{keyword.arg}'")
                if decorator.args or not isinstance(values, ast.Tuple) or len(values.elts) != 3:
                    raise context.error(decorator, "kernel requires workgroup_size=(x, y, z)")
                parsed: list[int] = []
                for value in values.elts:
                    if not isinstance(value, ast.Constant) or not isinstance(value.value, int) or value.value <= 0:
                        raise context.error(value, "workgroup dimensions must be positive integer literals")
                    parsed.append(value.value)
                workgroup_size = tuple(parsed)  # type: ignore[assignment]
            elif isinstance(decorator, ast.Call) and (_name(decorator.func) or "").split(".")[-1] == "func":
                if function_kind is not None:
                    raise context.error(decorator, "DSL functions require exactly one function decorator")
                function_kind = "func"
                if decorator.args or len(decorator.keywords) != 1:
                    raise context.error(decorator, "@func accepts only shared=True")
                keyword = decorator.keywords[0]
                if (
                    keyword.arg != "shared"
                    or not isinstance(keyword.value, ast.Constant)
                    or keyword.value.value is not True
                ):
                    raise context.error(decorator, "@func accepts only shared=True")
            else:
                raise context.error(decorator, "unsupported decorator syntax")
            if len(node.decorator_list) > 1:
                raise context.error(decorator, "DSL functions require exactly one function decorator")
        if function_kind is None:
            raise context.error(node, "DSL functions require @func, @vertex, @fragment, or @kernel")
        return stage, workgroup_size


def compile_source(source: str, filename: str = "<string>") -> str:
    return Compiler().compile(source, filename)


def compile_file(input_path: str | Path, *, features: Iterable[str] = (), entry: str | None = None) -> str:
    return Compiler().compile_file(input_path, features=features, entry=entry)
