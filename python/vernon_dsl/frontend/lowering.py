from __future__ import annotations

import ast
import itertools
from typing import Callable, Iterable

from ..diagnostics import copy_location
from ..language.ast_utils import dotted_name, tensor_update_parts
from ..language.slicing import (
    AxisIndex,
    AxisRange,
    AxisSelection,
    constant_integer,
    plan_selection,
    selection_shape,
)
from ..language.stage_registry import GRAPHICS_STAGES
from ..language.syntax import INTRINSIC_METHODS
from ..shader_contracts import (
    ATOMIC_OPERATION_NAMES,
    BUILTIN_CONTRACTS,
    FRAGMENT_DISCARD_OPERATION,
    GENERATED_INTERFACE_CONTRACTS,
    TypeContract,
)
from .abi import attribute_layout, value_leaves
from .aggregate_lowering import lower_aggregate_constructor, lower_tuple
from .control_flow_lowering import (
    emit_source_block,
    lower_bool_op,
    lower_conditional_expression,
    lower_if,
    lower_termination,
)
from .interfaces import plan_generated_interface
from .loop_lowering import lower_for, lower_while
from .lowering_types import DslType, FunctionSignature, ModuleContext, Value, element_type
from .model import (
    AccessMode,
    ConcreteType,
    InterfaceMetadata,
    StorageEffect,
    StorageOwnerKind,
    StorageRegionKind,
    TypedExpression,
    TypedFunctionInstance,
    TypedStatement,
)
from .numeric_lowering import lower_binary, lower_compare, lower_constant, lower_unary
from .params import params_root
from .resource_lowering import lower_texture_sample, lower_texture_size
from .storage_lowering import (
    lower_buffer_index,
    lower_shape_attribute,
    lower_storage_store,
    lower_tensor_view_indices,
)
from .type_parser import AnnotatedType
from .type_solver import can_convert

_SWIZZLES = set("xyzwrgba")


def _dsl_type_from_contract(contract: TypeContract) -> DslType:
    if contract.kind == "tensor":
        element = DslType("scalar", contract.name)
        return DslType("tensor", "Tensor", (element, *contract.shape))
    return DslType(contract.kind, contract.name)


def _placement_attributes(set_index: int, binding: int) -> tuple[str, ...]:
    """Only the authored placement; the compiler maps whatever is missing."""
    attributes = [f"vernon.set = {set_index} : i64"] if set_index >= 0 else []
    if binding >= 0:
        attributes.append(f"vernon.binding = {binding} : i64")
    return tuple(attributes)


class _FunctionEmitter:
    def __init__(
        self,
        context: ModuleContext,
        signature: FunctionSignature,
        argument_annotations: list[AnnotatedType],
        result_annotation: AnnotatedType | None,
        typed_function: TypedFunctionInstance,
        stage: str | None,
        workgroup_size: tuple[int, int, int] | None,
        planning_policy: str | None = None,
        textures_read_elsewhere: frozenset[str] = frozenset(),
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
        self.planning_policy = planning_policy
        self.lines: list[str] = []
        self.indent = 1
        self.next_value = 0
        self.environment: dict[str, Value] = {}
        self.generated_values: dict[str, Value] = {}
        self.implicit_samplers: dict[str, Value] = {}
        self.interface_plan = plan_generated_interface(
            context, self.node, argument_annotations, stage, typed_function.effects, textures_read_elsewhere
        )
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
        if not contract.supports(self.stage or "", direction):
            allowed = sorted(contract.uses)
            requirement = (
                f"a {allowed[0][0]} {allowed[0][1]}"
                if len(allowed) == 1
                else "one of [" + ", ".join(f"{stage} {use_direction}" for stage, use_direction in allowed) + "]"
            )
            raise self.context.error(
                self.node,
                f"builtin '{builtin}' requires {requirement}, but {label} is a {self.stage or 'non-entry'} {direction}",
            )
        if value_type != expected_type:
            raise self.context.error(
                self.node,
                f"builtin '{builtin}' requires type {expected_type.spelling}, "
                f"but {label} has type {value_type.spelling}",
            )

    def _validate_builtin_annotation(self, annotation: AnnotatedType, direction: str, label: str) -> None:
        builtins = [str(item.arguments[0]) for item in annotation.metadata if item.kind == "builtin"]
        if len(builtins) > 1:
            raise self.context.error(self.node, f"{label} has more than one builtin annotation")
        if builtins:
            self._validate_builtin_contract(builtins[0], annotation.type, direction, label)

    def _entry_result_fields(self) -> tuple[tuple[str, AnnotatedType], ...] | None:
        if self.stage not in GRAPHICS_STAGES or self.signature.result is None or self.signature.result.kind != "struct":
            return None
        fields = self.context.structs[self.signature.result.name]
        planned: list[tuple[str, AnnotatedType]] = []
        builtins: set[str] = set()
        next_location = 0

        def struct_fields(name: str) -> tuple[tuple[str, ConcreteType], ...]:
            return tuple((field_name, field.type) for field_name, field in self.context.structs[name])

        for field_name, annotation in fields:
            self._validate_builtin_annotation(annotation, "output", f"output field '{field_name}'")
            builtin_values = [str(item.arguments[0]) for item in annotation.metadata if item.kind == "builtin"]
            if self.stage == "fragment" and builtin_values:
                raise self.context.error(self.node, f"fragment output field '{field_name}' cannot be a builtin")
            if builtin_values:
                builtin = builtin_values[0]
                if builtin in builtins:
                    raise self.context.error(self.node, f"shader output builtin '{builtin}' is used more than once")
                builtins.add(builtin)
                planned.append((field_name, annotation))
                continue
            try:
                span = attribute_layout(annotation.type, struct_fields).location_span
            except ValueError as error:
                raise self.context.error(
                    self.node, f"output field '{field_name}' is not a numeric shader interface value: {error}"
                ) from None
            metadata = (*annotation.metadata, InterfaceMetadata("attribute", (next_location, 0)))
            planned.append((field_name, AnnotatedType(annotation.type, metadata)))
            next_location += span
        return tuple(planned)

    def emit(self) -> list[str]:
        emit_value_abi_metadata = self.stage is not None or self.node.name in self.context.shared_functions
        argument_locations: dict[int, int] = {}
        # Every entry stage's Value parameters carry locations. The module graph assigns them to source parameters,
        # but params leaves exist only after flattening, so unlocated leaves take the free locations here.
        if self.stage is not None:
            candidates: list[tuple[int, AnnotatedType, int, int | None]] = []

            def struct_fields(name: str) -> tuple[tuple[str, ConcreteType], ...]:
                return tuple((field_name, field.type) for field_name, field in self.context.structs[name])

            for index, annotation in enumerate(self.argument_annotations):
                if any(item.kind in {"builtin", "uniform", "resource"} for item in annotation.metadata):
                    continue
                value_type = annotation.type
                if value_type.kind not in {"scalar", "tensor", "tuple", "struct"}:
                    continue
                try:
                    span = attribute_layout(value_type, struct_fields).location_span
                except ValueError as error:
                    if self.stage not in GRAPHICS_STAGES or self.stage == "fragment":
                        span = 1
                    else:
                        raise self.context.error(self.node.args.args[index], str(error)) from None
                attribute_metadata = next(
                    (item for item in annotation.metadata if item.kind == "attribute"),
                    None,
                )
                explicit = (
                    int(attribute_metadata.arguments[0])
                    if attribute_metadata is not None and int(attribute_metadata.arguments[0]) >= 0
                    else None
                )
                candidates.append((index, annotation, span, explicit))

            occupied: set[int] = set()
            for index, _, span, explicit in candidates:
                if explicit is None:
                    continue
                slots = set(range(explicit, explicit + span))
                overlap = occupied & slots
                if overlap:
                    raise self.context.error(
                        self.node.args.args[index], f"interface location overlap at {min(overlap)}"
                    )
                occupied.update(slots)
                argument_locations[index] = explicit

            next_location = 0
            # Params leaves follow every other attribute, matching the module graph, which places plain parameters
            # before the entry is flattened.
            automatic = [candidate for candidate in candidates if candidate[3] is None]
            automatic.sort(key=lambda candidate: params_root(self.node.args.args[candidate[0]]) is not None)
            for index, _, span, _ in automatic:
                while occupied & set(range(next_location, next_location + span)):
                    next_location += 1
                argument_locations[index] = next_location
                occupied.update(range(next_location, next_location + span))
                next_location += span

        arguments: list[str] = []
        for index, (argument, annotation) in enumerate(
            zip(self.node.args.args, self.argument_annotations, strict=True)
        ):
            self._validate_builtin_annotation(annotation, "input", f"argument '{argument.arg}'")
            value_type = annotation.type
            typed_parameter = self.typed_function.parameters[index]
            access = typed_parameter.access
            value = Value(
                f"%arg{index}",
                value_type,
                access=access,
            )
            self.environment[argument.arg] = value
            if self.stage == "compute" and any(
                item.kind == "resource" or (item.kind == "uniform" and any(int(value) >= 0 for value in item.arguments))
                for item in annotation.metadata
            ):
                raise self.context.error(
                    argument,
                    f"kernel parameters bind in argument order; set and binding apply only to graphics stages: "
                    f"'{argument.arg}'",
                )
            attributes = self._metadata_attributes(annotation.metadata, stage=self.stage, is_result=False)
            if index in argument_locations and not any(
                attribute.startswith("vernon.location") for attribute in attributes
            ):
                attributes.append(f"vernon.location = {argument_locations[index]} : i64")
            attributes.append(f'vernon.source_name = "{argument.arg}"')
            attributes.extend(self._dtype_attributes(annotation.type))
            if annotation.type.kind == "tensor_view":
                element_type = annotation.type.arguments[0]
                assert isinstance(element_type, DslType)
                attributes.append(self._abi_leaf_dtypes_attribute(element_type, "vernon.element_abi_leaf_dtypes"))
            if emit_value_abi_metadata and value_type.kind in {"scalar", "tensor", "tuple", "struct"}:
                attributes.extend(self._abi_attributes(value_type))
            if value_type.kind in {"tensor_view", "texture", "sampler"} and self.stage is not None:
                attributes = [
                    attribute
                    for attribute in attributes
                    if not attribute.startswith(("vernon.interface", "vernon.location"))
                ]
                attributes.append('vernon.interface = "resource"')
                # Graphics stages compile separately but link into one pipeline; only the compiler's binding
                # planner sees the pipeline-wide binding base, so it owns every unannotated graphics binding.
                # Kernels have no such planner and bind in argument order.
                if self.stage == "compute":
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
            has_result_slot = any(item.kind in {"attribute", "builtin"} for item in result_metadata)
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
            if self.planning_policy is not None:
                function_attributes.append(f'vernon.ad.planning_policy = "{self.planning_policy}"')
            reflected_effects: list[str] = []
            for effect in self.typed_function.effects:
                if not isinstance(effect, StorageEffect) or effect.owner.kind is not StorageOwnerKind.PARAMETER:
                    continue
                fields = [
                    f'kind = "{effect.kind.value}"',
                    f'owner = "{effect.owner.name}"',
                    f'region = "{effect.region.kind.value}"',
                ]
                if effect.region.kind is StorageRegionKind.ELEMENT:
                    indices = ", ".join(str(index) for index in effect.region.indices)
                    fields.append(f"indices = array<i64: {indices}>" if indices else "indices = array<i64>")
                if effect.atomic:
                    fields.append("atomic = true")
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
        emit_source_block(self, self.node.body)
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

        attributes = [self._abi_leaf_dtypes_attribute(value_type, "vernon.abi_leaf_dtypes")]
        if value_type.kind == "tensor":
            element_type = value_type.arguments[0]
            assert isinstance(element_type, DslType)
            attributes.append(self._abi_leaf_dtypes_attribute(element_type, "vernon.element_abi_leaf_dtypes"))
        return attributes

    @staticmethod
    def _dtype_attributes(value_type: DslType) -> list[str]:
        if value_type.kind in {"tensor", "tensor_view"}:
            element = value_type.arguments[0]
        elif value_type.kind == "texture" and value_type.arguments[3] == "sampled":
            element = value_type.arguments[1]
        else:
            element = value_type
        assert isinstance(element, DslType)
        return [f'vernon.dtype = "{element.name}"'] if element.kind == "scalar" else []

    def _abi_leaf_dtypes_attribute(self, value_type: DslType, attribute: str) -> str:
        def fields(name: str) -> tuple[tuple[str, DslType], ...]:
            return tuple((field_name, annotation.type) for field_name, annotation in self.context.structs[name])

        leaves = value_leaves(value_type, fields)
        leaf_dtypes = ", ".join(f'"{leaf.dtype}"' for leaf in leaves)
        return f"{attribute} = [{leaf_dtypes}]"

    def _append_generated_arguments(self, arguments: list[str]) -> None:
        for sampler_plan in self.interface_plan.implicit_samplers:
            texture_name = sampler_plan.texture_name
            index = len(arguments)
            value = Value(f"%arg{index}", DslType("sampler", "Sampler"))
            self.implicit_samplers[self.environment[texture_name].name] = value
            source_name = f"__vernon_implicit_sampler_{texture_name}"
            texture_index = next(
                index for index, argument in enumerate(self.node.args.args) if argument.arg == texture_name
            )
            texture_set = next(
                (
                    int(item.arguments[0])
                    for item in self.argument_annotations[texture_index].metadata
                    if item.kind == "resource"
                ),
                -1,
            )
            set_attribute = f", vernon.set = {texture_set} : i64" if texture_set >= 0 else ""
            arguments.append(
                f'{value.name}: !vernon.sampler {{vernon.interface = "resource", '
                f'vernon.source_name = "{source_name}", '
                f'vernon.implicit = "sampler", '
                f'vernon.implicit_texture = "{texture_name}"{set_attribute}}}'
            )

        for api in self.interface_plan.generated_apis:
            contract = GENERATED_INTERFACE_CONTRACTS[api]
            index = len(arguments)
            value_type = _dsl_type_from_contract(contract.type)
            value = Value(f"%arg{index}", value_type)
            self.generated_values[api] = value
            attributes = [
                f'vernon.interface = "{contract.interface}"',
                f'vernon.source_name = "_vernon_{api}"',
                f'vernon.implicit = "{api}"',
            ]
            if contract.builtin is not None:
                attributes.append(f'vernon.builtin = "{contract.builtin}"')
            attributes.extend(self._dtype_attributes(value_type))
            if value_type.kind == "scalar":
                attributes.append(self._abi_leaf_dtypes_attribute(value_type, "vernon.abi_leaf_dtypes"))
            elif value_type.kind == "tensor":
                attributes.extend(self._abi_attributes(value_type))
            arguments.append(f"{value.name}: {value.type.mlir} {{{', '.join(attributes)}}}")

    def _line(self, text: str) -> None:
        self.lines.append("  " * self.indent + text)

    def _fresh(self) -> str:
        value = f"%{self.next_value}"
        self.next_value += 1
        return value

    @staticmethod
    def _metadata_attributes(
        metadata: Iterable[InterfaceMetadata],
        *,
        stage: str | None,
        is_result: bool,
        default_location: int | None = None,
    ) -> list[str]:
        attributes: list[str] = []
        if stage is None:
            return attributes
        items = tuple(metadata)
        interface = "output" if is_result else "input"
        if any(item.kind == "resource" for item in items):
            interface = "resource"
        elif any(item.kind == "uniform" for item in items):
            interface = "uniform"

        has_slot = any(item.kind in {"attribute", "builtin"} for item in items)
        attributes.append(f'vernon.interface = "{interface}"')
        if interface in {"input", "output"} and not has_slot:
            if is_result and stage == "vertex":
                attributes.append('vernon.builtin = "position"')
            elif default_location is not None:
                attributes.append(f"vernon.location = {default_location} : i64")

        for item in items:
            if item.kind == "attribute":
                location, divisor = (int(value) for value in item.arguments)
                if location >= 0:
                    attributes.append(f"vernon.location = {location} : i64")
                if divisor:
                    attributes.append(f"vernon.instance_divisor = {divisor} : i64")
            elif item.kind in {"resource", "uniform"}:
                attributes.extend(_placement_attributes(*(int(value) for value in item.arguments)))
            elif item.kind == "varying":
                attributes.append("vernon.varying = true")
                if item.arguments:
                    attributes.append(f'vernon.interpolation = "{item.arguments[0]}"')
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
                lower_storage_store(self, node.targets[0], self._expression(node.value))
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
            copy_location(operation, node)
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
                    ast.FloorDiv: "floordiv",
                    ast.Mod: "mod",
                    ast.Pow: "pow",
                    ast.BitAnd: "bitand",
                    ast.BitOr: "bitor",
                    ast.BitXor: "bitxor",
                    ast.LShift: "lshift",
                    ast.RShift: "rshift",
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
                lower_storage_store(self, node.target, value)
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
        if lower_termination(self, node):
            return
        if isinstance(node, ast.If):
            lower_if(self, node)
            return
        if isinstance(node, ast.For):
            lower_for(self, node)
            return
        if isinstance(node, ast.While):
            lower_while(self, node)
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
        raise self.context.error(node, f"cannot create a control-flow payload for {value_type.spelling}")

    def _expression(self, node: ast.expr, expected: DslType | None = None) -> Value:
        typed = self._typed_expression(node)
        if expected is not None and not can_convert(typed.type, expected):
            raise self.context.error(
                node,
                f"typed model mismatch: cannot use {typed.type.spelling} as {expected.spelling}",
            )
        if isinstance(node, ast.Name):
            if node.id not in self.environment:
                raise self.context.error(node, f"unknown local value '{node.id}'")
            return self.environment[node.id]
        if isinstance(node, ast.Constant):
            return lower_constant(self, node, typed.type)
        if isinstance(node, ast.BinOp):
            return lower_binary(self, node)
        if isinstance(node, ast.UnaryOp):
            return lower_unary(self, node)
        if isinstance(node, ast.BoolOp):
            return lower_bool_op(self, node)
        if isinstance(node, ast.Compare):
            return lower_compare(self, node)
        if isinstance(node, ast.Tuple):
            return lower_tuple(self, node, node.elts)
        if isinstance(node, ast.Call):
            return self._call(node)
        if isinstance(node, ast.Attribute):
            return self._attribute(node)
        if isinstance(node, ast.Subscript):
            return self._index(node)
        if isinstance(node, ast.IfExp):
            return lower_conditional_expression(self, node)
        raise self.context.error(node, f"unsupported expression syntax: {type(node).__name__}")

    def _splat(self, node: ast.AST, value: Value, tensor_type: DslType) -> Value:
        element = tensor_type.arguments[0]
        assert isinstance(element, DslType)
        self._require_same_type(node, element, value.type)
        result = self._fresh()
        self._line(f"{result} = tensor.splat {value.name} : {tensor_type.mlir}")
        return Value(result, tensor_type)

    # Selections lower to element extraction, select and `construct`, which every backend and the autodiff rules
    # already handle; a dedicated insert operation can replace them without changing the language. Each axis
    # evaluates to a constant position, a dynamic position (an `index` value, negative ones wrapped), constant
    # positions (a `range`), or a dynamic window `(start, length)` whose start is clamped so the window fits.

    def _constant_pool(self) -> Callable[[int], str]:
        pool: dict[int, str] = {}

        def constant(number: int) -> str:
            if number not in pool:
                pool[number] = self._fresh()
                self._line(f"{pool[number]} = arith.constant {number} : index")
            return pool[number]

        return constant

    def _index_operand(self, node: ast.expr) -> tuple[str, bool]:
        """The `index` value of an integer expression, and whether its type is signed."""

        value = self._expression(node)
        if not value.type.is_integer:
            raise self.context.error(node, "tensor index must be an integer")
        if value.type.mlir == "index":
            return value.name, True
        cast = self._fresh()
        self._line(f"{cast} = arith.index_cast {value.name} : {value.type.mlir} to index")
        return cast, not value.type.name.startswith("u")

    def _select_index(self, condition: str, true_value: str, false_value: str, type_mlir: str = "index") -> str:
        result = self._fresh()
        self._line(f"{result} = arith.select {condition}, {true_value}, {false_value} : {type_mlir}")
        return result

    def _wrapped_index(self, index: str, signed: bool, extent: int, constant: Callable[[int], str]) -> str:
        if not signed:
            return index
        negative = self._fresh()
        self._line(f"{negative} = arith.cmpi slt, {index}, {constant(0)} : index")
        wrapped = self._fresh()
        self._line(f"{wrapped} = arith.addi {index}, {constant(extent)} : index")
        return self._select_index(negative, wrapped, index)

    def _selection_axes(
        self, plan: list[AxisSelection], extents: list[int], constant: Callable[[int], str]
    ) -> list[int | str | range | tuple[str, int]]:
        axes: list[int | str | range | tuple[str, int]] = []
        for axis, extent in zip(plan, extents, strict=True):
            if isinstance(axis, AxisRange):
                axes.append(range(axis.start, axis.start + axis.step * axis.length, axis.step))
            elif isinstance(axis, AxisIndex):
                literal = constant_integer(axis.node)
                if literal is not None:
                    axes.append(literal % extent if -extent <= literal < extent else literal)
                else:
                    axes.append(self._wrapped_index(*self._index_operand(axis.node), extent, constant))
            else:
                start = self._wrapped_index(*self._index_operand(axis.start), extent, constant)
                below = self._fresh()
                self._line(f"{below} = arith.cmpi slt, {start}, {constant(0)} : index")
                start = self._select_index(below, constant(0), start)
                last = constant(extent - axis.length)
                above = self._fresh()
                self._line(f"{above} = arith.cmpi sgt, {start}, {last} : index")
                axes.append((self._select_index(above, last, start), axis.length))
        return axes

    def _extract(self, tensor: Value, coordinate: Iterable[int | str], constant: Callable[[int], str]) -> Value:
        element = tensor.type.arguments[0]
        assert isinstance(element, DslType)
        indices = ", ".join(constant(item) if isinstance(item, int) else item for item in coordinate)
        result = self._fresh()
        self._line(f"{result} = tensor.extract {tensor.name}[{indices}] : {tensor.type.mlir}")
        return Value(result, element)

    def _selection(self, node: ast.Subscript, tensor: Value, indices: list[ast.expr]) -> Value:
        """`v[...]` of a Tensor of scalars: the element, or a Tensor of the selected elements."""

        extents = [int(extent) for extent in tensor.type.arguments[1:]]  # type: ignore[arg-type]
        constant = self._constant_pool()
        axes = self._selection_axes(plan_selection(node, indices, extents), extents, constant)
        options: list[list[int | str]] = []
        for axis in axes:
            if isinstance(axis, tuple):
                start, length = axis
                positions: list[int | str] = [start]
                for offset in range(1, length):
                    shifted = self._fresh()
                    self._line(f"{shifted} = arith.addi {start}, {constant(offset)} : index")
                    positions.append(shifted)
                options.append(positions)
            else:
                options.append(list(axis) if isinstance(axis, range) else [axis])
        elements = [self._extract(tensor, coordinate, constant) for coordinate in itertools.product(*options)]
        result_type = self._typed_expression(node).type
        if result_type.kind == "scalar":
            return elements[0]
        return self._intrinsic(node, "construct", elements, result_type)

    def _tensor_update(self, node: ast.Call, tensor_node: ast.expr, index_nodes: list[ast.expr], method: str) -> Value:
        """`v.at[...].set(x)` or `.add(x)`. Each element is the old one unless the selection covers it. A dynamic
        index covers position p when it wraps to p, so one out of range changes nothing; a window covers p from each
        start that puts p inside it, and the block element is picked by which start the clamped window has."""

        tensor = self._expression(tensor_node)
        element = tensor.type.arguments[0]
        assert isinstance(element, DslType)
        extents = [int(extent) for extent in tensor.type.arguments[1:]]  # type: ignore[arg-type]
        assert isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Subscript)
        plan = plan_selection(node.func.value, index_nodes, extents)
        constant = self._constant_pool()
        axes = self._selection_axes(plan, extents, constant)
        shape = selection_shape(plan)
        value = self._expression(node.args[0])
        if value.type.kind == "scalar":
            value = self._coerce_implicit(node.args[0], value, element)
        else:
            value = self._coerce_implicit(node.args[0], value, DslType("tensor", "Tensor", (element, *shape)))
        add = "arith.addf" if element.is_float else "arith.addi"

        def conjunction(conditions: list[str]) -> str | None:
            result: str | None = None
            for condition in conditions:
                if result is None:
                    result = condition
                else:
                    both = self._fresh()
                    self._line(f"{both} = arith.andi {result}, {condition} : i1")
                    result = both
            return result

        elements: list[Value] = []
        for position in itertools.product(*(range(extent) for extent in extents)):
            old = self._extract(tensor, position, constant)
            matches: list[str] = []
            block_options: list[list[tuple[str | None, int]]] = []
            covered = True
            for axis, p, extent in zip(axes, position, extents, strict=True):
                if isinstance(axis, int):
                    covered = axis == p
                elif isinstance(axis, str):
                    match = self._fresh()
                    self._line(f"{match} = arith.cmpi eq, {axis}, {constant(p)} : index")
                    matches.append(match)
                elif isinstance(axis, range):
                    covered = p in axis
                    if covered:
                        block_options.append([(None, axis.index(p))])
                else:
                    start, length = axis
                    starts: list[tuple[str | None, int]] = []
                    for offset in range(length):
                        if 0 <= p - offset <= extent - length:
                            match = self._fresh()
                            self._line(f"{match} = arith.cmpi eq, {start}, {constant(p - offset)} : index")
                            starts.append((match, offset))
                    covered = bool(starts)
                    block_options.append(starts)
                if not covered:
                    break
            if not covered:
                elements.append(old)
                continue
            result = old
            for choice in itertools.product(*block_options):
                source = value
                if value.type.kind == "tensor":
                    source = self._extract(value, [offset for _, offset in choice], constant)
                new = source
                if method == "add":
                    new = Value(self._fresh(), element)
                    self._line(f"{new.name} = {add} {old.name}, {source.name} : {element.mlir}")
                condition = conjunction(matches + [match for match, _ in choice if match is not None])
                result = (
                    new
                    if condition is None
                    else Value(self._select_index(condition, new.name, result.name, element.mlir), element)
                )
            elements.append(result)
        return self._intrinsic(node, "construct", elements, tensor.type)

    def _struct_replace(self, node: ast.Call, result_type: DslType) -> Value:
        struct = self._expression(node.args[0])
        fields = self._struct_value_fields(node, struct)
        changes = {keyword.arg: keyword.value for keyword in node.keywords}
        values = [
            self._coerce_implicit(changes[name], self._expression(changes[name]), annotation.type)
            if name in changes
            else field
            for (name, annotation), field in zip(self.context.structs[result_type.name], fields, strict=True)
        ]
        result = self._fresh()
        self._line(
            f'{result} = "vernon.struct_create"({", ".join(value.name for value in values)}) '
            f'{{type_name = "{result_type.name}"}} : '
            f"({', '.join(value.type.mlir for value in values)}) -> {result_type.mlir}"
        )
        return Value(result, result_type, tuple(values))

    def _call(self, node: ast.Call) -> Value:
        if (update := tensor_update_parts(node)) is not None:
            return self._tensor_update(node, *update)
        typed_call = self._typed_expression(node)
        name = typed_call.operation or ""
        if (
            (dotted_name(node.func) or "").split(".")[-1] == "replace"
            and typed_call.type.kind == "struct"
            and node.keywords
            and "replace" not in self.context.signatures
        ):
            return self._struct_replace(node, typed_call.type)
        if node.keywords and name != "workgroup_storage":
            raise self.context.error(node, "function calls do not support keyword arguments")
        if name == "workgroup_storage":
            if self.stage != "compute":
                raise self.context.error(node, "workgroup storage is supported only in compute kernels")
            result = self._fresh()
            self._line(f'{result} = "vernon.workgroup_alloc"() : () -> {typed_call.type.mlir}')
            return Value(result, typed_call.type, access=AccessMode.READ_WRITE)
        if name in {"workgroup_barrier", "storage_barrier"}:
            if self.stage != "compute":
                raise self.context.error(node, "barriers are supported only in compute kernels")
            scope = "workgroup" if name == "workgroup_barrier" else "device"
            self._line(f'"vernon.barrier"() {{ordering = "acquire_release", scope = "{scope}"}} : () -> ()')
            return Value("", typed_call.type)
        if name == FRAGMENT_DISCARD_OPERATION:
            if self.node.name in self.context.shared_functions:
                raise self.context.error(
                    node, f"shared function '{self.node.name}' uses device-only operation '{name}'"
                )
            self._line(f'"vernon.{name}"() : () -> ()')
            return Value("", typed_call.type)
        if name in {"Tensor", "Vector", "Matrix"}:
            return lower_aggregate_constructor(self, node, name)
        if name == "Tuple":
            return lower_tuple(self, node, node.args)
        if name in ATOMIC_OPERATION_NAMES:
            if self.stage != "compute" or len(node.args) != 3:
                raise self.context.error(node, f"{name} requires compute storage, index, and value")
            storage = self._expression(node.args[0])
            if storage.type.kind != "tensor_view":
                raise self.context.error(node.args[0], f"{name} requires a writable TensorView")
            if storage.type.arguments[2] == "read":
                raise self.context.error(node.args[0], f"{name} requires a writable TensorView")
            shape = storage.type.arguments[1]
            assert isinstance(shape, tuple)
            index_nodes = list(node.args[1].elts) if isinstance(node.args[1], ast.Tuple) else [node.args[1]]
            if len(index_nodes) != len(shape):
                raise self.context.error(node.args[1], f"{name} requires one index per TensorView dimension")
            indices = [lower_buffer_index(self, index_node) for index_node in index_nodes]
            element = storage.type.arguments[0]
            assert isinstance(element, DslType)
            value = self._coerce_implicit(node.args[2], self._expression(node.args[2]), element)
            result = self._fresh()
            atomic_kind = name.removeprefix("atomic_")
            if atomic_kind in {"min", "max"} and element.name == "u32":
                atomic_kind = f"u{atomic_kind}"
            operands = ", ".join((storage.name, *(index.name for index in indices), value.name))
            operand_types = ", ".join((storage.type.mlir, *(["index"] * len(indices)), element.mlir))
            attributes = [f'atomic_kind = "{atomic_kind}"', 'ordering = "relaxed"']
            self._line(
                f'{result} = "vernon.atomic"({operands}) '
                f"{{{', '.join(attributes)}}} "
                f": ({operand_types}) -> {element.mlir}"
            )
            return Value(result, element)
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
            "acos": "math.acos",
            "sin": "math.sin",
            "cos": "math.cos",
            "exp": "math.exp",
            "floor": "math.floor",
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
        if name == "atan2":
            if len(arguments) != 2:
                raise self.context.error(node, "atan2 requires two floating-point arguments")
            result_type = self._typed_expression(node).type
            arguments = [
                self._coerce_numeric(source, argument, result_type)
                for source, argument in zip(node.args, arguments, strict=True)
            ]
            if not result_type.is_float:
                raise self.context.error(node, "atan2 requires two floating-point arguments")
            result = self._fresh()
            self._line(f"{result} = math.atan2 {arguments[0].name}, {arguments[1].name} : {result_type.mlir}")
            return Value(result, result_type)
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
        if name == "dot":
            if len(arguments) != 2:
                raise self.context.error(node, "dot requires two Tensor arguments")
            self._require_same_type(node, arguments[0].type, arguments[1].type)
            tensor = arguments[0].type
            if tensor.kind != "tensor" or len(tensor.arguments) < 2 or not tensor.is_float:
                raise self.context.error(node, "dot requires equal floating-point Tensors")
            return self._intrinsic(node, name, arguments, typed_call.type)
        if name == "cross":
            if len(arguments) != 2:
                raise self.context.error(node, "cross requires two vector arguments")
            self._require_same_type(node, arguments[0].type, arguments[1].type)
            vector = arguments[0].type
            if vector.kind != "tensor" or len(vector.arguments) != 2 or not vector.is_float:
                raise self.context.error(node, "cross requires floating-point vectors")
            if vector.arguments[1] != 3:
                raise self.context.error(node, "cross requires three-component vectors")
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
            result_type = self._typed_expression(node).type
            element = element_type(result_type)
            coerced: list[Value] = []
            for source, argument in zip(node.args, arguments, strict=True):
                if argument.type.kind != "tensor":
                    raise self.context.error(source, "matmul operands must be Tensors")
                operand_type = DslType("tensor", "Tensor", (element, *argument.type.arguments[1:]))
                coerced.append(self._coerce_implicit(source, argument, operand_type))
            return self._intrinsic(node, name, coerced, result_type)
        if name == "texture_sample":
            return lower_texture_sample(self, node, arguments, typed_call.type)
        if name == "texture_size":
            return lower_texture_size(self, node, arguments, typed_call.type)
        if name == "texture_load":
            return self._intrinsic(node, name, arguments, typed_call.type)
        if name == "texture_store":
            operand_types = ", ".join(argument.type.mlir for argument in arguments)
            self._line(
                f'"vernon.intrinsic"({", ".join(argument.name for argument in arguments)}) '
                f'{{name = "texture_store"}} : ({operand_types}) -> ()'
            )
            return Value("", typed_call.type)
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
            if self._entry_result_fields() is not None and self.signature.result == result_type:
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

    def _cast(self, node: ast.AST, value: Value, target: DslType) -> Value:
        if value.type == target:
            return value
        source_scalar = element_type(value.type)
        target_scalar = element_type(target)
        if value.type.kind == "index" and target_scalar.is_float:
            integer = DslType("scalar", "i32")
            intermediate = self._fresh()
            self._line(f"{intermediate} = arith.index_cast {value.name} : {value.type.mlir} to {integer.mlir}")
            value = Value(intermediate, integer)
            source_scalar = integer
            operation = "arith.sitofp"
        elif value.type.kind == "index":
            operation = "arith.index_cast"
        elif source_scalar.is_integer and target_scalar.is_float:
            # Both i32 and u32 operands are signless i32 in MLIR; the selected
            # conversion operation carries the source signedness semantics.
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
            raise self.context.error(node, f"cannot convert {value.type.spelling} to {target.spelling}")
        result = self._fresh()
        self._line(f"{result} = {operation} {value.name} : {value.type.mlir} to {target.mlir}")
        return Value(result, target)

    def _coerce_numeric(self, node: ast.AST, value: Value, target: DslType) -> Value:
        if value.type == target:
            return value
        target_element = element_type(target)
        if value.type.kind == "scalar" and target.kind == "tensor":
            value = self._coerce_implicit(node, value, target_element)
            return self._splat(node, value, target)
        if value.type.kind == "tensor" and target.kind == "tensor":
            source_shape = value.type.arguments[1:]
            element_target = DslType("tensor", "Tensor", (target_element, *source_shape))
            value = self._coerce_implicit(node, value, element_target)
            if source_shape != target.arguments[1:]:
                return self._intrinsic(node, "broadcast", [value], target)
            return value
        return self._coerce_implicit(node, value, target)

    def _coerce_implicit(self, node: ast.AST, value: Value, target: DslType) -> Value:
        if value.type == target:
            return value
        if can_convert(value.type, target):
            return self._cast(node, value, target)
        raise self.context.error(node, f"unsafe implicit conversion from {value.type.spelling} to {target.spelling}")

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
        if node.attr == "shape":
            return lower_shape_attribute(self, node, value)
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
            extent = shape[0]
            assert isinstance(extent, int)
            if any(index >= extent for index in indices):
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
            indices = lower_tensor_view_indices(self, node, value)
            element = value.type.arguments[0]
            assert isinstance(element, DslType)
            result = self._fresh()
            operands = ", ".join((value.name, *(index.name for index in indices)))
            operand_types = ", ".join((value.abi_type.mlir, *(["index"] * len(indices))))
            self._line(f'{result} = "vernon.load"({operands}) : ({operand_types}) -> {element.mlir}')
            return Value(result, element)
        if value.type.kind != "tensor":
            raise self.context.error(node, "indexing requires a Tensor or Storage value")
        indices = list(node.slice.elts) if isinstance(node.slice, ast.Tuple) else [node.slice]
        element = value.type.arguments[0]
        assert isinstance(element, DslType)
        if element.kind == "scalar":
            return self._selection(node, value, indices)
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

    def _require_same_type(self, node: ast.AST, expected: DslType, actual: DslType) -> None:
        if expected != actual:
            raise self.context.error(node, f"type mismatch: expected {expected.spelling}, got {actual.spelling}")
