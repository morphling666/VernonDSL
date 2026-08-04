from __future__ import annotations

import ast
import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, NewType

from ..bundle import canonical_json
from ..language.ast_utils import dotted_name
from ..language.scalar_types import SCALAR_TYPES
from .model import ConcreteType, TypedFunctionInstance, TypedStatement

Error = Callable[[ast.AST, str], Exception]
StructFields = Mapping[str, tuple[tuple[str, ConcreteType], ...]]
NodeId = NewType("NodeId", int)


class OpCode(Enum):
    PARAMETER = "parameter"
    BUILTIN = "builtin"
    CONSTANT = "constant"
    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    DIV = "div"
    POW = "pow"
    NEG = "neg"
    IDENTITY = "identity"
    SIN = "sin"
    COS = "cos"
    ACOS = "acos"
    ATAN2 = "atan2"
    EXP = "exp"
    LOG = "log"
    SQRT = "sqrt"
    ABS = "abs"
    DOT = "dot"
    CROSS = "cross"
    MATMUL = "matmul"
    NORM = "norm"
    NORMALIZE = "normalize"
    REFLECT = "reflect"
    SPLAT = "splat"
    BROADCAST = "broadcast"
    CONDITIONAL = "conditional"
    TUPLE = "tuple"
    TYPED_TUPLE = "Tuple"
    TENSOR = "Tensor"
    VECTOR = "Vector"
    MATRIX = "Matrix"
    STRUCT = "struct"
    FIELD = "field"
    INDEX = "index"
    INDEX_DYNAMIC = "index_dynamic"
    STORE = "store"
    STORE_DYNAMIC = "store_dynamic"
    COMPARE = "compare"
    INTRINSIC = "intrinsic"


class AccumulationMode(Enum):
    """Grid-independent mathematical operation for combining invocation gradients."""

    REDUCE_SUM = "reduce_sum"
    SCATTER_ADD = "scatter_add"


class AccessPatternEvidence(Enum):
    DISJOINT_SCATTER = "disjoint_scatter"
    INJECTIVE_GLOBAL_INDEX = "injective_global_index"
    NON_INJECTIVE_INDEX = "non_injective_index"
    STATIC_INDEX_CONFLICT = "static_index_conflict"
    SHARED_VALUE = "shared_value"


@dataclass(frozen=True)
class BuiltinOp:
    name: str


@dataclass(frozen=True)
class LiteralOp:
    value: bool | int | float


@dataclass(frozen=True)
class NamedOp:
    name: str


@dataclass(frozen=True)
class StaticIndicesOp:
    indices: tuple[int, ...]


OperationPayload = BuiltinOp | LiteralOp | NamedOp | StaticIndicesOp


# Saved operand positions, followed by whether the result is saved.
_DERIVATIVE_RULES: dict[OpCode, tuple[tuple[int, ...], bool]] = {
    OpCode.ADD: ((), False),
    OpCode.SUB: ((), False),
    OpCode.MUL: ((0, 1), False),
    OpCode.DIV: ((0, 1), False),
    OpCode.POW: ((0, 1), True),
    OpCode.NEG: ((), False),
    OpCode.IDENTITY: ((), False),
    OpCode.SIN: ((0,), False),
    OpCode.COS: ((0,), False),
    OpCode.ACOS: ((0,), False),
    OpCode.ATAN2: ((0, 1), False),
    OpCode.EXP: ((), True),
    OpCode.LOG: ((0,), False),
    OpCode.SQRT: ((), True),
    OpCode.ABS: ((0,), False),
    OpCode.DOT: ((0, 1), False),
    OpCode.CROSS: ((0, 1), False),
    OpCode.MATMUL: ((0, 1), False),
    OpCode.NORM: ((0,), True),
    OpCode.NORMALIZE: ((0,), True),
    OpCode.REFLECT: ((0, 1), False),
    OpCode.SPLAT: ((), False),
    OpCode.BROADCAST: ((), False),
    OpCode.CONDITIONAL: ((0,), False),
    OpCode.TUPLE: ((), False),
    OpCode.TYPED_TUPLE: ((), False),
    OpCode.TENSOR: ((), False),
    OpCode.VECTOR: ((), False),
    OpCode.MATRIX: ((), False),
}


@dataclass(frozen=True)
class ProgramGraphNode:
    id: NodeId
    operation: OpCode
    inputs: tuple[NodeId, ...]
    type: ConcreteType
    source_line: int
    source_name: str | None = None
    payload: OperationPayload | None = None

    def to_dict(self) -> dict[str, Any]:
        operation: dict[str, Any] = {"code": self.operation.value}
        if isinstance(self.payload, BuiltinOp):
            operation["builtin"] = self.payload.name
        elif isinstance(self.payload, LiteralOp):
            operation["literal"] = self.payload.value
        elif isinstance(self.payload, NamedOp):
            operation["name"] = self.payload.name
        elif isinstance(self.payload, StaticIndicesOp):
            operation["indices"] = list(self.payload.indices)
        result: dict[str, Any] = {
            "id": self.id,
            "operation": operation,
            "inputs": list(self.inputs),
            "type": self.type.mlir,
            "source_line": self.source_line,
        }
        if self.source_name is not None:
            result["source_name"] = self.source_name
        return result


@dataclass(frozen=True)
class TapeSlot:
    value: NodeId
    offset: int
    size: int
    alignment: int
    type: ConcreteType

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "offset": self.offset,
            "size": self.size,
            "alignment": self.alignment,
            "type": self.type.mlir,
        }


@dataclass(frozen=True)
class SemanticProgramGraph:
    entry: str
    structs: tuple[tuple[str, tuple[tuple[str, ConcreteType], ...]], ...]
    nodes: tuple[ProgramGraphNode, ...]
    outputs: tuple[NodeId, ...]
    storage_outputs: tuple[tuple[str, NodeId], ...]
    wrt: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry": self.entry,
            "structs": [
                {
                    "name": name,
                    "fields": [{"name": field_name, "type": field_type.mlir} for field_name, field_type in fields],
                }
                for name, fields in self.structs
            ],
            "nodes": [node.to_dict() for node in self.nodes],
            "outputs": list(self.outputs),
            "storage_outputs": [{"parameter": parameter, "value": value} for parameter, value in self.storage_outputs],
            "wrt": list(self.wrt),
        }


@dataclass(frozen=True)
class ReversePlan:
    saved_values: tuple[NodeId, ...]
    reverse_order: tuple[NodeId, ...]
    reverse_dependencies: tuple[tuple[NodeId, tuple[NodeId, ...]], ...]
    cotangent_paths: tuple[str, ...]
    gradient_paths: tuple[str, ...]
    derivative_rules: tuple[str, ...]
    derivative_rules_version: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "saved_values": list(self.saved_values),
            "reverse_order": list(self.reverse_order),
            "reverse_dependencies": [
                {"value": value, "consumers": list(consumers)} for value, consumers in self.reverse_dependencies
            ],
            "cotangent_paths": list(self.cotangent_paths),
            "gradient_paths": list(self.gradient_paths),
            "derivative_rules": list(self.derivative_rules),
            "derivative_rules_version": self.derivative_rules_version,
        }


@dataclass(frozen=True)
class TapeLayout:
    slots: tuple[TapeSlot, ...]
    bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "slots": [slot.to_dict() for slot in self.slots],
            "bytes": self.bytes,
        }


@dataclass(frozen=True)
class AccumulationPlan:
    path: str
    mode: AccumulationMode
    evidence: tuple[AccessPatternEvidence, ...]
    invocation_axes: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "mode": self.mode.value,
            "evidence": [item.value for item in self.evidence],
            "invocation_axes": list(self.invocation_axes),
        }


@dataclass(frozen=True)
class LaunchPlan:
    workgroup_size: tuple[int, int, int]
    accumulation_plans: tuple[AccumulationPlan, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "workgroup_size": list(self.workgroup_size),
            "accumulation_plans": [plan.to_dict() for plan in self.accumulation_plans],
        }


@dataclass(frozen=True)
class AutodiffProgram:
    semantic: SemanticProgramGraph
    reverse: ReversePlan
    tape: TapeLayout
    launch: LaunchPlan

    def to_dict(self) -> dict[str, Any]:
        return {
            "semantic": self.semantic.to_dict(),
            "reverse": self.reverse.to_dict(),
            "tape": self.tape.to_dict(),
        }

    @property
    def identity(self) -> str:
        return hashlib.sha256(canonical_json(self.to_dict()).encode("utf-8")).hexdigest()


def _floating_paths(
    value_type: ConcreteType,
    structs: StructFields,
    prefix: tuple[str, ...] = (),
) -> tuple[str, ...]:
    if value_type.kind == "scalar":
        return (".".join(prefix),) if value_type.is_float else ()
    if value_type.kind in {"tensor", "tensor_view"}:
        element = value_type.arguments[0]
        if not isinstance(element, ConcreteType):
            return ()
        # A Tensor is one logical cotangent leaf; its shape is carried by type.
        return (".".join(prefix),) if _floating_paths(element, structs, prefix) else ()
    if value_type.kind == "tuple":
        return tuple(
            path
            for index, element in enumerate(value_type.arguments)
            if isinstance(element, ConcreteType)
            for path in _floating_paths(element, structs, (*prefix, str(index)))
        )
    if value_type.kind == "struct":
        return tuple(
            path
            for name, field in structs[value_type.name]
            for path in _floating_paths(field, structs, (*prefix, name))
        )
    return ()


class _ProgramGraphBuilder:
    def __init__(
        self,
        function: TypedFunctionInstance,
        functions: Mapping[str, TypedFunctionInstance],
        wrt: tuple[str, ...],
        structs: StructFields,
        derivative_rules_version: int,
        workgroup_size: tuple[int, int, int],
        error: Error,
    ):
        self.function = function
        self.functions = functions
        self.wrt = wrt
        self.structs = structs
        self.derivative_rules_version = derivative_rules_version
        self.workgroup_size = workgroup_size
        self.error = error
        self.nodes: list[ProgramGraphNode] = []
        self.environment: dict[str, NodeId] = {}
        self.expression_types: dict[int, ConcreteType] = {}
        self.outputs: tuple[NodeId, ...] = ()
        self.saved: set[NodeId] = set()
        self.active_helpers: list[str] = []
        for typed_function in functions.values():
            self._index_expressions(typed_function.body)
        self.builtin_parameters = {
            parameter.name: parameter.builtin for parameter in function.parameters if parameter.builtin is not None
        }

    def _index_expressions(self, statements: tuple[TypedStatement, ...]) -> None:
        for statement in statements:
            self.expression_types.update(
                (id(expression.source), expression.type) for expression in statement.expressions
            )
            self._index_expressions(statement.children)

    def _add(
        self,
        operation: OpCode,
        inputs: tuple[NodeId, ...],
        value_type: ConcreteType,
        source: ast.AST,
        source_name: str | None = None,
        payload: OperationPayload | None = None,
    ) -> NodeId:
        node_id = NodeId(len(self.nodes))
        self.nodes.append(
            ProgramGraphNode(
                node_id,
                operation,
                inputs,
                value_type,
                getattr(source, "lineno", 0),
                source_name,
                payload,
            )
        )
        return node_id

    def _type(self, expression: ast.expr) -> ConcreteType:
        value_type = self.expression_types.get(id(expression))
        if value_type is None:
            raise self.error(expression, "autodiff ProgramGraph expression has no concrete type")
        return value_type

    def _expression(self, expression: ast.expr) -> NodeId:
        if isinstance(expression, ast.Name):
            try:
                return self.environment[expression.id]
            except KeyError:
                raise self.error(expression, f"autodiff ProgramGraph cannot resolve '{expression.id}'") from None
        if isinstance(expression, ast.Constant):
            if not isinstance(expression.value, (bool, int, float)):
                raise self.error(expression, "autodiff constants must be numeric or Boolean")
            return self._add(
                OpCode.CONSTANT,
                (),
                self._type(expression),
                expression,
                payload=LiteralOp(expression.value),
            )
        if isinstance(expression, ast.BinOp):
            inputs = (self._expression(expression.left), self._expression(expression.right))
        elif isinstance(expression, ast.UnaryOp):
            inputs = (self._expression(expression.operand),)
        elif isinstance(expression, (ast.Tuple, ast.List)):
            inputs = tuple(self._expression(item) for item in expression.elts)
        elif isinstance(expression, ast.Attribute):
            inputs = (self._expression(expression.value),)
        elif isinstance(expression, ast.Subscript):
            indices = expression.slice.elts if isinstance(expression.slice, ast.Tuple) else [expression.slice]
            all_static = all(
                isinstance(index, ast.Constant) and isinstance(index.value, int) and not isinstance(index.value, bool)
                for index in indices
            )
            inputs = (
                (self._expression(expression.value),)
                if all_static
                else (self._expression(expression.value), *(self._expression(index) for index in indices))
            )
        elif isinstance(expression, ast.Compare):
            if len(expression.ops) != 1:
                raise self.error(expression, "autodiff supports one comparison per condition")
            inputs = (
                self._expression(expression.left),
                self._expression(expression.comparators[0]),
            )
        elif isinstance(expression, ast.IfExp):
            inputs = (
                self._expression(expression.test),
                self._expression(expression.body),
                self._expression(expression.orelse),
            )
        elif isinstance(expression, ast.Call):
            if expression.keywords:
                raise self.error(expression, "autodiff does not support keyword arguments inside differentiated code")
            name = (dotted_name(expression.func) or "").split(".")[-1]
            helper = self.functions.get(name)
            if helper is not None and helper is not self.function:
                return self._inline_helper(expression, helper)
            inputs = tuple(self._expression(item) for item in expression.args)
        else:
            raise self.error(
                expression,
                f"autodiff derivative rule is unavailable for {type(expression).__name__}",
            )
        operation, payload = self._operation(expression)
        value_type = self._type(expression)
        if operation in {OpCode.ADD, OpCode.SUB, OpCode.MUL, OpCode.DIV, OpCode.POW}:
            inputs = tuple(self._broadcast_input(value, value_type, expression) for value in inputs)
        rule = (
            ((), False)
            if operation in {OpCode.FIELD, OpCode.INDEX, OpCode.STRUCT, OpCode.COMPARE, OpCode.INDEX_DYNAMIC}
            else _DERIVATIVE_RULES.get(operation)
        )
        if rule is None:
            operation_name = (
                payload.name if operation is OpCode.INTRINSIC and isinstance(payload, NamedOp) else operation.value
            )
            raise self.error(expression, f"autodiff derivative rule is unavailable for '{operation_name}'")
        node_id = self._add(operation, inputs, value_type, expression, payload=payload)
        if operation is OpCode.INDEX_DYNAMIC:
            self.saved.update(inputs[1:])
        saved_operands, save_result = rule
        for index in saved_operands:
            if index >= len(inputs):
                raise self.error(expression, f"invalid built-in derivative rule for '{operation.value}'")
            self.saved.add(inputs[index])
        if save_result:
            self.saved.add(node_id)
        return node_id

    def _broadcast_input(self, value: NodeId, target: ConcreteType, source: ast.AST) -> NodeId:
        value_type = self.nodes[value].type
        if value_type == target:
            return value
        if target.kind != "tensor":
            raise self.error(source, f"autodiff cannot broadcast {value_type.mlir} to {target.mlir}")
        if value_type.kind == "scalar":
            return self._add(OpCode.SPLAT, (value,), target, source)
        if value_type.kind == "tensor":
            return self._add(OpCode.BROADCAST, (value,), target, source)
        raise self.error(source, f"autodiff cannot broadcast {value_type.mlir} to {target.mlir}")

    def _inline_helper(self, call: ast.Call, helper: TypedFunctionInstance) -> NodeId:
        if helper.symbol in self.active_helpers:
            raise self.error(call, f"autodiff does not support recursive helper '{helper.symbol}'")
        if helper.effects:
            raise self.error(call, f"autodiff helper '{helper.symbol}' must be pure")
        if len(call.args) != len(helper.parameters):
            raise self.error(call, f"autodiff helper '{helper.symbol}' argument count does not match")
        arguments = tuple(self._expression(argument) for argument in call.args)
        previous_environment = self.environment
        previous_outputs = self.outputs
        self.environment = {
            parameter.name: argument for parameter, argument in zip(helper.parameters, arguments, strict=True)
        }
        self.outputs = ()
        self.active_helpers.append(helper.symbol)
        try:
            self._statements(helper.source.body)
            if len(self.outputs) != 1:
                raise self.error(call, f"autodiff helper '{helper.symbol}' must return one Value")
            return self.outputs[0]
        finally:
            self.active_helpers.pop()
            self.environment = previous_environment
            self.outputs = previous_outputs

    def _operation(self, expression: ast.expr) -> tuple[OpCode, OperationPayload | None]:
        if isinstance(expression, ast.BinOp):
            operation = {
                ast.Add: OpCode.ADD,
                ast.Sub: OpCode.SUB,
                ast.Mult: OpCode.MUL,
                ast.Div: OpCode.DIV,
                ast.Pow: OpCode.POW,
            }.get(type(expression.op))
            if operation is None:
                raise self.error(expression, f"autodiff has no opcode for {type(expression.op).__name__}")
            return operation, None
        if isinstance(expression, ast.UnaryOp):
            return (OpCode.NEG if isinstance(expression.op, ast.USub) else OpCode.IDENTITY), None
        if isinstance(expression, ast.Call):
            name = (dotted_name(expression.func) or "").split(".")[-1]
            if name in self.structs:
                return OpCode.STRUCT, NamedOp(name)
            try:
                return OpCode(name), None
            except ValueError:
                return OpCode.INTRINSIC, NamedOp(name)
        if isinstance(expression, ast.Attribute):
            return OpCode.FIELD, NamedOp(expression.attr)
        if isinstance(expression, ast.Subscript):
            indices = expression.slice.elts if isinstance(expression.slice, ast.Tuple) else [expression.slice]
            values: list[int] = []
            for index in indices:
                if (
                    not isinstance(index, ast.Constant)
                    or not isinstance(index.value, int)
                    or isinstance(index.value, bool)
                ):
                    return OpCode.INDEX_DYNAMIC, None
                values.append(index.value)
            return OpCode.INDEX, StaticIndicesOp(tuple(values))
        if isinstance(expression, ast.Compare):
            return OpCode.COMPARE, NamedOp(type(expression.ops[0]).__name__)
        if isinstance(expression, ast.IfExp):
            return OpCode.CONDITIONAL, None
        return OpCode.TUPLE, None

    def build(self) -> AutodiffProgram:
        for parameter in self.function.parameters:
            builtin = self.builtin_parameters.get(parameter.name)
            operation = OpCode.BUILTIN if builtin is not None else OpCode.PARAMETER
            node_id = self._add(
                operation,
                (),
                parameter.type,
                self.function.source,
                parameter.name,
                BuiltinOp(builtin) if builtin is not None else None,
            )
            self.environment[parameter.name] = node_id
        self._statements(self.function.source.body)
        if not self.outputs:
            raise self.error(self.function.source, "autodiff ProgramGraph has no returned Value")
        consumers: dict[NodeId, list[NodeId]] = {node.id: [] for node in self.nodes}
        for node in self.nodes:
            for value in node.inputs:
                consumers[value].append(node.id)
        nodes_by_id = {node.id: node for node in self.nodes}
        active: set[NodeId] = set()

        def activate(value: NodeId) -> None:
            if value in active:
                return
            active.add(value)
            for dependency in nodes_by_id[value].inputs:
                activate(dependency)

        for output in self.outputs:
            activate(output)
        parameter_nodes = {
            node.source_name: node.id
            for node in self.nodes
            if node.operation in {OpCode.PARAMETER, OpCode.BUILTIN} and node.source_name is not None
        }
        storage_outputs = tuple(
            (parameter.name, self.environment[parameter.name])
            for parameter in self.function.parameters
            if parameter.type.kind == "tensor_view"
            and self.environment[parameter.name] != parameter_nodes[parameter.name]
        )
        for _, value in storage_outputs:
            activate(value)
        # TensorView nodes name resource state, not copyable Values.  Reverse
        # execution replays their resource effects and tapes only scalar/Tensor
        # values (including branch predicates and dynamic indices).
        saved_values = tuple(
            value for value in sorted(self.saved & active) if nodes_by_id[value].type.kind != "tensor_view"
        )

        def global_index_axis(value: NodeId) -> int | None:
            node = nodes_by_id[value]
            if (
                node.operation is not OpCode.INDEX
                or len(node.inputs) != 1
                or not isinstance(node.payload, StaticIndicesOp)
                or len(node.payload.indices) != 1
            ):
                return None
            source = nodes_by_id[node.inputs[0]]
            if source.operation is not OpCode.BUILTIN or source.payload != BuiltinOp("global_invocation_id"):
                return None
            axis = node.payload.indices[0]
            return axis if 0 <= axis < 3 else None

        def global_index_mapping(indices: tuple[NodeId, ...]) -> tuple[tuple[str, int], ...] | None:
            mapping: list[tuple[str, int]] = []
            for value in indices:
                axis = global_index_axis(value)
                if axis is not None:
                    mapping.append(("gid", axis))
                elif nodes_by_id[value].operation is OpCode.CONSTANT:
                    mapping.append(("constant", int(value)))
                else:
                    return None
            return tuple(mapping)

        def storage_owner(value: NodeId) -> str | None:
            current = nodes_by_id[value]
            while current.operation is not OpCode.PARAMETER:
                if not current.inputs:
                    return None
                current = nodes_by_id[current.inputs[0]]
            return current.source_name if current.type.kind == "tensor_view" else None

        accumulation_plans: list[AccumulationPlan] = []
        for path in self.wrt:
            parameter_name = path.split(".", 1)[0]
            parameter = next(value for value in self.function.parameters if value.name == parameter_name)
            if parameter.type.kind != "tensor_view":
                accumulation_plans.append(
                    AccumulationPlan(
                        path,
                        AccumulationMode.REDUCE_SUM,
                        (AccessPatternEvidence.SHARED_VALUE,),
                    )
                )
                continue
            evidence: set[AccessPatternEvidence] = set()
            invocation_axes: set[int] = set()
            access_mappings: list[tuple[tuple[str, int], ...]] = []
            for node in self.nodes:
                if (
                    node.id not in active
                    or node.operation not in {OpCode.INDEX, OpCode.INDEX_DYNAMIC, OpCode.STORE_DYNAMIC}
                    or storage_owner(node.inputs[0]) != parameter_name
                ):
                    continue
                if node.operation is OpCode.INDEX:
                    evidence.add(AccessPatternEvidence.STATIC_INDEX_CONFLICT)
                elif node.operation is OpCode.INDEX_DYNAMIC:
                    mapping = global_index_mapping(node.inputs[1:])
                    axes = {value for kind, value in mapping if kind == "gid"} if mapping is not None else set()
                    if axes:
                        evidence.add(AccessPatternEvidence.INJECTIVE_GLOBAL_INDEX)
                        invocation_axes.update(axes)
                        access_mappings.append(mapping)
                    else:
                        evidence.add(AccessPatternEvidence.NON_INJECTIVE_INDEX)
                else:
                    mapping = global_index_mapping(node.inputs[1:-1])
                    axes = {value for kind, value in mapping if kind == "gid"} if mapping is not None else set()
                    if axes:
                        evidence.add(AccessPatternEvidence.INJECTIVE_GLOBAL_INDEX)
                        invocation_axes.update(axes)
                        access_mappings.append(mapping)
                    else:
                        evidence.add(AccessPatternEvidence.NON_INJECTIVE_INDEX)
            mode = AccumulationMode.SCATTER_ADD
            if (
                invocation_axes == {0, 1, 2}
                and access_mappings
                and all(mapping == access_mappings[0] for mapping in access_mappings[1:])
                and AccessPatternEvidence.STATIC_INDEX_CONFLICT not in evidence
                and AccessPatternEvidence.NON_INJECTIVE_INDEX not in evidence
            ):
                evidence.add(AccessPatternEvidence.DISJOINT_SCATTER)
            accumulation_plans.append(
                AccumulationPlan(
                    path,
                    mode,
                    tuple(sorted(evidence, key=lambda item: item.value)),
                    tuple(sorted(invocation_axes)),
                )
            )
        tape_slots, tape_bytes = self._plan_tape(saved_values, nodes_by_id)
        reverse_order = tuple(
            node.id
            for node in reversed(self.nodes)
            if node.id in active and node.operation not in {OpCode.CONSTANT, OpCode.PARAMETER, OpCode.BUILTIN}
        )
        derivative_rules = tuple(
            sorted(
                {
                    node.operation.value
                    for node in self.nodes
                    if node.id in active and node.operation not in {OpCode.CONSTANT, OpCode.PARAMETER, OpCode.BUILTIN}
                }
            )
        )
        result_type = self.function.result_type
        assert result_type is not None
        cotangent_paths = tuple(sorted(_floating_paths(result_type, self.structs, ("output",))))
        gradient_paths = tuple(sorted(path for wrt_path in self.wrt for path in self._gradient_paths(wrt_path)))
        return AutodiffProgram(
            SemanticProgramGraph(
                self.function.symbol,
                tuple((name, self.structs[name]) for name in sorted(self.structs)),
                tuple(self.nodes),
                self.outputs,
                storage_outputs,
                self.wrt,
            ),
            ReversePlan(
                saved_values,
                reverse_order,
                tuple((node.id, tuple(consumers[node.id])) for node in self.nodes if consumers[node.id]),
                cotangent_paths,
                gradient_paths,
                derivative_rules,
                self.derivative_rules_version,
            ),
            TapeLayout(tape_slots, tape_bytes),
            LaunchPlan(
                self.workgroup_size,
                tuple(accumulation_plans),
            ),
        )

    def _plan_tape(
        self,
        saved_values: tuple[NodeId, ...],
        nodes: Mapping[NodeId, ProgramGraphNode],
    ) -> tuple[tuple[TapeSlot, ...], int]:
        offset = 0
        tape_alignment = 1
        slots: list[TapeSlot] = []
        for value in saved_values:
            size, alignment = self._value_layout(nodes[value].type)
            offset = ((offset + alignment - 1) // alignment) * alignment
            slots.append(TapeSlot(value, offset, size, alignment, nodes[value].type))
            offset += size
            tape_alignment = max(tape_alignment, alignment)
        total = ((offset + tape_alignment - 1) // tape_alignment) * tape_alignment
        return tuple(slots), total

    def _value_layout(self, value_type: ConcreteType) -> tuple[int, int]:
        if value_type.kind == "scalar":
            size = max(SCALAR_TYPES[value_type.name].width // 8, 1)
            return size, size
        if value_type.kind == "tensor":
            element = value_type.arguments[0]
            assert isinstance(element, ConcreteType)
            count = 1
            for extent in value_type.arguments[1:]:
                if not isinstance(extent, int) or extent <= 0:
                    raise self.error(
                        self.function.source,
                        "autodiff tape requires positive static Tensor shapes",
                    )
                count *= extent
            element_size, element_alignment = self._value_layout(element)
            stride = ((element_size + element_alignment - 1) // element_alignment) * element_alignment
            return count * stride, element_alignment
        if value_type.kind in {"tuple", "struct"}:
            elements = (
                tuple(element for element in value_type.arguments if isinstance(element, ConcreteType))
                if value_type.kind == "tuple"
                else tuple(field for _, field in self.structs[value_type.name])
            )
            offset = 0
            aggregate_alignment = 1
            for element in elements:
                size, alignment = self._value_layout(element)
                offset = ((offset + alignment - 1) // alignment) * alignment
                offset += size
                aggregate_alignment = max(aggregate_alignment, alignment)
            return (
                ((offset + aggregate_alignment - 1) // aggregate_alignment) * aggregate_alignment,
                aggregate_alignment,
            )
        raise self.error(
            self.function.source,
            f"autodiff cannot save {value_type.kind} in the typed tape",
        )

    def _gradient_paths(self, path: str) -> tuple[str, ...]:
        components = path.split(".")
        parameter = next(parameter for parameter in self.function.parameters if parameter.name == components[0])
        value_type = parameter.type
        prefix = components[:1]
        for component in components[1:]:
            prefix.append(component)
            if value_type.kind == "struct":
                value_type = dict(self.structs[value_type.name])[component]
            elif value_type.kind == "tuple":
                value_type = value_type.arguments[int(component)]  # type: ignore[assignment]
                assert isinstance(value_type, ConcreteType)
        return _floating_paths(value_type, self.structs, tuple(prefix))

    def _statements(self, statements: list[ast.stmt]) -> None:
        for statement_index, statement in enumerate(statements):
            if isinstance(statement, ast.Assign):
                if len(statement.targets) != 1:
                    raise self.error(statement, "autodiff assignments require one target")
                target = statement.targets[0]
                if isinstance(target, ast.Name):
                    self.environment[target.id] = self._expression(statement.value)
                elif isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
                    owner = target.value.id
                    storage = self.environment.get(owner)
                    if storage is None or self.nodes[storage].type.kind != "tensor_view":
                        raise self.error(target, "autodiff indexed assignment requires TensorView storage")
                    value = self._expression(statement.value)
                    index_operation, index_payload = self._operation(target)
                    if index_operation is OpCode.INDEX_DYNAMIC:
                        indices = target.slice.elts if isinstance(target.slice, ast.Tuple) else [target.slice]
                        index_values = tuple(self._expression(index) for index in indices)
                        operation = OpCode.STORE_DYNAMIC
                        payload = None
                        self.saved.update(index_values)
                    else:
                        index_values = ()
                        operation = OpCode.STORE
                        payload = index_payload
                    storage_type = self.nodes[storage].type
                    self.environment[owner] = self._add(
                        operation,
                        (storage, *index_values, value),
                        storage_type,
                        statement,
                        owner,
                        payload,
                    )
                else:
                    raise self.error(statement, "autodiff assignment target must be a local or TensorView element")
            elif isinstance(statement, ast.AnnAssign):
                if not isinstance(statement.target, ast.Name) or statement.value is None:
                    raise self.error(statement, "autodiff supports only initialized local annotations")
                self.environment[statement.target.id] = self._expression(statement.value)
            elif isinstance(statement, ast.AugAssign):
                if not isinstance(statement.target, ast.Name):
                    raise self.error(statement, "autodiff supports only simple augmented assignments")
                synthetic = ast.BinOp(
                    left=ast.copy_location(ast.Name(id=statement.target.id, ctx=ast.Load()), statement.target),
                    op=statement.op,
                    right=statement.value,
                )
                ast.copy_location(synthetic, statement)
                # Augmented assignments have no standalone typed synthetic node.
                left = self.environment[statement.target.id]
                right = self._expression(statement.value)
                operation, payload = self._operation(synthetic)
                rule = _DERIVATIVE_RULES.get(operation)
                if rule is None:
                    raise self.error(statement, f"autodiff derivative rule is unavailable for '{operation.value}'")
                result = self._add(operation, (left, right), self.nodes[left].type, statement, payload=payload)
                for index in rule[0]:
                    self.saved.add((left, right)[index])
                if rule[1]:
                    self.saved.add(result)
                self.environment[statement.target.id] = result
            elif isinstance(statement, ast.Return) and statement.value is not None:
                self.outputs = (self._expression(statement.value),)
                break
            elif isinstance(statement, ast.If):
                if any(isinstance(node, ast.Return) for node in ast.walk(statement)):
                    self._returning_if(statement, statements[statement_index + 1 :])
                    break
                self._if(statement)
            elif isinstance(statement, ast.For):
                self._for(statement)
            elif isinstance(statement, ast.Pass):
                continue
            else:
                raise self.error(
                    statement,
                    f"autodiff bounded control-flow lowering is unavailable for {type(statement).__name__}",
                )

    def _returning_if(self, statement: ast.If, continuation: list[ast.stmt]) -> None:
        if any(parameter.type.kind == "tensor_view" for parameter in self.function.parameters):
            raise self.error(
                statement,
                "autodiff branch-local returns with Storage effects are unavailable",
            )
        condition = self._expression(statement.test)
        before = dict(self.environment)
        previous_outputs = self.outputs

        def branch_output(branch: list[ast.stmt]) -> NodeId:
            self.environment = dict(before)
            self.outputs = ()
            self._statements(branch)
            if not self.outputs:
                self._statements(continuation)
            if len(self.outputs) != 1:
                raise self.error(statement, "autodiff early-return branch does not produce one Value")
            return self.outputs[0]

        true_output = branch_output(statement.body)
        false_output = branch_output(statement.orelse)
        true_type = self.nodes[true_output].type
        false_type = self.nodes[false_output].type
        if true_type != false_type:
            raise self.error(statement, "autodiff early-return branches have incompatible types")
        self.environment = before
        self.outputs = (
            self._add(
                OpCode.CONDITIONAL,
                (condition, true_output, false_output),
                true_type,
                statement,
                "return",
            ),
        )
        self.saved.add(condition)
        if previous_outputs:
            raise self.error(statement, "autodiff early-return state is ambiguous")

    def _if(self, statement: ast.If) -> None:
        condition = self._expression(statement.test)
        before = dict(self.environment)

        self.environment = dict(before)
        self._statements(statement.body)
        true_environment = dict(self.environment)

        self.environment = dict(before)
        self._statements(statement.orelse)
        false_environment = dict(self.environment)

        merged = dict(before)
        for name in sorted(set(true_environment) | set(false_environment)):
            true_value = true_environment.get(name, before.get(name))
            false_value = false_environment.get(name, before.get(name))
            if true_value is None or false_value is None:
                raise self.error(statement, f"autodiff branch local '{name}' is not defined on every path")
            if true_value == false_value:
                merged[name] = true_value
                continue
            true_type = self.nodes[true_value].type
            false_type = self.nodes[false_value].type
            if true_type != false_type:
                raise self.error(statement, f"autodiff branch local '{name}' has incompatible types")
            merged[name] = self._add(
                OpCode.CONDITIONAL,
                (condition, true_value, false_value),
                true_type,
                statement,
                name,
            )
            self.saved.add(condition)
        self.environment = merged

    def _for(self, statement: ast.For) -> None:
        if not isinstance(statement.target, ast.Name):
            raise self.error(statement.target, "autodiff bounded loops require one induction variable")
        if (
            not isinstance(statement.iter, ast.Call)
            or (dotted_name(statement.iter.func) or "").split(".")[-1] != "range"
            or statement.iter.keywords
            or not 1 <= len(statement.iter.args) <= 3
            or any(
                not isinstance(argument, ast.Constant)
                or not isinstance(argument.value, int)
                or isinstance(argument.value, bool)
                for argument in statement.iter.args
            )
        ):
            raise self.error(statement.iter, "autodiff loops require a literal range with one to three arguments")
        if statement.orelse:
            raise self.error(statement, "autodiff loop else lowering is unavailable")
        if any(isinstance(node, (ast.Continue, ast.Return)) for node in ast.walk(statement)):
            raise self.error(statement, "autodiff bounded loops do not support continue or return")
        arguments = [int(argument.value) for argument in statement.iter.args if isinstance(argument, ast.Constant)]
        try:
            iterations = tuple(range(*arguments))
        except ValueError as error:
            raise self.error(statement.iter, f"invalid autodiff loop range: {error}") from None
        if len(iterations) > 1024:
            raise self.error(statement.iter, "autodiff loop exceeds the static iteration cap 1024")
        break_guards = [
            child
            for child in ast.walk(statement)
            if isinstance(child, ast.If) and len(child.body) == 1 and isinstance(child.body[0], ast.Break)
        ]
        if any(isinstance(node, ast.Break) for node in ast.walk(statement)):
            if (
                len(break_guards) != 1
                or not statement.body
                or statement.body[0] is not break_guards[0]
                or break_guards[0].orelse
                or any(isinstance(node, ast.Break) for child in statement.body[1:] for node in ast.walk(child))
            ):
                raise self.error(
                    statement,
                    "autodiff dynamic loops require one leading 'if condition: break' guard",
                )
            if any(parameter.type.kind == "tensor_view" for parameter in self.function.parameters):
                raise self.error(
                    statement,
                    "autodiff dynamic loops with Storage effects are unavailable",
                )
            if len(iterations) > 256:
                raise self.error(statement.iter, "autodiff dynamic loop exceeds the iteration cap 256")
            guard = break_guards[0]
            body = statement.body[1:]

            def unroll(iteration_index: int, environment: dict[str, NodeId]) -> dict[str, NodeId]:
                if iteration_index == len(iterations):
                    return environment
                self.environment = dict(environment)
                induction = self._add(
                    OpCode.CONSTANT,
                    (),
                    ConcreteType("scalar", "i32"),
                    statement.target,
                    statement.target.id,
                    LiteralOp(iterations[iteration_index]),
                )
                self.environment[statement.target.id] = induction
                condition = self._expression(guard.test)
                before = dict(self.environment)
                self._statements(body)
                continued = unroll(iteration_index + 1, dict(self.environment))
                merged = dict(before)
                for name in sorted(set(before) | set(continued)):
                    stopped_value = before.get(name)
                    continued_value = continued.get(name)
                    if stopped_value is None or continued_value is None:
                        raise self.error(statement, f"autodiff loop local '{name}' is not defined on every path")
                    if stopped_value == continued_value:
                        merged[name] = stopped_value
                        continue
                    stopped_type = self.nodes[stopped_value].type
                    continued_type = self.nodes[continued_value].type
                    if stopped_type != continued_type:
                        raise self.error(statement, f"autodiff loop local '{name}' has incompatible types")
                    merged[name] = self._add(
                        OpCode.CONDITIONAL,
                        (condition, stopped_value, continued_value),
                        stopped_type,
                        guard,
                        name,
                    )
                    self.saved.add(condition)
                return merged

            self.environment = unroll(0, dict(self.environment))
            return
        for value in iterations:
            induction = self._add(
                OpCode.CONSTANT,
                (),
                ConcreteType("scalar", "i32"),
                statement.target,
                statement.target.id,
                LiteralOp(value),
            )
            self.environment[statement.target.id] = induction
            self._statements(statement.body)


def build_program_graph(
    function: TypedFunctionInstance,
    functions: Mapping[str, TypedFunctionInstance],
    wrt: tuple[str, ...],
    structs: StructFields,
    derivative_rules_version: int,
    workgroup_size: tuple[int, int, int],
    error: Error,
) -> AutodiffProgram:
    return _ProgramGraphBuilder(
        function,
        functions,
        wrt,
        structs,
        derivative_rules_version,
        workgroup_size,
        error,
    ).build()


__all__ = [
    "AccessPatternEvidence",
    "AccumulationMode",
    "AccumulationPlan",
    "BuiltinOp",
    "AutodiffProgram",
    "LaunchPlan",
    "LiteralOp",
    "NamedOp",
    "NodeId",
    "OpCode",
    "ProgramGraphNode",
    "ReversePlan",
    "SemanticProgramGraph",
    "StaticIndicesOp",
    "TapeLayout",
    "TapeSlot",
    "build_program_graph",
]
