from __future__ import annotations

from collections.abc import Callable
from enum import Enum, auto
from itertools import product

from .autodiff import (
    AccessPatternEvidence,
    AccumulationMode,
    AutodiffProgram,
    BuiltinOp,
    LiteralOp,
    NamedOp,
    NodeId,
    OpCode,
    ProgramGraphNode,
    StaticIndicesOp,
    storage_parameter_owner,
)
from .autodiff_native_abi import (
    ResourceBindingPlan,
    backward_resources,
    forward_resources,
    gradient_resource_type,
    gradient_type,
    launch_value_type,
    native_type,
    resource_type,
    value_dtype,
)
from .autodiff_native_common import (
    AutodiffNativeLoweringError,
    Names,
    builtin_argument,
    entry_header,
    index_constants,
    invocation_indices,
    module,
)
from .autodiff_native_math import emit_forward_math, emit_math_vjp
from .autodiff_profiles import AutodiffProfilePlan
from .model import ConcreteType


class DirectGpuLoweringError(AutodiffNativeLoweringError):
    pass


def _static_indices(node: ProgramGraphNode) -> tuple[int, ...]:
    if not isinstance(node.payload, StaticIndicesOp) or not node.payload.indices:
        raise DirectGpuLoweringError(f"native autodiff requires static integer indices for '{node.operation.value}'")
    return node.payload.indices


def _literal(node: ProgramGraphNode) -> str:
    if not isinstance(node.payload, LiteralOp):
        raise DirectGpuLoweringError("native autodiff constant has no literal")
    value = node.payload.value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        text = f"{value:.17g}"
        return text if "." in text or "e" in text.lower() else text + ".0"
    raise DirectGpuLoweringError("native autodiff constant has no literal")


def _value_indices(
    operands: tuple[str, ...],
    operand_types: tuple[ConcreteType, ...],
    names: Names,
    lines: list[str],
) -> tuple[str, ...]:
    results: list[str] = []
    for operand, value_type in zip(operands, operand_types, strict=True):
        if value_type.kind == "scalar" and value_type.name == "index":
            results.append(operand)
            continue
        result = names.fresh()
        lines.append(f"    {result} = arith.index_cast {operand} : {native_type(value_type)} to index")
        results.append(result)
    return tuple(results)


def _emit_operation(
    node: ProgramGraphNode,
    operands: tuple[str, ...],
    operand_types: tuple[ConcreteType, ...],
    names: Names,
    lines: list[str],
) -> str:
    result = names.fresh()
    result_type = native_type(node.type)
    integer = node.type.kind == "scalar" and node.type.name in {"i32", "u32"}
    binary = (
        {
            OpCode.ADD: "arith.addi",
            OpCode.SUB: "arith.subi",
            OpCode.MUL: "arith.muli",
            OpCode.DIV: "arith.divui" if node.type.name == "u32" else "arith.divsi",
        }
        if integer
        else {
            OpCode.ADD: "arith.addf",
            OpCode.SUB: "arith.subf",
            OpCode.MUL: "arith.mulf",
            OpCode.DIV: "arith.divf",
        }
    )
    if node.operation in binary:
        lines.append(f"    {result} = {binary[node.operation]} {operands[0]}, {operands[1]} : {result_type}")
    elif node.operation is OpCode.COMPARE:
        if not isinstance(node.payload, NamedOp):
            raise DirectGpuLoweringError("comparison node is missing its typed payload")
        integer_comparison = operand_types[0].kind == "scalar" and operand_types[0].name in {"i32", "u32"}
        unsigned = operand_types[0].name == "u32"
        predicates = (
            {
                "Eq": "eq",
                "NotEq": "ne",
                "Lt": "ult" if unsigned else "slt",
                "LtE": "ule" if unsigned else "sle",
                "Gt": "ugt" if unsigned else "sgt",
                "GtE": "uge" if unsigned else "sge",
            }
            if integer_comparison
            else {"Eq": "oeq", "NotEq": "une", "Lt": "olt", "LtE": "ole", "Gt": "ogt", "GtE": "oge"}
        )
        predicate = predicates.get(node.payload.name)
        if predicate is None:
            raise DirectGpuLoweringError(f"cannot emit native comparison '{node.payload.name}'")
        comparison = "arith.cmpi" if integer_comparison else "arith.cmpf"
        lines.append(
            f"    {result} = {comparison} {predicate}, {operands[0]}, {operands[1]} : {native_type(operand_types[0])}"
        )
    elif node.operation is OpCode.CONSTANT:
        literal = _literal(node)
        value = f"dense<{literal}>" if node.type.kind == "tensor" else literal
        lines.append(f"    {result} = arith.constant {value} : {result_type}")
    elif node.operation in {OpCode.INDEX, OpCode.INDEX_DYNAMIC}:
        indices = (
            index_constants(_static_indices(node), names, lines)
            if node.operation is OpCode.INDEX
            else _value_indices(operands[1:], operand_types[1:], names, lines)
        )
        lines.append(
            f"    {result} = tensor.extract {operands[0]}[{', '.join(indices)}] : {native_type(operand_types[0])}"
        )
    elif node.operation is OpCode.NEG:
        zero = names.fresh()
        literal = "dense<0.0>" if node.type.kind == "tensor" else "0.0"
        lines.append(f"    {zero} = arith.constant {literal} : {result_type}")
        lines.append(f"    {result} = arith.subf {zero}, {operands[0]} : {result_type}")
    elif node.operation is OpCode.IDENTITY:
        return operands[0]
    elif node.operation is OpCode.SPLAT:
        lines.append(f"    {result} = tensor.splat {operands[0]} : {result_type}")
    elif node.operation is OpCode.BROADCAST:
        lines.append(
            f'    {result} = "vernon.intrinsic"({operands[0]}) {{name = "broadcast"}} : '
            f"({native_type(operand_types[0])}) -> {result_type}"
        )
    elif node.operation in {OpCode.SIN, OpCode.COS, OpCode.EXP, OpCode.LOG, OpCode.SQRT}:
        lines.append(f"    {result} = math.{node.operation.value} {operands[0]} : {result_type}")
    else:
        math_result = emit_forward_math(node, operands, operand_types, names, lines)
        if math_result is None:
            raise DirectGpuLoweringError(f"cannot emit direct value operation '{node.operation.value}'")
        return math_result
    return result


def _resource_arguments(resources: tuple[ResourceBindingPlan, ...]) -> str:
    return ", ".join(resource.argument(binding) for binding, resource in enumerate(resources))


_GLOBAL_ID_TYPE = launch_value_type()


def _global_id_arguments(
    builtin_parameters: tuple[ProgramGraphNode, ...],
) -> tuple[tuple[str, ...], int]:
    arguments = [builtin_argument(node, index) for index, node in enumerate(builtin_parameters)]
    global_id = next(
        (index for index, node in enumerate(builtin_parameters) if node.payload == BuiltinOp("global_invocation_id")),
        None,
    )
    if global_id is None:
        global_id = len(arguments)
        arguments.append(
            f"%builtin{global_id}: {_GLOBAL_ID_TYPE.mlir} "
            '{vernon.interface = "input", vernon.source_name = "__vernon_global_invocation_id", '
            'vernon.builtin = "global_invocation_id"}'
        )
    return tuple(arguments), global_id


def _guard_runtime_grid(
    launch_binding: int,
    global_id: str,
    names: Names,
    lines: list[str],
) -> tuple[tuple[str, str, str], str]:
    xyz = invocation_indices(global_id, _GLOBAL_ID_TYPE.mlir, names, lines)
    zero = names.fresh()
    launch = names.fresh()
    lines.append(f"    {zero} = arith.constant 0 : index")
    lines.append(
        f'    {launch} = "vernon.load"(%resource{launch_binding}, {zero}) : '
        f"({resource_type(_GLOBAL_ID_TYPE, 'read')}, index) -> {_GLOBAL_ID_TYPE.mlir}"
    )
    valid: str | None = None
    for axis, coordinate in enumerate(xyz):
        axis_index = names.fresh()
        extent_u32 = names.fresh()
        extent = names.fresh()
        inside = names.fresh()
        lines.append(f"    {axis_index} = arith.constant {axis} : index")
        lines.append(f"    {extent_u32} = tensor.extract {launch}[{axis_index}] : {_GLOBAL_ID_TYPE.mlir}")
        lines.append(f"    {extent} = arith.index_castui {extent_u32} : i32 to index")
        lines.append(f"    {inside} = arith.cmpi ult, {coordinate}, {extent} : index")
        if valid is None:
            valid = inside
        else:
            combined = names.fresh()
            lines.append(f"    {combined} = arith.andi {valid}, {inside} : i1")
            valid = combined
    assert valid is not None
    return xyz, valid


def _owner(node_id: NodeId, nodes: dict[NodeId, ProgramGraphNode]) -> ProgramGraphNode:
    owner = storage_parameter_owner(node_id, nodes)
    if owner is None:
        raise DirectGpuLoweringError("Storage operation has no TensorView parameter owner")
    return owner


def _physical_indices(
    node: ProgramGraphNode,
    nodes: dict[NodeId, ProgramGraphNode],
    primal: Callable[[NodeId], str],
    names: Names,
    lines: list[str],
) -> tuple[str, ...]:
    if node.operation in {OpCode.INDEX, OpCode.STORE}:
        return index_constants(_static_indices(node), names, lines)
    ids = node.inputs[1:] if node.operation is OpCode.INDEX_DYNAMIC else node.inputs[1:-1]
    results: list[str] = []
    for value_id in ids:
        source = primal(value_id)
        value_type = nodes[value_id].type
        if value_type.kind == "scalar" and value_type.name == "index":
            results.append(source)
            continue
        result = names.fresh()
        lines.append(f"    {result} = arith.index_cast {source} : {native_type(value_type)} to index")
        results.append(result)
    return tuple(results)


def _value_index_vjp(
    node: ProgramGraphNode,
    seed: str,
    primal: Callable[[NodeId], str],
    nodes: dict[NodeId, ProgramGraphNode],
    names: Names,
    lines: list[str],
) -> str:
    value_type = gradient_type(nodes[node.inputs[0]].type)
    element_type = value_type.arguments[0]
    shape = value_type.arguments[1:]
    assert isinstance(element_type, ConcreteType)
    assert all(isinstance(extent, int) for extent in shape)
    zero = _zero(element_type, names, lines)
    static_indices = _static_indices(node) if node.operation is OpCode.INDEX else None
    dynamic_ids = node.inputs[1:] if node.operation is OpCode.INDEX_DYNAMIC else ()
    dynamic_values = tuple(primal(value) for value in dynamic_ids)
    elements: list[str] = []
    for current in product(*(range(extent) for extent in shape)):
        if static_indices is not None:
            elements.append(seed if current == static_indices else zero)
            continue
        comparisons: list[str] = []
        for dynamic, value_id, index in zip(dynamic_values, dynamic_ids, current, strict=True):
            index_type = nodes[value_id].type
            constant = names.fresh()
            lines.append(f"    {constant} = arith.constant {index} : {native_type(index_type)}")
            comparison = names.fresh()
            lines.append(f"    {comparison} = arith.cmpi eq, {dynamic}, {constant} : {native_type(index_type)}")
            comparisons.append(comparison)
        condition = comparisons[0]
        for comparison in comparisons[1:]:
            combined = names.fresh()
            lines.append(f"    {combined} = arith.andi {condition}, {comparison} : i1")
            condition = combined
        selected = names.fresh()
        lines.append(f"    {selected} = arith.select {condition}, {seed}, {zero} : {element_type.mlir}")
        elements.append(selected)
    result = names.fresh()
    lines.append(
        f'    {result} = "vernon.intrinsic"({", ".join(elements)}) {{name = "construct"}} : '
        f"({', '.join(element_type.mlir for _ in elements)}) -> {native_type(value_type)}"
    )
    return result


def _storage_effects(program: AutodiffProgram) -> tuple[str, ...]:
    nodes = {node.id: node for node in program.semantic.nodes}
    effects: list[str] = []
    for node in program.semantic.nodes:
        if node.operation not in {OpCode.INDEX, OpCode.INDEX_DYNAMIC, OpCode.STORE, OpCode.STORE_DYNAMIC}:
            continue
        if nodes[node.inputs[0]].type.kind != "tensor_view":
            continue
        owner = _owner(node.inputs[0], nodes).source_name or ""
        kind = "write" if node.operation in {OpCode.STORE, OpCode.STORE_DYNAMIC} else "read"
        if node.operation in {OpCode.INDEX, OpCode.STORE}:
            values = ", ".join(str(value) for value in _static_indices(node))
            effects.append(
                f'{{kind = "{kind}", owner = "{owner}", region = "element", indices = array<i64: {values}>}}'
            )
        else:
            effects.append(f'{{kind = "{kind}", owner = "{owner}", region = "unknown"}}')
    return tuple(effects)


class _ForwardPhase(Enum):
    CAPTURE_TAPE = auto()
    COMMIT_STORAGE = auto()
    OBSERVE_OUTPUT = auto()


class _GpuForwardEmitter:
    def __init__(
        self,
        nodes: dict[NodeId, ProgramGraphNode],
        saved_values: tuple[NodeId, ...],
        names: Names,
        base_values: dict[NodeId, str],
    ) -> None:
        self.nodes = nodes
        self.saved_values = frozenset(saved_values)
        self.names = names
        self.caches = {phase: dict(base_values) for phase in _ForwardPhase}
        self.dependency_cache: dict[NodeId, frozenset[NodeId]] = {}

    def emit(self, value: NodeId, phase: _ForwardPhase, lines: list[str]) -> str:
        return self._emit(value, phase, self.caches[phase], lines)

    def captured(self, value: NodeId) -> str:
        return self.caches[_ForwardPhase.CAPTURE_TAPE][value]

    def _dependencies(self, value: NodeId) -> frozenset[NodeId]:
        cached = self.dependency_cache.get(value)
        if cached is not None:
            return cached
        result = frozenset((value,)).union(*(self._dependencies(operand) for operand in self.nodes[value].inputs))
        self.dependency_cache[value] = result
        return result

    def _zero(self, value: NodeId, lines: list[str]) -> str:
        result = self.names.fresh()
        value_type = self.nodes[value].type
        literal = (
            "0"
            if value_type.kind == "scalar" and value_type.name == "bool"
            else "dense<0.0>"
            if value_type.kind == "tensor"
            else "0.0"
        )
        lines.append(f"    {result} = arith.constant {literal} : {native_type(value_type)}")
        return result

    def _storage_owner(self, node: ProgramGraphNode, phase: _ForwardPhase, scope: dict[NodeId, str]) -> str:
        owner = _owner(node.id, self.nodes)
        result = scope.get(owner.id)
        if result is None:
            result = self.caches[phase].get(owner.id)
        if result is None:
            raise DirectGpuLoweringError("Storage owner is absent from the forward phase inputs")
        scope[node.id] = result
        return result

    def _capture_storage_read(
        self,
        node: ProgramGraphNode,
        scope: dict[NodeId, str],
        lines: list[str],
    ) -> str:
        read_indices = _physical_indices(
            node,
            self.nodes,
            lambda item: self._emit(item, _ForwardPhase.CAPTURE_TAPE, scope, lines),
            self.names,
            lines,
        )
        storage_type = self.nodes[node.inputs[0]].type
        result_type = value_dtype(storage_type)

        def read_state(state_id: NodeId, state_scope: dict[NodeId, str], state_lines: list[str]) -> str:
            state = self.nodes[state_id]
            if state.operation is OpCode.PARAMETER:
                result = self.names.fresh()
                state_lines.append(
                    f'    {result} = "vernon.load"({state_scope[state.id]}, {", ".join(read_indices)}) : '
                    f"({resource_type(storage_type, str(storage_type.arguments[2]))}, "
                    f"{', '.join('index' for _ in read_indices)}) -> {result_type}"
                )
                return result
            if state.operation in {OpCode.STORE, OpCode.STORE_DYNAMIC}:
                previous = read_state(state.inputs[0], state_scope, state_lines)
                stored_id = state.inputs[1] if state.operation is OpCode.STORE else state.inputs[-1]
                stored = self._emit(stored_id, _ForwardPhase.CAPTURE_TAPE, state_scope, state_lines)
                stored_indices = _physical_indices(
                    state,
                    self.nodes,
                    lambda item: self._emit(item, _ForwardPhase.CAPTURE_TAPE, state_scope, state_lines),
                    self.names,
                    state_lines,
                )
                comparisons: list[str] = []
                for read_index, stored_index in zip(read_indices, stored_indices, strict=True):
                    comparison = self.names.fresh()
                    state_lines.append(f"    {comparison} = arith.cmpi eq, {read_index}, {stored_index} : index")
                    comparisons.append(comparison)
                selected = comparisons[0]
                for comparison in comparisons[1:]:
                    combined = self.names.fresh()
                    state_lines.append(f"    {combined} = arith.andi {selected}, {comparison} : i1")
                    selected = combined
                result = self.names.fresh()
                state_lines.append(f"    {result} = arith.select {selected}, {stored}, {previous} : {result_type}")
                return result
            if state.operation is OpCode.CONDITIONAL and state.type.kind == "tensor_view":
                condition = self._emit(state.inputs[0], _ForwardPhase.CAPTURE_TAPE, state_scope, state_lines)
                result = self.names.fresh()
                state_lines.append(f"    {result} = scf.if {condition} -> ({result_type}) {{")
                for branch_index in (1, 2):
                    branch_scope = dict(state_scope)
                    branch_lines: list[str] = []
                    branch_value = read_state(state.inputs[branch_index], branch_scope, branch_lines)
                    state_lines.extend(branch_lines)
                    state_lines.append(f"    scf.yield {branch_value} : {result_type}")
                    if branch_index == 1:
                        state_lines.append("    } else {")
                state_lines.append("    }")
                return result
            raise DirectGpuLoweringError(
                f"cannot capture a tape read from Storage state '{state.operation.value}' without side effects"
            )

        return read_state(node.inputs[0], scope, lines)

    def _emit(
        self,
        value: NodeId,
        phase: _ForwardPhase,
        scope: dict[NodeId, str],
        lines: list[str],
    ) -> str:
        previous = scope.get(value)
        if previous is not None:
            return previous
        node = self.nodes[value]
        if node.operation is OpCode.CONDITIONAL:
            if node.type.kind == "tensor_view":
                true_owner = _owner(node.inputs[1], self.nodes)
                false_owner = _owner(node.inputs[2], self.nodes)
                if true_owner.id != false_owner.id:
                    raise DirectGpuLoweringError("conditional Storage states must have one parameter owner")
                if phase is _ForwardPhase.CAPTURE_TAPE:
                    raise DirectGpuLoweringError(
                        "reverse tape Value depends on an uncommitted conditional Storage state"
                    )
                if phase is _ForwardPhase.OBSERVE_OUTPUT:
                    return self._storage_owner(node, phase, scope)
                condition = self._emit(node.inputs[0], phase, scope, lines)
                lines.append(f"    scf.if {condition} {{")
                self._emit(node.inputs[1], phase, dict(scope), lines)
                lines.append("    } else {")
                self._emit(node.inputs[2], phase, dict(scope), lines)
                lines.append("    }")
                return self._storage_owner(node, phase, scope)
            condition = self._emit(node.inputs[0], phase, scope, lines)
            true_dependencies = self._dependencies(node.inputs[1])
            false_dependencies = self._dependencies(node.inputs[2])
            for shared in sorted(true_dependencies & false_dependencies):
                self._emit(shared, phase, scope, lines)
            saved = sorted(((true_dependencies ^ false_dependencies) & self.saved_values).difference(scope))
            result_ids = [node.id, *saved]
            result_names = [self.names.fresh() for _ in result_ids]
            result_types = [native_type(self.nodes[item].type) for item in result_ids]
            lines.append(f"    {', '.join(result_names)} = scf.if {condition} -> ({', '.join(result_types)}) {{")
            for branch_index, branch_dependencies in ((1, true_dependencies), (2, false_dependencies)):
                branch_scope = dict(scope)
                branch_lines: list[str] = []
                yielded = [self._emit(node.inputs[branch_index], phase, branch_scope, branch_lines)]
                yielded.extend(
                    self._emit(item, phase, branch_scope, branch_lines)
                    if item in branch_dependencies
                    else self._zero(item, branch_lines)
                    for item in saved
                )
                lines.extend(branch_lines)
                lines.append(f"    scf.yield {', '.join(yielded)} : {', '.join(result_types)}")
                if branch_index == 1:
                    lines.append("    } else {")
            lines.append("    }")
            scope.update(zip(result_ids, result_names, strict=True))
            return result_names[0]
        if (
            node.operation in {OpCode.INDEX, OpCode.INDEX_DYNAMIC}
            and self.nodes[node.inputs[0]].type.kind == "tensor_view"
        ):
            if phase is _ForwardPhase.CAPTURE_TAPE:
                result = self._capture_storage_read(node, scope, lines)
                scope[value] = result
                return result
            indices = _physical_indices(
                node,
                self.nodes,
                lambda item: self._emit(item, phase, scope, lines),
                self.names,
                lines,
            )
            result = self.names.fresh()
            storage_type = self.nodes[node.inputs[0]].type
            lines.append(
                f'    {result} = "vernon.load"({self._emit(node.inputs[0], phase, scope, lines)}, '
                f"{', '.join(indices)}) : ({resource_type(storage_type, str(storage_type.arguments[2]))}, "
                f"{', '.join('index' for _ in indices)}) -> {value_dtype(storage_type)}"
            )
            scope[value] = result
            return result
        if node.operation in {OpCode.STORE, OpCode.STORE_DYNAMIC}:
            if phase is _ForwardPhase.CAPTURE_TAPE:
                raise DirectGpuLoweringError("reverse tape Value depends on an uncommitted Storage write")
            if phase is _ForwardPhase.OBSERVE_OUTPUT:
                return self._storage_owner(node, phase, scope)
            storage = self._emit(node.inputs[0], phase, scope, lines)
            indices = _physical_indices(
                node,
                self.nodes,
                lambda item: self._emit(item, phase, scope, lines),
                self.names,
                lines,
            )
            stored_id = node.inputs[1] if node.operation is OpCode.STORE else node.inputs[-1]
            stored = self._emit(stored_id, phase, scope, lines)
            storage_type = self.nodes[node.inputs[0]].type
            lines.append(
                f'    "vernon.store"({stored}, {storage}, {", ".join(indices)}) : '
                f"({value_dtype(storage_type)}, {resource_type(storage_type, str(storage_type.arguments[2]))}, "
                f"{', '.join('index' for _ in indices)}) -> ()"
            )
            scope[value] = storage
            return storage
        operands = tuple(self._emit(operand, phase, scope, lines) for operand in node.inputs)
        result = _emit_operation(
            node,
            operands,
            tuple(self.nodes[operand].type for operand in node.inputs),
            self.names,
            lines,
        )
        scope[value] = result
        return result


def emit_forward(program: AutodiffProgram, plan: AutodiffProfilePlan) -> str:
    graph = program.semantic
    reverse = program.reverse
    nodes = {node.id: node for node in graph.nodes}
    profile = next(value for value in plan.profiles if value.name == "forward_with_tape")
    resources = forward_resources(program)
    parameters = tuple(node for node in graph.nodes if node.operation in {OpCode.PARAMETER, OpCode.BUILTIN})
    resource_parameters = tuple(node for node in parameters if node.operation is OpCode.PARAMETER)
    builtin_parameters = tuple(node for node in parameters if node.operation is OpCode.BUILTIN)
    builtin_arguments, global_id_binding = _global_id_arguments(builtin_parameters)
    arguments = ", ".join(
        (
            *(_resource_arguments(resources),),
            *builtin_arguments,
        )
    )
    lines = [
        entry_header(
            profile.symbol,
            arguments,
            program.launch.workgroup_size,
            _storage_effects(program),
        )
    ]
    names = Names()
    carrier_xyz, valid = _guard_runtime_grid(
        len(resource_parameters),
        f"%builtin{global_id_binding}",
        names,
        lines,
    )
    carrier_indices = (carrier_xyz[2], carrier_xyz[1], carrier_xyz[0])
    lines.append(f"    scf.if {valid} {{")
    zero_index = names.fresh()
    lines.append(f"    {zero_index} = arith.constant 0 : index")
    resource_binding = {node.id: index for index, node in enumerate(resource_parameters)}
    builtin_binding = {node.id: index for index, node in enumerate(builtin_parameters)}
    base_values: dict[NodeId, str] = {}
    for parameter in parameters:
        if parameter.operation is OpCode.BUILTIN:
            base_values[parameter.id] = f"%builtin{builtin_binding[parameter.id]}"
        elif parameter.type.kind == "tensor_view":
            base_values[parameter.id] = f"%resource{resource_binding[parameter.id]}"
        else:
            value = names.fresh()
            lines.append(
                f'    {value} = "vernon.load"(%resource{resource_binding[parameter.id]}, {zero_index}) : '
                f"({resource_type(parameter.type, 'read')}, index) -> {native_type(parameter.type)}"
            )
            base_values[parameter.id] = value

    emitter = _GpuForwardEmitter(nodes, reverse.saved_values, names, base_values)
    for value in reverse.saved_values:
        if nodes[value].type.kind == "tensor_view":
            raise DirectGpuLoweringError("TensorView state must not be materialized on the reverse tape")
        emitter.emit(value, _ForwardPhase.CAPTURE_TAPE, lines)
    for _, value in graph.storage_outputs:
        emitter.emit(value, _ForwardPhase.COMMIT_STORAGE, lines)
    output_value = emitter.emit(graph.outputs[0], _ForwardPhase.OBSERVE_OUTPUT, lines)
    output_binding = len(resource_parameters) + 1
    output = nodes[graph.outputs[0]]
    lines.append(
        f'    "vernon.store"({output_value}, %resource{output_binding}, {", ".join(carrier_indices)}) : '
        f"({native_type(output.type)}, {resource_type(output.type, 'write', True)}, index, index, index) -> ()"
    )
    for tape_index, value_id in enumerate(reverse.saved_values):
        binding = output_binding + 1 + tape_index
        value_type = nodes[value_id].type
        lines.append(
            f'    "vernon.store"({emitter.captured(value_id)}, %resource{binding}, {", ".join(carrier_indices)}) : '
            f"({native_type(value_type)}, {resource_type(value_type, 'write', True)}, index, index, index) -> ()"
        )
    lines.extend(("    }", "    func.return", "  }"))
    return module(lines, "forward_with_tape", plan)


def _zero(value_type: ConcreteType, names: Names, lines: list[str]) -> str:
    result = names.fresh()
    literal = "dense<0.0>" if value_type.kind == "tensor" else "0.0"
    lines.append(f"    {result} = arith.constant {literal} : {native_type(value_type)}")
    return result


def emit_backward(program: AutodiffProgram, plan: AutodiffProfilePlan) -> str:
    graph = program.semantic
    reverse = program.reverse
    nodes = {node.id: node for node in graph.nodes}
    profile = next(value for value in plan.profiles if value.name == "backward")
    resources = backward_resources(program)
    builtin_parameters = tuple(node for node in graph.nodes if node.operation is OpCode.BUILTIN)
    builtin_arguments, global_id_binding = _global_id_arguments(builtin_parameters)
    arguments = ", ".join(
        (
            *(_resource_arguments(resources),),
            *builtin_arguments,
        )
    )
    lines = [entry_header(profile.symbol, arguments, program.launch.workgroup_size)]
    names = Names()
    carrier_xyz, valid = _guard_runtime_grid(0, f"%builtin{global_id_binding}", names, lines)
    carrier_indices = (carrier_xyz[2], carrier_xyz[1], carrier_xyz[0])
    lines.append(f"    scf.if {valid} {{")
    primal_values: dict[NodeId, str] = {}
    for tape_index, value_id in enumerate(reverse.saved_values):
        binding = tape_index + 1
        value_type = nodes[value_id].type
        if value_type.kind == "tensor_view":
            raise DirectGpuLoweringError("TensorView state must not be materialized on the reverse tape")
        value = names.fresh()
        lines.append(
            f'    {value} = "vernon.load"(%resource{binding}, {", ".join(carrier_indices)}) : '
            f"({resource_type(value_type, 'read', True)}, index, index, index) -> {native_type(value_type)}"
        )
        derivative_type = gradient_type(value_type)
        if derivative_type != value_type:
            promoted = names.fresh()
            lines.append(
                f"    {promoted} = arith.extf {value} : {native_type(value_type)} to {native_type(derivative_type)}"
            )
            value = promoted
        primal_values[value_id] = value
    cotangent_binding = len(reverse.saved_values) + 1
    output = nodes[graph.outputs[0]]
    cotangent_type = gradient_type(output.type)
    cotangent = names.fresh()
    lines.append(
        f'    {cotangent} = "vernon.load"(%resource{cotangent_binding}, {", ".join(carrier_indices)}) : '
        f"({resource_type(cotangent_type, 'read', True)}, index, index, index) -> {native_type(cotangent_type)}"
    )

    def primal(value_id: NodeId) -> str:
        previous = primal_values.get(value_id)
        if previous is not None:
            return previous
        node = nodes[value_id]
        if node.operation is OpCode.CONSTANT:
            result = _emit_operation(node, (), (), names, lines)
            primal_values[value_id] = result
            return result
        raise DirectGpuLoweringError(f"reverse rule requires unsaved primal node {int(value_id)}")

    cotangents: dict[NodeId, str] = {graph.outputs[0]: cotangent}

    def accumulate(value_id: NodeId, contribution: str) -> None:
        previous = cotangents.get(value_id)
        if previous is None:
            cotangents[value_id] = contribution
            return
        result = names.fresh()
        lines.append(
            f"    {result} = arith.addf {previous}, {contribution} : {native_type(gradient_type(nodes[value_id].type))}"
        )
        cotangents[value_id] = result

    StorageKey = tuple[NodeId, tuple[tuple[str, int], ...]]
    storage_adjoints: dict[StorageKey, str] = {}

    def storage_indices(node: ProgramGraphNode) -> tuple[tuple[str, int], ...]:
        if node.operation in {OpCode.INDEX, OpCode.STORE}:
            return tuple(("static", value) for value in _static_indices(node))
        ids = node.inputs[1:] if node.operation is OpCode.INDEX_DYNAMIC else node.inputs[1:-1]
        return tuple(("dynamic", int(value)) for value in ids)

    def storage_key(node: ProgramGraphNode, state: NodeId | None = None) -> StorageKey:
        return (node.inputs[0] if state is None else state), storage_indices(node)

    def add_storage(key: StorageKey, contribution: str, value_type: ConcreteType) -> None:
        previous = storage_adjoints.get(key)
        if previous is None:
            storage_adjoints[key] = contribution
            return
        result = names.fresh()
        lines.append(
            f"    {result} = arith.addf {previous}, {contribution} : {value_dtype(gradient_resource_type(value_type))}"
        )
        storage_adjoints[key] = result

    for value_id in reverse.reverse_order:
        node = nodes[value_id]
        if node.operation in {OpCode.STORE, OpCode.STORE_DYNAMIC}:
            stored = node.inputs[1] if node.operation is OpCode.STORE else node.inputs[-1]
            overwritten = storage_indices(node)
            for key in tuple(storage_adjoints):
                if key[0] != value_id:
                    continue
                contribution = storage_adjoints.pop(key)
                if key[1] == overwritten:
                    accumulate(stored, contribution)
                else:
                    add_storage((node.inputs[0], key[1]), contribution, nodes[node.inputs[0]].type)
            continue
        if node.operation is OpCode.CONDITIONAL and node.type.kind == "tensor_view":
            condition = primal(node.inputs[0])
            element_type = gradient_resource_type(node.type).arguments[0]
            assert isinstance(element_type, ConcreteType)
            for key in tuple(storage_adjoints):
                if key[0] != value_id:
                    continue
                contribution = storage_adjoints.pop(key)
                zero = _zero(element_type, names, lines)
                true_seed = names.fresh()
                false_seed = names.fresh()
                lines.append(
                    f"    {true_seed}, {false_seed} = scf.if {condition} -> "
                    f"({element_type.mlir}, {element_type.mlir}) {{"
                )
                lines.append(f"    scf.yield {contribution}, {zero} : {element_type.mlir}, {element_type.mlir}")
                lines.append("    } else {")
                lines.append(f"    scf.yield {zero}, {contribution} : {element_type.mlir}, {element_type.mlir}")
                lines.append("    }")
                add_storage((node.inputs[1], key[1]), true_seed, node.type)
                add_storage((node.inputs[2], key[1]), false_seed, node.type)
            continue
        seed = cotangents.get(value_id)
        if seed is None:
            continue
        if node.operation in {OpCode.INDEX, OpCode.INDEX_DYNAMIC} and nodes[node.inputs[0]].type.kind == "tensor_view":
            add_storage(storage_key(node), seed, nodes[node.inputs[0]].type)
            continue
        if node.operation in {OpCode.INDEX, OpCode.INDEX_DYNAMIC}:
            accumulate(node.inputs[0], _value_index_vjp(node, seed, primal, nodes, names, lines))
            continue
        if node.operation is OpCode.ADD:
            accumulate(node.inputs[0], seed)
            accumulate(node.inputs[1], seed)
        elif node.operation is OpCode.SUB:
            accumulate(node.inputs[0], seed)
            zero = _zero(gradient_type(node.type), names, lines)
            negative = names.fresh()
            lines.append(f"    {negative} = arith.subf {zero}, {seed} : {native_type(gradient_type(node.type))}")
            accumulate(node.inputs[1], negative)
        elif node.operation is OpCode.MUL:
            left = names.fresh()
            right = names.fresh()
            lines.append(
                f"    {left} = arith.mulf {seed}, {primal(node.inputs[1])} : {native_type(gradient_type(node.type))}"
            )
            lines.append(
                f"    {right} = arith.mulf {seed}, {primal(node.inputs[0])} : {native_type(gradient_type(node.type))}"
            )
            accumulate(node.inputs[0], left)
            accumulate(node.inputs[1], right)
        elif node.operation is OpCode.DIV:
            zero = _zero(gradient_type(node.type), names, lines)
            one = names.fresh()
            active = names.fresh()
            denominator = names.fresh()
            left, square, numerator, quotient, negative = (names.fresh() for _ in range(5))
            value_type = native_type(gradient_type(node.type))
            one_literal = "dense<1.0>" if node.type.kind == "tensor" else "1.0"
            lines.append(f"    {one} = arith.constant {one_literal} : {value_type}")
            if node.type.kind == "tensor":
                squared_seed = names.fresh()
                activity = names.fresh()
                element = gradient_type(node.type).arguments[0]
                assert isinstance(element, ConcreteType)
                lines.append(f"    {squared_seed} = arith.mulf {seed}, {seed} : {value_type}")
                lines.append(
                    f'    {activity} = "vernon.intrinsic"({squared_seed}) {{name = "reduce_sum_to_shape"}} : '
                    f"({value_type}) -> {native_type(element)}"
                )
                scalar_zero = names.fresh()
                lines.append(f"    {scalar_zero} = arith.constant 0.0 : {native_type(element)}")
                lines.append(f"    {active} = arith.cmpf une, {activity}, {scalar_zero} : {native_type(element)}")
            else:
                lines.append(f"    {active} = arith.cmpf une, {seed}, {zero} : {value_type}")
            lines.append(f"    {denominator} = scf.if {active} -> ({value_type}) {{")
            lines.append(f"    scf.yield {primal(node.inputs[1])} : {value_type}")
            lines.append("    } else {")
            lines.append(f"    scf.yield {one} : {value_type}")
            lines.append("    }")
            lines.append(f"    {left} = arith.divf {seed}, {denominator} : {value_type}")
            lines.append(f"    {square} = arith.mulf {denominator}, {denominator} : {value_type}")
            lines.append(f"    {numerator} = arith.mulf {seed}, {primal(node.inputs[0])} : {value_type}")
            lines.append(f"    {quotient} = arith.divf {numerator}, {square} : {value_type}")
            lines.append(f"    {negative} = arith.subf {zero}, {quotient} : {value_type}")
            accumulate(node.inputs[0], left)
            accumulate(node.inputs[1], negative)
        elif node.operation is OpCode.NEG:
            zero = _zero(gradient_type(node.type), names, lines)
            negative = names.fresh()
            lines.append(f"    {negative} = arith.subf {zero}, {seed} : {native_type(gradient_type(node.type))}")
            accumulate(node.inputs[0], negative)
        elif node.operation is OpCode.IDENTITY:
            accumulate(node.inputs[0], seed)
        elif node.operation is OpCode.CONDITIONAL:
            condition = primal(node.inputs[0])
            zero = _zero(gradient_type(node.type), names, lines)
            true_seed = names.fresh()
            false_seed = names.fresh()
            value_type = native_type(gradient_type(node.type))
            lines.append(f"    {true_seed}, {false_seed} = scf.if {condition} -> ({value_type}, {value_type}) {{")
            lines.append(f"    scf.yield {seed}, {zero} : {value_type}, {value_type}")
            lines.append("    } else {")
            lines.append(f"    scf.yield {zero}, {seed} : {value_type}, {value_type}")
            lines.append("    }")
            accumulate(node.inputs[1], true_seed)
            accumulate(node.inputs[2], false_seed)
        elif node.operation in {OpCode.SIN, OpCode.COS, OpCode.EXP, OpCode.LOG, OpCode.SQRT}:
            factor = names.fresh()
            if node.operation is OpCode.SIN:
                lines.append(
                    f"    {factor} = math.cos {primal(node.inputs[0])} : {native_type(gradient_type(node.type))}"
                )
            elif node.operation is OpCode.COS:
                sine = names.fresh()
                zero = _zero(gradient_type(node.type), names, lines)
                lines.append(
                    f"    {sine} = math.sin {primal(node.inputs[0])} : {native_type(gradient_type(node.type))}"
                )
                lines.append(f"    {factor} = arith.subf {zero}, {sine} : {native_type(gradient_type(node.type))}")
            elif node.operation is OpCode.EXP:
                factor = primal(node.id)
            elif node.operation is OpCode.LOG:
                one = names.fresh()
                lines.append(f"    {one} = arith.constant 1.0 : {native_type(gradient_type(node.type))}")
                lines.append(
                    f"    {factor} = arith.divf {one}, {primal(node.inputs[0])} : "
                    f"{native_type(gradient_type(node.type))}"
                )
            else:
                half = names.fresh()
                lines.append(f"    {half} = arith.constant 0.5 : {native_type(gradient_type(node.type))}")
                lines.append(
                    f"    {factor} = arith.divf {half}, {primal(node.id)} : {native_type(gradient_type(node.type))}"
                )
            contribution = names.fresh()
            lines.append(f"    {contribution} = arith.mulf {seed}, {factor} : {native_type(gradient_type(node.type))}")
            accumulate(node.inputs[0], contribution)
        elif node.operation in {OpCode.SPLAT, OpCode.BROADCAST}:
            reduced = names.fresh()
            lines.append(
                f'    {reduced} = "vernon.intrinsic"({seed}) {{name = "reduce_sum_to_shape"}} : '
                f"({native_type(gradient_type(node.type))}) -> "
                f"{native_type(gradient_type(nodes[node.inputs[0]].type))}"
            )
            accumulate(node.inputs[0], reduced)
        elif node.operation not in {OpCode.CONSTANT, OpCode.COMPARE}:
            contributions = emit_math_vjp(node, seed, primal, nodes, names, lines, gradient_type)
            if contributions is None:
                raise DirectGpuLoweringError(f"cannot emit direct backward rule for '{node.operation.value}'")
            for operand, contribution in zip(node.inputs, contributions, strict=True):
                if contribution is not None:
                    accumulate(operand, contribution)

    gradient_nodes = {node.source_name: node for node in graph.nodes if node.operation is OpCode.PARAMETER}
    gradient_plans = {item.path: item for item in program.launch.accumulation_plans}
    gradient_binding = {path: cotangent_binding + 1 + index for index, path in enumerate(graph.wrt)}

    def key_indices(key: StorageKey) -> tuple[str, ...]:
        results: list[str] = []
        for kind, value in key[1]:
            if kind == "static":
                results.extend(index_constants((value,), names, lines))
            else:
                value_id = NodeId(value)
                physical = names.fresh()
                lines.append(
                    f"    {physical} = arith.index_cast {primal(value_id)} : "
                    f"{native_type(nodes[value_id].type)} to index"
                )
                results.append(physical)
        return tuple(results)

    def accumulate_resource(
        contribution: str,
        binding: int,
        value_type: ConcreteType,
        indices: tuple[str, ...],
        mode: AccumulationMode,
        disjoint: bool = False,
    ) -> None:
        derivative_value_type = gradient_type(value_type)
        if value_type.kind == "tensor":
            element = derivative_value_type.arguments[0]
            shape = derivative_value_type.arguments[1:]
            assert isinstance(element, ConcreteType)
            gradient_value_type = ConcreteType(
                "tensor_view",
                "TensorView",
                (element, tuple(shape), "read_write", "device"),
            )
            gradient_resource = resource_type(gradient_value_type, "read_write")
            lower = names.fresh()
            step = names.fresh()
            lines.append(f"    {lower} = arith.constant 0 : index")
            lines.append(f"    {step} = arith.constant 1 : index")
            loop_indices: list[str] = []
            for depth, extent in enumerate(shape):
                upper = names.fresh()
                index = names.fresh()
                indent = "    " + "  " * depth
                lines.append(f"{indent}{upper} = arith.constant {extent} : index")
                lines.append(f"{indent}scf.for {index} = {lower} to {upper} step {step} {{")
                loop_indices.append(index)
            indent = "    " + "  " * len(shape)
            component = names.fresh()
            lines.append(
                f"{indent}{component} = tensor.extract {contribution}"
                f"[{', '.join(loop_indices)}] : {native_type(derivative_value_type)}"
            )
            lines.append(
                f'{indent}"vernon.reduce_sum"({component}, %resource{binding}, '
                f"{', '.join(loop_indices)}) "
                "{deterministic = false} : "
                f"({element.mlir}, {gradient_resource}, "
                f"{', '.join('index' for _ in loop_indices)}) -> ()"
            )
            for depth in reversed(range(len(shape))):
                lines.append("    " + "  " * depth + "}")
            return
        access = "read_write"
        resource = f"%resource{binding}"
        resource_value_type = gradient_resource_type(value_type)
        resource_mlir = resource_type(resource_value_type, access)
        index_types = ", ".join("index" for _ in indices)
        payload_type = (
            value_dtype(resource_value_type) if value_type.kind == "tensor_view" else native_type(derivative_value_type)
        )
        operation = "reduce_sum" if mode is AccumulationMode.REDUCE_SUM else "scatter_add"
        evidence = ", disjoint" if disjoint else ""
        lines.append(
            f'    "vernon.{operation}"({contribution}, {resource}, {", ".join(indices)}) '
            f"{{deterministic = false{evidence}}} : "
            f"({payload_type}, {resource_mlir}, {index_types}) -> ()"
        )

    for key, contribution in storage_adjoints.items():
        state = nodes[key[0]]
        if state.operation is not OpCode.PARAMETER or state.source_name is None:
            raise DirectGpuLoweringError("reverse Storage adjoint did not reach a parameter state")
        path = state.source_name
        if path not in gradient_binding:
            continue
        node = gradient_nodes[path]
        accumulate_resource(
            contribution,
            gradient_binding[path],
            node.type,
            key_indices(key),
            gradient_plans[path].mode,
            AccessPatternEvidence.DISJOINT_SCATTER in gradient_plans[path].evidence,
        )
    for path in graph.wrt:
        node = gradient_nodes[path]
        if node.type.kind == "tensor_view":
            continue
        contribution = cotangents.get(node.id)
        if contribution is None:
            contribution = _zero(gradient_type(node.type), names, lines)
        mode = gradient_plans[path].mode
        indices = index_constants((0,), names, lines)
        accumulate_resource(contribution, gradient_binding[path], node.type, indices, mode)
    lines.append("    }")
    lines.extend(("    func.return", "  }"))
    return module(lines, "backward", plan)


__all__ = ["DirectGpuLoweringError", "emit_backward", "emit_forward"]
