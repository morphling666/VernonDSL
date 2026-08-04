from __future__ import annotations

from collections.abc import Callable

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
)
from .autodiff_native_abi import (
    ResourceBindingPlan,
    backward_resources,
    forward_resources,
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
        raise DirectGpuLoweringError(f"cannot emit direct value operation '{node.operation.value}'")
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


def _runtime_grid_extents(
    launch_binding: int,
    names: Names,
    lines: list[str],
) -> tuple[str, str, str]:
    zero = names.fresh()
    launch = names.fresh()
    lines.append(f"    {zero} = arith.constant 0 : index")
    lines.append(
        f'    {launch} = "vernon.load"(%resource{launch_binding}, {zero}) : '
        f"({resource_type(_GLOBAL_ID_TYPE, 'read')}, index) -> {_GLOBAL_ID_TYPE.mlir}"
    )
    extents: list[str] = []
    for axis in range(3):
        axis_index = names.fresh()
        extent_u32 = names.fresh()
        extent = names.fresh()
        lines.append(f"    {axis_index} = arith.constant {axis} : index")
        lines.append(f"    {extent_u32} = tensor.extract {launch}[{axis_index}] : {_GLOBAL_ID_TYPE.mlir}")
        lines.append(f"    {extent} = arith.index_castui {extent_u32} : i32 to index")
        extents.append(extent)
    return extents[0], extents[1], extents[2]


def _requires_serial_backward(program: AutodiffProgram, target: str) -> bool:
    parameters = {
        node.source_name: node
        for node in program.semantic.nodes
        if node.operation is OpCode.PARAMETER and node.source_name is not None
    }
    for accumulation in program.launch.accumulation_plans:
        if AccessPatternEvidence.DISJOINT_SCATTER in accumulation.evidence:
            continue
        parameter = parameters[accumulation.path]
        value_type = parameter.type
        if value_type.kind in {"tensor", "tensor_view"}:
            element = value_type.arguments[0]
            assert isinstance(element, ConcreteType)
            value_type = element
        if target != "cuda" or value_type.name != "f32":
            return True
    return False


def _owner(node_id: NodeId, nodes: dict[NodeId, ProgramGraphNode]) -> ProgramGraphNode:
    current = nodes[node_id]
    while current.operation is not OpCode.PARAMETER:
        if not current.inputs:
            raise DirectGpuLoweringError("Storage operation has no TensorView parameter owner")
        current = nodes[current.inputs[0]]
    if current.type.kind != "tensor_view":
        raise DirectGpuLoweringError("Storage operation has no TensorView parameter owner")
    return current


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
    values: dict[NodeId, str] = {}
    for parameter in parameters:
        if parameter.operation is OpCode.BUILTIN:
            values[parameter.id] = f"%builtin{builtin_binding[parameter.id]}"
        elif parameter.type.kind == "tensor_view":
            values[parameter.id] = f"%resource{resource_binding[parameter.id]}"
        else:
            value = names.fresh()
            lines.append(
                f'    {value} = "vernon.load"(%resource{resource_binding[parameter.id]}, {zero_index}) : '
                f"({resource_type(parameter.type, 'read')}, index) -> {native_type(parameter.type)}"
            )
            values[parameter.id] = value

    def dependencies(value: NodeId) -> set[NodeId]:
        result = {value}
        for operand in nodes[value].inputs:
            result.update(dependencies(operand))
        return result

    def zero(value: NodeId, target_lines: list[str]) -> str:
        result = names.fresh()
        value_type = nodes[value].type
        literal = (
            "0"
            if value_type.kind == "scalar" and value_type.name == "bool"
            else "dense<0.0>"
            if value_type.kind == "tensor"
            else "0.0"
        )
        target_lines.append(f"    {result} = arith.constant {literal} : {native_type(value_type)}")
        return result

    def emit(value: NodeId, scope: dict[NodeId, str], target_lines: list[str]) -> str:
        previous = scope.get(value)
        if previous is not None:
            return previous
        node = nodes[value]
        if node.operation is OpCode.CONDITIONAL:
            condition = emit(node.inputs[0], scope, target_lines)
            true_dependencies = dependencies(node.inputs[1])
            false_dependencies = dependencies(node.inputs[2])
            for shared in sorted(true_dependencies & false_dependencies):
                emit(shared, scope, target_lines)
            saved = sorted((true_dependencies ^ false_dependencies) & set(reverse.saved_values))
            result_ids = [node.id, *saved]
            result_names = [names.fresh() for _ in result_ids]
            result_types = [native_type(nodes[item].type) for item in result_ids]
            target_lines.append(f"    {', '.join(result_names)} = scf.if {condition} -> ({', '.join(result_types)}) {{")
            for branch_index, branch_dependencies in ((1, true_dependencies), (2, false_dependencies)):
                branch_scope = dict(scope)
                branch_lines: list[str] = []
                yielded = [emit(node.inputs[branch_index], branch_scope, branch_lines)]
                yielded.extend(
                    emit(item, branch_scope, branch_lines) if item in branch_dependencies else zero(item, branch_lines)
                    for item in saved
                )
                target_lines.extend(branch_lines)
                target_lines.append(f"    scf.yield {', '.join(yielded)} : {', '.join(result_types)}")
                if branch_index == 1:
                    target_lines.append("    } else {")
            target_lines.append("    }")
            scope.update(zip(result_ids, result_names, strict=True))
            return result_names[0]
        if node.operation in {OpCode.INDEX, OpCode.INDEX_DYNAMIC} and nodes[node.inputs[0]].type.kind == "tensor_view":
            indices = _physical_indices(node, nodes, lambda item: emit(item, scope, target_lines), names, target_lines)
            result = names.fresh()
            storage_type = nodes[node.inputs[0]].type
            target_lines.append(
                f'    {result} = "vernon.load"({emit(node.inputs[0], scope, target_lines)}, '
                f"{', '.join(indices)}) : ({resource_type(storage_type, str(storage_type.arguments[2]))}, "
                f"{', '.join('index' for _ in indices)}) -> {value_dtype(storage_type)}"
            )
            scope[value] = result
            return result
        if node.operation in {OpCode.STORE, OpCode.STORE_DYNAMIC}:
            storage = emit(node.inputs[0], scope, target_lines)
            indices = _physical_indices(node, nodes, lambda item: emit(item, scope, target_lines), names, target_lines)
            stored_id = node.inputs[1] if node.operation is OpCode.STORE else node.inputs[-1]
            stored = emit(stored_id, scope, target_lines)
            storage_type = nodes[node.inputs[0]].type
            target_lines.append(
                f'    "vernon.store"({stored}, {storage}, {", ".join(indices)}) : '
                f"({value_dtype(storage_type)}, {resource_type(storage_type, str(storage_type.arguments[2]))}, "
                f"{', '.join('index' for _ in indices)}) -> ()"
            )
            scope[value] = storage
            return storage
        operands = tuple(emit(operand, scope, target_lines) for operand in node.inputs)
        result = _emit_operation(
            node,
            operands,
            tuple(nodes[operand].type for operand in node.inputs),
            names,
            target_lines,
        )
        scope[value] = result
        return result

    output_value = emit(graph.outputs[0], values, lines)
    for _, value in graph.storage_outputs:
        emit(value, values, lines)
    for value in reverse.saved_values:
        if nodes[value].type.kind == "tensor_view":
            raise DirectGpuLoweringError("TensorView state must not be materialized on the reverse tape")
        emit(value, values, lines)
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
            f'    "vernon.store"({values[value_id]}, %resource{binding}, {", ".join(carrier_indices)}) : '
            f"({native_type(value_type)}, {resource_type(value_type, 'write', True)}, index, index, index) -> ()"
        )
    lines.extend(("    }", "    func.return", "  }"))
    return module(lines, "forward_with_tape", plan)


def _zero(value_type: ConcreteType, names: Names, lines: list[str]) -> str:
    result = names.fresh()
    literal = "dense<0.0>" if value_type.kind == "tensor" else "0.0"
    lines.append(f"    {result} = arith.constant {literal} : {native_type(value_type)}")
    return result


def emit_backward(
    program: AutodiffProgram,
    plan: AutodiffProfilePlan,
    *,
    target: str = "cuda",
) -> str:
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
    serial = _requires_serial_backward(program, target)
    lines = [entry_header(profile.symbol, arguments, (1, 1, 1) if serial else program.launch.workgroup_size)]
    names = Names()
    if serial:
        grid_x, grid_y, grid_z = _runtime_grid_extents(0, names, lines)
        lower = names.fresh()
        step = names.fresh()
        z = names.fresh()
        y = names.fresh()
        x = names.fresh()
        lines.append(f"    {lower} = arith.constant 0 : index")
        lines.append(f"    {step} = arith.constant 1 : index")
        lines.append(f"    scf.for {z} = {lower} to {grid_z} step {step} {{")
        lines.append(f"      scf.for {y} = {lower} to {grid_y} step {step} {{")
        lines.append(f"        scf.for {x} = {lower} to {grid_x} step {step} {{")
        carrier_indices = (z, y, x)
    else:
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
        primal_values[value_id] = value
    cotangent_binding = len(reverse.saved_values) + 1
    output = nodes[graph.outputs[0]]
    cotangent = names.fresh()
    lines.append(
        f'    {cotangent} = "vernon.load"(%resource{cotangent_binding}, {", ".join(carrier_indices)}) : '
        f"({resource_type(output.type, 'read', True)}, index, index, index) -> {native_type(output.type)}"
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
        lines.append(f"    {result} = arith.addf {previous}, {contribution} : {native_type(nodes[value_id].type)}")
        cotangents[value_id] = result

    StorageKey = tuple[str, tuple[tuple[str, int], ...]]
    storage_adjoints: dict[StorageKey, str] = {}

    def storage_key(node: ProgramGraphNode) -> StorageKey:
        owner = _owner(node.inputs[0], nodes).source_name or ""
        if node.operation in {OpCode.INDEX, OpCode.STORE}:
            indices = tuple(("static", value) for value in _static_indices(node))
        else:
            ids = node.inputs[1:] if node.operation is OpCode.INDEX_DYNAMIC else node.inputs[1:-1]
            indices = tuple(("dynamic", int(value)) for value in ids)
        return owner, indices

    def add_storage(key: StorageKey, contribution: str, value_type: ConcreteType) -> None:
        previous = storage_adjoints.get(key)
        if previous is None:
            storage_adjoints[key] = contribution
            return
        result = names.fresh()
        lines.append(f"    {result} = arith.addf {previous}, {contribution} : {value_dtype(value_type)}")
        storage_adjoints[key] = result

    for value_id in reverse.reverse_order:
        node = nodes[value_id]
        if node.operation in {OpCode.STORE, OpCode.STORE_DYNAMIC}:
            key = storage_key(node)
            value_gradient = storage_adjoints.pop(key, None)
            if value_gradient is not None:
                stored = node.inputs[1] if node.operation is OpCode.STORE else node.inputs[-1]
                accumulate(stored, value_gradient)
            continue
        seed = cotangents.get(value_id)
        if seed is None:
            continue
        if node.operation in {OpCode.INDEX, OpCode.INDEX_DYNAMIC} and nodes[node.inputs[0]].type.kind == "tensor_view":
            add_storage(storage_key(node), seed, nodes[node.inputs[0]].type)
            continue
        if node.operation is OpCode.ADD:
            accumulate(node.inputs[0], seed)
            accumulate(node.inputs[1], seed)
        elif node.operation is OpCode.SUB:
            accumulate(node.inputs[0], seed)
            zero = _zero(node.type, names, lines)
            negative = names.fresh()
            lines.append(f"    {negative} = arith.subf {zero}, {seed} : {native_type(node.type)}")
            accumulate(node.inputs[1], negative)
        elif node.operation is OpCode.MUL:
            left = names.fresh()
            right = names.fresh()
            lines.append(f"    {left} = arith.mulf {seed}, {primal(node.inputs[1])} : {native_type(node.type)}")
            lines.append(f"    {right} = arith.mulf {seed}, {primal(node.inputs[0])} : {native_type(node.type)}")
            accumulate(node.inputs[0], left)
            accumulate(node.inputs[1], right)
        elif node.operation is OpCode.DIV:
            zero = _zero(node.type, names, lines)
            one = names.fresh()
            active = names.fresh()
            denominator = names.fresh()
            left, square, numerator, quotient, negative = (names.fresh() for _ in range(5))
            value_type = native_type(node.type)
            one_literal = "dense<1.0>" if node.type.kind == "tensor" else "1.0"
            lines.append(f"    {one} = arith.constant {one_literal} : {value_type}")
            if node.type.kind == "tensor":
                squared_seed = names.fresh()
                activity = names.fresh()
                element = node.type.arguments[0]
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
            zero = _zero(node.type, names, lines)
            negative = names.fresh()
            lines.append(f"    {negative} = arith.subf {zero}, {seed} : {native_type(node.type)}")
            accumulate(node.inputs[0], negative)
        elif node.operation is OpCode.IDENTITY:
            accumulate(node.inputs[0], seed)
        elif node.operation is OpCode.CONDITIONAL:
            condition = primal(node.inputs[0])
            zero = _zero(node.type, names, lines)
            true_seed = names.fresh()
            false_seed = names.fresh()
            value_type = native_type(node.type)
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
                lines.append(f"    {factor} = math.cos {primal(node.inputs[0])} : {native_type(node.type)}")
            elif node.operation is OpCode.COS:
                sine = names.fresh()
                zero = _zero(node.type, names, lines)
                lines.append(f"    {sine} = math.sin {primal(node.inputs[0])} : {native_type(node.type)}")
                lines.append(f"    {factor} = arith.subf {zero}, {sine} : {native_type(node.type)}")
            elif node.operation is OpCode.EXP:
                factor = primal(node.id)
            elif node.operation is OpCode.LOG:
                one = names.fresh()
                lines.append(f"    {one} = arith.constant 1.0 : {native_type(node.type)}")
                lines.append(f"    {factor} = arith.divf {one}, {primal(node.inputs[0])} : {native_type(node.type)}")
            else:
                half = names.fresh()
                lines.append(f"    {half} = arith.constant 0.5 : {native_type(node.type)}")
                lines.append(f"    {factor} = arith.divf {half}, {primal(node.id)} : {native_type(node.type)}")
            contribution = names.fresh()
            lines.append(f"    {contribution} = arith.mulf {seed}, {factor} : {native_type(node.type)}")
            accumulate(node.inputs[0], contribution)
        elif node.operation in {OpCode.SPLAT, OpCode.BROADCAST}:
            reduced = names.fresh()
            lines.append(
                f'    {reduced} = "vernon.intrinsic"({seed}) {{name = "reduce_sum_to_shape"}} : '
                f"({native_type(node.type)}) -> {native_type(nodes[node.inputs[0]].type)}"
            )
            accumulate(node.inputs[0], reduced)
        elif node.operation not in {OpCode.CONSTANT, OpCode.COMPARE}:
            raise DirectGpuLoweringError(f"cannot emit direct backward rule for '{node.operation.value}'")

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
        if value_type.kind == "tensor":
            element = value_type.arguments[0]
            shape = value_type.arguments[1:]
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
                f"[{', '.join(loop_indices)}] : {native_type(value_type)}"
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
        resource_mlir = resource_type(value_type, access)
        index_types = ", ".join("index" for _ in indices)
        payload_type = value_dtype(value_type) if value_type.kind == "tensor_view" else native_type(value_type)
        operation = "reduce_sum" if mode is AccumulationMode.REDUCE_SUM else "scatter_add"
        evidence = ", disjoint" if disjoint else ""
        lines.append(
            f'    "vernon.{operation}"({contribution}, {resource}, {", ".join(indices)}) '
            f"{{deterministic = false{evidence}}} : "
            f"({payload_type}, {resource_mlir}, {index_types}) -> ()"
        )

    for key, contribution in storage_adjoints.items():
        path = key[0]
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
            contribution = _zero(node.type, names, lines)
        mode = gradient_plans[path].mode
        indices = index_constants((0,), names, lines)
        accumulate_resource(contribution, gradient_binding[path], node.type, indices, mode)
    lines.extend((("        }", "      }", "    }") if serial else ("    }",)))
    lines.extend(("    func.return", "  }"))
    return module(lines, "backward", plan)


__all__ = ["DirectGpuLoweringError", "emit_backward", "emit_forward"]
