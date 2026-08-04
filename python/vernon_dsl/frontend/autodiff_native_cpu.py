from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import product

from .._versions import COMPILER_CONTRACT_VERSION, PIPELINE_VERSION
from .autodiff import (
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
from .autodiff_native_abi import gradient_type as native_gradient_type
from .autodiff_native_common import AutodiffNativeLoweringError
from .autodiff_native_math import emit_forward_math, emit_math_vjp
from .autodiff_profiles import AutodiffProfilePlan
from .model import ConcreteType


@dataclass
class _Names:
    index: int = 0

    def fresh(self) -> str:
        value = f"%ad{self.index}"
        self.index += 1
        return value


def _module(
    function: list[str],
    profile: str,
    plan: AutodiffProfilePlan,
    program: AutodiffProgram,
) -> str:
    attributes = (
        f"vernon.compiler_contract_version = {COMPILER_CONTRACT_VERSION} : i64, "
        f"vernon.pipeline_version = {PIPELINE_VERSION} : i64, "
        f'vernon.ad_profile = "{profile}", '
        f'vernon.ad_profiles_identity = "{plan.identity}"'
    )
    declarations: list[str] = []
    structs = dict(program.semantic.structs)
    for name in sorted(structs):
        fields = structs[name]
        field_text = ", ".join(json.dumps(f"{field_name}:{field_type.mlir}") for field_name, field_type in fields)
        leaf_dtypes = ", ".join(f'"{dtype}"' for dtype in _value_dtypes(ConcreteType("struct", name), structs))
        declarations.append(
            f'  "vernon.struct"() {{abi_leaf_dtypes = [{leaf_dtypes}], '
            f'fields = [{field_text}], sym_name = "{name}"}} : () -> ()'
        )
    return "\n".join([f"module attributes {{{attributes}}} {{", *declarations, *function, "}", ""])


def _entry_attributes(storage_effects: tuple[str, ...] = ()) -> str:
    effects = ", ".join(storage_effects)
    return (
        'attributes {vernon.entry, vernon.stage = "compute", '
        f"vernon.storage_effects = [{effects}], "
        "vernon.workgroup_size = array<i32: 1, 1, 1>}"
    )


def _abi_attributes(
    name: str,
    value_type: ConcreteType,
    interface: str,
    location: int,
    leaf_dtypes: tuple[str, ...],
    builtin: str | None = None,
) -> str:
    dtype = leaf_dtypes[0] if len(leaf_dtypes) == 1 else ""
    builtin_attribute = f', vernon.builtin = "{builtin}"' if builtin is not None else ""
    encoded_dtypes = ", ".join(f'"{value}"' for value in leaf_dtypes)
    element_attribute = f', vernon.element_abi_leaf_dtypes = ["{dtype}"]' if value_type.kind == "tensor" else ""
    return (
        f'{{vernon.interface = "{interface}", vernon.source_name = "{name}", '
        f'vernon.dtype = "{dtype}", vernon.abi_leaf_dtypes = [{encoded_dtypes}]{element_attribute}, '
        f"vernon.location = {location} : i64{builtin_attribute}}}"
    )


def _tensor_view_value_type(value_type: ConcreteType) -> ConcreteType:
    if value_type.kind != "tensor_view":
        return value_type
    element, shape, _, _ = value_type.arguments
    if not isinstance(element, ConcreteType) or not isinstance(shape, tuple):
        raise AutodiffNativeLoweringError("TensorView type is unresolved")
    return ConcreteType("tensor", "Tensor", (element, *shape))


def _native_mlir(value_type: ConcreteType) -> str:
    return _tensor_view_value_type(value_type).mlir


def _tensor_view_argument(name: str, value_type: ConcreteType, binding: int) -> str:
    return (
        f"%arg{binding}: {value_type.mlir} "
        f'{{vernon.interface = "resource", vernon.source_name = "{name}", '
        f'vernon.dtype = "{_value_dtype(value_type)}", '
        f'vernon.element_abi_leaf_dtypes = ["{_value_dtype(value_type)}"], '
        f"vernon.set = 0 : i64, vernon.binding = {binding} : i64}}"
    )


def _literal(node: ProgramGraphNode) -> str:
    if not isinstance(node.payload, LiteralOp):
        raise AutodiffNativeLoweringError("native autodiff constant has no literal")
    value = node.payload.value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        text = f"{value:.17g}"
        return text if "." in text or "e" in text.lower() else text + ".0"
    raise AutodiffNativeLoweringError("native autodiff constant has no literal")


def _value_dtype(value_type: ConcreteType) -> str:
    element = value_type.arguments[0] if value_type.kind in {"tensor", "tensor_view"} else value_type
    if not isinstance(element, ConcreteType) or element.kind != "scalar":
        raise AutodiffNativeLoweringError(f"{value_type.mlir} has no native floating element type")
    return element.name


def _input_abi_attributes(
    node: ProgramGraphNode,
    index: int,
    structs: dict[str, tuple[tuple[str, ConcreteType], ...]],
) -> str:
    builtin = node.payload.name if isinstance(node.payload, BuiltinOp) else None
    return _abi_attributes(
        node.source_name or "", node.type, "input", index, _value_dtypes(node.type, structs), builtin
    )


def _value_dtypes(
    value_type: ConcreteType,
    structs: dict[str, tuple[tuple[str, ConcreteType], ...]],
) -> tuple[str, ...]:
    if value_type.kind == "scalar":
        return (value_type.name,)
    if value_type.kind in {"tensor", "tensor_view"}:
        return (_value_dtype(value_type),)
    if value_type.kind == "tuple":
        return tuple(
            dtype
            for element in value_type.arguments
            if isinstance(element, ConcreteType)
            for dtype in _value_dtypes(element, structs)
        )
    if value_type.kind == "struct":
        return tuple(dtype for _, field in structs[value_type.name] for dtype in _value_dtypes(field, structs))
    return ()


def _constant(value: str, value_type: ConcreteType) -> str:
    return f"dense<{value}>" if value_type.kind in {"tensor", "tensor_view"} else value


def _static_indices(node: ProgramGraphNode) -> tuple[int, ...]:
    if not isinstance(node.payload, StaticIndicesOp) or not node.payload.indices:
        raise AutodiffNativeLoweringError(
            f"native autodiff requires static integer indices for '{node.operation.value}'"
        )
    return node.payload.indices


def _index_constants(indices: tuple[int, ...], names: _Names, lines: list[str]) -> tuple[str, ...]:
    results = tuple(names.fresh() for _ in indices)
    for result, index in zip(results, indices, strict=True):
        lines.append(f"    {result} = arith.constant {index} : index")
    return results


def _replace_tensor_element(
    tensor: str,
    value_type: ConcreteType,
    indices: tuple[int, ...],
    replacement: str,
    names: _Names,
    lines: list[str],
) -> str:
    tensor_type = _tensor_view_value_type(value_type)
    shape = tensor_type.arguments[1:]
    elements: list[str] = []
    for current_indices in product(*(range(extent) for extent in shape)):
        if current_indices == indices:
            elements.append(replacement)
            continue
        index_values = _index_constants(current_indices, names, lines)
        element = names.fresh()
        lines.append(f"    {element} = tensor.extract {tensor}[{', '.join(index_values)}] : {tensor_type.mlir}")
        elements.append(element)
    result = names.fresh()
    element_type = _value_dtype(value_type)
    operand_types = ", ".join(element_type for _ in elements)
    lines.append(
        f'    {result} = "vernon.intrinsic"({", ".join(elements)}) {{name = "construct"}} : '
        f"({operand_types}) -> {tensor_type.mlir}"
    )
    return result


def _dynamic_index_condition(
    indices: tuple[str, ...],
    index_types: tuple[ConcreteType, ...],
    current: tuple[int, ...],
    names: _Names,
    lines: list[str],
) -> str:
    comparisons: list[str] = []
    for operand, value_type, literal in zip(indices, index_types, current, strict=True):
        constant = names.fresh()
        lines.append(f"    {constant} = arith.constant {literal} : {_native_mlir(value_type)}")
        comparison = names.fresh()
        lines.append(f"    {comparison} = arith.cmpi eq, {operand}, {constant} : {_native_mlir(value_type)}")
        comparisons.append(comparison)
    condition = comparisons[0]
    for comparison in comparisons[1:]:
        combined = names.fresh()
        lines.append(f"    {combined} = arith.andi {condition}, {comparison} : i1")
        condition = combined
    return condition


def _replace_tensor_element_dynamic(
    tensor: str,
    value_type: ConcreteType,
    indices: tuple[str, ...],
    index_types: tuple[ConcreteType, ...],
    replacement: str,
    names: _Names,
    lines: list[str],
) -> str:
    tensor_type = _tensor_view_value_type(value_type)
    shape = tensor_type.arguments[1:]
    elements: list[str] = []
    for current in product(*(range(extent) for extent in shape)):
        index_values = _index_constants(current, names, lines)
        element = names.fresh()
        lines.append(f"    {element} = tensor.extract {tensor}[{', '.join(index_values)}] : {tensor_type.mlir}")
        condition = _dynamic_index_condition(indices, index_types, current, names, lines)
        selected = names.fresh()
        lines.append(
            f"    {selected} = arith.select {condition}, {replacement}, {element} : {_value_dtype(value_type)}"
        )
        elements.append(selected)
    result = names.fresh()
    element_types = ", ".join(_value_dtype(value_type) for _ in elements)
    lines.append(
        f'    {result} = "vernon.intrinsic"({", ".join(elements)}) {{name = "construct"}} : '
        f"({element_types}) -> {tensor_type.mlir}"
    )
    return result


def _extract_tensor_element_dynamic(
    tensor: str,
    value_type: ConcreteType,
    indices: tuple[str, ...],
    index_types: tuple[ConcreteType, ...],
    names: _Names,
    lines: list[str],
) -> str:
    tensor_type = _tensor_view_value_type(value_type)
    shape = tensor_type.arguments[1:]
    result: str | None = None
    for current in product(*(range(extent) for extent in shape)):
        index_values = _index_constants(current, names, lines)
        element = names.fresh()
        lines.append(f"    {element} = tensor.extract {tensor}[{', '.join(index_values)}] : {tensor_type.mlir}")
        if result is None:
            result = element
            continue
        condition = _dynamic_index_condition(indices, index_types, current, names, lines)
        selected = names.fresh()
        lines.append(f"    {selected} = arith.select {condition}, {element}, {result} : {_value_dtype(value_type)}")
        result = selected
    assert result is not None
    return result


def _storage_effect(kind: str, parameter: str, indices: tuple[int, ...]) -> str:
    values = ", ".join(str(index) for index in indices)
    return f'{{kind = "{kind}", owner = "{parameter}", region = "element", indices = array<i64: {values}>}}'


def _emit_operation(
    node: ProgramGraphNode,
    operands: tuple[str, ...],
    operand_types: tuple[ConcreteType, ...],
    names: _Names,
    lines: list[str],
    structs: dict[str, tuple[tuple[str, ConcreteType], ...]],
) -> str:
    result = names.fresh()
    value_type = _native_mlir(node.type)
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
        lines.append(f"    {result} = {binary[node.operation]} {operands[0]}, {operands[1]} : {value_type}")
    elif node.operation is OpCode.COMPARE:
        assert isinstance(node.payload, NamedOp)
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
            else {
                "Eq": "oeq",
                "NotEq": "une",
                "Lt": "olt",
                "LtE": "ole",
                "Gt": "ogt",
                "GtE": "oge",
            }
        )
        predicate = predicates.get(node.payload.name)
        if predicate is None:
            raise AutodiffNativeLoweringError(f"cannot emit native comparison '{node.payload.name}'")
        comparison = "arith.cmpi" if integer_comparison else "arith.cmpf"
        lines.append(f"    {result} = {comparison} {predicate}, {operands[0]}, {operands[1]} : {operand_types[0].mlir}")
    elif node.operation is OpCode.CONSTANT:
        literal = _literal(node)
        lines.append(
            f"    {result} = arith.constant "
            f"{f'dense<{literal}>' if node.type.kind == 'tensor' else literal} : {value_type}"
        )
    elif node.operation is OpCode.INDEX:
        static_indices = _static_indices(node)
        if operand_types[0].kind == "tuple":
            index = static_indices[0]
            lines.append(
                f'    {result} = "vernon.tuple_get"({operands[0]}) {{index = {index} : i64}} : '
                f"({_native_mlir(operand_types[0])}) -> {value_type}"
            )
        else:
            indices = _index_constants(static_indices, names, lines)
            lines.append(
                f"    {result} = tensor.extract {operands[0]}[{', '.join(indices)}] : {_native_mlir(operand_types[0])}"
            )
    elif node.operation is OpCode.INDEX_DYNAMIC:
        return _extract_tensor_element_dynamic(
            operands[0],
            operand_types[0],
            operands[1:],
            operand_types[1:],
            names,
            lines,
        )
    elif node.operation is OpCode.STORE:
        return _replace_tensor_element(
            operands[0],
            operand_types[0],
            _static_indices(node),
            operands[1],
            names,
            lines,
        )
    elif node.operation is OpCode.STORE_DYNAMIC:
        return _replace_tensor_element_dynamic(
            operands[0],
            operand_types[0],
            operands[1:-1],
            operand_types[1:-1],
            operands[-1],
            names,
            lines,
        )
    elif node.operation is OpCode.NEG:
        zero = names.fresh()
        lines.append(f"    {zero} = arith.constant {_constant('0.0', node.type)} : {value_type}")
        lines.append(f"    {result} = arith.subf {zero}, {operands[0]} : {value_type}")
    elif node.operation is OpCode.IDENTITY:
        return operands[0]
    elif node.operation is OpCode.FIELD:
        assert isinstance(node.payload, NamedOp)
        owner = operand_types[0]
        fields = structs[owner.name]
        index = next(index for index, (name, _) in enumerate(fields) if name == node.payload.name)
        lines.append(
            f'{result} = "vernon.struct_get"({operands[0]}) '
            f'{{field = "{node.payload.name}", index = {index} : i64}} : '
            f"({_native_mlir(owner)}) -> {value_type}"
        )
    elif node.operation is OpCode.SPLAT:
        lines.append(f"    {result} = tensor.splat {operands[0]} : {value_type}")
    elif node.operation is OpCode.BROADCAST:
        lines.append(
            f'    {result} = "vernon.intrinsic"({operands[0]}) {{name = "broadcast"}} : '
            f"({_native_mlir(operand_types[0])}) -> {value_type}"
        )
    elif node.operation in {OpCode.SIN, OpCode.COS, OpCode.EXP, OpCode.LOG, OpCode.SQRT}:
        lines.append(f"    {result} = math.{node.operation.value} {operands[0]} : {value_type}")
    else:
        math_result = emit_forward_math(node, operands, operand_types, names, lines)
        if math_result is None:
            raise AutodiffNativeLoweringError(f"cannot emit native operation '{node.operation.value}'")
        return math_result
    return result


def _forward_function(
    program: AutodiffProgram,
    plan: AutodiffProfilePlan,
    *,
    symbol: str | None = None,
    entry: bool = True,
    return_storage: bool = False,
) -> list[str]:
    graph = program.semantic
    reverse = program.reverse
    profile = next(value for value in plan.profiles if value.name == "forward_with_tape")
    symbol = symbol or profile.symbol
    nodes = {node.id: node for node in graph.nodes}
    structs = dict(graph.structs)
    parameters = tuple(node for node in graph.nodes if node.operation in {OpCode.PARAMETER, OpCode.BUILTIN})
    arguments = ", ".join(
        (
            _tensor_view_argument(node.source_name or "", node.type, index)
            if entry and node.type.kind == "tensor_view"
            else f"%arg{index}: {_native_mlir(node.type)}"
            + (f" {_input_abi_attributes(node, index, structs)}" if entry else "")
        )
        for index, node in enumerate(parameters)
    )
    tape_value_types = tuple(nodes[value].type for value in reverse.saved_values)
    tape_types = tuple(_native_mlir(value_type) for value_type in tape_value_types)
    tape_type = f"tuple<{', '.join(tape_types)}>"
    output = nodes[graph.outputs[0]]
    storage_result_types = tuple(_native_mlir(nodes[value].type) for _, value in graph.storage_outputs)
    tape_dtypes = ", ".join(f'"{_value_dtype(value_type)}"' for value_type in tape_value_types)
    result_type = (
        f"tuple<{', '.join((_native_mlir(output.type), tape_type, *storage_result_types))}>"
        if return_storage
        else f"tuple<{_native_mlir(output.type)}, {tape_type}>"
    )
    result_dtypes = ", ".join(value for value in (f'"{_value_dtype(output.type)}"', tape_dtypes) if value)
    result_attributes = (
        '{vernon.interface = "output", vernon.source_name = "forward_result", '
        f"vernon.abi_leaf_dtypes = [{result_dtypes}], vernon.location = 0 : i64}}"
    )
    result = f"({result_type} {result_attributes})" if entry else result_type

    def storage_owner(value: NodeId) -> str:
        owner = storage_parameter_owner(value, nodes)
        if owner is None or owner.source_name is None:
            raise AutodiffNativeLoweringError("native Storage operation has no TensorView parameter owner")
        return owner.source_name

    storage_effects = tuple(
        (
            _storage_effect(
                "write" if node.operation is OpCode.STORE else "read",
                storage_owner(node.inputs[0]),
                _static_indices(node),
            )
            if node.operation in {OpCode.INDEX, OpCode.STORE}
            else (
                f'{{kind = "{"write" if node.operation is OpCode.STORE_DYNAMIC else "read"}", '
                f'owner = "{storage_owner(node.inputs[0])}", region = "unknown"}}'
            )
        )
        for node in graph.nodes
        if node.operation in {OpCode.INDEX, OpCode.STORE, OpCode.INDEX_DYNAMIC, OpCode.STORE_DYNAMIC}
        and nodes[node.inputs[0]].type.kind == "tensor_view"
    )
    attributes = f" {_entry_attributes(storage_effects)}" if entry else ""
    lines = [f"  func.func @{symbol}({arguments}) -> {result}{attributes} {{"]
    values = {
        node.id: f"%arg{index}" for index, node in enumerate(parameters) if not entry or node.type.kind != "tensor_view"
    }
    names = _Names()
    if entry:
        for index, parameter in enumerate(parameters):
            if parameter.type.kind != "tensor_view":
                continue
            tensor_type = _tensor_view_value_type(parameter.type)
            zero = names.fresh()
            lines.append(f"    {zero} = arith.constant dense<0.0> : {tensor_type.mlir}")
            current = zero
            _, shape, access, _ = parameter.type.arguments
            assert isinstance(shape, tuple)
            if access != "write":
                loaded_elements: list[str] = []
                for indices in product(*(range(extent) for extent in shape)):
                    index_values = _index_constants(indices, names, lines)
                    loaded = names.fresh()
                    lines.append(
                        f'    {loaded} = "vernon.load"(%arg{index}, {", ".join(index_values)}) : '
                        f"({parameter.type.mlir}, {', '.join('index' for _ in indices)}) -> "
                        f"{_value_dtype(parameter.type)}"
                    )
                    loaded_elements.append(loaded)
                constructed = names.fresh()
                element_type = _value_dtype(parameter.type)
                operand_types = ", ".join(element_type for _ in loaded_elements)
                lines.append(
                    f'    {constructed} = "vernon.intrinsic"({", ".join(loaded_elements)}) '
                    f'{{name = "construct"}} : ({operand_types}) -> {tensor_type.mlir}'
                )
                current = constructed
            values[parameter.id] = current

    def dependencies(value: NodeId) -> set[NodeId]:
        result = {value}
        for operand in nodes[value].inputs:
            result.update(dependencies(operand))
        return result

    def zero(value: NodeId, target_lines: list[str]) -> str:
        result = names.fresh()
        value_type = nodes[value].type
        literal = "0" if value_type.kind == "scalar" and value_type.name == "bool" else _constant("0.0", value_type)
        target_lines.append(f"    {result} = arith.constant {literal} : {_native_mlir(value_type)}")
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
            exclusive = ((true_dependencies ^ false_dependencies) & set(reverse.saved_values)).difference(scope)
            saved = sorted(exclusive)
            result_ids = [node.id, *saved]
            result_names = [names.fresh() for _ in result_ids]
            result_types = [_native_mlir(nodes[item].type) for item in result_ids]
            target_lines.append(f"    {', '.join(result_names)} = scf.if {condition} -> ({', '.join(result_types)}) {{")
            for branch_index, branch_dependencies in ((1, true_dependencies), (2, false_dependencies)):
                branch_scope = dict(scope)
                branch_lines: list[str] = []
                branch_value = emit(node.inputs[branch_index], branch_scope, branch_lines)
                yielded = [branch_value]
                for saved_value in saved:
                    yielded.append(
                        emit(saved_value, branch_scope, branch_lines)
                        if saved_value in branch_dependencies
                        else zero(saved_value, branch_lines)
                    )
                target_lines.extend(branch_lines)
                target_lines.append(f"    scf.yield {', '.join(yielded)} : {', '.join(result_types)}")
                if branch_index == 1:
                    target_lines.append("    } else {")
            target_lines.append("    }")
            scope.update(zip(result_ids, result_names, strict=True))
            return result_names[0]
        operands = tuple(emit(operand, scope, target_lines) for operand in node.inputs)
        result = _emit_operation(
            node,
            operands,
            tuple(nodes[operand].type for operand in node.inputs),
            names,
            target_lines,
            structs,
        )
        scope[value] = result
        return result

    for _, value in graph.storage_outputs:
        emit(value, values, lines)
    emit(graph.outputs[0], values, lines)
    for value in reverse.saved_values:
        emit(value, values, lines)
    if entry:
        parameter_indices = {
            node.source_name: index
            for index, node in enumerate(parameters)
            if node.source_name is not None and node.type.kind == "tensor_view"
        }
        for parameter, value in graph.storage_outputs:
            parameter_node = next(
                node for node in parameters if node.source_name == parameter and node.type.kind == "tensor_view"
            )
            _, shape, _, _ = parameter_node.type.arguments
            assert isinstance(shape, tuple)
            for indices in product(*(range(extent) for extent in shape)):
                index_values = _index_constants(indices, names, lines)
                element = names.fresh()
                tensor_type = _tensor_view_value_type(parameter_node.type)
                lines.append(
                    f"    {element} = tensor.extract {values[value]}[{', '.join(index_values)}] : {tensor_type.mlir}"
                )
                resource_index = parameter_indices[parameter]
                lines.append(
                    f'    "vernon.store"({element}, %arg{resource_index}, {", ".join(index_values)}) : '
                    f"({_value_dtype(parameter_node.type)}, {parameter_node.type.mlir}, "
                    f"{', '.join('index' for _ in indices)}) -> ()"
                )
    tape = names.fresh()
    tape_operands = ", ".join(values[value] for value in reverse.saved_values)
    tape_operand_types = ", ".join(tape_types)
    lines.append(f'    {tape} = "vernon.tuple_create"({tape_operands}) : ({tape_operand_types}) -> {tape_type}')
    result = names.fresh()
    result_values = (values[graph.outputs[0]], tape)
    result_value_types = (_native_mlir(output.type), tape_type)
    if return_storage:
        result_values = (*result_values, *(values[value] for _, value in graph.storage_outputs))
        result_value_types = (*result_value_types, *storage_result_types)
    lines.append(
        f'    {result} = "vernon.tuple_create"({", ".join(result_values)}) : '
        f"({', '.join(result_value_types)}) -> {result_type}"
    )
    lines.append(f"    func.return {result} : {result_type}")
    lines.append("  }")
    return lines


def emit_forward(program: AutodiffProgram, plan: AutodiffProfilePlan) -> str:
    return _module(_forward_function(program, plan), "forward_with_tape", plan, program)


def _backward_function(
    program: AutodiffProgram,
    plan: AutodiffProfilePlan,
    *,
    symbol: str | None = None,
    entry: bool = True,
) -> list[str]:
    graph = program.semantic
    reverse = program.reverse
    profile = next(value for value in plan.profiles if value.name == "backward")
    symbol = symbol or profile.symbol
    nodes = {node.id: node for node in graph.nodes}
    structs = dict(graph.structs)
    output = nodes[graph.outputs[0]]
    tape_value_types = tuple(nodes[value].type for value in reverse.saved_values)
    tape_types = tuple(_native_mlir(value_type) for value_type in tape_value_types)
    tape_type = f"tuple<{', '.join(tape_types)}>"
    tape_dtypes = ", ".join(f'"{_value_dtype(value_type)}"' for value_type in tape_value_types)
    tape_attributes = (
        '{vernon.interface = "input", vernon.source_name = "tape", '
        f"vernon.abi_leaf_dtypes = [{tape_dtypes}], vernon.location = 0 : i64}}"
    )
    cotangent_attributes = _abi_attributes(
        "output",
        native_gradient_type(output.type),
        "input",
        1,
        _value_dtypes(native_gradient_type(output.type), structs),
    )
    parameter_types = {
        node.source_name: node.type
        for node in graph.nodes
        if node.operation is OpCode.PARAMETER and node.source_name is not None
    }

    def resolve_path(path: str) -> ConcreteType:
        components = path.split(".")
        value_type = parameter_types[components[0]]
        for component in components[1:]:
            if value_type.kind == "struct":
                value_type = dict(structs[value_type.name])[component]
            elif value_type.kind == "tuple":
                element = value_type.arguments[int(component)]
                if not isinstance(element, ConcreteType):
                    raise AutodiffNativeLoweringError(f"cannot resolve native autodiff path '{path}'")
                value_type = element
            else:
                raise AutodiffNativeLoweringError(f"cannot resolve native autodiff path '{path}'")
        return value_type

    node_paths: dict[NodeId, str] = {}
    nodes_by_path: dict[str, list[ProgramGraphNode]] = {}
    for node in graph.nodes:
        path: str | None = None
        if node.operation is OpCode.PARAMETER:
            path = node.source_name
        elif node.operation is OpCode.FIELD and isinstance(node.payload, NamedOp):
            owner = node_paths.get(node.inputs[0])
            if owner is not None:
                path = f"{owner}.{node.payload.name}"
        elif node.operation is OpCode.INDEX and nodes[node.inputs[0]].type.kind == "tuple":
            owner = node_paths.get(node.inputs[0])
            indices = _static_indices(node)
            if owner is not None and len(indices) == 1:
                path = f"{owner}.{indices[0]}"
        if path is not None:
            node_paths[node.id] = path
            nodes_by_path.setdefault(path, []).append(node)

    gradient_targets = [
        (
            binding.path,
            native_gradient_type(resolve_path(binding.path)),
            tuple(nodes_by_path.get(binding.path, ())),
        )
        for binding in profile.outputs
    ]
    gradient_type = (
        _native_mlir(gradient_targets[0][1])
        if len(gradient_targets) == 1
        else f"tuple<{', '.join(_native_mlir(value_type) for _, value_type, _ in gradient_targets)}>"
    )
    gradient_dtypes = ", ".join(f'"{_value_dtype(value_type)}"' for _, value_type, _ in gradient_targets)
    gradient_attributes = (
        '{vernon.interface = "output", vernon.source_name = "gradients", '
        f"vernon.abi_leaf_dtypes = [{gradient_dtypes}], vernon.location = 0 : i64}}"
    )
    tape_annotation = f" {tape_attributes}" if entry else ""
    cotangent_annotation = f" {cotangent_attributes}" if entry else ""
    result = f"({gradient_type} {gradient_attributes})" if entry else gradient_type
    attributes = f" {_entry_attributes()}" if entry else ""
    lines = [
        f"  func.func @{symbol}("
        f"%tape: {tape_type}{tape_annotation}, "
        f"%cotangent: {_native_mlir(native_gradient_type(output.type))}{cotangent_annotation}) -> "
        f"{result}{attributes} {{"
    ]
    names = _Names()
    primal: dict[NodeId, str] = {}
    for index, value in enumerate(reverse.saved_values):
        extracted = names.fresh()
        value_type = _native_mlir(tape_value_types[index])
        lines.append(
            f'    {extracted} = "vernon.tuple_get"(%tape) {{index = {index} : i64}} : ({tape_type}) -> {value_type}'
        )
        derivative_type = native_gradient_type(tape_value_types[index])
        if derivative_type != tape_value_types[index]:
            promoted = names.fresh()
            lines.append(f"    {promoted} = arith.extf {extracted} : {value_type} to {_native_mlir(derivative_type)}")
            extracted = promoted
        primal[value] = extracted
    gradient_nodes = tuple(node for _, _, target_nodes in gradient_targets for node in target_nodes)
    gradient_node_ids = {node.id for node in gradient_nodes}

    def accumulate(target: dict[NodeId, str], value: NodeId, contribution: str, target_lines: list[str]) -> None:
        previous = target.get(value)
        if previous is None:
            target[value] = contribution
            return
        result = names.fresh()
        target_lines.append(
            f"    {result} = arith.addf {previous}, {contribution} : "
            f"{_native_mlir(native_gradient_type(nodes[value].type))}"
        )
        target[value] = result

    def merge(target: dict[NodeId, str], source: dict[NodeId, str], target_lines: list[str]) -> None:
        for value, contribution in source.items():
            accumulate(target, value, contribution, target_lines)

    def constant(value_type: ConcreteType, value: str, target_lines: list[str]) -> str:
        result = names.fresh()
        target_lines.append(
            f"    {result} = arith.constant {_constant(value, value_type)} : {_native_mlir(value_type)}"
        )
        return result

    def unary(operation: str, operand: str, value_type: ConcreteType, target_lines: list[str]) -> str:
        result = names.fresh()
        target_lines.append(f"    {result} = math.{operation} {operand} : {_native_mlir(value_type)}")
        return result

    def backprop(value: NodeId, seed: str, target_lines: list[str]) -> dict[NodeId, str]:
        node = nodes[value]
        value_type = native_gradient_type(node.type)
        if value in gradient_node_ids:
            return {value: seed}
        if node.operation is OpCode.PARAMETER:
            return {}
        if node.operation in {OpCode.CONSTANT, OpCode.COMPARE}:
            return {}
        if node.operation is OpCode.CONDITIONAL:
            condition = primal[node.inputs[0]]
            result_names = [names.fresh() for _ in gradient_nodes]
            result_types = [_native_mlir(native_gradient_type(gradient.type)) for gradient in gradient_nodes]
            target_lines.append(f"    {', '.join(result_names)} = scf.if {condition} -> ({', '.join(result_types)}) {{")
            for branch_index in (1, 2):
                branch_lines: list[str] = []
                gradients = backprop(node.inputs[branch_index], seed, branch_lines)
                yielded = [
                    gradients.get(gradient.id) or constant(native_gradient_type(gradient.type), "0.0", branch_lines)
                    for gradient in gradient_nodes
                ]
                target_lines.extend(branch_lines)
                target_lines.append(f"    scf.yield {', '.join(yielded)} : {', '.join(result_types)}")
                if branch_index == 1:
                    target_lines.append("    } else {")
            target_lines.append("    }")
            return {gradient.id: result for gradient, result in zip(gradient_nodes, result_names, strict=True)}

        contributions: tuple[str | None, ...]
        if node.operation is OpCode.ADD:
            contributions = (seed, seed)
        elif node.operation is OpCode.SUB:
            contributions = (seed, constant(value_type, "0.0", target_lines))
            negative = names.fresh()
            target_lines.append(f"    {negative} = arith.subf {contributions[1]}, {seed} : {value_type.mlir}")
            contributions = (seed, negative)
        elif node.operation is OpCode.MUL:
            contributions = tuple(names.fresh() for _ in range(2))
            target_lines.append(
                f"    {contributions[0]} = arith.mulf {seed}, {primal[node.inputs[1]]} : {value_type.mlir}"
            )
            target_lines.append(
                f"    {contributions[1]} = arith.mulf {seed}, {primal[node.inputs[0]]} : {value_type.mlir}"
            )
        elif node.operation is OpCode.DIV:
            left, square, numerator, quotient, negative = (names.fresh() for _ in range(5))
            target_lines.append(f"    {left} = arith.divf {seed}, {primal[node.inputs[1]]} : {value_type.mlir}")
            target_lines.append(
                f"    {square} = arith.mulf {primal[node.inputs[1]]}, {primal[node.inputs[1]]} : {value_type.mlir}"
            )
            target_lines.append(f"    {numerator} = arith.mulf {seed}, {primal[node.inputs[0]]} : {value_type.mlir}")
            target_lines.append(f"    {quotient} = arith.divf {numerator}, {square} : {value_type.mlir}")
            zero = constant(value_type, "0.0", target_lines)
            target_lines.append(f"    {negative} = arith.subf {zero}, {quotient} : {value_type.mlir}")
            contributions = (left, negative)
        elif node.operation is OpCode.NEG:
            zero = constant(value_type, "0.0", target_lines)
            negative = names.fresh()
            target_lines.append(f"    {negative} = arith.subf {zero}, {seed} : {value_type.mlir}")
            contributions = (negative,)
        elif node.operation is OpCode.IDENTITY:
            contributions = (seed,)
        elif node.operation is OpCode.INDEX:
            storage_type = native_gradient_type(nodes[node.inputs[0]].type)
            if storage_type.kind == "tuple":
                contributions = (None,)
            else:
                gradient = constant(storage_type, "0.0", target_lines)
                inserted = _replace_tensor_element(
                    gradient,
                    storage_type,
                    _static_indices(node),
                    seed,
                    names,
                    target_lines,
                )
                contributions = (inserted,)
        elif node.operation is OpCode.INDEX_DYNAMIC:
            storage_type = native_gradient_type(nodes[node.inputs[0]].type)
            gradient = constant(storage_type, "0.0", target_lines)
            inserted = _replace_tensor_element_dynamic(
                gradient,
                storage_type,
                tuple(primal[input_id] for input_id in node.inputs[1:]),
                tuple(nodes[input_id].type for input_id in node.inputs[1:]),
                seed,
                names,
                target_lines,
            )
            contributions = (inserted, *(None for _ in node.inputs[1:]))
        elif node.operation is OpCode.STORE:
            storage_type = native_gradient_type(nodes[node.inputs[0]].type)
            indices = _index_constants(_static_indices(node), names, target_lines)
            value_gradient = names.fresh()
            target_lines.append(
                f"    {value_gradient} = tensor.extract {seed}[{', '.join(indices)}] : {_native_mlir(storage_type)}"
            )
            zero = constant(native_gradient_type(nodes[node.inputs[1]].type), "0.0", target_lines)
            storage_gradient = _replace_tensor_element(
                seed,
                storage_type,
                _static_indices(node),
                zero,
                names,
                target_lines,
            )
            contributions = (storage_gradient, value_gradient)
        elif node.operation is OpCode.STORE_DYNAMIC:
            storage_type = native_gradient_type(nodes[node.inputs[0]].type)
            dynamic_values = tuple(primal[input_id] for input_id in node.inputs[1:-1])
            dynamic_types = tuple(nodes[input_id].type for input_id in node.inputs[1:-1])
            value_gradient = _extract_tensor_element_dynamic(
                seed,
                storage_type,
                dynamic_values,
                dynamic_types,
                names,
                target_lines,
            )
            zero = constant(native_gradient_type(nodes[node.inputs[-1]].type), "0.0", target_lines)
            storage_gradient = _replace_tensor_element_dynamic(
                seed,
                storage_type,
                dynamic_values,
                dynamic_types,
                zero,
                names,
                target_lines,
            )
            contributions = (
                storage_gradient,
                *(None for _ in node.inputs[1:-1]),
                value_gradient,
            )
        elif node.operation in {OpCode.SPLAT, OpCode.BROADCAST}:
            reduced = names.fresh()
            operand_type = native_gradient_type(nodes[node.inputs[0]].type)
            target_lines.append(
                f'    {reduced} = "vernon.intrinsic"({seed}) {{name = "reduce_sum_to_shape"}} : '
                f"({_native_mlir(value_type)}) -> {_native_mlir(operand_type)}"
            )
            contributions = (reduced,)
        elif node.operation is OpCode.FIELD:
            contributions = (None,)
        elif node.operation in {OpCode.SIN, OpCode.COS, OpCode.EXP, OpCode.LOG, OpCode.SQRT}:
            if node.operation is OpCode.SIN:
                factor = unary("cos", primal[node.inputs[0]], value_type, target_lines)
            elif node.operation is OpCode.COS:
                sine = unary("sin", primal[node.inputs[0]], value_type, target_lines)
                zero = constant(value_type, "0.0", target_lines)
                factor = names.fresh()
                target_lines.append(f"    {factor} = arith.subf {zero}, {sine} : {value_type.mlir}")
            elif node.operation is OpCode.EXP:
                factor = primal[node.id]
            elif node.operation is OpCode.LOG:
                one = constant(value_type, "1.0", target_lines)
                factor = names.fresh()
                target_lines.append(f"    {factor} = arith.divf {one}, {primal[node.inputs[0]]} : {value_type.mlir}")
            else:
                half = constant(value_type, "0.5", target_lines)
                factor = names.fresh()
                target_lines.append(f"    {factor} = arith.divf {half}, {primal[node.id]} : {value_type.mlir}")
            contribution = names.fresh()
            target_lines.append(f"    {contribution} = arith.mulf {seed}, {factor} : {value_type.mlir}")
            contributions = (contribution,)
        else:
            contributions = emit_math_vjp(
                node,
                seed,
                lambda item: primal[item],
                nodes,
                names,
                target_lines,
                native_gradient_type,
            )
            if contributions is None:
                raise AutodiffNativeLoweringError(f"cannot emit backward rule for '{node.operation.value}'")

        result: dict[NodeId, str] = {}
        for operand, contribution in zip(node.inputs, contributions, strict=True):
            if contribution is None:
                continue
            merge(result, backprop(operand, contribution, target_lines), target_lines)
        return result

    adjoints = backprop(graph.outputs[0], "%cotangent", lines)
    result_values: list[str] = []
    for _, value_type, target_nodes in gradient_targets:
        gradient: str | None = None
        for node in target_nodes:
            contribution = adjoints.get(node.id)
            if contribution is None:
                continue
            if gradient is None:
                gradient = contribution
            else:
                combined = names.fresh()
                lines.append(f"    {combined} = arith.addf {gradient}, {contribution} : {_native_mlir(value_type)}")
                gradient = combined
        if gradient is None:
            gradient = names.fresh()
            lines.append(f"    {gradient} = arith.constant {_constant('0.0', value_type)} : {_native_mlir(value_type)}")
        result_values.append(gradient)
    if len(result_values) == 1:
        result = result_values[0]
    else:
        result = names.fresh()
        lines.append(
            f'    {result} = "vernon.tuple_create"({", ".join(result_values)}) : '
            f"({', '.join(_native_mlir(value_type) for _, value_type, _ in gradient_targets)}) -> {gradient_type}"
        )
    lines.append(f"    func.return {result} : {gradient_type}")
    lines.append("  }")
    return lines


def emit_backward(program: AutodiffProgram, plan: AutodiffProfilePlan) -> str:
    return _module(_backward_function(program, plan), "backward", plan, program)


__all__ = ["emit_backward", "emit_forward"]
