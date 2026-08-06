from __future__ import annotations

import ast
from typing import Protocol

from ..language.ast_utils import dotted_name
from .control_flow_lowering import assigned_names, emit_loop_body, emit_source_block, return_state_items, yield_values
from .lowering_types import DslType, FunctionSignature, ModuleContext, Value
from .model import TypedStatement


class LoopEmitter(Protocol):
    context: ModuleContext
    signature: FunctionSignature
    indent: int
    environment: dict[str, Value]
    loop_controls: list[str]
    return_flag_name: str | None
    typed_statements: dict[int, TypedStatement]

    def _expression(self, node: ast.expr, expected: DslType | None = None) -> Value: ...

    def _coerce_implicit(self, node: ast.AST, value: Value, target: DslType) -> Value: ...

    def _require_same_type(self, node: ast.AST, expected: DslType, actual: DslType) -> None: ...

    def _hidden_name(self, prefix: str) -> str: ...

    def _control_constant(self, value: int) -> Value: ...

    def _bool_constant(self, value: bool) -> Value: ...

    def _fresh(self) -> str: ...

    def _line(self, text: str) -> None: ...


def lower_for(emitter: LoopEmitter, node: ast.For) -> None:
    if (
        not isinstance(node.target, ast.Name)
        or not isinstance(node.iter, ast.Call)
        or dotted_name(node.iter.func) != "range"
    ):
        raise emitter.context.error(node, "for loops must have the form 'for name in range(...)'")
    if not 1 <= len(node.iter.args) <= 3 or node.iter.keywords:
        raise emitter.context.error(node.iter, "range requires one to three positional i32 arguments")
    integer_type = DslType("scalar", "i32")
    arguments = [
        emitter._coerce_implicit(argument, emitter._expression(argument, integer_type), integer_type)
        for argument in node.iter.args
    ]
    zero = emitter._control_constant(0)
    one = emitter._control_constant(1)
    if len(arguments) == 1:
        start, stop, step = zero, arguments[0], one
    elif len(arguments) == 2:
        start, stop = arguments
        step = one
    else:
        start, stop, step = arguments
    literal_step = _literal_integer(node.iter.args[2]) if len(node.iter.args) == 3 else None
    if literal_step == 0:
        raise emitter.context.error(node.iter.args[2], "range step must not be zero")
    if len(node.iter.args) == 3 and literal_step is None:
        nonzero = emitter._fresh()
        emitter._line(f"{nonzero} = arith.cmpi ne, {step.name}, {zero.name} : i32")
        emitter._line(f'cf.assert {nonzero}, "range step must not be zero"')

    outer = emitter.environment.copy()
    typed_statement = emitter.typed_statements[id(node)]
    control_name = emitter._hidden_name("loop_control")
    current_name = emitter._hidden_name("range_current")
    active_name = emitter._hidden_name("range_active")
    outer[control_name] = emitter._control_constant(0)
    outer[current_name] = start
    outer[active_name] = emitter._bool_constant(True)
    carried_names = [merge.name for merge in typed_statement.branch_merges]
    carried_types = [merge.type for merge in typed_statement.branch_merges]
    carried_names.extend((current_name, active_name, control_name))
    carried_types.extend((integer_type, DslType("scalar", "bool"), integer_type))
    _append_return_state(emitter, carried_names, carried_types)
    for name, value_type in zip(carried_names, carried_types, strict=True):
        outer[name] = emitter._coerce_implicit(node, outer[name], value_type)

    results = [emitter._fresh() for _ in carried_names]
    before_arguments = [emitter._fresh() for _ in carried_names]
    operand_types = ", ".join(value_type.mlir for value_type in carried_types)
    assignments = ", ".join(
        f"{argument} = {outer[name].name}" for argument, name in zip(before_arguments, carried_names, strict=True)
    )
    emitter._line(f"{', '.join(results)} = scf.while ({assignments}) : ({operand_types}) -> ({operand_types}) {{")
    emitter.indent += 1
    emitter.environment = outer.copy()
    for name, value_type, argument in zip(carried_names, carried_types, before_arguments, strict=True):
        emitter.environment[name] = Value(argument, value_type)
    step_positive = emitter._fresh()
    emitter._line(f"{step_positive} = arith.cmpi sgt, {step.name}, {zero.name} : i32")
    forward = emitter._fresh()
    emitter._line(f"{forward} = arith.cmpi slt, {emitter.environment[current_name].name}, {stop.name} : i32")
    backward = emitter._fresh()
    emitter._line(f"{backward} = arith.cmpi sgt, {emitter.environment[current_name].name}, {stop.name} : i32")
    range_condition = emitter._fresh()
    emitter._line(f"{range_condition} = arith.select {step_positive}, {forward}, {backward} : i1")
    break_value = emitter._control_constant(1)
    control_active = emitter._fresh()
    emitter._line(
        f"{control_active} = arith.cmpi ne, {emitter.environment[control_name].name}, {break_value.name} : i32"
    )
    condition = emitter._fresh()
    emitter._line(f"{condition} = arith.andi {range_condition}, {emitter.environment[active_name].name} : i1")
    combined = emitter._fresh()
    emitter._line(f"{combined} = arith.andi {condition}, {control_active} : i1")
    condition = _guard_return(emitter, combined)
    forwarded = ", ".join(emitter.environment[name].name for name in carried_names)
    emitter._line(f"scf.condition({condition}) {forwarded} : {operand_types}")
    emitter.indent -= 1
    emitter._line("} do {")
    emitter.indent += 1
    emitter.environment = outer.copy()
    after_arguments = [emitter._fresh() for _ in carried_names]
    block_arguments = ", ".join(
        f"{argument}: {value_type.mlir}" for argument, value_type in zip(after_arguments, carried_types, strict=True)
    )
    emitter._line(f"^bb0({block_arguments}):")
    for name, value_type, argument in zip(carried_names, carried_types, after_arguments, strict=True):
        emitter.environment[name] = Value(argument, value_type)
    induction = emitter._fresh()
    emitter._line(f"{induction} = arith.index_cast {emitter.environment[current_name].name} : i32 to index")
    emitter.environment[node.target.id] = Value(induction, DslType("index", "index"))
    emitter.loop_controls.append(control_name)
    emit_loop_body(emitter, node.body, control_name, typed_statement.loop_depth + 1)
    emitter.loop_controls.pop()

    next_value = emitter._fresh()
    emitter._line(f"{next_value} = arith.addi {emitter.environment[current_name].name}, {step.name} : i32")
    positive_overflow = emitter._fresh()
    emitter._line(f"{positive_overflow} = arith.cmpi slt, {next_value}, {emitter.environment[current_name].name} : i32")
    negative_overflow = emitter._fresh()
    emitter._line(f"{negative_overflow} = arith.cmpi sgt, {next_value}, {emitter.environment[current_name].name} : i32")
    body_step_positive = emitter._fresh()
    emitter._line(f"{body_step_positive} = arith.cmpi sgt, {step.name}, {zero.name} : i32")
    overflow = emitter._fresh()
    emitter._line(f"{overflow} = arith.select {body_step_positive}, {positive_overflow}, {negative_overflow} : i1")
    false_value = emitter._bool_constant(False)
    no_overflow = emitter._fresh()
    emitter._line(f"{no_overflow} = arith.cmpi eq, {overflow}, {false_value.name} : i1")
    emitter.environment[control_name] = _normalize_continue(emitter, emitter.environment[control_name])
    emitter.environment[current_name] = Value(next_value, integer_type)
    emitter.environment[active_name] = Value(no_overflow, DslType("scalar", "bool"))
    yielded = ", ".join(emitter.environment[name].name for name in carried_names)
    emitter._line(f"scf.yield {yielded} : {operand_types}")
    emitter.indent -= 1
    emitter._line(f"}}{_loop_state_attributes(emitter, carried_names, control_name)}")
    emitter.environment = outer
    internal_names = {current_name, active_name, control_name}
    for name, result, value_type in zip(carried_names, results, carried_types, strict=True):
        emitter.environment[name] = Value(result, value_type)
    _emit_loop_else(emitter, node.orelse, control_name)
    for name in internal_names:
        emitter.environment.pop(name, None)


def lower_while(emitter: LoopEmitter, node: ast.While) -> None:
    outer = emitter.environment.copy()
    typed_statement = emitter.typed_statements[id(node)]
    control_name = emitter._hidden_name("loop_control")
    outer[control_name] = emitter._control_constant(0)
    carried_names = [merge.name for merge in typed_statement.branch_merges]
    carried_types = [merge.type for merge in typed_statement.branch_merges]
    carried_names.append(control_name)
    carried_types.append(DslType("scalar", "i32"))
    _append_return_state(emitter, carried_names, carried_types)
    for name, value_type in zip(carried_names, carried_types, strict=True):
        outer[name] = emitter._coerce_implicit(node, outer[name], value_type)
    results = [emitter._fresh() for _ in carried_names]
    operand_types = ", ".join(value.mlir for value in carried_types)
    before_arguments = [emitter._fresh() for _ in carried_names]
    assignments = ", ".join(
        f"{argument} = {outer[name].name}" for argument, name in zip(before_arguments, carried_names, strict=True)
    )
    result_prefix = f"{', '.join(results)} = " if results else ""
    signature = f" ({assignments}) : ({operand_types}) -> ({operand_types})" if carried_names else ""
    emitter._line(f"{result_prefix}scf.while{signature} {{")
    emitter.indent += 1
    emitter.environment = outer.copy()
    for name, value_type, argument in zip(carried_names, carried_types, before_arguments, strict=True):
        emitter.environment[name] = Value(argument, value_type)
    break_value = emitter._control_constant(1)
    active = emitter._fresh()
    emitter._line(f"{active} = arith.cmpi ne, {emitter.environment[control_name].name}, {break_value.name} : i32")
    active = _guard_return(emitter, active)
    condition_result = emitter._fresh()
    emitter._line(f"{condition_result} = scf.if {active} -> (i1) {{")
    emitter.indent += 1
    condition = emitter._expression(node.test)
    emitter._require_same_type(node.test, DslType("scalar", "bool"), condition.type)
    emitter._line(f"scf.yield {condition.name} : i1")
    emitter.indent -= 1
    emitter._line("} else {")
    emitter.indent += 1
    false_value = emitter._fresh()
    emitter._line(f"{false_value} = arith.constant false")
    emitter._line(f"scf.yield {false_value} : i1")
    emitter.indent -= 1
    emitter._line("}")
    forwarded = ", ".join(emitter.environment[name].name for name in carried_names)
    suffix = f" : {operand_types}" if carried_names else ""
    emitter._line(f"scf.condition({condition_result}) {forwarded}{suffix}")
    emitter.indent -= 1
    emitter._line("} do {")
    emitter.indent += 1
    emitter.environment = outer.copy()
    after_arguments = []
    for name, value_type in zip(carried_names, carried_types, strict=True):
        argument = emitter._fresh()
        after_arguments.append(argument)
        emitter.environment[name] = Value(argument, value_type)
    if after_arguments:
        block_arguments = ", ".join(
            f"{name}: {value_type.mlir}" for name, value_type in zip(after_arguments, carried_types, strict=True)
        )
        emitter._line(f"^bb0({block_arguments}):")
    emitter.loop_controls.append(control_name)
    emit_loop_body(emitter, node.body, control_name, typed_statement.loop_depth + 1)
    emitter.loop_controls.pop()
    emitter.environment[control_name] = _normalize_continue(emitter, emitter.environment[control_name])
    yielded = ", ".join(emitter.environment[name].name for name in carried_names)
    emitter._line(f"scf.yield {yielded}{suffix}")
    emitter.indent -= 1
    emitter._line(f"}}{_loop_state_attributes(emitter, carried_names, control_name)}")
    emitter.environment = outer
    for name, result, result_type in zip(carried_names, results, carried_types, strict=True):
        emitter.environment[name] = Value(result, result_type)
    _emit_loop_else(emitter, node.orelse, control_name)
    emitter.environment.pop(control_name, None)


def _append_return_state(
    emitter: LoopEmitter,
    names: list[str],
    types: list[DslType],
) -> None:
    for name, value_type in return_state_items(emitter):
        if name not in names:
            names.append(name)
            types.append(value_type)


def _loop_state_attributes(emitter: LoopEmitter, carried_names: list[str], control_name: str) -> str:
    attributes = [f"vernon.loop_control_index = {carried_names.index(control_name)} : i64"]
    if emitter.return_flag_name is not None and emitter.return_flag_name in carried_names:
        attributes.append(f"vernon.return_flag_index = {carried_names.index(emitter.return_flag_name)} : i64")
    return " attributes {" + ", ".join(attributes) + "}"


def _guard_return(emitter: LoopEmitter, active: str) -> str:
    if emitter.return_flag_name is None:
        return active
    false_value = emitter._bool_constant(False)
    not_returned = emitter._fresh()
    emitter._line(
        f"{not_returned} = arith.cmpi eq, {emitter.environment[emitter.return_flag_name].name}, {false_value.name} : i1"
    )
    combined = emitter._fresh()
    emitter._line(f"{combined} = arith.andi {active}, {not_returned} : i1")
    return combined


def _normalize_continue(emitter: LoopEmitter, control: Value) -> Value:
    continue_value = emitter._control_constant(2)
    normal_value = emitter._control_constant(0)
    is_continue = emitter._fresh()
    emitter._line(f"{is_continue} = arith.cmpi eq, {control.name}, {continue_value.name} : i32")
    normalized = emitter._fresh()
    emitter._line(f"{normalized} = arith.select {is_continue}, {normal_value.name}, {control.name} : i32")
    return Value(normalized, DslType("scalar", "i32"))


def _emit_loop_else(
    emitter: LoopEmitter,
    statements: list[ast.stmt],
    control_name: str,
) -> None:
    if not statements:
        return
    outer = emitter.environment.copy()
    assigned = assigned_names(statements)
    carried_names = [
        name
        for name in sorted(assigned & outer.keys())
        if outer[name].type.kind not in {"sampler", "tensor_storage", "tensor_view", "texture"}
    ]
    for name, _ in return_state_items(emitter):
        if name in outer and name not in carried_names:
            carried_names.append(name)
    carried_types = [outer[name].type for name in carried_names]
    break_value = emitter._control_constant(1)
    normal = emitter._fresh()
    emitter._line(f"{normal} = arith.cmpi ne, {outer[control_name].name}, {break_value.name} : i32")
    normal = _guard_return(emitter, normal)

    outer_lines = emitter.lines
    outer_indent = emitter.indent
    emitter.lines = []
    emitter.indent = outer_indent + 1
    emitter.environment = outer.copy()
    emit_source_block(emitter, statements)
    yield_values(emitter, statements[0], carried_names, carried_types)
    then_lines = emitter.lines

    emitter.lines = []
    emitter.environment = outer.copy()
    yield_values(emitter, statements[0], carried_names, carried_types)
    else_lines = emitter.lines

    results = [emitter._fresh() for _ in carried_names]
    prefix = f"{', '.join(results)} = " if results else ""
    result_types = f" -> ({', '.join(value.mlir for value in carried_types)})" if results else ""
    emitter.lines = outer_lines
    emitter.indent = outer_indent
    emitter.environment = outer
    emitter._line(f"{prefix}scf.if {normal}{result_types} {{")
    emitter.lines.extend(then_lines)
    emitter._line("} else {")
    emitter.lines.extend(else_lines)
    emitter._line("}")
    for name, result, value_type in zip(carried_names, results, carried_types, strict=True):
        emitter.environment[name] = Value(result, value_type)


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
