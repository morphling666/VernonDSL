from __future__ import annotations

import ast
from typing import Protocol

from .lowering_types import DslType, FunctionSignature, ModuleContext, Value
from .model import Termination, TypedExpression, TypedStatement


class ControlFlowEmitter(Protocol):
    context: ModuleContext
    indent: int
    environment: dict[str, Value]
    lines: list[str]
    loop_controls: list[str]
    typed_statements: dict[int, TypedStatement]
    signature: FunctionSignature
    use_return_state: bool
    return_flag_name: str | None
    return_value_name: str | None
    returned: bool

    def _typed_expression(self, node: ast.expr) -> TypedExpression: ...

    def _expression(self, node: ast.expr, expected: DslType | None = None) -> Value: ...

    def _coerce_implicit(self, node: ast.AST, value: Value, target: DslType) -> Value: ...

    def _require_same_type(self, node: ast.AST, expected: DslType, actual: DslType) -> None: ...

    def _fresh(self) -> str: ...

    def _line(self, text: str) -> None: ...

    def _bool_constant(self, value: bool) -> Value: ...

    def _control_constant(self, value: int) -> Value: ...

    def _emit_function_return(self, node: ast.AST, value: Value | None) -> None: ...

    def _default_value(self, node: ast.AST, value_type: DslType) -> Value: ...

    def _statement(self, typed_statement: TypedStatement) -> None: ...


def lower_termination(emitter: ControlFlowEmitter, node: ast.stmt) -> bool:
    if isinstance(node, ast.Return):
        value = _lower_return_value(emitter, node)
        if emitter.use_return_state:
            assert emitter.return_flag_name is not None
            emitter.environment[emitter.return_flag_name] = emitter._bool_constant(True)
            if emitter.return_value_name is not None:
                assert value is not None
                emitter.environment[emitter.return_value_name] = value
        else:
            emitter._emit_function_return(node, value)
            emitter.returned = True
        return True
    if isinstance(node, (ast.Break, ast.Continue)):
        if not emitter.loop_controls:
            keyword = "break" if isinstance(node, ast.Break) else "continue"
            raise emitter.context.error(node, f"{keyword} is only valid inside a loop")
        state = 1 if isinstance(node, ast.Break) else 2
        emitter.environment[emitter.loop_controls[-1]] = emitter._control_constant(state)
        return True
    return False


def _lower_return_value(emitter: ControlFlowEmitter, node: ast.Return) -> Value | None:
    if node.value is None:
        if emitter.signature.result is not None:
            raise emitter.context.error(node, "return value is required")
        return None
    if emitter.signature.result is None:
        raise emitter.context.error(node, "void function cannot return a value")
    value = emitter._expression(node.value, emitter.signature.result)
    return emitter._coerce_implicit(node.value, value, emitter.signature.result)


def lower_if(emitter: ControlFlowEmitter, node: ast.If) -> None:
    condition = emitter._expression(node.test)
    emitter._require_same_type(node.test, DslType("scalar", "bool"), condition.type)
    typed_statement = emitter.typed_statements[id(node)]
    merged_names = [merge.name for merge in typed_statement.branch_merges]
    merged_types = [merge.type for merge in typed_statement.branch_merges]
    for control_name in emitter.loop_controls:
        if control_name not in merged_names:
            merged_names.append(control_name)
            merged_types.append(DslType("scalar", "i32"))
    for return_name, return_type in return_state_items(emitter):
        if return_name not in merged_names:
            merged_names.append(return_name)
            merged_types.append(return_type)
    outer_environment = emitter.environment.copy()
    outer_lines = emitter.lines
    outer_indent = emitter.indent

    emitter.lines = []
    emitter.indent = outer_indent + 1
    emitter.environment = outer_environment.copy()
    emit_source_block(emitter, node.body)
    then_lines = emitter.lines
    then_environment = emitter.environment.copy()

    emitter.lines = []
    emitter.environment = outer_environment.copy()
    emit_source_block(emitter, node.orelse)
    else_lines = emitter.lines
    else_environment = emitter.environment.copy()

    emitter.lines = then_lines
    emitter.environment = then_environment
    _yield_merged(emitter, node, merged_names, merged_types)
    emitter.lines = else_lines
    emitter.environment = else_environment
    _yield_merged(emitter, node, merged_names, merged_types)

    results = [emitter._fresh() for _ in merged_names]
    lhs = f"{', '.join(results)} = " if results else ""
    result_types = f" -> ({', '.join(value.mlir for value in merged_types)})" if results else ""
    emitter.lines = outer_lines
    emitter.indent = outer_indent
    emitter.environment = outer_environment
    emitter._line(f"{lhs}scf.if {condition.name}{result_types} {{")
    emitter.lines.extend(then_lines)
    emitter._line("} else {")
    emitter.lines.extend(else_lines)
    emitter._line("}")
    for name, result, result_type in zip(merged_names, results, merged_types, strict=True):
        emitter.environment[name] = Value(result, result_type)


def _yield_merged(
    emitter: ControlFlowEmitter,
    node: ast.If,
    names: list[str],
    types: list[DslType],
) -> None:
    values = [
        emitter._coerce_implicit(node, emitter.environment[name], expected)
        for name, expected in zip(names, types, strict=True)
    ]
    if values:
        emitter._line(
            f"scf.yield {', '.join(value.name for value in values)} : {', '.join(value.type.mlir for value in values)}"
        )
    else:
        emitter._line("scf.yield")


def lower_conditional_expression(emitter: ControlFlowEmitter, node: ast.IfExp) -> Value:
    typed = emitter._typed_expression(node)
    bool_type = DslType("scalar", "bool")
    condition = emitter._expression(node.test, bool_type)
    emitter._require_same_type(node.test, bool_type, condition.type)
    result = emitter._fresh()
    emitter._line(f"{result} = scf.if {condition.name} -> ({typed.type.mlir}) {{")
    emitter.indent += 1
    then_value = emitter._coerce_implicit(
        node.body,
        emitter._expression(node.body, typed.type),
        typed.type,
    )
    emitter._line(f"scf.yield {then_value.name} : {typed.type.mlir}")
    emitter.indent -= 1
    emitter._line("} else {")
    emitter.indent += 1
    else_value = emitter._coerce_implicit(
        node.orelse,
        emitter._expression(node.orelse, typed.type),
        typed.type,
    )
    emitter._line(f"scf.yield {else_value.name} : {typed.type.mlir}")
    emitter.indent -= 1
    emitter._line("}")
    return Value(result, typed.type)


def emit_source_block(emitter: ControlFlowEmitter, statements: list[ast.stmt]) -> None:
    needs_guard = False
    for statement in statements:
        typed = emitter.typed_statements.get(id(statement))
        if typed is None:
            break
        if needs_guard:
            control_name = emitter.loop_controls[-1] if emitter.loop_controls else None
            emit_guarded_statement(emitter, typed, control_name)
        else:
            emitter._statement(typed)
        if contains_return(typed) or (emitter.loop_controls and contains_loop_exit(typed, typed.loop_depth)):
            needs_guard = True
        if typed.termination is not Termination.FALLTHROUGH:
            break


def contains_loop_exit(statement: TypedStatement, target_depth: int) -> bool:
    if statement.termination in {Termination.BREAK, Termination.CONTINUE} and statement.loop_depth == target_depth:
        return True
    if isinstance(statement.source, (ast.For, ast.While)):
        return False
    return any(contains_loop_exit(child, target_depth) for child in statement.children)


def contains_return(statement: TypedStatement) -> bool:
    if statement.termination is Termination.RETURN:
        return True
    return any(contains_return(child) for child in statement.children)


def return_state_items(emitter: ControlFlowEmitter) -> list[tuple[str, DslType]]:
    items: list[tuple[str, DslType]] = []
    if emitter.return_flag_name is not None:
        items.append((emitter.return_flag_name, DslType("scalar", "bool")))
    if emitter.return_value_name is not None:
        assert emitter.signature.result is not None
        items.append((emitter.return_value_name, emitter.signature.result))
    return items


def emit_guarded_statement(
    emitter: ControlFlowEmitter,
    typed: TypedStatement,
    control_name: str | None,
) -> None:
    outer = emitter.environment.copy()
    assigned = assigned_names([typed.source])
    lvalue_types = {value.name: value.type for value in typed.lvalues if value.kind == "local"}
    for name in sorted(assigned - outer.keys()):
        value_type = lvalue_types.get(name)
        if value_type is not None:
            outer[name] = emitter._default_value(typed.source, value_type)
    carried_names = [
        name
        for name in sorted(assigned & outer.keys())
        if outer[name].type.kind not in {"sampler", "tensor_storage", "tensor_view", "texture"}
    ]
    for name in emitter.loop_controls:
        if name in outer and name not in carried_names:
            carried_names.append(name)
    for name, _ in return_state_items(emitter):
        if name in outer and name not in carried_names:
            carried_names.append(name)
    carried_types = [outer[name].type for name in carried_names]
    active: Value | None = None
    if control_name is not None:
        zero = emitter._control_constant(0)
        control_active = emitter._fresh()
        emitter._line(f"{control_active} = arith.cmpi eq, {outer[control_name].name}, {zero.name} : i32")
        active = Value(control_active, DslType("scalar", "bool"))
    if emitter.return_flag_name is not None:
        false_value = emitter._bool_constant(False)
        not_returned = emitter._fresh()
        emitter._line(
            f"{not_returned} = arith.cmpi eq, {outer[emitter.return_flag_name].name}, {false_value.name} : i1"
        )
        if active is None:
            active = Value(not_returned, DslType("scalar", "bool"))
        else:
            combined = emitter._fresh()
            emitter._line(f"{combined} = arith.andi {active.name}, {not_returned} : i1")
            active = Value(combined, DslType("scalar", "bool"))
    if active is None:
        emitter._statement(typed)
        return

    outer_lines = emitter.lines
    outer_indent = emitter.indent
    emitter.lines = []
    emitter.indent = outer_indent + 1
    emitter.environment = outer.copy()
    emitter._statement(typed)
    yield_values(emitter, typed.source, carried_names, carried_types)
    then_lines = emitter.lines

    emitter.lines = []
    emitter.environment = outer.copy()
    yield_values(emitter, typed.source, carried_names, carried_types)
    else_lines = emitter.lines

    results = [emitter._fresh() for _ in carried_names]
    result_prefix = f"{', '.join(results)} = " if results else ""
    result_types = f" -> ({', '.join(value.mlir for value in carried_types)})" if results else ""
    emitter.lines = outer_lines
    emitter.indent = outer_indent
    emitter.environment = outer
    emitter._line(f"{result_prefix}scf.if {active.name}{result_types} {{")
    emitter.lines.extend(then_lines)
    emitter._line("} else {")
    emitter.lines.extend(else_lines)
    emitter._line("}")
    for name, result, value_type in zip(carried_names, results, carried_types, strict=True):
        emitter.environment[name] = Value(result, value_type)


def yield_values(
    emitter: ControlFlowEmitter,
    node: ast.AST,
    names: list[str],
    types: list[DslType],
) -> None:
    values = [
        emitter._coerce_implicit(node, emitter.environment[name], value_type)
        for name, value_type in zip(names, types, strict=True)
    ]
    if values:
        emitter._line(
            f"scf.yield {', '.join(value.name for value in values)} : {', '.join(value.type.mlir for value in values)}"
        )
    else:
        emitter._line("scf.yield")


def emit_loop_body(
    emitter: ControlFlowEmitter,
    statements: list[ast.stmt],
    control_name: str,
    target_depth: int,
) -> None:
    needs_guard = False
    for source in statements:
        typed = emitter.typed_statements.get(id(source))
        if typed is None:
            break
        if needs_guard:
            emit_guarded_statement(emitter, typed, control_name)
        else:
            emitter._statement(typed)
        if contains_loop_exit(typed, target_depth) or contains_return(typed):
            needs_guard = True
        if typed.termination is not Termination.FALLTHROUGH:
            break


def assigned_names(statements: list[ast.stmt]) -> set[str]:
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


def lower_bool_op(emitter: ControlFlowEmitter, node: ast.BoolOp) -> Value:
    result_type = emitter._typed_expression(node).type
    emitter._require_same_type(node, DslType("scalar", "bool"), result_type)
    current = emitter._expression(node.values[0], result_type)
    emitter._require_same_type(node.values[0], result_type, current.type)
    for source in node.values[1:]:
        result = emitter._fresh()
        emitter._line(f"{result} = scf.if {current.name} -> ({result_type.mlir}) {{")
        emitter.indent += 1
        if isinstance(node.op, ast.And):
            alternative = emitter._expression(source, result_type)
            emitter._line(f"scf.yield {alternative.name} : {result_type.mlir}")
        else:
            emitter._line(f"scf.yield {current.name} : {result_type.mlir}")
        emitter.indent -= 1
        emitter._line("} else {")
        emitter.indent += 1
        if isinstance(node.op, ast.And):
            emitter._line(f"scf.yield {current.name} : {result_type.mlir}")
        else:
            alternative = emitter._expression(source, result_type)
            emitter._line(f"scf.yield {alternative.name} : {result_type.mlir}")
        emitter.indent -= 1
        emitter._line("}")
        current = Value(result, result_type)
    return current
