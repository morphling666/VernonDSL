from __future__ import annotations

import ast
from typing import Protocol

from .lowering_types import DslType, ModuleContext, Value
from .model import TypedExpression


class NumericEmitter(Protocol):
    context: ModuleContext

    def _typed_expression(self, node: ast.expr) -> TypedExpression: ...

    def _expression(self, node: ast.expr, expected: DslType | None = None) -> Value: ...

    def _coerce_numeric(self, node: ast.AST, value: Value, target: DslType) -> Value: ...

    def _intrinsic(self, node: ast.AST, name: str, arguments: list[Value], result_type: DslType) -> Value: ...

    def _fresh(self) -> str: ...

    def _line(self, text: str) -> None: ...

    def _require_same_type(self, node: ast.AST, expected: DslType, actual: DslType) -> None: ...

    def _default_value(self, node: ast.AST, value_type: DslType) -> Value: ...


def lower_constant(emitter: NumericEmitter, node: ast.Constant, expected: DslType | None) -> Value:
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
        raise emitter.context.error(node, "only bool, integer, and float constants are supported")
    result = emitter._fresh()
    emitter._line(f"{result} = arith.constant {literal} : {value_type.mlir}")
    return Value(result, value_type)


def lower_binary(emitter: NumericEmitter, node: ast.BinOp) -> Value:
    result_type = emitter._typed_expression(node).type
    left = emitter._coerce_numeric(node.left, emitter._expression(node.left), result_type)
    right = emitter._coerce_numeric(node.right, emitter._expression(node.right), result_type)
    floating = left.type.is_float
    if isinstance(node.op, ast.Pow):
        if not floating:
            raise emitter.context.error(node, "power requires floating-point operands")
        return emitter._intrinsic(node, "pow", [left, right], result_type)
    operations = {
        ast.Add: "arith.addf" if floating else "arith.addi",
        ast.Sub: "arith.subf" if floating else "arith.subi",
        ast.Mult: "arith.mulf" if floating else "arith.muli",
        ast.Div: "arith.divf" if floating else ("arith.divui" if left.type.name == "u32" else "arith.divsi"),
        ast.FloorDiv: None if floating else ("arith.divui" if left.type.name == "u32" else "arith.divsi"),
        ast.Mod: "arith.remf" if floating else ("arith.remui" if left.type.name == "u32" else "arith.remsi"),
    }
    operation = operations.get(type(node.op))
    if operation is None or not (floating or left.type.is_integer):
        raise emitter.context.error(node, f"unsupported binary operation for {left.type.name}")
    result = emitter._fresh()
    emitter._line(f"{result} = {operation} {left.name}, {right.name} : {left.type.mlir}")
    return Value(result, result_type)


def lower_unary(emitter: NumericEmitter, node: ast.UnaryOp) -> Value:
    operand = emitter._expression(node.operand)
    result_type = emitter._typed_expression(node).type
    result = emitter._fresh()
    if isinstance(node.op, ast.Not):
        emitter._require_same_type(node, DslType("scalar", "bool"), operand.type)
        truth = emitter._fresh()
        emitter._line(f"{truth} = arith.constant true")
        emitter._line(f"{result} = arith.xori {operand.name}, {truth} : i1")
        return Value(result, result_type)
    if isinstance(node.op, ast.USub) and (operand.type.is_float or operand.type.is_integer):
        zero = emitter._default_value(node, operand.type)
        operation = "arith.subf" if operand.type.is_float else "arith.subi"
        emitter._line(f"{result} = {operation} {zero.name}, {operand.name} : {operand.type.mlir}")
        return Value(result, result_type)
    if isinstance(node.op, ast.UAdd):
        return Value(operand.name, result_type, operand.fields, operand.access)
    raise emitter.context.error(node, "unsupported unary operation")


def lower_compare(emitter: NumericEmitter, node: ast.Compare) -> Value:
    if len(node.ops) != 1:
        raise emitter.context.error(node, "chained comparisons are not supported")
    typed = emitter._typed_expression(node)
    if len(typed.operand_types) != 2:
        raise emitter.context.error(node, "internal error: comparison has no typed operands")
    common = typed.operand_types[0]
    left = emitter._coerce_numeric(node.left, emitter._expression(node.left), common)
    right = emitter._coerce_numeric(node.comparators[0], emitter._expression(node.comparators[0]), common)
    predicates = {
        ast.Eq: ("oeq", "eq"),
        ast.NotEq: ("one", "ne"),
        ast.Lt: ("olt", "slt"),
        ast.LtE: ("ole", "sle"),
        ast.Gt: ("ogt", "sgt"),
        ast.GtE: ("oge", "sge"),
    }
    selected = predicates.get(type(node.ops[0]))
    if selected is None:
        raise emitter.context.error(node, f"unsupported comparison: {type(node.ops[0]).__name__}")
    floating_predicate, integer_predicate = selected
    if left.type.is_float:
        operation, predicate = "arith.cmpf", floating_predicate
    elif left.type.is_integer or left.type.name == "bool":
        operation, predicate = "arith.cmpi", integer_predicate
    else:
        raise emitter.context.error(node, "comparison requires scalar or tensor numeric operands")
    result = emitter._fresh()
    emitter._line(f"{result} = {operation} {predicate}, {left.name}, {right.name} : {left.type.mlir}")
    return Value(result, typed.type)
