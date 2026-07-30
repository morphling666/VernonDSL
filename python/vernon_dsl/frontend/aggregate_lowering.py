from __future__ import annotations

import ast
from typing import Protocol

from ..language.ast_utils import rectangular_literal
from .lowering_types import DslType, ModuleContext, Value, element_type
from .model import TypedExpression


class AggregateEmitter(Protocol):
    context: ModuleContext

    def _typed_expression(self, node: ast.expr) -> TypedExpression: ...

    def _expression(self, node: ast.expr, expected: DslType | None = None) -> Value: ...

    def _coerce_implicit(self, node: ast.AST, value: Value, target: DslType) -> Value: ...

    def _intrinsic(self, node: ast.AST, name: str, arguments: list[Value], result_type: DslType) -> Value: ...

    def _fresh(self) -> str: ...

    def _line(self, text: str) -> None: ...


def lower_tuple(emitter: AggregateEmitter, node: ast.expr, sources: list[ast.expr]) -> Value:
    result_type = emitter._typed_expression(node).type
    elements = [element for element in result_type.arguments if isinstance(element, DslType)]
    values = [
        emitter._coerce_implicit(source, emitter._expression(source, expected), expected)
        for source, expected in zip(sources, elements, strict=True)
    ]
    result = emitter._fresh()
    emitter._line(
        f'{result} = "vernon.tuple_create"({", ".join(value.name for value in values)}) : '
        f"({', '.join(value.type.mlir for value in values)}) -> {result_type.mlir}"
    )
    return Value(result, result_type, tuple(values))


def lower_aggregate_constructor(emitter: AggregateEmitter, node: ast.Call, name: str) -> Value:
    if len(node.args) != 1:
        raise emitter.context.error(node, f"{name} requires one sequence literal")
    if name == "Vector":
        sequence = node.args[0]
        if not isinstance(sequence, (ast.List, ast.Tuple)) or not sequence.elts:
            raise emitter.context.error(node, "Vector requires a non-empty sequence literal")
        result_type = emitter._typed_expression(node).type
        constructor_element_type = element_type(result_type)
        values = [
            _coerce_constructor_argument(emitter, value, emitter._expression(value), constructor_element_type)
            for value in sequence.elts
        ]
        return emitter._intrinsic(node, "construct", values, result_type)
    literal = rectangular_literal(node.args[0])
    if literal is None:
        raise emitter.context.error(node, f"{name} requires a non-empty rectangular sequence literal")
    elements, shape = literal
    expected_rank = {"Vector": 1, "Matrix": 2}.get(name)
    if expected_rank is not None and len(shape) != expected_rank:
        raise emitter.context.error(node, f"{name} requires a rank-{expected_rank} sequence literal")
    values = [emitter._expression(value) for value in elements]
    result_type = emitter._typed_expression(node).type
    constructor_element_type = element_type(result_type)
    values = [
        emitter._coerce_implicit(source, value, constructor_element_type)
        for source, value in zip(elements, values, strict=True)
    ]
    return emitter._intrinsic(node, "construct", values, result_type)


def _coerce_constructor_argument(
    emitter: AggregateEmitter,
    node: ast.AST,
    value: Value,
    element: DslType,
) -> Value:
    if value.type.kind == "tensor":
        target = DslType("tensor", "Tensor", (element, *value.type.arguments[1:]))
        return emitter._coerce_implicit(node, value, target)
    return emitter._coerce_implicit(node, value, element)
