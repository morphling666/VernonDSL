import json
from collections.abc import Iterable
from typing import Any

from .model import (
    AccessMode,
    BranchMerge,
    Effect,
    LValue,
    StorageClass,
    Termination,
    TypedExpression,
    TypedFunctionInstance,
    TypedParameter,
    TypedStatement,
)


def typed_model_data(functions: Iterable[TypedFunctionInstance]) -> list[dict[str, Any]]:
    def statement_data(statement: TypedStatement) -> dict[str, Any]:
        return {
            "kind": type(statement.source).__name__,
            "line": getattr(statement.source, "lineno", 0),
            "effect": statement.effect.value,
            "termination": statement.termination.value,
            "expressions": [
                {
                    "kind": type(expression.source).__name__,
                    "line": getattr(expression.source, "lineno", 0),
                    "operation": expression.operation,
                    "type": expression.type.mlir,
                    "operand_types": [value.mlir for value in expression.operand_types],
                    "storage": expression.storage.value,
                    "access": expression.access.value,
                }
                for expression in statement.expressions
            ],
            "lvalues": [
                {
                    "kind": value.kind,
                    "name": value.name,
                    "type": value.type.mlir,
                    "storage": value.storage.value,
                    "access": value.access.value,
                }
                for value in statement.lvalues
            ],
            "branch_merges": [{"name": merge.name, "type": merge.type.mlir} for merge in statement.branch_merges],
            "children": [statement_data(child) for child in statement.children],
        }

    return [
        {
            "qualified_name": function.qualified_name,
            "symbol": function.symbol,
            "parameters": [
                {
                    "name": parameter.name,
                    "type": parameter.type.mlir,
                    "storage": parameter.storage.value,
                    "access": parameter.access.value,
                }
                for parameter in function.parameters
            ],
            "result": function.result_type.mlir if function.result_type is not None else None,
            "features": list(function.enabled_features),
            "body": [statement_data(statement) for statement in function.body],
        }
        for function in functions
    ]


def dump_typed_model(functions: Iterable[TypedFunctionInstance]) -> str:
    return json.dumps(typed_model_data(functions), indent=2, sort_keys=True)


__all__ = [
    "AccessMode",
    "BranchMerge",
    "Effect",
    "LValue",
    "StorageClass",
    "Termination",
    "TypedExpression",
    "TypedFunctionInstance",
    "TypedParameter",
    "TypedStatement",
    "dump_typed_model",
    "typed_model_data",
]
