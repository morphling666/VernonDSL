import json
from collections.abc import Iterable
from typing import Any

from .model import (
    AccessMode,
    AtomicEffect,
    BarrierEffect,
    BranchMerge,
    Effect,
    EffectScope,
    LValue,
    MemoryOrdering,
    ResourceEffect,
    StorageEffect,
    StorageEffectKind,
    StorageOwner,
    StorageOwnerKind,
    StorageRegion,
    StorageRegionKind,
    Termination,
    TypedEffect,
    TypedExpression,
    TypedFunctionInstance,
    TypedParameter,
    TypedStatement,
)


def typed_effect_data(effect: TypedEffect) -> dict[str, Any]:
    if isinstance(effect, ResourceEffect):
        return {
            "kind": "resource_read",
            "operation": effect.operation,
            "owner": effect.owner,
            "stages": sorted(effect.stages),
            "has_lod": effect.has_lod,
        }
    if isinstance(effect, StorageEffect):
        return {
            "kind": effect.kind.value,
            "owner": effect.owner.name,
            "owner_kind": effect.owner.kind.value,
            "region": {
                "kind": effect.region.kind.value,
                "indices": list(effect.region.indices),
            },
        }
    if isinstance(effect, AtomicEffect):
        return {
            "kind": "atomic",
            "operation": effect.operation,
            "owner": effect.owner.name,
            "owner_kind": effect.owner.kind.value,
            "region": {
                "kind": effect.region.kind.value,
                "indices": list(effect.region.indices),
            },
            "ordering": effect.ordering.value,
            "scope": effect.scope.value,
        }
    return {
        "kind": "barrier",
        "ordering": effect.ordering.value,
        "scope": effect.scope.value,
    }


def typed_model_data(functions: Iterable[TypedFunctionInstance]) -> list[dict[str, Any]]:
    def statement_data(statement: TypedStatement) -> dict[str, Any]:
        return {
            "kind": type(statement.source).__name__,
            "line": getattr(statement.source, "lineno", 0),
            "effect": statement.effect.value,
            "effects": [typed_effect_data(effect) for effect in statement.effects],
            "termination": statement.termination.value,
            "loop_depth": statement.loop_depth,
            "return_type": statement.return_type.mlir if statement.return_type is not None else None,
            "expressions": [
                {
                    "kind": type(expression.source).__name__,
                    "line": getattr(expression.source, "lineno", 0),
                    "operation": expression.operation,
                    "type": expression.type.mlir,
                    "operand_types": [value.mlir for value in expression.operand_types],
                    "access": expression.access.value,
                }
                for expression in statement.expressions
            ],
            "lvalues": [
                {
                    "kind": value.kind,
                    "name": value.name,
                    "type": value.type.mlir,
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
                    "access": parameter.access.value,
                    "interface": [
                        {"kind": item.kind, "arguments": list(item.arguments)} for item in parameter.interface
                    ],
                }
                for parameter in function.parameters
            ],
            "result": function.result_type.mlir if function.result_type is not None else None,
            "features": list(function.enabled_features),
            "effects": [typed_effect_data(effect) for effect in function.effects],
            "body": [statement_data(statement) for statement in function.body],
        }
        for function in functions
    ]


def dump_typed_model(functions: Iterable[TypedFunctionInstance]) -> str:
    return json.dumps(typed_model_data(functions), indent=2, sort_keys=True)


__all__ = [
    "AccessMode",
    "AtomicEffect",
    "BarrierEffect",
    "BranchMerge",
    "Effect",
    "EffectScope",
    "LValue",
    "MemoryOrdering",
    "ResourceEffect",
    "StorageEffect",
    "StorageEffectKind",
    "StorageOwner",
    "StorageOwnerKind",
    "StorageRegion",
    "StorageRegionKind",
    "Termination",
    "TypedExpression",
    "TypedEffect",
    "TypedFunctionInstance",
    "TypedParameter",
    "TypedStatement",
    "dump_typed_model",
    "typed_effect_data",
    "typed_model_data",
]
