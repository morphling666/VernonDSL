"""Storage activity derived from the typed frontend statement model."""

from __future__ import annotations

import ast

from .model import AccessMode, StorageActivitySummary, TypedFunctionInstance, TypedStatement


def analyze_storage_activity(entry: TypedFunctionInstance) -> StorageActivitySummary:
    readable = frozenset(
        parameter.name
        for parameter in entry.parameters
        if parameter.type.kind == "tensor_view" and parameter.access is not AccessMode.WRITE
    )
    writable = frozenset(
        parameter.name
        for parameter in entry.parameters
        if parameter.type.kind == "tensor_view" and parameter.access is not AccessMode.READ
    )
    outputs: dict[str, set[str]] = {name: set() for name in writable}
    initial = {name: {name} for name in readable}

    def dependencies(node: ast.AST | None, environment: dict[str, set[str]]) -> set[str]:
        if node is None:
            return set()
        if isinstance(node, ast.Name):
            return set(environment.get(node.id, ()))
        return set().union(*(dependencies(child, environment) for child in ast.iter_child_nodes(node)))

    def root(node: ast.AST) -> str | None:
        current = node
        while isinstance(current, (ast.Subscript, ast.Attribute)):
            current = current.value
        return current.id if isinstance(current, ast.Name) else None

    def assign(target: ast.AST, value: set[str], environment: dict[str, set[str]]) -> None:
        if isinstance(target, ast.Name):
            environment[target.id] = set(value)
            return
        if isinstance(target, (ast.Tuple, ast.List)):
            for member in target.elts:
                assign(member, value, environment)
            return
        owner = root(target)
        if owner in outputs:
            outputs[owner].update(value)

    def execute(statements: tuple[TypedStatement, ...], incoming: dict[str, set[str]]) -> dict[str, set[str]]:
        environment = {name: set(value) for name, value in incoming.items()}
        for typed in statements:
            statement = typed.source
            if isinstance(statement, ast.Assign):
                value = dependencies(statement.value, environment)
                for target in statement.targets:
                    assign(target, value, environment)
            elif isinstance(statement, ast.AnnAssign):
                assign(statement.target, dependencies(statement.value, environment), environment)
            elif isinstance(statement, ast.AugAssign):
                assign(
                    statement.target,
                    dependencies(statement.target, environment) | dependencies(statement.value, environment),
                    environment,
                )
            elif isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
                call_dependencies = dependencies(statement.value, environment)
                for effect in typed.effects:
                    owner = getattr(effect, "owner", None)
                    owner_name = getattr(owner, "name", None)
                    if owner_name in outputs:
                        outputs[owner_name].update(call_dependencies)
            if typed.children:
                nested = execute(typed.children, environment)
                for name, value in nested.items():
                    environment.setdefault(name, set()).update(value)
        return environment

    execute(entry.body, initial)
    return StorageActivitySummary(
        readable,
        writable,
        tuple((name, frozenset(outputs[name])) for name in sorted(outputs)),
    )


__all__ = ["analyze_storage_activity"]
