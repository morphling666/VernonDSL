from __future__ import annotations

import ast


def dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return None


def decorator_name(node: ast.expr) -> str:
    target = node.func if isinstance(node, ast.Call) else node
    return (dotted_name(target) or "").split(".")[-1]


def subscript_items(node: ast.Subscript) -> list[ast.AST]:
    return list(node.slice.elts) if isinstance(node.slice, ast.Tuple) else [node.slice]


def assigned_names(statements: tuple[ast.stmt, ...] | list[ast.stmt]) -> set[str]:
    """Local names assigned anywhere in the statements, including nested block bodies."""
    names: set[str] = set()

    def target_names(target: ast.expr) -> set[str]:
        if isinstance(target, ast.Name):
            return {target.id}
        if isinstance(target, (ast.Tuple, ast.List)):
            return set().union(*(target_names(item) for item in target.elts))
        return set()

    for statement in statements:
        for node in ast.walk(statement):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    names.update(target_names(target))
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                names.add(node.target.id)
            elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
                names.add(node.target.id)
    return names


def rectangular_literal(node: ast.AST) -> tuple[list[ast.expr], tuple[int, ...]] | None:
    """Flatten a non-empty rectangular list/tuple literal in row-major order."""
    if not isinstance(node, (ast.List, ast.Tuple)) or not node.elts:
        return None
    flattened: list[ast.expr] = []
    child_shape: tuple[int, ...] | None = None
    for child in node.elts:
        if isinstance(child, (ast.List, ast.Tuple)):
            nested = rectangular_literal(child)
            if nested is None:
                return None
            elements, shape = nested
        else:
            elements, shape = [child], ()
        if child_shape is None:
            child_shape = shape
        elif shape != child_shape:
            return None
        flattened.extend(elements)
    assert child_shape is not None
    return flattened, (len(node.elts), *child_shape)
