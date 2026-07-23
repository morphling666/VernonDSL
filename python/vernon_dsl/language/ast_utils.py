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
