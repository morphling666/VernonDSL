from __future__ import annotations

import ast
import copy

# Nodes without fields or a location, which the parser shares across every tree.
STATELESS_NODES = (ast.expr_context, ast.boolop, ast.operator, ast.unaryop, ast.cmpop)


def strip_docstrings(tree: ast.Module) -> None:
    """Removes the docstrings of the module and of its functions and classes, which device code has no value for. A
    body that held only a docstring becomes `pass`."""

    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and (
            ast.get_docstring(node, clean=False) is not None
        ):
            docstring = node.body.pop(0)
            if not node.body:
                node.body.append(ast.copy_location(ast.Pass(), docstring))


def detach_shared_nodes(tree: ast.AST) -> None:
    """Copies every node `tree` reaches a second time, so each node has one parent. The frontend keys typing and
    lowering facts by node identity, and rewrites that insert one node in several places must not share those facts.
    Stateless nodes stay shared."""

    seen: set[int] = set()
    pending = [tree]
    while pending:
        node = pending.pop()
        for name, value in ast.iter_fields(node):
            children = value if isinstance(value, list) else [value]
            for index, child in enumerate(children):
                if not isinstance(child, ast.AST) or isinstance(child, STATELESS_NODES):
                    continue
                if id(child) in seen:
                    child = copy.deepcopy(child)
                    if isinstance(value, list):
                        value[index] = child
                    else:
                        setattr(node, name, child)
                seen.add(id(child))
                pending.append(child)


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


def tensor_update_parts(node: ast.AST) -> tuple[ast.expr, list[ast.expr], str] | None:
    """`(v, [i, ...], method)` for a call `v.at[i, ...].method(...)`; None for any other node."""

    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
        return None
    subscript = node.func.value
    if not isinstance(subscript, ast.Subscript):
        return None
    at = subscript.value
    if not isinstance(at, ast.Attribute) or at.attr != "at":
        return None
    indices = list(subscript.slice.elts) if isinstance(subscript.slice, ast.Tuple) else [subscript.slice]
    return at.value, indices, node.func.attr


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
