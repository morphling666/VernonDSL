from __future__ import annotations

import ast
import copy

from ..diagnostics import CompileError, SourceLocation, copy_location, fix_missing_locations, node_location
from ..language.ast_utils import dotted_name
from .request import FrontendCompileRequest


def specialize_frontend_module(module: ast.Module, request: FrontendCompileRequest) -> ast.Module:
    """A copy of `module` with the request's tensor shapes and captured constants substituted."""

    tree = copy.deepcopy(module)
    function = next(
        (node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == request.entry),
        None,
    )
    if function is None:
        raise CompileError(
            f"entry function '{request.entry}' was not found",
            SourceLocation(str(request.source_path), 1, 1),
        )
    shapes = {name: shape for name, _, shape in request.tensor_shapes}
    for argument in function.args.args:
        shape = shapes.get(argument.arg)
        if shape is None or argument.annotation is None:
            continue
        annotation = argument.annotation
        if (
            isinstance(annotation, ast.Subscript)
            and (dotted_name(annotation.value) or "").split(".")[-1] == "Annotated"
        ):
            values = list(annotation.slice.elts) if isinstance(annotation.slice, ast.Tuple) else [annotation.slice]
            annotation = values[0]
        if not (
            isinstance(annotation, ast.Subscript)
            and (dotted_name(annotation.value) or "").split(".")[-1] in {"Tensor", "TensorView"}
        ):
            continue
        items = list(annotation.slice.elts) if isinstance(annotation.slice, ast.Tuple) else [annotation.slice]
        if len(items) < 2:
            continue
        shape_node = items[1]
        shape_nodes = list(shape_node.elts) if isinstance(shape_node, ast.Tuple) else items[1:]
        if len(shape_nodes) != len(shape):
            raise CompileError(
                f"Tensor argument '{argument.arg}' rank does not match annotation",
                node_location(argument, str(request.source_path)),
            )
        for index, (declared, concrete) in enumerate(zip(shape_nodes, shape, strict=True)):
            if (isinstance(declared, ast.Constant) and declared.value is None) or (dotted_name(declared) or "").split(
                "."
            )[-1] == "dyn":
                shape_nodes[index] = copy_location(ast.Constant(value=concrete), declared)
            elif not (isinstance(declared, ast.Constant) and declared.value == concrete):
                raise CompileError(
                    f"Tensor argument '{argument.arg}' shape does not match annotation",
                    node_location(argument, str(request.source_path)),
                )
        items[1] = ast.Tuple(elts=shape_nodes, ctx=ast.Load())
        annotation.slice = ast.Tuple(elts=items, ctx=ast.Load())

    constants = dict(request.captured_constants)

    class ConstantSpecializer(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.expr:
            if isinstance(node.ctx, ast.Load) and node.id in constants:
                return copy_location(ast.Constant(constants[node.id]), node)
            return node

    specialized = ConstantSpecializer().visit(tree)
    assert isinstance(specialized, ast.Module)
    return fix_missing_locations(specialized)
