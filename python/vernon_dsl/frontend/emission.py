from __future__ import annotations

import ast
import json
from collections.abc import Callable, Mapping

from ..diagnostics import CompileError
from ..language.syntax import FRONTEND_VERSION
from .abi import value_abi_layout
from .model import ConcreteType
from .type_parser import AnnotatedType


def emit_mlir_module(
    module: ast.Module,
    *,
    dependencies: tuple[tuple[str, str], ...],
    declared_features: tuple[str, ...],
    enabled_features: tuple[str, ...],
    structs: Mapping[str, tuple[tuple[str, AnnotatedType], ...]],
    emit_function: Callable[[ast.FunctionDef], list[str]],
    error: Callable[[ast.AST, str], CompileError],
) -> str:
    attributes = [
        'vernon.frontend = "python"',
        f"vernon.frontend_version = {FRONTEND_VERSION} : i64",
        "vernon.value_abi_version = 1 : i64",
    ]
    if dependencies:
        encoded = ", ".join(json.dumps(f"{path}={digest}") for path, digest in dependencies)
        attributes.append(f"vernon.source_dependencies = [{encoded}]")
    if declared_features:
        declarations = ", ".join(json.dumps(name) for name in sorted(declared_features))
        attributes.append(f"vernon.feature_declarations = [{declarations}]")
    if enabled_features:
        variant = ", ".join(json.dumps(name) for name in sorted(enabled_features))
        attributes.append(f"vernon.variant_key = [{variant}]")
    body: list[str] = [f"module attributes {{{', '.join(attributes)}}} {{"]

    def struct_field_types(name: str) -> tuple[tuple[str, ConcreteType], ...]:
        return tuple((field_name, annotation.type) for field_name, annotation in structs[name])

    for name in sorted(structs):
        fields = structs[name]
        field_text = ", ".join(json.dumps(f"{field_name}:{annotation.type.mlir}") for field_name, annotation in fields)
        layout = value_abi_layout(ConcreteType("struct", name), struct_field_types)
        leaf_dtypes = ", ".join(f'"{leaf.dtype}"' for leaf in layout.leaves)
        body.append(
            f'  "vernon.struct"() {{abi_leaf_dtypes = [{leaf_dtypes}], '
            f'fields = [{field_text}], sym_name = "{name}"}} : () -> ()'
        )

    for node in module.body:
        if isinstance(node, ast.FunctionDef):
            body.extend(emit_function(node))
        elif isinstance(node, (ast.Import, ast.ImportFrom, ast.ClassDef)):
            continue
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            continue
        else:
            raise error(node, f"unsupported module-level syntax: {type(node).__name__}")
    body.append("}")
    return "\n".join(body) + "\n"
