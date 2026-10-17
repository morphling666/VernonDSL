from __future__ import annotations

import ast
import copy
from dataclasses import dataclass

from .diagnostics import CompileError, copy_location, fix_missing_locations, node_location


def _name(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _name(node.value)
        return f"{prefix}.{node.attr}" if prefix else None
    return None


def _decorator(node: ast.expr, expected: str, filename: str) -> bool:
    target = node.func if isinstance(node, ast.Call) else node
    if (_name(target) or "").split(".")[-1] != expected:
        _error(filename, node, f"expected @{expected} decorator")
    if not isinstance(node, ast.Call):
        return False
    if node.args:
        _error(filename, node, f"@{expected} accepts only the shared keyword option")
    shared = False
    seen = False
    for keyword in node.keywords:
        if (
            keyword.arg != "shared"
            or seen
            or not isinstance(keyword.value, ast.Constant)
            or not isinstance(keyword.value.value, bool)
        ):
            _error(filename, keyword, f"@{expected} accepts only shared=True or shared=False")
        shared = keyword.value.value
        seen = True
    return shared


def _annotation_name(node: ast.expr | None) -> str | None:
    if node is None:
        return None
    if isinstance(node, ast.Subscript) and (_name(node.value) or "").split(".")[-1] == "Annotated":
        items = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
        return _annotation_name(items[0])
    return _name(node)


def _error(filename: str, node: ast.AST, message: str) -> None:
    raise CompileError(message, node_location(node, filename))


@dataclass(frozen=True)
class _StructInfo:
    fields: dict[str, ast.expr]
    methods: dict[str, str]
    kind: str


def _is_self_attribute(node: ast.AST) -> bool:
    while isinstance(node, ast.Attribute):
        if isinstance(node.value, ast.Name) and node.value.id == "self":
            return True
        node = node.value
    return False


def _class_kind(node: ast.ClassDef) -> str | None:
    for decorator in node.decorator_list:
        name = (_name(decorator.func if isinstance(decorator, ast.Call) else decorator) or "").split(".")[-1]
        if name in {"struct", "binding_group"}:
            return name
    return None


def _validate_method(method: ast.FunctionDef, kind: str, shared_owner: bool, filename: str) -> bool:
    owner = "struct" if kind == "struct" else "binding group"
    if len(method.decorator_list) != 1:
        _error(filename, method, f"{owner} methods require exactly one @func decorator")
    shared = _decorator(method.decorator_list[0], "func", filename)
    if shared and kind == "binding_group":
        _error(filename, method, "binding group cannot be shared; put host/device code on its Value Structs")
    if shared and not shared_owner:
        _error(filename, method, "a shared method requires @struct(shared=True)")
    if not method.args.args or method.args.args[0].arg != "self":
        _error(filename, method, f"{owner} methods require 'self' as the first argument")
    if method.args.args[0].annotation is not None:
        _error(filename, method.args.args[0], f"the self type is supplied by the containing @{kind}")
    for node in ast.walk(method):
        targets: list[ast.AST] = []
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            if isinstance(node, ast.Assign):
                targets.extend(node.targets)
            else:
                targets.append(node.target)
        if any(_is_self_attribute(target) for target in targets):
            if kind == "binding_group":
                _error(filename, node, "binding group cannot be reassigned through its fields")
            _error(filename, node, "struct methods cannot mutate self fields")
    return shared


class _MethodCallRewriter(ast.NodeTransformer):
    def __init__(self, structs: dict[str, _StructInfo], function_results: dict[str, str], filename: str):
        self.structs = structs
        self.function_results = function_results
        self.filename = filename
        self.environments: list[dict[str, str]] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        environment: dict[str, str] = {}
        for argument in node.args.args:
            type_name = _annotation_name(argument.annotation)
            if type_name in self.structs:
                environment[argument.arg] = type_name
        self.environments.append(environment)
        node.body = [self.visit(statement) for statement in node.body]
        self.environments.pop()
        return node

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        node.value = self.visit(node.value)
        inferred = self._expression_struct(node.value)
        for target in node.targets:
            target = self.visit(target)
            if isinstance(target, ast.Name):
                if inferred is None:
                    self.environments[-1].pop(target.id, None)
                else:
                    self.environments[-1][target.id] = inferred
        return node

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AST:
        if node.value is not None:
            node.value = self.visit(node.value)
        if isinstance(node.target, ast.Name) and self.environments:
            type_name = _annotation_name(node.annotation)
            if type_name in self.structs:
                self.environments[-1][node.target.id] = type_name
        return node

    def visit_Call(self, node: ast.Call) -> ast.expr:
        if isinstance(node.func, ast.Attribute):
            receiver = node.func.value
            candidates = [name for name, info in self.structs.items() if node.func.attr in info.methods]
            struct_name = self._expression_struct(receiver) if candidates else None
            # A receiver whose type is not visible in source, such as an unannotated generic parameter, resolves on
            # its inferred type during helper specialization.
            if struct_name is not None:
                info = self.structs[struct_name]
                if node.func.attr not in info.methods:
                    _error(self.filename, node.func, f"'{struct_name}' has no method '{node.func.attr}'")
                rewritten = ast.Call(
                    func=ast.Name(id=info.methods[node.func.attr], ctx=ast.Load()),
                    args=[self.visit(receiver), *(self.visit(value) for value in node.args)],
                    keywords=[self.visit(keyword) for keyword in node.keywords],
                )
                return copy_location(rewritten, node)
        return self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> ast.expr:
        if not isinstance(getattr(node, "_method_callee", None), bool):
            struct_name = self._expression_struct(node.value)
            info = self.structs[struct_name] if struct_name is not None else None
            if info is not None and node.attr in info.methods:
                owner = "struct" if info.kind == "struct" else "binding group"
                _error(self.filename, node, f"{owner} methods cannot be used as first-class values")
        return self.generic_visit(node)

    def _expression_struct(self, node: ast.expr) -> str | None:
        if isinstance(node, ast.Name):
            return self.environments[-1].get(node.id) if self.environments else None
        if isinstance(node, ast.Call):
            called = _name(node.func)
            if called in self.structs:
                return called
            return self.function_results.get(called or "")
        if isinstance(node, ast.Attribute):
            owner = self._expression_struct(node.value)
            if owner is None:
                return None
            annotation = self.structs[owner].fields.get(node.attr)
            type_name = _annotation_name(annotation)
            return type_name if type_name in self.structs else None
        return None


def method_tables(module: ast.Module) -> dict[str, dict[str, str]]:
    """Maps each Struct or binding group to its method helpers; set by `normalize_struct_methods`."""
    return getattr(module, "_vernon_methods", {})


def normalize_struct_methods(module: ast.Module, filename: str) -> ast.Module:
    """Hoists Struct and binding-group methods into `Class__method` helpers and rewrites calls on typed receivers."""
    structs: dict[str, _StructInfo] = {}
    extracted: list[ast.FunctionDef] = []
    for node in module.body:
        if not isinstance(node, ast.ClassDef):
            continue
        kind = _class_kind(node)
        if kind is None:
            continue
        if node.bases or node.keywords:
            _error(filename, node, f"@{kind} classes do not support inheritance")
        if len(node.decorator_list) != 1:
            _error(filename, node, f"DSL classes require exactly one @{kind} decorator")
        if kind == "binding_group":
            if isinstance(node.decorator_list[0], ast.Call):
                _error(filename, node, "binding group cannot be shared; put host/device code on its Value Structs")
            shared_struct = False
        else:
            shared_struct = _decorator(node.decorator_list[0], "struct", filename)
        fields: dict[str, ast.expr] = {}
        methods: dict[str, str] = {}
        kept: list[ast.stmt] = []
        for statement in node.body:
            if (
                isinstance(statement, ast.Expr)
                and isinstance(statement.value, ast.Constant)
                and isinstance(statement.value.value, str)
            ):
                kept.append(statement)
                continue
            if isinstance(statement, ast.Pass):
                kept.append(statement)
                continue
            if (
                isinstance(statement, ast.AnnAssign)
                and isinstance(statement.target, ast.Name)
                and statement.value is None
            ):
                if statement.target.id in fields:
                    _error(filename, statement, f"duplicate {kind.replace('_', ' ')} field '{statement.target.id}'")
                fields[statement.target.id] = copy.deepcopy(statement.annotation)
                kept.append(statement)
                continue
            if not isinstance(statement, ast.FunctionDef):
                _error(filename, statement, f"@{kind} bodies may contain only fields and @func methods")
            if statement.name in methods or statement.name in fields:
                _error(filename, statement, f"duplicate {kind.replace('_', ' ')} member '{statement.name}'")
            _validate_method(statement, kind, shared_struct, filename)
            helper_name = f"{node.name}__{statement.name}"
            methods[statement.name] = helper_name
            helper = copy.deepcopy(statement)
            helper.name = helper_name
            helper.args.args[0].annotation = copy_location(
                ast.Name(id=node.name, ctx=ast.Load()),
                helper.args.args[0],
            )
            extracted.append(helper)
        node.body = kept
        structs[node.name] = _StructInfo(fields, methods, kind)

    module._vernon_methods = {name: dict(info.methods) for name, info in structs.items() if info.methods}
    if not extracted:
        return module
    module.body.extend(extracted)
    function_results = {
        node.name: result
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        if (result := _annotation_name(node.returns)) in structs
    }
    rewritten = _MethodCallRewriter(structs, function_results, filename).visit(module)
    assert isinstance(rewritten, ast.Module)
    return fix_missing_locations(rewritten)
