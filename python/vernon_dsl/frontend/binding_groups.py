"""Flattens `@binding_group` parameters into dotted leaf parameters before typed inference.

A binding group is never a Value, so its type is always visible in source: it can only reach code as an annotated
parameter, a forwarded argument, or a field of another group. That makes flattening exact without inference.
Leaf parameters are named by their dotted path (`material.noise`); a dot cannot occur in a source identifier, so leaf
names never collide with user names, and lowering, reflection, and host binding see the dotted names directly.
"""

from __future__ import annotations

import ast
import copy
import hashlib
from collections.abc import Callable, Iterator
from dataclasses import dataclass

from ..language.ast_utils import decorator_name, dotted_name
from ..language.stage_registry import ENTRY_DECORATOR_STAGES, GRAPHICS_STAGES
from ..struct_methods import method_tables

Error = Callable[[ast.AST, str], Exception]

_RESOURCE_CONSTRUCTORS = frozenset({"TensorView", "Texture", "Sampler"})


def _head(annotation: ast.AST) -> str:
    target = annotation.value if isinstance(annotation, ast.Subscript) else annotation
    return (dotted_name(target) or "").split(".")[-1]


def _is_group_declaration(node: ast.AST) -> bool:
    return isinstance(node, ast.ClassDef) and any(
        decorator_name(decorator) == "binding_group" for decorator in node.decorator_list
    )


def _is_shared(function: ast.FunctionDef) -> bool:
    return any(
        isinstance(decorator, ast.Call)
        and decorator_name(decorator) == "func"
        and any(
            keyword.arg == "shared" and isinstance(keyword.value, ast.Constant) and keyword.value.value is True
            for keyword in decorator.keywords
        )
        for decorator in function.decorator_list
    )


@dataclass(frozen=True)
class _Field:
    name: str
    annotation: ast.expr
    group: str | None


@dataclass(frozen=True)
class _GroupRef:
    """A group occurrence reached through a parameter; `path` is its dotted leaf prefix."""

    group: str
    path: str


@dataclass(frozen=True)
class _MethodRef:
    group: str


@dataclass(frozen=True)
class _Signature:
    groups: tuple[str | None, ...]
    annotated: tuple[bool, ...]


class _Flattener:
    def __init__(self, module: ast.Module, error: Error):
        self.module = module
        self.error = error
        declarations = [node for node in module.body if _is_group_declaration(node)]
        names = {declaration.name for declaration in declarations}
        self.groups: dict[str, tuple[_Field, ...]] = {}
        for declaration in declarations:
            assert isinstance(declaration, ast.ClassDef)
            self.groups[declaration.name] = self._fields(declaration, names)
        self.methods = {name: table for name, table in method_tables(module).items() if name in self.groups}
        self.functions = {node.name: node for node in module.body if isinstance(node, ast.FunctionDef)}
        self.signatures = {name: self._signature(function) for name, function in self.functions.items()}
        self.originals = {
            name: copy.deepcopy(function)
            for name, function in self.functions.items()
            if any(argument.annotation is None for argument in function.args.args)
        }
        self.clones: dict[tuple[str, tuple[tuple[int, str], ...]], str] = {}

    def run(self) -> ast.Module:
        if not self.groups:
            return self.module
        self._reject_cycles()
        self._reject_type_uses()
        pending = [statement for statement in self.module.body if isinstance(statement, ast.FunctionDef)]
        while pending:
            pending.extend(self._flatten_function(pending.pop()))
        self.module.body = [statement for statement in self.module.body if not _is_group_declaration(statement)]
        return ast.fix_missing_locations(self.module)

    def _fields(self, declaration: ast.ClassDef, names: set[str]) -> tuple[_Field, ...]:
        fields: list[_Field] = []
        for statement in declaration.body:
            if isinstance(statement, ast.Pass) or (
                isinstance(statement, ast.Expr)
                and isinstance(statement.value, ast.Constant)
                and isinstance(statement.value.value, str)
            ):
                continue
            if (
                not isinstance(statement, ast.AnnAssign)
                or not isinstance(statement.target, ast.Name)
                or statement.value is not None
            ):
                raise self.error(statement, "@binding_group bodies may contain only annotation-only fields and methods")
            qualified = f"'{declaration.name}.{statement.target.id}'"
            head = _head(statement.annotation)
            if head == "Annotated":
                raise self.error(
                    statement.annotation, f"binding group field cannot declare interface metadata: {qualified}"
                )
            if head == "When":
                raise self.error(
                    statement.annotation, f"binding group field cannot declare a When condition: {qualified}"
                )
            annotation = statement.annotation
            group = annotation.id if isinstance(annotation, ast.Name) and annotation.id in names else None
            fields.append(_Field(statement.target.id, annotation, group))
        return tuple(fields)

    def _reject_cycles(self) -> None:
        states: dict[str, int] = {}

        def visit(name: str, chain: tuple[str, ...]) -> None:
            states[name] = 1
            for field in self.groups[name]:
                if field.group is None:
                    continue
                if states.get(field.group) == 1:
                    cycle = " -> ".join((*chain[chain.index(field.group) :], field.group))
                    raise self.error(field.annotation, f"binding group cannot contain itself: {cycle}")
                if states.get(field.group, 0) == 0:
                    visit(field.group, (*chain, field.group))
            states[name] = 2

        for name in self.groups:
            if states.get(name, 0) == 0:
                visit(name, (name,))

    def _group_names(self, annotation: ast.AST | None) -> list[ast.Name]:
        if annotation is None:
            return []
        return [node for node in ast.walk(annotation) if isinstance(node, ast.Name) and node.id in self.groups]

    def _parameter_group(self, annotation: ast.AST | None) -> str | None:
        if isinstance(annotation, ast.Name) and annotation.id in self.groups:
            return annotation.id
        return None

    def _reject_type_uses(self) -> None:
        for statement in self.module.body:
            if isinstance(statement, ast.ClassDef):
                for member in statement.body:
                    if not isinstance(member, ast.AnnAssign):
                        continue
                    nested = self._group_names(member.annotation)
                    if not nested:
                        continue
                    if not _is_group_declaration(statement):
                        raise self.error(nested[0], "binding group cannot be a Struct field")
                    if not isinstance(member.annotation, ast.Name):
                        raise self.error(nested[0], "binding group cannot be a Tensor, TensorView, or Tuple element")
                continue
            if not isinstance(statement, ast.FunctionDef):
                continue
            shared = _is_shared(statement)
            for argument in statement.args.args:
                nested = self._group_names(argument.annotation)
                if not nested:
                    continue
                if self._parameter_group(argument.annotation) is None:
                    assert argument.annotation is not None
                    if _head(argument.annotation) in {"Annotated", "When"}:
                        raise self.error(nested[0], "binding group cannot be annotated with interface metadata")
                    raise self.error(nested[0], "binding group cannot be a Tensor, TensorView, or Tuple element")
                if shared:
                    raise self.error(argument, f"binding group cannot be passed to shared function '{statement.name}'")
            if self._group_names(statement.returns):
                raise self.error(statement.returns or statement, "binding group cannot be returned")
            for node in ast.walk(statement):
                if isinstance(node, ast.AnnAssign) and self._group_names(node.annotation):
                    raise self.error(node, "binding group cannot be assigned to a local")

    def _signature(self, function: ast.FunctionDef) -> _Signature:
        return _Signature(
            tuple(self._parameter_group(argument.annotation) for argument in function.args.args),
            tuple(argument.annotation is not None for argument in function.args.args),
        )

    def leaves(self, group: str, prefix: str) -> Iterator[tuple[str, ast.expr]]:
        for field in self.groups[group]:
            path = f"{prefix}.{field.name}"
            if field.group is not None:
                yield from self.leaves(field.group, path)
            else:
                yield path, field.annotation

    def _flatten_function(self, function: ast.FunctionDef) -> list[ast.FunctionDef]:
        stages = [ENTRY_DECORATOR_STAGES.get(decorator_name(item)) for item in function.decorator_list]
        stage = next((value for value in stages if value is not None), None)
        environment: dict[str, _GroupRef] = {}
        arguments: list[ast.arg] = []
        for argument in function.args.args:
            group = self._parameter_group(argument.annotation)
            if group is None:
                arguments.append(argument)
                continue
            environment[argument.arg] = _GroupRef(group, argument.arg)
            for path, annotation in self.leaves(group, argument.arg):
                leaf = ast.copy_location(
                    ast.arg(arg=path, annotation=self._leaf_annotation(annotation, stage)), argument
                )
                leaf._vernon_binding_group = argument.arg
                arguments.append(leaf)
        rewriter = _BodyRewriter(self, environment)
        if environment:
            function.args.args = arguments
        function.body = [rewriter.visit(statement) for statement in function.body]
        return rewriter.clones

    @staticmethod
    def _leaf_annotation(annotation: ast.expr, stage: str | None) -> ast.expr:
        annotation = copy.deepcopy(annotation)
        if stage not in GRAPHICS_STAGES or _head(annotation) in _RESOURCE_CONSTRUCTORS:
            return annotation
        uniform = ast.Call(func=ast.Name(id="uniform", ctx=ast.Load()), args=[], keywords=[])
        return ast.Subscript(
            value=ast.Name(id="Annotated", ctx=ast.Load()),
            slice=ast.Tuple(elts=[annotation, uniform], ctx=ast.Load()),
            ctx=ast.Load(),
        )

    def specialize(self, name: str, positions: tuple[tuple[int, str], ...]) -> tuple[str, ast.FunctionDef | None]:
        key = (name, positions)
        existing = self.clones.get(key)
        if existing is not None:
            return existing, None
        digest = hashlib.sha256(repr(key).encode("utf-8")).hexdigest()[:12]
        clone = copy.deepcopy(self.originals[name])
        clone.name = f"{name}__{digest}"
        clone._vernon_qualified_name = name
        clone.decorator_list = [ast.Name(id="func", ctx=ast.Load())]
        for index, group in positions:
            clone.args.args[index].annotation = ast.Name(id=group, ctx=ast.Load())
        self.clones[key] = clone.name
        self.functions[clone.name] = clone
        self.signatures[clone.name] = self._signature(clone)
        if not all(self.signatures[clone.name].annotated):
            self.originals[clone.name] = copy.deepcopy(clone)
        self.module.body.append(clone)
        return clone.name, clone


class _BodyRewriter(ast.NodeTransformer):
    def __init__(self, owner: _Flattener, environment: dict[str, _GroupRef]):
        self.owner = owner
        self.environment = environment
        self.clones: list[ast.FunctionDef] = []

    def _resolve(self, node: ast.AST) -> _GroupRef | _MethodRef | str | None:
        """Returns the group occurrence, method, or dotted leaf path that `node` denotes, if any."""
        if isinstance(node, ast.Name):
            return self.environment.get(node.id)
        if not isinstance(node, ast.Attribute):
            return None
        base = self._resolve(node.value)
        if not isinstance(base, _GroupRef):
            return None
        for field in self.owner.groups[base.group]:
            if field.name == node.attr:
                path = f"{base.path}.{field.name}"
                return _GroupRef(field.group, path) if field.group is not None else path
        if node.attr in self.owner.methods.get(base.group, {}):
            return _MethodRef(base.group)
        raise self.owner.error(node, f"binding group '{base.group}' has no field '{node.attr}'")

    def _is_group(self, node: ast.AST | None) -> bool:
        return node is not None and isinstance(self._resolve(node), _GroupRef)

    def visit_Name(self, node: ast.Name) -> ast.expr:
        if node.id in self.environment:
            if not isinstance(node.ctx, ast.Load):
                raise self.owner.error(node, "binding group cannot be reassigned")
            raise self.owner.error(node, "binding group cannot be used as a value")
        return node

    def visit_Attribute(self, node: ast.Attribute) -> ast.expr:
        resolved = self._resolve(node)
        if resolved is None:
            return self.generic_visit(node)  # type: ignore[return-value]
        if not isinstance(node.ctx, ast.Load):
            raise self.owner.error(node, "binding group cannot be reassigned through its fields")
        if isinstance(resolved, str):
            return ast.copy_location(ast.Name(id=resolved, ctx=ast.Load()), node)
        if isinstance(resolved, _MethodRef):
            raise self.owner.error(node, "binding group methods cannot be used as first-class values")
        raise self.owner.error(node, "binding group cannot be used as a value")

    def visit_Return(self, node: ast.Return) -> ast.stmt:
        if self._is_group(node.value):
            raise self.owner.error(node, "binding group cannot be returned")
        return self.generic_visit(node)  # type: ignore[return-value]

    def visit_Assign(self, node: ast.Assign) -> ast.stmt:
        if self._is_group(node.value):
            raise self.owner.error(node, "binding group cannot be assigned to a local")
        return self.generic_visit(node)  # type: ignore[return-value]

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.stmt:
        if self._is_group(node.value):
            raise self.owner.error(node, "binding group cannot be assigned to a local")
        return self.generic_visit(node)  # type: ignore[return-value]

    def visit_NamedExpr(self, node: ast.NamedExpr) -> ast.expr:
        if self._is_group(node.value):
            raise self.owner.error(node, "binding group cannot be assigned to a local")
        return self.generic_visit(node)  # type: ignore[return-value]

    def visit_Compare(self, node: ast.Compare) -> ast.expr:
        if any(self._is_group(operand) for operand in (node.left, *node.comparators)):
            raise self.owner.error(node, "binding group cannot be compared")
        return self.generic_visit(node)  # type: ignore[return-value]

    def visit_Subscript(self, node: ast.Subscript) -> ast.expr:
        if self._is_group(node.value):
            raise self.owner.error(node, "binding group cannot be indexed")
        return self.generic_visit(node)  # type: ignore[return-value]

    def visit_Call(self, node: ast.Call) -> ast.expr:
        if isinstance(node.func, ast.Attribute):
            receiver = self._resolve(node.func.value)
            if isinstance(receiver, _GroupRef):
                helper = self.owner.methods.get(receiver.group, {}).get(node.func.attr)
                if helper is None:
                    raise self.owner.error(node.func, f"'{receiver.group}' has no method '{node.func.attr}'")
                node = ast.copy_location(
                    ast.Call(
                        func=ast.Name(id=helper, ctx=ast.Load()),
                        args=[node.func.value, *node.args],
                        keywords=node.keywords,
                    ),
                    node,
                )
        if isinstance(node.func, ast.Name) and node.func.id in self.owner.groups:
            raise self.owner.error(node, "binding group cannot be constructed in shader code")
        for keyword in node.keywords:
            if self._is_group(keyword.value):
                raise self.owner.error(keyword, "binding group cannot be passed by keyword")
        groups = [(index, self._resolve(argument)) for index, argument in enumerate(node.args)]
        positions = [(index, value) for index, value in groups if isinstance(value, _GroupRef)]
        if not positions:
            return self.generic_visit(node)  # type: ignore[return-value]
        callee = node.func.id if isinstance(node.func, ast.Name) else None
        signature = self.owner.signatures.get(callee or "")
        if callee is None or signature is None:
            raise self.owner.error(node, f"binding group cannot be passed to '{dotted_name(node.func) or 'call'}'")
        if len(node.args) != len(signature.groups):
            raise self.owner.error(node, f"function '{callee}' expects {len(signature.groups)} arguments")
        generic: list[tuple[int, str]] = []
        for index, reference in positions:
            assert isinstance(reference, _GroupRef)
            if not signature.annotated[index]:
                generic.append((index, reference.group))
            elif signature.groups[index] != reference.group:
                raise self.owner.error(
                    node.args[index],
                    f"binding group cannot be passed as argument {index} of '{callee}', "
                    f"which does not declare {reference.group}",
                )
        if generic:
            symbol, clone = self.owner.specialize(callee, tuple(generic))
            if clone is not None:
                self.clones.append(clone)
            node.func = ast.copy_location(ast.Name(id=symbol, ctx=ast.Load()), node.func)
        arguments: list[ast.expr] = []
        for argument, (_, reference) in zip(node.args, groups, strict=True):
            if isinstance(reference, _GroupRef):
                arguments.extend(
                    ast.copy_location(ast.Name(id=path, ctx=ast.Load()), argument)
                    for path, _ in self.owner.leaves(reference.group, reference.path)
                )
            else:
                arguments.append(self.visit(argument))
        node.args = arguments
        node.keywords = [self.visit(keyword) for keyword in node.keywords]
        return node


def flatten_binding_groups(module: ast.Module, error: Error) -> ast.Module:
    return _Flattener(module, error).run()


def binding_group_occurrence(argument: ast.arg) -> str | None:
    """The entry parameter path of the group occurrence a flattened leaf belongs to."""
    return getattr(argument, "_vernon_binding_group", None)


__all__ = ["binding_group_occurrence", "flatten_binding_groups"]
