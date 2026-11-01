"""Lowers `@host` classes to what the shared frontend types: a Struct of the class's fields, a `@host` function per
method taking `self` first, and a synthesized `__init__` writing the field defaults.

The Struct is only the layout; `HostStruct.place` keeps the class's values from being copied, and the emitter
passes `self` as the pointer the application owns, so field stores through it mutate the instance in place.
"""

from __future__ import annotations

import ast
import copy
import json
from dataclasses import dataclass

from ..diagnostics import CompileError, copy_location, fix_missing_locations, node_location
from ..language.ast_utils import decorator_name, dotted_name
from .model import HostClass, HostField, HostMethod, Metadata

_METADATA_SCALARS = (str, int, float, bool)


@dataclass(frozen=True)
class StructDocs:
    doc: str | None
    metadata: Metadata
    # Types are filled in once the frontend parses the fields.
    fields: tuple[HostField, ...]


@dataclass(frozen=True)
class GenericExtern:
    """A specialization of an extern generic over a Struct type: `type_index` is where the C call receives the
    type's reflection descriptor. `passed` says whether Host code passes the type as an argument; if not, the value
    at `type_index` binds it."""

    type_index: int
    bound: str
    passed: bool = True


@dataclass(frozen=True)
class LoweredClasses:
    classes: dict[str, HostClass]
    # Hoisted method symbol -> (class, method name).
    owners: dict[str, tuple[str, str]]
    # Field annotations as written, for diagnostics on field types.
    annotations: dict[str, dict[str, ast.expr]]
    structs: dict[str, StructDocs]
    generics: dict[str, GenericExtern]
    # Value-bound generic externs: the index of the value whose Struct type binds `T`, by extern name.
    by_value: dict[str, int]


def _error(filename: str, node: ast.AST, message: str) -> CompileError:
    return CompileError(message, node_location(node, filename))


def is_host_class(node: ast.stmt) -> bool:
    return isinstance(node, ast.ClassDef) and any(decorator_name(item) == "host" for item in node.decorator_list)


def _metadata_value(filename: str, node: ast.expr) -> object:
    if isinstance(node, ast.Constant) and isinstance(node.value, _METADATA_SCALARS):
        return node.value
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, (ast.USub, ast.UAdd))
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, (int, float))
        and not isinstance(node.operand.value, bool)
    ):
        return -node.operand.value if isinstance(node.op, ast.USub) else node.operand.value
    if isinstance(node, (ast.Tuple, ast.List)):
        return [_metadata_value(filename, item) for item in node.elts]
    raise _error(filename, node, "metadata values are str, int, float, or bool constants, or tuples of them")


def parse_metadata(filename: str, node: ast.expr) -> Metadata:
    if not isinstance(node, ast.Call) or decorator_name(node) != "meta":
        raise _error(filename, node, "Host metadata is written as vd.meta(key=value, ...)")
    if node.args:
        raise _error(filename, node, "vd.meta takes only keyword entries")
    entries: list[tuple[str, str]] = []
    for keyword in node.keywords:
        if keyword.arg is None:
            raise _error(filename, keyword, "vd.meta takes only keyword entries")
        entries.append((keyword.arg, json.dumps(_metadata_value(filename, keyword.value))))
    return tuple(entries)


def _split_annotation(filename: str, annotation: ast.expr) -> tuple[ast.expr, Metadata]:
    """The type of an `Annotated[T, vd.meta(...), ...]` field and its merged metadata."""

    if not (isinstance(annotation, ast.Subscript) and (dotted_name(annotation.value) or "").endswith("Annotated")):
        return annotation, ()
    items = list(annotation.slice.elts) if isinstance(annotation.slice, ast.Tuple) else [annotation.slice]
    if len(items) < 2:
        raise _error(filename, annotation, "Annotated needs a type and metadata")
    merged: dict[str, str] = {}
    for item in items[1:]:
        for key, value in parse_metadata(filename, item):
            if key in merged:
                raise _error(filename, item, f"metadata key '{key}' is given twice")
            merged[key] = value
    return items[0], tuple(merged.items())


def _docstring(node: ast.stmt | None) -> str | None:
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
        return node.value.value
    return None


def _class_options(filename: str, node: ast.ClassDef) -> Metadata:
    if len(node.decorator_list) != 1:
        raise _error(filename, node, "a @host class takes exactly one decorator")
    decorator = node.decorator_list[0]
    if not isinstance(decorator, ast.Call):
        return ()
    if decorator.args:
        raise _error(filename, decorator, "@host accepts only keyword options")
    metadata: Metadata = ()
    for keyword in decorator.keywords:
        if keyword.arg != "meta":
            raise _error(filename, keyword, f"a @host class takes only meta=, not '{keyword.arg}'")
        metadata = parse_metadata(filename, keyword.value)
    return metadata


def _method_effects(filename: str, method: ast.FunctionDef) -> ast.expr:
    """The `@host(effects=...)` decorator of a hoisted method; an undecorated method may do anything."""

    if not method.decorator_list:
        return ast.Call(
            func=ast.Name(id="host", ctx=ast.Load()),
            args=[],
            keywords=[ast.keyword(arg="effects", value=ast.Constant(value="io"))],
        )
    if len(method.decorator_list) != 1 or decorator_name(method.decorator_list[0]) != "host":
        raise _error(filename, method, "Host class methods take no decorator or one @host(effects=...)")
    decorator = method.decorator_list[0]
    return ast.Name(id="host", ctx=ast.Load()) if not isinstance(decorator, ast.Call) else decorator


def _struct_docs(filename: str, node: ast.ClassDef) -> StructDocs:
    """Takes the reflection-only parts out of a `@struct`: its `meta=` option, field metadata, and docstrings."""

    metadata: Metadata = ()
    decorator = node.decorator_list[0]
    if isinstance(decorator, ast.Call):
        for keyword in list(decorator.keywords):
            if keyword.arg == "meta":
                metadata = parse_metadata(filename, keyword.value)
                decorator.keywords.remove(keyword)
    doc = _docstring(node.body[0]) if node.body else None
    fields: list[HostField] = []
    kept: list[ast.stmt] = []
    for index, member in enumerate(node.body):
        if _docstring(member) is not None and (index == 0 or isinstance(node.body[index - 1], ast.AnnAssign)):
            continue
        if isinstance(member, ast.AnnAssign) and isinstance(member.target, ast.Name):
            member.annotation, field_metadata = _split_annotation(filename, member.annotation)
            following = node.body[index + 1] if index + 1 < len(node.body) else None
            fields.append(HostField(member.target.id, None, _docstring(following), field_metadata))  # type: ignore[arg-type]
        kept.append(member)
    node.body = kept or [ast.Pass()]
    return StructDocs(doc, metadata, tuple(fields))


def type_parameter(annotation: ast.expr | None) -> str | None:
    """The type variable of a `type[T]` annotation."""

    if (
        isinstance(annotation, ast.Subscript)
        and dotted_name(annotation.value) == "type"
        and isinstance(annotation.slice, ast.Name)
    ):
        return annotation.slice.id
    return None


def generic_parameter(node: ast.FunctionDef) -> tuple[int, str, bool] | None:
    """Where a generic extern's C function receives the bound Struct's descriptor, the type variable, and whether
    Host code passes the Struct there as a `type: type[T]` argument. Without one, the first parameter annotated `T`
    binds it: Host code passes a Struct value, whose type is the bound Struct, and the descriptor goes just before
    it. A `type[T]` parameter is needed where no value says the type, as for a getter."""

    typed = [
        (index, variable)
        for index, argument in enumerate(node.args.args)
        if (variable := type_parameter(argument.annotation)) is not None
    ]
    if typed:
        return typed[0][0], typed[0][1], True
    variables = getattr(node, "_vernon_type_variables", ())
    for index, argument in enumerate(node.args.args):
        if isinstance(argument.annotation, ast.Name) and argument.annotation.id in variables:
            return index, argument.annotation.id, False
    return None


class _Substitute(ast.NodeTransformer):
    def __init__(self, variable: str, bound: str):
        self.variable = variable
        self.bound = bound

    def visit_Name(self, node: ast.Name) -> ast.Name:
        return copy_location(ast.Name(id=self.bound, ctx=node.ctx), node) if node.id == self.variable else node


def _specialize_generic_externs(
    module: ast.Module, filename: str, struct_names: set[str]
) -> tuple[dict[str, GenericExtern], dict[str, int]]:
    """Replaces each generic extern by its specializations, one per bound Struct. A `type[T]` extern is specialized
    per Struct type its calls pass, dropping the type argument from the call; the emitter passes the type's
    descriptor in its place. A value-bound extern is specialized for every Struct, since only inference knows a
    value's type: inference points each call at its value's specialization (the returned index of the value by
    extern name), and the frontend drops the ones no call reaches."""

    generic: dict[str, tuple[ast.FunctionDef, int, str]] = {}
    by_value: dict[str, int] = {}
    specialized: dict[str, ast.FunctionDef] = {}
    generics: dict[str, GenericExtern] = {}
    for node in module.body:
        if not isinstance(node, ast.FunctionDef) or not any(
            decorator_name(item) == "extern" for item in node.decorator_list
        ):
            continue
        if sum(type_parameter(argument.annotation) is not None for argument in node.args.args) > 1:
            raise _error(filename, node, "a generic extern takes one type[T] parameter")
        found = generic_parameter(node)
        if found is None:
            continue
        index, variable, passed = found
        if passed:
            generic[node.name] = (node, index, variable)
            continue
        by_value[node.name] = index
        for bound in sorted(struct_names):
            name = f"{node.name}__{bound}"
            copy_node = _Substitute(variable, bound).visit(copy.deepcopy(node))
            copy_node.name = name
            specialized[name] = copy_node
            generics[name] = GenericExtern(index, bound, passed=False)
    for node in ast.walk(module):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id not in generic:
            continue
        declaration, index, variable = generic[node.func.id]
        if index >= len(node.args):
            raise _error(filename, node, f"'{node.func.id}' takes a type argument at position {index}")
        argument = node.args[index]
        bound = dotted_name(argument)
        if not isinstance(argument, ast.Name) or bound not in struct_names:
            raise _error(filename, argument, f"the type argument of '{node.func.id}' is a @struct type")
        name = f"{node.func.id}__{bound}"
        if name not in specialized:
            copy_node = _Substitute(variable, bound).visit(copy.deepcopy(declaration))
            del copy_node.args.args[index]
            copy_node.name = name
            specialized[name] = copy_node
            generics[name] = GenericExtern(index, bound)
        node.func = copy_location(ast.Name(id=name, ctx=ast.Load()), node.func)
        del node.args[index]
    if not specialized and not generic and not by_value:
        return {}, {}
    module.body = [
        statement
        for statement in module.body
        if not (isinstance(statement, ast.FunctionDef) and (statement.name in generic or statement.name in by_value))
    ] + list(specialized.values())
    return generics, by_value


def lower_host_classes(module: ast.Module, filename: str) -> tuple[ast.Module, LoweredClasses]:
    nodes = [node for node in module.body if is_host_class(node)]
    names = {node.name for node in nodes}
    struct_docs = {
        node.name: _struct_docs(filename, node)
        for node in module.body
        if isinstance(node, ast.ClassDef) and [decorator_name(item) for item in node.decorator_list] == ["struct"]
    }
    generics, by_value = _specialize_generic_externs(module, filename, set(struct_docs))
    classes: dict[str, HostClass] = {}
    owners: dict[str, tuple[str, str]] = {}
    annotations: dict[str, dict[str, ast.expr]] = {}
    body: list[ast.stmt] = []
    hoisted: list[ast.stmt] = []
    for statement in module.body:
        if statement not in nodes:
            body.append(statement)
            continue
        assert isinstance(statement, ast.ClassDef)
        if statement.bases or statement.keywords:
            raise _error(filename, statement, "Host classes do not support inheritance")
        metadata = _class_options(filename, statement)
        class_doc = _docstring(statement.body[0]) if statement.body else None
        members = statement.body[1:] if class_doc is not None else statement.body
        fields: list[HostField] = []
        layout: list[ast.stmt] = []
        defaults: list[tuple[str, ast.expr | None, ast.AnnAssign]] = []
        methods: list[HostMethod] = []
        field_annotations: dict[str, ast.expr] = {}
        for index, member in enumerate(members):
            if isinstance(member, ast.Pass) or (_docstring(member) is not None and index > 0):
                continue
            if isinstance(member, ast.AnnAssign) and isinstance(member.target, ast.Name):
                name = member.target.id
                if name in field_annotations:
                    raise _error(filename, member, f"duplicate field '{name}'")
                annotation, field_metadata = _split_annotation(filename, member.annotation)
                field_annotations[name] = annotation
                following = members[index + 1] if index + 1 < len(members) else None
                fields.append(HostField(name, None, _docstring(following), field_metadata))  # type: ignore[arg-type]
                layout.append(
                    copy_location(
                        ast.AnnAssign(
                            target=ast.Name(id=name, ctx=ast.Store()),
                            annotation=copy.deepcopy(annotation),
                            value=None,
                            simple=1,
                        ),
                        member,
                    )
                )
                annotation_name = dotted_name(annotation)
                if annotation_name in names:
                    if member.value is not None:
                        raise _error(
                            filename, member.value, f"field '{name}' initializes from {annotation_name}'s defaults"
                        )
                    defaults.append((name, None, member))
                elif member.value is not None:
                    defaults.append((name, member.value, member))
                continue
            if isinstance(member, ast.FunctionDef):
                if member.name.startswith("__") and member.name.endswith("__"):
                    raise _error(
                        filename, member, f"Host classes define no '{member.name}'; fields initialize from defaults"
                    )
                if not member.args.args or member.args.args[0].arg != "self":
                    raise _error(filename, member, "Host class methods take 'self' first")
                if member.args.args[0].annotation is not None:
                    raise _error(filename, member.args.args[0], "the type of self is the class")
                if any(method.name == member.name for method in methods) or member.name in field_annotations:
                    raise _error(filename, member, f"duplicate member '{member.name}'")
                symbol = f"{statement.name}__{member.name}"
                helper = copy.deepcopy(member)
                helper.name = symbol
                helper.decorator_list = [_method_effects(filename, member)]
                helper.args.args[0].annotation = copy_location(ast.Name(id=statement.name, ctx=ast.Load()), member)
                hoisted.append(helper)
                owners[symbol] = (statement.name, member.name)
                methods.append(HostMethod(member.name, symbol, not member.name.startswith("_")))
                continue
            raise _error(filename, member, "Host class bodies contain fields, their docstrings, and methods")
        init_symbol = f"{statement.name}____init__"
        init_body: list[ast.stmt] = []
        for name, value, source in defaults:
            target = ast.Attribute(value=ast.Name(id="self", ctx=ast.Load()), attr=name, ctx=ast.Store())
            if value is None:
                call = ast.Call(
                    func=ast.Attribute(
                        value=ast.Attribute(value=ast.Name(id="self", ctx=ast.Load()), attr=name, ctx=ast.Load()),
                        attr="__init__",
                        ctx=ast.Load(),
                    ),
                    args=[],
                    keywords=[],
                )
                init_body.append(copy_location(ast.Expr(value=call), source))
            else:
                init_body.append(copy_location(ast.Assign(targets=[target], value=copy.deepcopy(value)), source))
        init = ast.FunctionDef(
            name=init_symbol,
            args=ast.arguments(
                posonlyargs=[],
                args=[ast.arg(arg="self", annotation=ast.Name(id=statement.name, ctx=ast.Load()))],
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[],
            ),
            body=init_body or [ast.Pass()],
            decorator_list=[
                ast.Call(
                    func=ast.Name(id="host", ctx=ast.Load()),
                    args=[],
                    keywords=[ast.keyword(arg="effects", value=ast.Constant(value="pure"))],
                )
            ],
            returns=ast.Constant(value=None),
        )
        hoisted.append(copy_location(init, statement))
        owners[init_symbol] = (statement.name, "__init__")
        layout_class = copy_location(
            ast.ClassDef(
                name=statement.name,
                bases=[],
                keywords=[],
                body=layout or [ast.Pass()],
                decorator_list=[ast.Name(id="struct", ctx=ast.Load())],
            ),
            statement,
        )
        layout_class._vernon_host_class = True  # type: ignore[attr-defined]
        body.append(layout_class)
        classes[statement.name] = HostClass(
            statement.name, tuple(fields), tuple(methods), init_symbol, class_doc, metadata
        )
        annotations[statement.name] = field_annotations
    module.body = body + hoisted
    return fix_missing_locations(module), LoweredClasses(classes, owners, annotations, struct_docs, generics, by_value)


def method_table(lowered: LoweredClasses) -> dict[str, dict[str, str]]:
    tables: dict[str, dict[str, str]] = {}
    for symbol, (owner, method) in lowered.owners.items():
        tables.setdefault(owner, {})[method] = symbol
    return tables
