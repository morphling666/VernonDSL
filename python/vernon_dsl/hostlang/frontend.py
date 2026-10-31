"""Builds the typed Host program of one `@host` entry.

Host code shares the device frontend up to typed inference: the module graph, type parser, monomorphization, and
typed expression model are the same. It forks afterwards, because Host functions call C externs and return statuses
instead of running as shader stages.
"""

from __future__ import annotations

import ast
import copy
from dataclasses import replace
from pathlib import Path

from ..diagnostics import CompileError, node_location
from ..frontend.model import ConcreteType, LValue, TypedStatement, is_abi_stable_value
from ..frontend.monomorphize import infer_and_monomorphize_helpers
from ..frontend.type_parser import TypeParser
from ..frontend.type_solver import common_type
from ..language.ast_utils import (
    decorator_name,
    detach_shared_nodes,
    dotted_name,
    strip_docstrings,
    tensor_update_parts,
)
from ..module_graph import load_project
from ..shader_contracts import DEVICE_ONLY_OPERATION_NAMES
from ..struct_methods import method_tables, normalize_struct_methods
from .classes import is_host_class, lower_host_classes, method_table
from .format_spec import parse_format_spec, spec_text, value_kind
from .layout import OWNED_KINDS, layout
from .model import ExternAbi, HostEffect, HostExtern, HostFunctionInfo, HostHandle, HostProgram, HostStruct
from .values import element_of, key_of, literal_value

_DEVICE_KINDS = frozenset({"tensor_view", "tensor_storage", "texture", "sampler"})
_CASTS = frozenset({"bool", "int", "float", "i32", "u32", "i64", "u64", "f32", "f64"})
_INTRINSICS = frozenset(
    {
        "abs",
        "acos",
        "atan2",
        "clamp",
        "cos",
        "cross",
        "dot",
        "exp",
        "floor",
        "log",
        "max",
        "min",
        "norm",
        "normalize",
        "pow",
        "sin",
        "sqrt",
    }
)
_AGGREGATES = frozenset({"Vector", "Matrix", "Tensor", "Tuple"})
_LIST_METHODS = frozenset({"append", "pop", "clear"})
_DICT_METHODS = frozenset({"get", "pop", "clear", "keys", "values", "items"})
_STORAGE_METHODS = frozenset({"fill", "copy_from_values", "to_values"})
_STORAGE_LIKE = frozenset({"zeros_like", "empty_like"})
_STORAGE_CREATIONS = frozenset({"zeros", "empty", "from_values", *_STORAGE_LIKE})


def _targets(statement: ast.Assign | ast.AnnAssign) -> list[ast.expr]:
    return statement.targets if isinstance(statement, ast.Assign) else [statement.target]


_BITWISE = (ast.BitAnd, ast.BitOr, ast.BitXor, ast.LShift, ast.RShift)
_OPERATOR_SPELLING = {ast.BitAnd: "&", ast.BitOr: "|", ast.BitXor: "^", ast.LShift: "<<", ast.RShift: ">>"}
_SUPPORTED_BINARY = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, *_BITWISE)


class _HostContext:
    def __init__(self, filename: str):
        self.filename = filename
        self.structs: dict[str, tuple] = {}
        self.handles: set[str] = set()

    def error(self, node: ast.AST, message: str) -> CompileError:
        return CompileError(message, node_location(node, self.filename))


def _keyword_strings(context: _HostContext, decorator: ast.expr, allowed: set[str]) -> dict[str, str]:
    if not isinstance(decorator, ast.Call):
        return {}
    if decorator.args:
        raise context.error(decorator, f"@{decorator_name(decorator)} accepts only keyword options")
    options: dict[str, str] = {}
    for keyword in decorator.keywords:
        if keyword.arg not in allowed:
            raise context.error(keyword, f"unknown @{decorator_name(decorator)} option '{keyword.arg}'")
        if not isinstance(keyword.value, ast.Constant) or not isinstance(keyword.value.value, str):
            raise context.error(keyword.value, f"@{decorator_name(decorator)} option '{keyword.arg}' must be a string")
        options[keyword.arg] = keyword.value.value
    return options


def _shared(decorator: ast.expr) -> bool:
    return isinstance(decorator, ast.Call) and any(
        keyword.arg == "shared" and isinstance(keyword.value, ast.Constant) and keyword.value.value is True
        for keyword in decorator.keywords
    )


def _parse_effect(context: _HostContext, node: ast.AST, value: str) -> HostEffect:
    try:
        return HostEffect.parse(value)
    except ValueError as error:
        raise context.error(node, str(error)) from None


def load_host_program(source_path: str | Path, entry: str) -> HostProgram:
    """The `@host` function `entry` of `source_path` with everything it reaches."""

    return _load(Path(source_path).resolve(), entry)


def load_host_module(source_path: str | Path) -> HostProgram:
    """Every `@host` function and class `source_path` defines, exported together with what they reach."""

    return _load(Path(source_path).resolve(), None)


def _root_exports(path: Path) -> tuple[list[str], list[str]]:
    """The names of the `@host` functions and classes the file itself defines, in source order."""

    tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
    functions = [
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and any(decorator_name(item) == "host" for item in node.decorator_list)
    ]
    classes = [node.name for node in tree.body if is_host_class(node)]
    return functions, classes


def _load(path: Path, entry: str | None) -> HostProgram:
    project = load_project(path, (), entry)
    module = copy.deepcopy(project.module)
    detach_shared_nodes(module)
    module, lowered = lower_host_classes(module, str(path))
    strip_docstrings(module)
    module = normalize_struct_methods(module, str(path))
    module._vernon_methods = {**method_tables(module), **method_table(lowered)}
    context = _HostContext(str(path))
    parser = TypeParser(context, host=True)

    structs: dict[str, HostStruct] = {}
    handles: dict[str, HostHandle] = {}
    class_nodes = [node for node in module.body if isinstance(node, ast.ClassDef)]
    for node in class_nodes:
        kinds = {decorator_name(item) for item in node.decorator_list}
        if kinds == {"handle"}:
            options = _keyword_strings(context, node.decorator_list[0], {"c_type"})
            context.handles.add(node.name)
            handles[node.name] = HostHandle(node.name, options.get("c_type"))
        elif kinds == {"struct"}:
            context.structs[node.name] = ()
        else:
            raise context.error(node, "Host programs declare classes only with @struct or @handle")
    for node in class_nodes:
        if node.name in handles:
            if any(not isinstance(item, (ast.Pass, ast.Expr)) for item in node.body):
                raise context.error(node, f"handle '{node.name}' has no fields or methods")
            continue
        decorator = node.decorator_list[0]
        options: dict[str, str] = {}
        if isinstance(decorator, ast.Call):
            for keyword in decorator.keywords:
                if keyword.arg == "c_type" and isinstance(keyword.value, ast.Constant):
                    options["c_type"] = str(keyword.value.value)
        if node.name not in lowered.classes and not _shared(decorator):
            raise context.error(
                node,
                f"Host code runs as Python, which constructs Struct '{node.name}'; declare it @struct(shared=True)",
            )
        fields: list[tuple[str, ConcreteType]] = []
        for statement in node.body:
            if isinstance(statement, ast.Pass):
                continue
            if (
                not isinstance(statement, ast.AnnAssign)
                or not isinstance(statement.target, ast.Name)
                or statement.value is not None
            ):
                raise context.error(statement, "@struct bodies may contain only annotation-only fields")
            fields.append((statement.target.id, parser.parse_type(statement.annotation)))
        context.structs[node.name] = tuple(fields)
        docs = lowered.structs.get(node.name)
        structs[node.name] = HostStruct(
            node.name,
            tuple(fields),
            options.get("c_type"),
            node.name in lowered.classes,
            docs.doc if docs is not None else None,
            docs.metadata if docs is not None else (),
            tuple(replace(info, type=field_type) for info, (_, field_type) in zip(docs.fields, fields, strict=True))
            if docs is not None
            else (),
        )

    # Value checks see a container field as its record of u64s: only classes hold them, and their fields are
    # checked one by one below.
    def record(value_type: ConcreteType) -> ConcreteType:
        words = layout(value_type, lambda name: ()).size // 8
        return ConcreteType("tensor", "Tensor", (ConcreteType("scalar", "u64"), words))

    def struct_field_types(name: str) -> tuple[ConcreteType, ...]:
        return tuple(record(value) if value.kind in OWNED_KINDS else value for _, value in structs[name].fields)

    def contains_place(value_type: ConcreteType) -> bool:
        if value_type.kind == "tensor":
            element = value_type.arguments[0]
            return isinstance(element, ConcreteType) and contains_place(element)
        return value_type.kind == "struct" and structs[value_type.name].place

    _reject_containment_cycles(context, structs, class_nodes)
    for node in class_nodes:
        if node.name not in structs:
            continue
        owner = structs[node.name]
        kind = "class" if owner.place else "Struct"
        for field_name, field_type in owner.fields:
            if field_type.kind in OWNED_KINDS:
                annotation = lowered.annotations.get(node.name, {}).get(field_name, node)
                if not owner.place:
                    raise context.error(
                        annotation,
                        f"Struct field '{node.name}.{field_name}' cannot be a {field_type.spelling}: a Struct is a "
                        "value, and only class instances own memory",
                    )
                element = field_type.arguments[-1] if field_type.kind in {"list", "dict"} else None
                if isinstance(element, ConcreteType) and (
                    not _host_value(element, struct_field_types) or contains_place(element)
                ):
                    what = "list elements" if field_type.kind == "list" else "dict values"
                    raise context.error(
                        annotation, f"{what} are Host Values without classes; {element.spelling} is not one"
                    )
                key = field_type.arguments[0] if field_type.kind == "dict" else None
                if isinstance(key, ConcreteType) and not (
                    key.kind in {"handle", "str"} or (key.kind == "scalar" and not key.is_float)
                ):
                    raise context.error(
                        annotation, f"dict keys are integers, bools, handles or str; {key.spelling} is not one"
                    )
                continue
            if not _host_value(field_type, struct_field_types):
                raise context.error(node, f"{kind} field '{node.name}.{field_name}' must be an ABI-stable Host Value")
            if contains_place(field_type) and (not owner.place or field_type.kind != "struct"):
                where = "a Struct, which is a value," if not owner.place else "an array"
                raise context.error(
                    lowered.annotations.get(node.name, {}).get(field_name, node),
                    f"Host class {field_type.spelling} cannot be held in {where}; hold it as a class field",
                )
    classes = {
        name: replace(
            host_class,
            fields=tuple(
                replace(field, type=field_type)
                for field, (_, field_type) in zip(host_class.fields, structs[name].fields, strict=True)
            ),
        )
        for name, host_class in lowered.classes.items()
    }

    externs: dict[str, HostExtern] = {}
    declared_effects: dict[str, tuple[str, HostEffect]] = {}
    for node in module.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        if len(node.decorator_list) != 1:
            raise context.error(node, "Host program functions require exactly one decorator")
        decorator = node.decorator_list[0]
        kind = decorator_name(decorator)
        if kind not in {"host", "extern", "func"}:
            raise context.error(decorator, f"@{kind} entries cannot run in a Host program")
        if node.args.posonlyargs or node.args.kwonlyargs or node.args.vararg or node.args.kwarg or node.args.defaults:
            raise context.error(node, "Host functions support only required positional arguments")
        if kind == "extern":
            extern = _extern(context, parser, node, struct_field_types)
            generic = lowered.generics.get(node.name)
            if generic is not None:
                extern = replace(extern, type_argument=(generic.type_index, generic.bound))
            externs[node.name] = extern
            continue
        if kind == "host":
            options = _keyword_strings(context, decorator, {"effects"})
            effect = _parse_effect(context, decorator, options.get("effects", "io"))
        else:
            if not _shared(decorator):
                raise context.error(
                    node, f"Host code runs as Python, which calls helper '{node.name}'; declare it @func(shared=True)"
                )
            effect = HostEffect.PURE
        declared_effects[node.name] = (kind, effect)
        for index, argument in enumerate(node.args.args):
            if argument.annotation is None:
                if kind == "host":
                    raise context.error(argument, f"Host argument '{argument.arg}' requires a type annotation")
                continue
            annotated = parser.parse(argument.annotation)
            _require_host_value(context, argument.annotation, annotated.type, struct_field_types)
            if annotated.metadata:
                raise context.error(argument.annotation, "Host arguments carry no interface metadata")
            if contains_place(annotated.type) and not (index == 0 and node.name in lowered.owners):
                raise context.error(
                    argument.annotation,
                    f"Host class {annotated.type.spelling} is passed only as self; make this a method of the class",
                )
        if kind == "host" and node.returns is None:
            raise context.error(node, f"Host function '{node.name}' requires a result annotation")
        if node.returns is not None and not (isinstance(node.returns, ast.Constant) and node.returns.value is None):
            result_type = parser.parse_type(node.returns)
            _require_host_value(context, node.returns, result_type, struct_field_types)
            if contains_place(result_type):
                raise context.error(node.returns, f"Host class {result_type.spelling} cannot be returned by value")

    module = infer_and_monomorphize_helpers(
        module,
        parser.parse_type,
        lambda annotation: parser.parse(annotation).metadata,
        context.error,
        (),
        host=True,
    )

    functions: dict[str, HostFunctionInfo] = {}
    for typed in getattr(module, "_vernon_typed_functions", ()):
        # Monomorphized helper copies have new symbols and are always pure @func helpers.
        kind, effect = declared_effects.get(typed.symbol, ("func", HostEffect.PURE))
        info = HostFunctionInfo(typed, kind, effect)
        _index(info, typed.body)
        functions[typed.symbol] = info
    if entry is not None:
        if entry not in functions or functions[entry].kind != "host" or entry in lowered.owners:
            raise CompileError(f"'{entry}' is not a @host function", node_location(module, str(path)))
        exports: tuple[tuple[str, str], ...] = ((entry, entry),)
        module_name = ""
    else:
        root_functions, root_classes = _root_exports(path)
        listed: list[tuple[str, str]] = [(name, name) for name in root_functions]
        for class_name in root_classes:
            host_class = classes[class_name]
            listed.append((f"{class_name}__init", host_class.init))
            listed.extend(
                (f"{class_name}_{method.name}", method.symbol) for method in host_class.methods if method.exported
            )
        if not listed:
            raise CompileError("the module defines no @host function or class", node_location(module, str(path)))
        exports = tuple(listed)
        entry = exports[0][1]
        module_name = path.stem

    program = HostProgram(
        path,
        entry,
        structs,
        handles,
        externs,
        functions,
        project.dependency_files,
        exports,
        classes,
        module_name,
    )
    validator = _Validator(context, program, lowered.owners)
    for info in functions.values():
        validator.validate(info)
    validator.propagate_effects()
    return program


def _reject_containment_cycles(
    context: _HostContext, structs: dict[str, HostStruct], nodes: list[ast.ClassDef]
) -> None:
    state: dict[str, int] = {}
    by_name = {node.name: node for node in nodes}

    def visit(name: str) -> None:
        if state.get(name) == 2:
            return
        if state.get(name) == 1:
            raise context.error(by_name[name], f"Host class '{name}' contains itself")
        state[name] = 1
        for _, field_type in structs[name].fields:
            while field_type.kind == "tensor":
                field_type = field_type.arguments[0]  # type: ignore[assignment]
            if field_type.kind == "struct":
                visit(field_type.name)
        state[name] = 2

    for name in structs:
        visit(name)


def _host_value(value_type: ConcreteType, struct_fields) -> bool:
    if value_type.kind in _DEVICE_KINDS or value_type.kind == "str":
        return False
    if value_type.kind == "scalar" and value_type.name == "f16":
        return False
    if value_type.kind == "tensor":
        element = value_type.arguments[0]
        return isinstance(element, ConcreteType) and element.kind == "scalar" and element.name != "f16"
    return is_abi_stable_value(value_type, struct_fields)


def _require_host_value(context: _HostContext, node: ast.AST, value_type: ConcreteType, struct_fields) -> None:
    if value_type.kind in _DEVICE_KINDS:
        raise context.error(node, f"{value_type.spelling} is a device type; Host code cannot use it")
    if value_type.kind == "storage":
        raise context.error(
            node, f"a {value_type.spelling} is a class field or a local, which owns its memory; it is not passed"
        )
    if value_type.kind in OWNED_KINDS:
        raise context.error(node, f"a {value_type.spelling} lives only in a class field, which owns its memory")
    if not _host_value(value_type, struct_fields):
        raise context.error(node, f"{value_type.spelling} is not a Host Value")


def _extern(context: _HostContext, parser: TypeParser, node: ast.FunctionDef, struct_fields) -> HostExtern:
    decorator = node.decorator_list[0]
    options = _keyword_strings(context, decorator, {"symbol", "effects", "abi", "schema"})
    body = [statement for statement in node.body if not isinstance(statement, ast.Pass)]
    if not all(isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant) for statement in body):
        raise context.error(node, f"extern '{node.name}' declares a C function; its body must be '...'")
    symbol = options.get("symbol", node.name)
    if not symbol.isidentifier():
        raise context.error(decorator, f"extern symbol '{symbol}' is not a C identifier")
    abi = options.get("abi", ExternAbi.DIRECT)
    if abi not in ExternAbi.ALL:
        raise context.error(decorator, "extern abi must be 'direct' or 'status'")
    parameters: list[tuple[str, ConcreteType]] = []
    for argument in node.args.args:
        if argument.annotation is None:
            raise context.error(argument, f"extern argument '{argument.arg}' requires a type annotation")
        value_type = parser.parse_type(argument.annotation)
        if value_type.kind != "str":
            _require_host_value(context, argument.annotation, value_type, struct_fields)
        parameters.append((argument.arg, value_type))
    result = None
    if node.returns is None:
        raise context.error(node, f"extern '{node.name}' requires a result annotation")
    if not (isinstance(node.returns, ast.Constant) and node.returns.value is None):
        result = parser.parse_type(node.returns)
        _require_host_value(context, node.returns, result, struct_fields)
        if abi == ExternAbi.DIRECT and result.kind not in {"scalar", "handle"}:
            raise context.error(node.returns, "a direct extern returns a scalar or handle; use abi='status'")
    return HostExtern(
        node.name,
        symbol,
        tuple(parameters),
        result,
        _parse_effect(context, decorator, options.get("effects", "io")),
        abi,
        options.get("schema"),
        node,
    )


def _index(info: HostFunctionInfo, statements: tuple[TypedStatement, ...]) -> None:
    for statement in statements:
        info.statements[id(statement.source)] = statement
        for expression in statement.expressions:
            info.expressions[id(expression.source)] = expression
        _index(info, statement.children)


class _Validator:
    """Host rules the shared inference does not enforce, because device code either cannot express them or gives
    them device meaning."""

    def __init__(self, context: _HostContext, program: HostProgram, owners: dict[str, tuple[str, str]]):
        self.context = context
        self.program = program
        self.owners = owners
        self.calls: dict[str, set[str]] = {}

    def error(self, node: ast.AST, message: str) -> CompileError:
        return self.context.error(node, message)

    def validate(self, info: HostFunctionInfo) -> None:
        # Decorators hold option strings, so only the body is Host code.
        body = [node for statement in info.source.body for node in ast.walk(statement)]
        parents: dict[int, ast.AST] = {}
        for parent in body:
            for child in ast.iter_child_nodes(parent):
                parents[id(child)] = parent
        self.calls[info.symbol] = set()
        self.info_parameters = info.parameters
        info.local_types.update(info.parameters)
        self._locals(info, info.typed.body)
        for node in body:
            if isinstance(node, ast.For):
                for name, target in self._loop_targets(info, node):
                    self._record_local(info, node, LValue("local", name, target))
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                try:
                    node.value.encode("utf-8")
                except UnicodeEncodeError:
                    raise self.error(node, "str literals must be valid UTF-8") from None
            elif isinstance(node, ast.FormattedValue):
                self._validate_formatted(info, node)
            elif (
                isinstance(node, ast.Constant)
                and isinstance(node.value, (bool, int, float))
                # A Tuple index is a position, not a Value.
                and id(node) in info.expressions
            ):
                literal_type = info.expression_type(node)
                try:
                    literal_value(node.value, literal_type.name)
                except OverflowError as error:
                    raise self.error(node, str(error)) from None
            elif isinstance(node, ast.Call):
                self._validate_call(info, node)
            elif isinstance(node, ast.BinOp):
                self._validate_binary(info, node)
            elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Invert):
                operand = info.expression_type(node)
                element = operand.arguments[0] if operand.kind == "tensor" else operand
                if not isinstance(element, ConcreteType) or not element.is_integer:
                    raise self.error(node, f"~ requires integer operands; {operand.spelling} is not one")
            elif isinstance(node, ast.Compare) and len(node.ops) != 1:
                raise self.error(node, "chained comparisons are not supported")
            elif isinstance(node, (ast.Lambda, ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp)):
                raise self.error(node, f"{type(node).__name__} is not Host syntax")
            elif isinstance(node, ast.Dict) and (node.keys or not isinstance(parents.get(id(node)), ast.Assign)):
                raise self.error(node, "dict displays are not Host Values; {} only empties a dict field")
            elif isinstance(node, ast.Set):
                raise self.error(node, "Set is not a Host Value")
            elif isinstance(node, ast.Delete):
                for target in node.targets:
                    if not isinstance(target, ast.Subscript) or self._place_root(info, target) is None:
                        raise self.error(target, "del removes an entry of a dict field, as in del self.counts[key]")
            elif isinstance(node, (ast.Global, ast.Nonlocal, ast.With, ast.Try, ast.Raise, ast.Assert)):
                raise self.error(node, f"{type(node).__name__} is not Host syntax")
            elif isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Store):
                self._validate_store(info, node)
            elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store):
                self._validate_field_store(info, node)
            elif (
                isinstance(node, ast.AugAssign)
                and id(node.target) in info.expressions
                and info.expression_type(node.target).kind == "str"
            ):
                if not isinstance(node.op, ast.Add):
                    raise self.error(node, "a str is updated only with +=")
            elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Attribute):
                target, value = info.expression_type(node.target), info.expression_type(node.value)
                self._validate_operator(node, node.op, target)
                if common_type(target, value) != target:
                    raise self.error(
                        node,
                        f"'{ast.unparse(node.target)}' is {target.spelling}; this update would make it "
                        f"{value.spelling}, so convert the operand explicitly",
                    )
            elif isinstance(node, ast.AugAssign) and id(node.target) in info.expressions:
                self._validate_operator(node, node.op, info.expression_type(node.target))
            if (
                isinstance(node, (ast.Name, ast.Attribute))
                and isinstance(node.ctx, ast.Load)
                and id(node) in info.expressions
                and self.program.is_place(info.expression_type(node))
            ):
                self._validate_place_use(info, node, parents)
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.ctx, ast.Load)
                and id(node) in info.expressions
                and info.expression_type(node).kind in {"list", "dict"}
            ):
                self._validate_container_use(info, node, parents)
            if (
                isinstance(node, (ast.Name, ast.Attribute, ast.Constant, ast.JoinedStr, ast.BinOp, ast.IfExp))
                and not isinstance(getattr(node, "ctx", None), (ast.Store, ast.Del))
                and id(node) in info.expressions
                and info.expression_type(node).kind == "str"
            ):
                self._validate_str_use(info, node, parents)
            if (
                isinstance(node, ast.expr)
                and not isinstance(getattr(node, "ctx", None), (ast.Store, ast.Del))
                and id(node) in info.expressions
                and info.expression_type(node).kind == "storage"
            ):
                self._validate_storage_use(info, node, parents)

    def _validate_storage_use(self, info: HostFunctionInfo, node: ast.expr, parents: dict[int, ast.AST]) -> None:
        """A storage is used in place: Python would alias another storage where C copies the record, so a storage
        field or local is assigned only a newly created storage and is otherwise only indexed, measured, or the
        receiver of a method."""

        if info.kind != "host":
            raise self.error(node, f"@func helper '{info.symbol}' cannot use TensorStorage values; they are Host-only")
        parent = parents.get(id(node))
        if isinstance(node, ast.Call):
            if (
                isinstance(parent, (ast.Assign, ast.AnnAssign))
                and parent.value is node
                and all(isinstance(target, (ast.Name, ast.Attribute)) for target in _targets(parent))
            ):
                return
            raise self.error(
                node,
                f"'{ast.unparse(node.func)}' creates a TensorStorage only as the value of a storage local or field",
            )
        if isinstance(node, (ast.Name, ast.Attribute)):
            if isinstance(parent, (ast.Subscript, ast.Attribute)) and parent.value is node:
                return
            if (
                isinstance(parent, ast.Call)
                and parent.args == [node]
                and (dotted_name(parent.func) or "").split(".")[-1] in {"len", *_STORAGE_LIKE}
            ):
                return
        raise self.error(
            node,
            f"'{ast.unparse(node)}' is a TensorStorage; index it, read its shape, call len, fill, copy_from_values, "
            "to_values, zeros_like or empty_like; assign a storage only from a creation, as it is not copied",
        )

    def _validate_container_use(self, info: HostFunctionInfo, node: ast.Attribute, parents: dict[int, ast.AST]) -> None:
        """A list or str field is used in place: it is never copied, so it appears only where an operation reads
        or changes the field's own memory."""

        value_type = info.expression_type(node)
        parent = parents.get(id(node))
        if isinstance(parent, ast.Call) and dotted_name(parent.func) == "len" and parent.args == [node]:
            return
        if value_type.kind in {"list", "dict"}:
            if isinstance(parent, ast.Subscript) and parent.value is node:
                return
            if isinstance(parent, ast.For) and parent.iter is node:
                return
            if isinstance(parent, ast.Attribute) and parent.attr in (
                _LIST_METHODS if value_type.kind == "list" else _DICT_METHODS
            ):
                return
            if (
                isinstance(parent, ast.Compare)
                and value_type.kind == "dict"
                and isinstance(parent.ops[0], (ast.In, ast.NotIn))
                and parent.comparators[0] is node
            ):
                return
            if value_type.kind == "list":
                raise self.error(
                    node,
                    f"'{ast.unparse(node)}' is a list field; index it, iterate it, or call len, append, pop or "
                    "clear; it is not copied",
                )
            raise self.error(
                node,
                f"'{ast.unparse(node)}' is a dict field; index it, test 'in', iterate it, or call len, get, pop, "
                "clear, keys, values or items; it is not copied",
            )

    def _validate_str_use(self, info: HostFunctionInfo, node: ast.expr, parents: dict[int, ast.AST]) -> None:
        """A str is read where it appears: copied into a str local or field, compared, formatted, joined, used as a
        dict key, measured, or passed to an extern. Host functions take and return no strs, so a str never leaves
        the function or instance owning its bytes."""

        if info.kind != "host":
            raise self.error(node, f"@func helper '{info.symbol}' cannot use str values; they are Host-only")
        parent = parents.get(id(node))
        if isinstance(parent, (ast.Assign, ast.AnnAssign, ast.AugAssign)) and parent.value is node:
            return
        if isinstance(parent, ast.FormattedValue) and parent.value is node:
            return
        if isinstance(parent, ast.BinOp) and info.expression_type(parent).kind == "str":
            return
        if isinstance(parent, ast.IfExp) and parent.test is not node:
            return
        if isinstance(parent, ast.Call) and dotted_name(parent.func) == "len" and parent.args == [node]:
            return
        if self._is_dict_key(info, node, parent):
            return
        if isinstance(parent, ast.Compare) and not isinstance(parent.ops[0], (ast.In, ast.NotIn)):
            if any(info.expression_type(operand).kind != "str" for operand in [parent.left, *parent.comparators]):
                raise self.error(parent, "a str compares only with a str")
            if not isinstance(parent.ops[0], (ast.Eq, ast.NotEq)):
                raise self.error(parent, "str values compare only with == and !=")
            return
        if isinstance(parent, ast.Call) and node in parent.args:
            callee = parent.func.id if isinstance(parent.func, ast.Name) else None
            extern = self.program.externs.get(callee or "")
            if extern is not None and extern.parameters[parent.args.index(node)][1].kind == "str":
                return
        raise self.error(
            node,
            f"'{ast.unparse(node)}' is a str; assign it to a str local or field, compare it with == or !=, format it "
            "in an f-string, join it with +, use it as a dict key, or pass it to len or an extern str parameter",
        )

    def _validate_formatted(self, info: HostFunctionInfo, node: ast.FormattedValue) -> None:
        if node.conversion != -1:
            raise self.error(node, "f-string conversions (!r, !s, !a, and the = form) are not supported")
        value_type = info.expression_type(node.value)
        kind = value_kind(value_type)
        if kind is None:
            raise self.error(
                node.value, f"f-strings format scalars, handles and strs; {value_type.spelling} is not one"
            )
        text = spec_text(node.format_spec)
        if text is None:
            raise self.error(node, "format specs cannot hold replacement fields")
        try:
            parse_format_spec(text, kind)
        except ValueError as error:
            raise self.error(node, str(error)) from None

    def _is_dict_key(self, info: HostFunctionInfo, node: ast.expr, parent: ast.AST | None) -> bool:
        """Whether `node` is the key of a dict subscript, `in` test, or get or pop."""

        def is_dict(value: ast.expr) -> bool:
            return id(value) in info.expressions and info.expression_type(value).kind == "dict"

        if isinstance(parent, ast.Subscript):
            return parent.slice is node and is_dict(parent.value)
        if isinstance(parent, ast.Compare) and isinstance(parent.ops[0], (ast.In, ast.NotIn)):
            return parent.left is node
        return (
            isinstance(parent, ast.Call)
            and isinstance(parent.func, ast.Attribute)
            and parent.func.attr in {"get", "pop"}
            and bool(parent.args)
            and parent.args[0] is node
            and is_dict(parent.func.value)
        )

    def _loop_targets(self, info: HostFunctionInfo, node: ast.For) -> list[tuple[str, ConcreteType]]:
        """The locals a for loop binds and their types."""

        if isinstance(node.iter, ast.Call) and dotted_name(node.iter.func) == "range":
            assert isinstance(node.target, ast.Name)
            return [(node.target.id, ConcreteType("scalar", "i32"))]
        iterated = info.expression_type(node.iter)
        if iterated.kind == "list":
            assert isinstance(node.target, ast.Name)
            return [(node.target.id, element_of(iterated))]
        view = node.iter.func.attr if isinstance(node.iter, ast.Call) else "keys"  # type: ignore[attr-defined]
        if view == "items":
            assert isinstance(node.target, ast.Tuple)
            names = [item.id for item in node.target.elts]  # type: ignore[attr-defined]
            return [(names[0], key_of(iterated)), (names[1], element_of(iterated))]
        assert isinstance(node.target, ast.Name)
        return [(node.target.id, key_of(iterated) if view == "keys" else element_of(iterated))]

    def _locals(self, info: HostFunctionInfo, statements: tuple[TypedStatement, ...]) -> None:
        for statement in statements:
            for lvalue in statement.lvalues:
                self._record_local(info, statement.source, lvalue)
            self._locals(info, statement.children)

    def _record_local(self, info: HostFunctionInfo, node: ast.stmt, lvalue: LValue) -> None:
        if lvalue.kind != "local":
            return
        if lvalue.type.kind == "void":
            raise self.error(node, f"local '{lvalue.name}' is assigned a value of no type")
        if lvalue.type.kind in {"list", "dict"}:
            raise self.error(
                node, f"a {lvalue.type.spelling} lives only in a class field; local '{lvalue.name}' cannot hold one"
            )
        if lvalue.name in {name for name, value in self.info_parameters if self.program.is_place(value)}:
            raise self.error(node, f"'{lvalue.name}' names the instance the method runs on; it cannot be rebound")
        if self.program.is_place(lvalue.type):
            raise self.error(node, f"Host class {lvalue.type.spelling} instances are not copied into locals")
        previous = info.local_types.setdefault(lvalue.name, lvalue.type)
        if previous != lvalue.type:
            raise self.error(
                node,
                f"Host local '{lvalue.name}' changes type from {previous.spelling} to {lvalue.type.spelling}; "
                "convert explicitly or annotate its first assignment",
            )

    def _validate_call(self, info: HostFunctionInfo, node: ast.Call) -> None:
        name = node.func.id if isinstance(node.func, ast.Name) else (dotted_name(node.func) or "").split(".")[-1]
        if name in _STORAGE_CREATIONS and info.expression_type(node).kind == "storage":
            return
        if tensor_update_parts(node) is not None:
            return
        if name == "replace" and name not in self.program.functions and info.expression_type(node).kind == "struct":
            return
        if node.keywords:
            raise self.error(node, "Host calls take only positional arguments")
        if name in DEVICE_ONLY_OPERATION_NAMES:
            raise self.error(node, f"'{name}' is a device operation; Host code cannot use it")
        if name in self.program.externs:
            if info.kind != "host":
                raise self.error(node, f"@func helper '{info.symbol}' cannot call extern '{name}'")
            self.calls[info.symbol].add(name)
            for argument, (_, parameter_type) in zip(node.args, self.program.externs[name].parameters, strict=True):
                if parameter_type.kind == "str" and info.expression_type(argument).kind != "str":
                    raise self.error(argument, f"extern '{name}' takes a str here")
            return
        if name == "len" and isinstance(node.func, ast.Name):
            return
        if isinstance(node.func, ast.Attribute) and id(node.func.value) in info.expressions:
            receiver = info.expression_type(node.func.value).kind
            if (
                (receiver == "list" and name in _LIST_METHODS)
                or (receiver == "dict" and name in _DICT_METHODS)
                or (receiver == "storage" and name in _STORAGE_METHODS)
            ):
                return
        if name in self.program.functions:
            callee = self.program.functions[name]
            if callee.kind == "host" and info.kind != "host":
                raise self.error(node, f"@func helper '{info.symbol}' cannot call @host function '{name}'")
            self.calls[info.symbol].add(name)
            return
        if name in self.program.structs or name in self.program.handles or name in _AGGREGATES:
            return
        if name in _CASTS:
            if len(node.args) != 1:
                raise self.error(node, f"{name}() converts exactly one scalar")
            source = info.expression_type(node.args[0])
            if source.kind != "scalar":
                raise self.error(node.args[0], f"{name}() converts scalars; {source.spelling} is not one")
            return
        if name in _INTRINSICS:
            return
        if name == "range" and isinstance(node.func, ast.Name):
            return
        raise self.error(node, f"Host code cannot call '{name}'")

    def _validate_operator(self, node: ast.AST, op: ast.operator, result: ConcreteType) -> None:
        if not isinstance(op, _SUPPORTED_BINARY):
            raise self.error(node, f"operator {type(op).__name__} is not supported in Host code")
        element = result.arguments[0] if result.kind == "tensor" else result
        if not isinstance(element, ConcreteType) or element.kind != "scalar" or element.name == "bool":
            raise self.error(node, f"arithmetic on {result.spelling} is not supported")
        if isinstance(op, ast.FloorDiv) and element.is_float:
            raise self.error(node, "floor division requires integer operands")
        if isinstance(op, _BITWISE) and element.is_float:
            raise self.error(node, f"{_OPERATOR_SPELLING[type(op)]} requires integer operands")
        if isinstance(op, ast.Pow) and not element.is_float:
            raise self.error(node, "power requires floating-point operands")

    def _validate_binary(self, info: HostFunctionInfo, node: ast.BinOp) -> None:
        result = info.expression_type(node)
        if result.kind == "str":
            if not isinstance(node.op, ast.Add):
                raise self.error(node, "str values join only with +")
            return
        self._validate_operator(node, node.op, result)
        for operand in (node.left, node.right):
            operand_type = info.expression_type(operand)
            if operand_type.kind == "tensor" and result.kind == "tensor":
                if operand_type.arguments[1:] != result.arguments[1:]:
                    raise self.error(node, "Host tensor arithmetic broadcasts only scalars")

    def _place_root(self, info: HostFunctionInfo, node: ast.expr) -> ast.Name | None:
        """The `self` parameter a chain of field and element accesses starts from, if it starts from one."""

        while isinstance(node, (ast.Attribute, ast.Subscript)):
            node = node.value
        if not isinstance(node, ast.Name):
            return None
        parameters = dict(info.parameters)
        return node if node.id in parameters and self.program.is_place(parameters[node.id]) else None

    def _validate_field_store(self, info: HostFunctionInfo, node: ast.Attribute) -> None:
        if self._place_root(info, node) is None:
            raise self.error(
                node, "only fields reached through self are writable; Struct values are immutable, so build a new one"
            )
        if not self.program.is_place(info.expression_type(node.value)):
            # Python holds a Struct field as an immutable value, so only the whole field can be replaced.
            raise self.error(
                node,
                f"'{ast.unparse(node.value)}' is a Struct value; assign the whole field with a new "
                f"{info.expression_type(node.value).spelling}",
            )
        if self.program.is_place(info.expression_type(node)):
            raise self.error(node, f"'{ast.unparse(node)}' is a class instance; assign its fields instead")

    def _validate_place_use(self, info: HostFunctionInfo, node: ast.expr, parents: dict[int, ast.AST]) -> None:
        parent = parents.get(id(node))
        if isinstance(parent, ast.Attribute) and parent.value is node:
            return
        if isinstance(parent, ast.Call) and node in parent.args:
            callee = self.program.functions.get(parent.func.id) if isinstance(parent.func, ast.Name) else None
            if callee is not None and callee.symbol in self.owners and parent.args.index(node) == 0:
                return
        raise self.error(
            node,
            f"'{ast.unparse(node)}' is a Host class instance; read its fields or call its methods, it is not copied",
        )

    def _validate_store(self, info: HostFunctionInfo, node: ast.Subscript) -> None:
        # Values are never written into (typing rejects element stores into Tensors), so an indexed store writes a
        # container: a storage, or a list or dict field of an instance.
        field = node.value
        if info.expression_type(field).kind == "storage":
            return
        if (
            self._place_root(info, node) is None
            or not isinstance(field, ast.Attribute)
            or not self.program.is_place(info.expression_type(field.value))
            or info.expression_type(field).kind not in {"list", "dict"}
        ):
            raise self.error(node, "indexed assignment writes a TensorStorage, or a list or dict field")

    def propagate_effects(self) -> None:
        resolved: dict[str, HostEffect] = {}

        def effect_of(symbol: str) -> HostEffect:
            if symbol in self.program.externs:
                return self.program.externs[symbol].effect
            if symbol in resolved:
                return resolved[symbol]
            resolved[symbol] = HostEffect.PURE
            value = max((effect_of(callee) for callee in self.calls.get(symbol, ())), default=HostEffect.PURE)
            resolved[symbol] = value
            return value

        # A fixpoint rather than a memoized walk, so functions calling each other see each other's externs.
        reached = {
            symbol: {callee for callee in self.calls.get(symbol, ()) if callee in self.program.externs}
            for symbol in self.program.functions
        }
        changed = True
        while changed:
            changed = False
            for symbol, externs in reached.items():
                size = len(externs)
                for callee in self.calls.get(symbol, ()):
                    externs |= reached.get(callee, set())
                changed |= len(externs) != size

        for symbol, info in self.program.functions.items():
            info.reached_externs = tuple(sorted(reached[symbol]))
            info.effect = effect_of(symbol)
            if info.effect > info.declared_effect:
                culprit = next(
                    callee for callee in sorted(self.calls[symbol]) if effect_of(callee) > info.declared_effect
                )
                raise self.error(
                    info.source,
                    f"@host function '{symbol}' declares effects='{info.declared_effect.spelling}' but calls "
                    f"'{culprit}', whose effect is '{effect_of(culprit).spelling}'",
                )
