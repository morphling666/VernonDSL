"""Builds the typed Host program of one `@host` entry.

Host code shares the device frontend up to typed inference: the module graph, type parser, monomorphization, and
typed expression model are the same. It forks afterwards, because Host functions call C externs and return statuses
instead of running as shader stages.
"""

from __future__ import annotations

import ast
import copy
from collections.abc import Callable, Mapping
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
from .classes import generic_parameter, is_host_class, lower_host_classes, method_table, parse_metadata
from .format_spec import parse_format_spec, spec_text, value_kind
from .layout import OWNED_KINDS, TRANSFERRED_KINDS, layout
from .model import (
    ExternAbi,
    HostEffect,
    HostEnum,
    HostExtern,
    HostFunctionInfo,
    HostHandle,
    HostProgram,
    HostStruct,
    Metadata,
)
from .ownership import OwnershipChecker
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
_LIST_METHODS = frozenset({"append", "pop", "clear", "copy"})
_DICT_METHODS = frozenset({"get", "pop", "clear", "copy", "keys", "values", "items"})
_DICT_VIEWS = frozenset({"keys", "values", "items"})
_STORAGE_METHODS = frozenset({"fill", "copy_from_values", "to_values", "copy"})
# `copy.deepcopy`, spelled through the module or imported from it.
_DEEPCOPY = frozenset({"deepcopy", "copy.deepcopy"})
_STORAGE_LIKE = frozenset({"zeros_like", "empty_like"})
_STORAGE_CREATIONS = frozenset({"zeros", "empty", "from_values", *_STORAGE_LIKE})


def _targets(statement: ast.Assign | ast.AnnAssign) -> list[ast.expr]:
    return statement.targets if isinstance(statement, ast.Assign) else [statement.target]


def _iterated(node: ast.expr) -> ast.expr:
    """The container a for loop iterates: `x` in `for k in x` and in `for k, v in x.items()`."""

    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in _DICT_VIEWS:
        return node.func.value
    return node


_BITWISE = (ast.BitAnd, ast.BitOr, ast.BitXor, ast.LShift, ast.RShift)
_OPERATOR_SPELLING = {ast.BitAnd: "&", ast.BitOr: "|", ast.BitXor: "^", ast.LShift: "<<", ast.RShift: ">>"}
_SUPPORTED_BINARY = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, *_BITWISE)


class _HostContext:
    def __init__(self, filename: str):
        self.filename = filename
        self.structs: dict[str, tuple] = {}
        self.handles: set[str] = set()
        self.enums: set[str] = set()

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


def _options_and_metadata(
    context: _HostContext, decorator: ast.expr, allowed: set[str]
) -> tuple[dict[str, str], Metadata]:
    """The string options of a decorator and its `meta=` entries."""

    if not isinstance(decorator, ast.Call):
        return {}, ()
    metadata: Metadata = ()
    keywords = []
    for keyword in decorator.keywords:
        if keyword.arg == "meta":
            metadata = parse_metadata(context.filename, keyword.value)
        else:
            keywords.append(keyword)
    stripped = ast.Call(func=decorator.func, args=decorator.args, keywords=keywords)
    ast.copy_location(stripped, decorator)
    return _keyword_strings(context, stripped, allowed), metadata


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


def load_host_library(source_path: str | Path) -> tuple[HostProgram, tuple[str, ...]]:
    """Every declaration of a `vd.library` module, which need not define `@host` code: the program holds the
    module's own non-generic externs, and generic externs, which only call sites type, come as their signatures."""

    from .library import generic_signature

    path = Path(source_path).resolve()
    program = _load(path, None, library=True)
    project = load_project(path, (), None)
    own = {name for name, (declaring, _, _) in project.library_externs.items() if declaring == path}
    generic_signatures = tuple(
        generic_signature(node)
        for node in project.module.body
        if isinstance(node, ast.FunctionDef) and node.name in own and generic_parameter(node) is not None
    )
    externs = {name: extern for name, extern in program.externs.items() if name in own}
    return replace(program, externs=externs), generic_signatures


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


def _check_python_imports(module: ast.Module, python_imports: Mapping[str, str], filename: str) -> None:
    """Rejects Host code naming what a Python import brought in: Python mode would run it, but the compiler sees
    only Host sources."""

    if not python_imports:
        return
    scopes: list[ast.AST] = []
    for node in module.body:
        if isinstance(node, ast.FunctionDef):
            scopes.append(node)
        elif isinstance(node, ast.ClassDef) and {decorator_name(item) for item in node.decorator_list} & {
            "host",
            "struct",
            "handle",
            "enum",
        }:
            scopes.extend(item for item in node.body if not isinstance(item, ast.Expr))
    for scope in scopes:
        decorators = {id(node) for decorator in getattr(scope, "decorator_list", []) for node in ast.walk(decorator)}
        for value in ast.walk(scope):
            if (
                isinstance(value, ast.Name)
                and isinstance(value.ctx, ast.Load)
                and value.id in python_imports
                and value.id not in _INTRINSICS
                and id(value) not in decorators
            ):
                raise CompileError(
                    f"'{value.id}' comes from '{python_imports[value.id]}', which is not a Host source: Host code "
                    "reaches another module only when it imports vernon_dsl and is found in the project or along "
                    "sys.path, as an installed package",
                    node_location(value, filename),
                )


def _generic_declarations(
    context: _HostContext,
    parser: TypeParser,
    source: ast.Module,
    structs: dict[str, HostStruct],
    struct_fields,
) -> list[HostExtern]:
    """Each generic extern the module declares, typed as written: its type variable stands as a memberless Struct
    of the variable's name, and `type_argument` holds the descriptor's position and that name."""

    declared: list[HostExtern] = []
    for original in source.body:
        if not isinstance(original, ast.FunctionDef) or not any(
            decorator_name(item) == "extern" for item in original.decorator_list
        ):
            continue
        found = generic_parameter(original)
        if found is None:
            continue
        index, variable, passed = found
        if variable in structs:
            raise context.error(original, f"the type variable '{variable}' of '{original.name}' names a Struct")
        node = copy.deepcopy(original)
        if passed:
            del node.args.args[index]
        context.structs[variable] = ()
        structs[variable] = HostStruct(variable, ())
        try:
            extern = _extern(context, parser, node, struct_fields)
        finally:
            del context.structs[variable]
            del structs[variable]
        # The source keeps the `type[T]` parameter, whose name binding generators spell.
        declared.append(replace(extern, type_argument=(index, variable), type_passed=passed, source=original))
    return declared


def load_generic_externs(source_path: str | Path) -> tuple[HostExtern, ...]:
    """The generic externs a `vd.library` module declares, typed as written (`_generic_declarations`), for binding
    generators: call sites specialize them, so `load_host_library` holds none."""

    generics: list[HostExtern] = []
    _load(Path(source_path).resolve(), None, library=True, generics=generics)
    return tuple(generics)


def _load(
    path: Path, entry: str | None, *, library: bool = False, generics: list[HostExtern] | None = None
) -> HostProgram:
    """`library` loads a `vd.library` module for its identity: it needs no `@host` exports, and its externs keep
    their declared schemas, since the identity is what their schema would become."""

    project = load_project(path, (), entry)
    _check_python_imports(project.module, project.python_imports, str(path))
    module = copy.deepcopy(project.module)
    detach_shared_nodes(module)
    module, lowered = lower_host_classes(module, str(path))
    enum_docs = {
        node.name: ast.get_docstring(node)
        for node in module.body
        if isinstance(node, ast.ClassDef) and any(decorator_name(item) == "enum" for item in node.decorator_list)
    }
    strip_docstrings(module)
    module = normalize_struct_methods(module, str(path))
    module._vernon_methods = {**method_tables(module), **method_table(lowered)}
    module._vernon_value_generics = lowered.by_value
    context = _HostContext(str(path))
    parser = TypeParser(context, host=True)

    structs: dict[str, HostStruct] = {}
    handles: dict[str, HostHandle] = {}
    enums: dict[str, HostEnum] = {}
    class_nodes = [node for node in module.body if isinstance(node, ast.ClassDef)]
    for node in class_nodes:
        kinds = {decorator_name(item) for item in node.decorator_list}
        if kinds == {"handle"}:
            options, metadata = _options_and_metadata(context, node.decorator_list[0], {"c_type"})
            context.handles.add(node.name)
            handles[node.name] = HostHandle(node.name, options.get("c_type"), metadata)
        elif kinds == {"enum"}:
            options, metadata = _options_and_metadata(context, node.decorator_list[0], {"c_type"})
            context.enums.add(node.name)
            enums[node.name] = _enum(context, node, options.get("c_type"), enum_docs.get(node.name), metadata)
        elif kinds == {"struct"}:
            context.structs[node.name] = ()
        else:
            raise context.error(node, "Host programs declare classes only with @struct, @handle or @enum")
    for node in class_nodes:
        if node.name in enums:
            continue
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
                _check_container(context, annotation, field_type, struct_field_types, contains_place)
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
                extern = replace(extern, type_argument=(generic.type_index, generic.bound), type_passed=generic.passed)
            declared_name = node.name.removesuffix(f"__{generic.bound}") if generic is not None else node.name
            declaring = project.library_externs.get(declared_name)
            if declaring is not None and extern.schema is None and not library:
                from .library import library_identity

                extern = replace(extern, schema=library_identity(*declaring))
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
        for argument in node.args.args:
            if argument.annotation is None:
                if kind == "host":
                    raise context.error(argument, f"Host argument '{argument.arg}' requires a type annotation")
                continue
            annotated = parser.parse(argument.annotation)
            _require_host_value(context, argument.annotation, annotated.type, struct_field_types)
            if annotated.metadata:
                raise context.error(argument.annotation, "Host arguments carry no interface metadata")
            _require_class_boundary(context, argument.annotation, annotated.type, kind, contains_place, "passed")
        if kind == "host" and node.returns is None:
            raise context.error(node, f"Host function '{node.name}' requires a result annotation")
        if node.returns is not None and not (isinstance(node.returns, ast.Constant) and node.returns.value is None):
            result_type = parser.parse_type(node.returns)
            _require_host_value(context, node.returns, result_type, struct_field_types)
            _require_class_boundary(context, node.returns, result_type, kind, contains_place, "returned")
    if generics is not None:
        generics.extend(_generic_declarations(context, parser, project.module, structs, struct_field_types))

    module = infer_and_monomorphize_helpers(
        module,
        parser.parse_type,
        lambda annotation: parser.parse(annotation).metadata,
        context.error,
        (),
        host=True,
    )
    if lowered.by_value:
        # A value-bound generic extern is declared for every Struct; inference pointed each call at one.
        called = {
            node.func.id for node in ast.walk(module) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        unused = {
            name
            for name in externs
            if name not in called
            and name in lowered.generics
            and name.removesuffix(f"__{lowered.generics[name].bound}") in lowered.by_value
        }
        externs = {name: extern for name, extern in externs.items() if name not in unused}
        module.body = [node for node in module.body if not (isinstance(node, ast.FunctionDef) and node.name in unused)]

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
        if not listed and not library:
            raise CompileError("the module defines no @host function or class", node_location(module, str(path)))
        exports = tuple(listed)
        entry = exports[0][1] if exports else ""
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
        enums,
    )
    validator = _Validator(context, program, lowered.owners)
    for info in functions.values():
        validator.validate(info)
    validator.ownership.finish()
    validator.propagate_effects()
    return program


def _enum(
    context: _HostContext, node: ast.ClassDef, c_type: str | None, doc: str | None, metadata: Metadata
) -> HostEnum:
    if len(node.bases) != 1 or not (dotted_name(node.bases[0]) or "").endswith("IntEnum") or node.keywords:
        raise context.error(node, f"enum '{node.name}' derives from enum.IntEnum and nothing else")
    members: list[tuple[str, int]] = []
    for statement in node.body:
        if isinstance(statement, ast.Pass) or (
            isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant)
        ):
            continue
        value = statement.value if isinstance(statement, ast.Assign) else None
        if (
            isinstance(value, ast.UnaryOp)
            and isinstance(value.op, ast.USub)
            and isinstance(value.operand, ast.Constant)
            and type(value.operand.value) is int
        ):
            value = ast.Constant(value=-value.operand.value)
        if (
            not isinstance(statement, ast.Assign)
            or len(statement.targets) != 1
            or not isinstance(statement.targets[0], ast.Name)
            or not isinstance(value, ast.Constant)
            or type(value.value) is not int
        ):
            raise context.error(statement, f"enum '{node.name}' members are written NAME = integer literal")
        name = statement.targets[0].id
        if name.startswith("_") or name in dict(members):
            raise context.error(statement, f"enum member '{name}' is private or declared twice")
        if not -(1 << 31) <= value.value < 1 << 31:
            raise context.error(statement, f"enum member '{name}' does not fit in an int32_t")
        members.append((name, value.value))
    if not members:
        raise context.error(node, f"enum '{node.name}' declares no members")
    return HostEnum(node.name, tuple(members), c_type, doc, metadata)


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


def _require_class_boundary(
    context: _HostContext,
    node: ast.AST,
    value_type: ConcreteType,
    kind: str,
    contains_place: Callable[[ConcreteType], bool],
    how: str,
) -> None:
    """A `@host` function takes a class instance by reference and returns a new one; a `@func` helper, which
    device code shares, handles no instances. A class inside a Tensor or Tuple is never a value."""

    if not contains_place(value_type):
        return
    if kind != "host":
        raise context.error(node, f"a @func helper is shared with device code; a Host class is not {how} to one")
    if value_type.kind != "struct":
        raise context.error(node, f"Host class instances are {how} one at a time; {value_type.spelling} holds them")


def _check_container(
    context: _HostContext,
    node: ast.AST,
    value_type: ConcreteType,
    struct_fields: Callable[[str], tuple[ConcreteType, ...]],
    contains_place: Callable[[ConcreteType], bool],
) -> None:
    """A list's elements and a dict's values are Host Values or Host class instances, and a dict's keys have one
    equality."""

    element = value_type.arguments[-1] if value_type.kind in {"list", "dict"} else None
    if isinstance(element, ConcreteType) and not (
        (element.kind == "struct" and contains_place(element))
        or (_host_value(element, struct_fields) and not contains_place(element))
    ):
        what = "list elements" if value_type.kind == "list" else "dict values"
        raise context.error(node, f"{what} are Host Values or Host class instances; {element.spelling} is neither")
    key = value_type.arguments[0] if value_type.kind == "dict" else None
    if isinstance(key, ConcreteType) and not (
        key.kind in {"handle", "enum", "str"} or (key.kind == "scalar" and not key.is_float)
    ):
        raise context.error(node, f"dict keys are integers, bools, handles, enums or str; {key.spelling} is not one")


def _require_host_value(context: _HostContext, node: ast.AST, value_type: ConcreteType, struct_fields) -> None:
    if value_type.kind in _DEVICE_KINDS:
        raise context.error(node, f"{value_type.spelling} is a device type; Host code cannot use it")
    if value_type.kind == "storage":
        raise context.error(
            node, f"a {value_type.spelling} is a class field or a local, which owns its memory; it is not passed"
        )
    if value_type.kind in OWNED_KINDS:
        raise context.error(
            node, f"a {value_type.spelling} lives in a class field or a local, which owns its memory; it is not passed"
        )
    if not _host_value(value_type, struct_fields):
        raise context.error(node, f"{value_type.spelling} is not a Host Value")


def _require_transfer(context: _HostContext, node: ast.AST, value_type: ConcreteType, struct_fields, what: str) -> None:
    """A list or dict crosses to C as a span or an owned result of plain values, which C reads as their bytes."""

    items = [element_of(value_type)] if value_type.kind == "list" else [key_of(value_type), element_of(value_type)]
    for item in items:
        if not _host_value(item, struct_fields):
            raise context.error(
                node,
                f"{value_type.spelling} crosses to C as {what} of plain values (scalars, enums, handles, Structs "
                f"and Tensors), so it cannot hold {item.spelling}",
            )


def _extern(context: _HostContext, parser: TypeParser, node: ast.FunctionDef, struct_fields) -> HostExtern:
    decorator = node.decorator_list[0]
    options, metadata = _options_and_metadata(context, decorator, {"symbol", "effects", "abi", "schema"})
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
        if value_type.kind in {"list", "dict"}:
            _require_transfer(context, argument.annotation, value_type, struct_fields, "a span")
        elif value_type.kind != "str":
            _require_host_value(context, argument.annotation, value_type, struct_fields)
        parameters.append((argument.arg, value_type))
    result = None
    if node.returns is None:
        raise context.error(node, f"extern '{node.name}' requires a result annotation")
    if not (isinstance(node.returns, ast.Constant) and node.returns.value is None):
        result = parser.parse_type(node.returns)
        if result.kind in TRANSFERRED_KINDS:
            if abi != ExternAbi.STATUS:
                raise context.error(
                    node.returns, f"an extern returns a {result.kind} only with abi='status', which owns it"
                )
            if result.kind != "str":
                _require_transfer(context, node.returns, result, struct_fields, "an owned result")
        else:
            _require_host_value(context, node.returns, result, struct_fields)
        if abi == ExternAbi.DIRECT and result.kind not in {"scalar", "handle", "enum"}:
            raise context.error(node.returns, "a direct extern returns a scalar, handle or enum; use abi='status'")
    return HostExtern(
        node.name,
        symbol,
        tuple(parameters),
        result,
        _parse_effect(context, decorator, options.get("effects", "io")),
        abi,
        options.get("schema"),
        node,
        metadata=metadata,
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
        self.ownership = OwnershipChecker(program, context.error)

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
                iterated = _iterated(node.iter)
                if isinstance(iterated, ast.Name) and self._is_container_local(info, iterated):
                    self._validate_loop_keeps_container(node, iterated)
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
                callee = self.program.externs.get(node.func.id) if isinstance(node.func, ast.Name) else None
                if callee is not None and callee.result is not None and callee.result.kind in TRANSFERRED_KINDS:
                    self._validate_owned_result(info, node, parents)
            elif isinstance(node, ast.BinOp):
                self._validate_binary(info, node)
            elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Invert):
                operand = info.expression_type(node)
                element = operand.arguments[0] if operand.kind == "tensor" else operand
                if not isinstance(element, ConcreteType) or not element.is_integer:
                    raise self.error(node, f"~ requires integer operands; {operand.spelling} is not one")
            elif isinstance(node, ast.Compare) and len(node.ops) != 1:
                raise self.error(node, "chained comparisons are not supported")
            elif (
                isinstance(node, ast.Compare)
                and not isinstance(node.ops[0], (ast.Eq, ast.NotEq, ast.In, ast.NotIn))
                and info.expression_type(node.left).kind in {"enum", "handle"}
            ):
                operand = info.expression_type(node.left).spelling
                raise self.error(node, f"{operand} values compare only with == and !=; convert an enum with int()")
            elif isinstance(node, (ast.Lambda, ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp)):
                raise self.error(node, f"{type(node).__name__} is not Host syntax")
            elif isinstance(node, ast.Dict) and (
                node.keys or not isinstance(parents.get(id(node)), (ast.Assign, ast.AnnAssign))
            ):
                raise self.error(
                    node,
                    "a dict is not passed as a value; {} declares a dict local (x: dict[K, V] = {}) or empties a "
                    "dict field or local",
                )
            elif isinstance(node, ast.Set):
                raise self.error(node, "Set is not a Host Value")
            elif isinstance(node, ast.Delete):
                for target in node.targets:
                    if not isinstance(target, ast.Subscript) or (
                        self._place_root(info, target) is None and not self._is_container_local(info, target.value)
                    ):
                        raise self.error(
                            target, "del removes an entry of a dict field or local, as in del self.counts[key]"
                        )
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
                isinstance(node, (ast.Attribute, ast.Name))
                and isinstance(node.ctx, ast.Load)
                and id(node) in info.expressions
                and info.expression_type(node).kind in {"list", "dict"}
            ):
                self._validate_container_use(info, node, parents)
            if (
                isinstance(node, ast.Call)
                and id(node) in info.expressions
                and info.expression_type(node).kind in {"list", "dict"}
                and (
                    (isinstance(node.func, ast.Attribute) and node.func.attr == "copy")
                    or dotted_name(node.func) in _DEEPCOPY
                )
            ):
                self._validate_container_copy(node, parents)
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
        self.ownership.check(info, parents)

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
                and (dotted_name(parent.func) or "").split(".")[-1] in {"len", "deepcopy", *_STORAGE_LIKE}
            ):
                return
        raise self.error(
            node,
            f"'{ast.unparse(node)}' is a TensorStorage; index it, read its shape, call len, fill, copy_from_values, "
            "to_values, copy, zeros_like or empty_like; assign a storage only from a creation or a copy, since "
            "Python would share it",
        )

    def _contains_place(self, value_type: ConcreteType) -> bool:
        if value_type.kind == "tensor":
            element = value_type.arguments[0]
            return isinstance(element, ConcreteType) and self._contains_place(element)
        return self.program.is_place(value_type)

    def _is_container_local(self, info: HostFunctionInfo, node: ast.expr | None) -> bool:
        return (
            isinstance(node, ast.Name)
            and node.id in info.local_types
            and node.id not in dict(info.parameters)
            and info.local_types[node.id].kind in {"list", "dict"}
        )

    def _validate_loop_keeps_container(self, loop: ast.For, container: ast.Name) -> None:
        """Python rebinds a name to a new container while its loop keeps iterating the old one, where compiled code
        empties the one record the loop reads; so a loop does not redeclare the local it iterates."""

        for node in (child for statement in loop.body for child in ast.walk(statement)):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            if any(isinstance(target, ast.Name) and target.id == container.id for target in _targets(node)):
                raise self.error(
                    node,
                    f"the loop iterates '{container.id}', so its body cannot rebind it; call {container.id}.clear()",
                )

    def _validate_container_use(
        self, info: HostFunctionInfo, node: ast.Attribute | ast.Name, parents: dict[int, ast.AST]
    ) -> None:
        """A list or dict field or local is used in place: it is never copied, so it appears only where an operation
        reads or changes its own memory."""

        value_type = info.expression_type(node)
        parent = parents.get(id(node))
        if isinstance(parent, ast.Call) and dotted_name(parent.func) in {"len", *_DEEPCOPY} and parent.args == [node]:
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
            if isinstance(parent, ast.Call) and self._extern_parameter(parent, node) is not None:
                # Python passes the container itself, which a later call could change or rebind after C has read
                # it.
                later = parent.args[parent.args.index(node) + 1 :]
                if any(self._may_change_state(info, item) for other in later for item in ast.walk(other)):
                    raise self.error(
                        node,
                        f"'{ast.unparse(node)}' is passed as a span, so no later argument may call a function, "
                        "method or extern; assign that argument to a local first",
                    )
                return
            where = "local" if isinstance(node, ast.Name) else "field"
            if value_type.kind == "list":
                raise self.error(
                    node,
                    f"'{ast.unparse(node)}' is a list {where}; index it, iterate it, or call len, append, pop, "
                    "clear or copy; Python would share it, so copy it explicitly",
                )
            raise self.error(
                node,
                f"'{ast.unparse(node)}' is a dict {where}; index it, test 'in', iterate it, or call len, get, pop, "
                "clear, copy, keys, values or items; Python would share it, so copy it explicitly",
            )

    def _validate_container_copy(self, node: ast.Call, parents: dict[int, ast.AST]) -> None:
        """A copy of a list or dict is a new container, which Host code holds only in a container local or field:
        functions take and return no containers."""

        parent = parents.get(id(node))
        if (
            isinstance(parent, (ast.Assign, ast.AnnAssign))
            and parent.value is node
            and len(_targets(parent)) == 1
            and isinstance(_targets(parent)[0], (ast.Name, ast.Attribute))
        ):
            return
        raise self.error(
            node, f"'{ast.unparse(node)}' copies a container only as the value of a list or dict local or field"
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
        if isinstance(parent, ast.Call) and self._extern_parameter(parent, node) is not None:
            return
        raise self.error(
            node,
            f"'{ast.unparse(node)}' is a str; assign it to a str local or field, compare it with == or !=, format it "
            "in an f-string, join it with +, use it as a dict key, or pass it to len or an extern str parameter",
        )

    def _may_change_state(self, info: HostFunctionInfo, node: ast.AST) -> bool:
        """Whether `node` calls a Host function, an extern or a method of a Host value, any of which may change a
        container; builtins and conversions change none."""

        if not isinstance(node, ast.Call):
            return False
        if isinstance(node.func, ast.Name):
            return node.func.id in self.program.functions or node.func.id in self.program.externs
        return isinstance(node.func, ast.Attribute) and id(node.func.value) in info.expressions

    def _extern_parameter(self, call: ast.Call, node: ast.expr) -> ConcreteType | None:
        """The str, list or dict parameter of the extern `call` that `node` is passed to, if it is one."""

        callee = self.program.externs.get(call.func.id) if isinstance(call.func, ast.Name) else None
        if callee is None or not any(argument is node for argument in call.args):
            return None
        parameter = callee.parameters[call.args.index(node)][1]
        return parameter if parameter.kind in TRANSFERRED_KINDS else None

    def _validate_owned_result(self, info: HostFunctionInfo, node: ast.Call, parents: dict[int, ast.AST]) -> None:
        """An extern's str, list or dict result is received like a copy, into a local or field of its type: Host
        functions take and return no containers, so there is nothing else to hold it."""

        result = info.expression_type(node)
        if info.kind != "host":
            raise self.error(node, f"@func helper '{info.symbol}' cannot call an extern returning a {result.kind}")
        parent = parents.get(id(node))
        if (
            isinstance(parent, (ast.Assign, ast.AnnAssign))
            and parent.value is node
            and len(_targets(parent)) == 1
            and isinstance(_targets(parent)[0], (ast.Name, ast.Attribute))
        ):
            return
        raise self.error(
            node,
            f"'{ast.unparse(node.func)}' returns {result.spelling}, which is received only as the value of an "
            f"assignment to a {result.kind} local or field",
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
            if info.kind != "host":
                raise self.error(
                    node, f"@func helper '{info.symbol}' cannot hold a {lvalue.type.spelling}; containers are Host-only"
                )
            _check_container(
                self.context,
                node,
                lvalue.type,
                lambda name: tuple(field for _, field in self.program.struct_fields(name)),
                self._contains_place,
            )
        if lvalue.name in {name for name, value in self.info_parameters if self.program.is_place(value)}:
            raise self.error(node, f"'{lvalue.name}' names a borrowed instance; it cannot be rebound")
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
        if dotted_name(node.func) in _DEEPCOPY and name not in self.program.functions:
            if info.kind != "host":
                raise self.error(node, f"@func helper '{info.symbol}' cannot copy instances or containers")
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
            if source.kind not in {"scalar", "enum"}:
                raise self.error(node.args[0], f"{name}() converts scalars and enums; {source.spelling} is not one")
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
        """The place a chain of field and element accesses starts from, if it starts from one: a class parameter or
        local, or a list or dict local."""

        while isinstance(node, (ast.Attribute, ast.Subscript)):
            node = node.value
        if not isinstance(node, ast.Name):
            return None
        local = info.local_types.get(node.id)
        if local is None or not (self.program.is_place(local) or self._is_container_local(info, node)):
            return None
        return node

    def _validate_field_store(self, info: HostFunctionInfo, node: ast.Attribute) -> None:
        if self._place_root(info, node) is None:
            raise self.error(
                node,
                "only fields of class instances are writable; Struct values are immutable, so build a new one",
            )
        if not self.program.is_place(info.expression_type(node.value)):
            # Python holds a Struct field as an immutable value, so only the whole field can be replaced.
            raise self.error(
                node,
                f"'{ast.unparse(node.value)}' is a Struct value; assign the whole field with a new "
                f"{info.expression_type(node.value).spelling}",
            )

    def _validate_store(self, info: HostFunctionInfo, node: ast.Subscript) -> None:
        # Values are never written into (typing rejects element stores into Tensors), so an indexed store writes a
        # container: a storage, a list or dict local, or a list or dict field of an instance.
        field = node.value
        if info.expression_type(field).kind == "storage" or self._is_container_local(info, field):
            return
        if (
            self._place_root(info, node) is None
            or not isinstance(field, ast.Attribute)
            or not self.program.is_place(info.expression_type(field.value))
            or info.expression_type(field).kind not in {"list", "dict"}
        ):
            raise self.error(node, "indexed assignment writes a TensorStorage, or a list or dict field or local")

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
