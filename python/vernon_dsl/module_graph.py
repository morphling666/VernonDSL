from __future__ import annotations

import ast
import copy
import re
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, NoReturn

from .diagnostics import CompileError, SourceLocation
from .frontend.abi import attribute_layout
from .frontend.model import ConcreteType
from .language.ast_utils import decorator_name as _decorator_name
from .language.ast_utils import dotted_name as _dotted_name
from .language.scalar_types import SCALAR_ALIASES, SCALAR_TYPES
from .language.syntax import ENTRY_DECORATORS, FUNCTION_DECORATORS
from .shader_contracts import DEVICE_ONLY_OPERATION_NAMES, DEVICE_ONLY_TYPE_NAMES
from .source_identity import (
    project_module_name,
    project_module_path,
    project_root,
    source_file_digest,
    source_text_digest,
)
from .struct_methods import method_tables, normalize_struct_methods

if TYPE_CHECKING:
    from .types import SpecializationAssignment

_HOST_MODULES = {
    "__future__",
    "argparse",
    "cv2",
    "importlib",
    "numpy",
    "os",
    "pathlib",
    "sys",
    "typing",
    "types",
    "unittest",
    "vernon_dsl",
}
_SPECIALIZATION_KINDS = frozenset({"bool", "i32", "u32", "f32", "f64", "type"})
_STAGE_DECORATORS = ENTRY_DECORATORS
_FUNCTION_DECORATORS = FUNCTION_DECORATORS
_CLASS_DECORATORS = frozenset({"struct", "binding_group"})


def _is_dsl_class(node: ast.AST | None) -> bool:
    return isinstance(node, ast.ClassDef) and any(
        _decorator_name(item) in _CLASS_DECORATORS for item in node.decorator_list
    )


def _is_struct_declaration(node: ast.AST | None) -> bool:
    return isinstance(node, ast.ClassDef) and any(_decorator_name(item) == "struct" for item in node.decorator_list)


def _is_binding_group(node: ast.AST | None) -> bool:
    return isinstance(node, ast.ClassDef) and any(
        _decorator_name(item) == "binding_group" for item in node.decorator_list
    )


@dataclass(frozen=True)
class _ImportSpec:
    node: ast.AST
    module_name: str
    level: int
    local_name: str
    symbol_name: str | None = None


@dataclass
class _Module:
    path: Path
    name: str
    source: str
    tree: ast.Module
    definitions: dict[str, ast.FunctionDef | ast.ClassDef] = field(default_factory=dict)
    host_functions: set[str] = field(default_factory=set)
    # Local name -> (specialization name, kind), where kind is a scalar name or "type".
    specializations: dict[str, tuple[str, str]] = field(default_factory=dict)
    constants: dict[str, int | float | bool] = field(default_factory=dict)
    import_specs: dict[str, _ImportSpec] = field(default_factory=dict)
    symbol_imports: dict[str, tuple["_Module", str]] = field(default_factory=dict)
    module_imports: dict[str, "_Module"] = field(default_factory=dict)

    def declares_specialization(self, name: str) -> bool:
        return name in self.specializations


@dataclass(frozen=True)
class _TypeValue:
    """A bound type specialization: a scalar type (no declaration) or a declared `@struct`/`@binding_group` class."""

    identity: str
    declaration: tuple[_Module, ast.ClassDef] | None = None


@dataclass(frozen=True)
class _TypeReference:
    """A type in a compile-time expression, where it only compares with `==` and `!=` against another type."""

    identity: str


@dataclass(frozen=True)
class LoadedProject:
    source: str
    dependencies: tuple[tuple[str, str], ...]
    features: tuple[str, ...]
    dependency_files: tuple[tuple[Path, str], ...] = ()


@dataclass(frozen=True)
class ProjectEntryResolution:
    name: str
    source_path: Path
    decorators: frozenset[str]


@dataclass(frozen=True)
class _ModuleGraphSnapshot:
    modules: dict[Path, _Module]
    order: tuple[_Module, ...]
    abi_structs: dict[str, tuple[_Module, ast.ClassDef]]
    reachable_symbols: frozenset[tuple[Path, str]]
    dependencies: tuple[tuple[Path, str], ...]


class _ModuleGraphCache:
    """Caches discovery; `ModuleGraph.load` applies the specialization key.

    Discovery depends on the key only through its type assignments, whose defining modules join the graph, so the
    type assignments are part of the cache key and scalar values are not.
    """

    def __init__(self) -> None:
        self._entries: dict[tuple[Path, str | None, tuple[tuple[str, str], ...]], _ModuleGraphSnapshot] = {}
        self._parsed_modules: dict[Path, tuple[str, _Module]] = {}
        self._lock = RLock()

    @staticmethod
    def _key(graph: "ModuleGraph") -> tuple[Path, str | None, tuple[tuple[str, str], ...]]:
        return graph.input_path, graph.entry, tuple(sorted(graph.type_assignments.items()))

    def restore(self, graph: "ModuleGraph") -> bool:
        key = self._key(graph)
        with self._lock:
            snapshot = self._entries.get(key)
        if snapshot is None:
            return False
        if not all(source_file_digest(path) == digest for path, digest in snapshot.dependencies):
            with self._lock:
                if self._entries.get(key) is snapshot:
                    self._entries.pop(key)
            return False
        graph.modules = snapshot.modules.copy()
        graph.order = list(snapshot.order)
        graph._abi_structs = snapshot.abi_structs.copy()
        graph.reachable_symbols = set(snapshot.reachable_symbols)
        return True

    def store(self, graph: "ModuleGraph") -> None:
        dependencies = tuple((module.path, source_text_digest(module.source)) for module in graph.modules.values())
        snapshot = _ModuleGraphSnapshot(
            graph.modules.copy(),
            tuple(graph.order),
            graph._abi_structs.copy(),
            frozenset(graph.reachable_symbols),
            dependencies,
        )
        key = self._key(graph)
        with self._lock:
            self._entries[key] = snapshot

    def restore_parsed_module(self, path: Path) -> _Module | None:
        with self._lock:
            cached = self._parsed_modules.get(path)
        if cached is None or source_file_digest(path) != cached[0]:
            return None
        return copy.deepcopy(cached[1])

    def store_parsed_module(self, module: _Module) -> None:
        with self._lock:
            self._parsed_modules[module.path] = (source_text_digest(module.source), copy.deepcopy(module))

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._parsed_modules.clear()


_module_graph_cache = _ModuleGraphCache()


_NOT_STATIC = object()
_STATIC_BINARY_OPERATORS = {
    ast.Add: lambda left, right: left + right,
    ast.Sub: lambda left, right: left - right,
    ast.Mult: lambda left, right: left * right,
    ast.FloorDiv: lambda left, right: left // right,
    ast.Mod: lambda left, right: left % right,
}
_STATIC_COMPARISONS = {
    ast.Eq: lambda left, right: left == right,
    ast.NotEq: lambda left, right: left != right,
    ast.Lt: lambda left, right: left < right,
    ast.LtE: lambda left, right: left <= right,
    ast.Gt: lambda left, right: left > right,
    ast.GtE: lambda left, right: left >= right,
}


class _Specializer(ast.NodeTransformer):
    """Applies one specialization key to a definition before method resolution, reachability, and typing.

    A test that references a specialization is decided here, so a discarded branch contributes no callees,
    resources, or diagnostics; that only holds because this runs before every later frontend stage.
    """

    def __init__(self, graph: "ModuleGraph", module: _Module, emitted_names: dict[tuple[Path, str], str]):
        self.graph = graph
        self.module = module
        self.emitted_names = emitted_names

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        self._infer_interface_locations(node)
        kept_arguments: list[ast.arg] = []
        disabled: set[str] = set()
        for argument in node.args.args:
            conditional = self._conditional_type(argument.annotation)
            if conditional is None:
                kept_arguments.append(argument)
                continue
            condition, wrapped_type = conditional
            if self._static_condition(condition, "When condition must be a compile-time expression"):
                argument.annotation = wrapped_type
                kept_arguments.append(argument)
            else:
                disabled.add(argument.arg)
        for argument in kept_arguments:
            if argument.annotation is not None:
                argument.annotation = self.visit(argument.annotation)
        if node.returns is not None:
            node.returns = self.visit(node.returns)
        node.args.args = kept_arguments
        node.body = self._specialize_statements(node.body)
        for value in ast.walk(ast.Module(body=node.body, type_ignores=[])):
            if isinstance(value, ast.Name) and value.id in disabled:
                self.graph._error(
                    self.module, value, f"disabled value '{value.id}' is used outside its specialization guard"
                )
        return node

    def _infer_interface_locations(self, node: ast.FunctionDef) -> None:
        stages = {_decorator_name(decorator) for decorator in node.decorator_list} & _STAGE_DECORATORS
        if not stages:
            return
        vertex = "vertex" in stages
        candidates: list[tuple[ast.arg, ast.expr, int, int | None]] = []
        for argument in node.args.args:
            annotation = argument.annotation
            if annotation is None:
                continue
            interface_type = self._unwrap_when(annotation)
            if self._has_metadata(interface_type, {"builtin", "uniform", "resource"}):
                continue
            # Binding-group leaves are uniforms and resources, which never take interface locations.
            declared_class = self.graph._resolve_class(self.module, interface_type)
            if declared_class is not None and _is_binding_group(declared_class[1]):
                continue
            candidates.append(
                (
                    argument,
                    interface_type,
                    self._location_span(interface_type, vertex),
                    self._explicit_location(interface_type),
                )
            )

        occupied: set[int] = set()
        for argument, _, span, explicit in candidates:
            if explicit is None:
                continue
            annotation = argument.annotation
            assert annotation is not None
            slots = set(range(explicit, explicit + span))
            overlap = occupied & slots
            if overlap:
                self.graph._error(self.module, annotation, f"interface location overlap at {min(overlap)}")
            occupied.update(slots)

        next_location = 0
        for argument, interface_type, span, explicit in candidates:
            if explicit is not None:
                continue
            while occupied & set(range(next_location, next_location + span)):
                next_location += 1
            location = next_location
            occupied.update(range(location, location + span))
            next_location = location + span
            annotation = argument.annotation
            assert annotation is not None
            located_type = self._insert_location(interface_type, location)
            if interface_type is annotation:
                argument.annotation = located_type
            else:
                when_items = (
                    list(annotation.slice.elts)
                    if isinstance(annotation, ast.Subscript) and isinstance(annotation.slice, ast.Tuple)
                    else [annotation.slice]
                )
                when_items[1] = located_type
                assert isinstance(annotation, ast.Subscript)
                annotation.slice = ast.Tuple(elts=when_items, ctx=ast.Load())

    def _unwrap_when(self, annotation: ast.expr) -> ast.expr:
        if isinstance(annotation, ast.Subscript) and (_dotted_name(annotation.value) or "").split(".")[-1] == "When":
            items = list(annotation.slice.elts) if isinstance(annotation.slice, ast.Tuple) else [annotation.slice]
            if len(items) == 2:
                return items[1]
        return annotation

    @staticmethod
    def _annotated_items(annotation: ast.expr) -> list[ast.expr] | None:
        if (
            not isinstance(annotation, ast.Subscript)
            or (_dotted_name(annotation.value) or "").split(".")[-1] != "Annotated"
        ):
            return None
        return list(annotation.slice.elts) if isinstance(annotation.slice, ast.Tuple) else [annotation.slice]

    def _has_metadata(self, annotation: ast.expr, kinds: set[str]) -> bool:
        items = self._annotated_items(annotation)
        if items is None:
            return False
        return any(
            isinstance(item, ast.Call) and (_dotted_name(item.func) or "").split(".")[-1] in kinds for item in items[1:]
        )

    def _explicit_location(self, annotation: ast.expr) -> int | None:
        items = self._annotated_items(annotation)
        if items is None:
            return None
        for item in items[1:]:
            if not isinstance(item, ast.Call):
                continue
            kind = (_dotted_name(item.func) or "").split(".")[-1]
            if kind == "attribute":
                if item.args and isinstance(item.args[0], ast.Constant) and isinstance(item.args[0].value, int):
                    return item.args[0].value
                for keyword in item.keywords:
                    if (
                        keyword.arg == "location"
                        and isinstance(keyword.value, ast.Constant)
                        and isinstance(keyword.value.value, int)
                    ):
                        return keyword.value.value
        return None

    def _insert_location(self, annotation: ast.expr, location: int) -> ast.expr:
        items = self._annotated_items(annotation)
        location_value = ast.Constant(value=location)
        if items is None:
            return ast.copy_location(
                ast.Subscript(
                    value=ast.Name(id="Annotated", ctx=ast.Load()),
                    slice=ast.Tuple(
                        elts=[
                            copy.deepcopy(annotation),
                            ast.Call(
                                func=ast.Name(id="attribute", ctx=ast.Load()),
                                args=[],
                                keywords=[ast.keyword(arg="location", value=location_value)],
                            ),
                        ],
                        ctx=ast.Load(),
                    ),
                    ctx=ast.Load(),
                ),
                annotation,
            )
        for item in items[1:]:
            if isinstance(item, ast.Call) and (_dotted_name(item.func) or "").split(".")[-1] == "attribute":
                item.keywords.append(ast.keyword(arg="location", value=location_value))
                return annotation
        items.append(
            ast.Call(
                func=ast.Name(id="attribute", ctx=ast.Load()),
                args=[],
                keywords=[ast.keyword(arg="location", value=location_value)],
            )
        )
        annotation.slice = ast.Tuple(elts=items, ctx=ast.Load())
        return annotation

    def _location_span(self, annotation: ast.expr, vertex: bool) -> int:
        items = self._annotated_items(annotation)
        value = items[0] if items else annotation
        try:
            value_type = self.graph._parse_abi_type(self.module, value)
            return attribute_layout(value_type, self.graph._struct_abi_fields).location_span
        except ValueError as error:
            constructor_node = value.value if isinstance(value, ast.Subscript) else value
            spelling = (_dotted_name(constructor_node) or "").split(".")[-1]
            scalar = SCALAR_ALIASES.get(spelling, spelling)
            # Only vertex inputs are attributes; elsewhere a non-attribute Value takes one location, as in lowering.
            if not vertex or spelling in DEVICE_ONLY_TYPE_NAMES or scalar == "bool":
                # These types need typed, stage-specific validation after
                # feature specialization; they never expand recursively here.
                return 1
            self.graph._error(self.module, value, str(error))

    def visit_If(self, node: ast.If) -> ast.AST | list[ast.stmt]:
        if not self._references_specialization(node.test):
            return self.generic_visit(node)
        selected = node.body if self._static_condition(node.test) else node.orelse
        return self._specialize_statements(selected)

    def visit_IfExp(self, node: ast.IfExp) -> ast.expr:
        if not self._references_specialization(node.test):
            return self.generic_visit(node)
        selected = self.visit(node.body if self._static_condition(node.test) else node.orelse)
        assert isinstance(selected, ast.expr)
        return selected

    def visit_Name(self, node: ast.Name) -> ast.expr:
        declaration = self.graph._resolve_specialization(self.module, node)
        if declaration is not None:
            return self._substituted(node, *declaration)
        if isinstance(node.ctx, ast.Load) and node.id in self.module.constants:
            return ast.copy_location(ast.Constant(value=self.module.constants[node.id]), node)
        return node

    def visit_Attribute(self, node: ast.Attribute) -> ast.expr:
        declaration = self.graph._resolve_specialization(self.module, node)
        if declaration is not None:
            return self._substituted(node, *declaration)
        return self.generic_visit(node)

    def _substituted(self, node: ast.expr, name: str, kind: str) -> ast.expr:
        if kind != "type":
            return ast.copy_location(ast.Constant(value=self.graph._scalar_value(self.module, node, name, kind)), node)
        bound = self.graph._type_value(self.module, node, name)
        symbol = (
            bound.identity
            if bound.declaration is None
            else self.emitted_names[(bound.declaration[0].path, bound.declaration[1].name)]
        )
        substituted = ast.copy_location(ast.Name(id=symbol, ctx=ast.Load()), node)
        # The symbol is already the emitted name, which `_ReferenceRewriter` must not resolve again in this module.
        substituted._vernon_emitted = True  # type: ignore[attr-defined]
        return substituted

    def _specialization_value(self, node: ast.expr, name: str, kind: str) -> object:
        if kind == "type":
            return _TypeReference(self.graph._type_value(self.module, node, name).identity)
        return self.graph._scalar_value(self.module, node, name, kind)

    def _references_specialization(self, expression: ast.expr) -> bool:
        return any(
            isinstance(value, (ast.Name, ast.Attribute))
            and self.graph._resolve_specialization(self.module, value) is not None
            for value in ast.walk(expression)
        )

    def _static_condition(
        self, expression: ast.expr, message: str = "specialization condition mixes runtime values"
    ) -> bool:
        value = self._static_value(expression)
        if value is _NOT_STATIC:
            self.graph._error(self.module, expression, message)
        self._require_untyped(expression, value)
        return bool(value)

    def _require_untyped(self, node: ast.expr, *values: object) -> None:
        if any(isinstance(value, _TypeReference) for value in values):
            self.graph._error(
                self.module, node, "a type specialization compares only with == and != against a type reference"
            )

    def _static_value(self, node: ast.expr) -> object:
        if isinstance(node, ast.Constant) and isinstance(node.value, (bool, int, float)):
            return node.value
        if isinstance(node, (ast.Name, ast.Attribute)):
            declaration = self.graph._resolve_specialization(self.module, node)
            if declaration is not None:
                return self._specialization_value(node, *declaration)
            if isinstance(node, ast.Name) and node.id in self.module.constants:
                return self.module.constants[node.id]
            identity = self.graph._type_reference(self.module, node)
            return _TypeReference(identity) if identity is not None else _NOT_STATIC
        if isinstance(node, ast.UnaryOp):
            operand = self._static_value(node.operand)
            if operand is _NOT_STATIC:
                return _NOT_STATIC
            self._require_untyped(node, operand)
            if isinstance(node.op, ast.Not):
                return not operand
            if isinstance(node.op, ast.USub):
                return -operand  # type: ignore[operator]
            if isinstance(node.op, ast.UAdd):
                return operand
            return _NOT_STATIC
        if isinstance(node, ast.BoolOp):
            values = [self._static_value(value) for value in node.values]
            if any(value is _NOT_STATIC for value in values):
                return _NOT_STATIC
            self._require_untyped(node, *values)
            return all(values) if isinstance(node.op, ast.And) else any(values)
        if isinstance(node, ast.BinOp):
            operator = _STATIC_BINARY_OPERATORS.get(type(node.op))
            left, right = self._static_value(node.left), self._static_value(node.right)
            if operator is None or left is _NOT_STATIC or right is _NOT_STATIC:
                return _NOT_STATIC
            self._require_untyped(node, left, right)
            try:
                return operator(left, right)
            except ZeroDivisionError:
                self.graph._error(self.module, node, "division by zero in a compile-time expression")
        if isinstance(node, ast.Compare):
            left = self._static_value(node.left)
            if left is _NOT_STATIC:
                return _NOT_STATIC
            for operator_node, comparator in zip(node.ops, node.comparators, strict=True):
                operator = _STATIC_COMPARISONS.get(type(operator_node))
                right = self._static_value(comparator)
                if operator is None or right is _NOT_STATIC:
                    return _NOT_STATIC
                typed = isinstance(left, _TypeReference), isinstance(right, _TypeReference)
                if any(typed) and (not all(typed) or not isinstance(operator_node, (ast.Eq, ast.NotEq))):
                    self._require_untyped(node, left, right)
                if not operator(left, right):
                    return False
                left = right
            return True
        return _NOT_STATIC

    def _specialize_statements(self, statements: list[ast.stmt]) -> list[ast.stmt]:
        result: list[ast.stmt] = []
        for statement in statements:
            transformed = self.visit(statement)
            if transformed is None:
                continue
            if isinstance(transformed, list):
                result.extend(transformed)
            else:
                assert isinstance(transformed, ast.stmt)
                result.append(transformed)
        return result

    def _conditional_type(self, annotation: ast.expr | None) -> tuple[ast.expr, ast.expr] | None:
        if not isinstance(annotation, ast.Subscript) or (_dotted_name(annotation.value) or "").split(".")[-1] != "When":
            return None
        items = list(annotation.slice.elts) if isinstance(annotation.slice, ast.Tuple) else [annotation.slice]
        if len(items) != 2:
            self.graph._error(self.module, annotation, "When requires a condition and a type")
        return items[0], items[1]


class _ReferenceRewriter(ast.NodeTransformer):
    def __init__(self, module: _Module, emitted_names: dict[tuple[Path, str], str]):
        self.module = module
        self.emitted_names = emitted_names

    def _resolve(self, node: ast.expr) -> ast.expr:
        if getattr(node, "_vernon_emitted", False):
            return node
        if isinstance(node, ast.Name):
            imported = self.module.symbol_imports.get(node.id)
            if imported:
                target, symbol = imported
                return ast.copy_location(ast.Name(id=self.emitted_names[(target.path, symbol)], ctx=node.ctx), node)
            if node.id in self.module.definitions:
                return ast.copy_location(
                    ast.Name(
                        id=self.emitted_names[(self.module.path, node.id)],
                        ctx=node.ctx,
                    ),
                    node,
                )
            return node
        if isinstance(node, ast.Attribute):
            dotted = _dotted_name(node) or ""
            for alias, target in self.module.module_imports.items():
                prefix = alias + "."
                if not dotted.startswith(prefix):
                    continue
                symbol = dotted.removeprefix(prefix)
                if "." in symbol:
                    continue
                if symbol not in target.definitions:
                    self._error(node, f"module '{target.name}' has no symbol '{symbol}'")
                return ast.copy_location(ast.Name(id=self.emitted_names[(target.path, symbol)], ctx=node.ctx), node)
        return node

    def _rewrite_annotation(self, node: ast.expr | None) -> ast.expr | None:
        if node is None:
            return None

        class AnnotationRewriter(ast.NodeTransformer):
            def __init__(self, owner: _ReferenceRewriter):
                self.owner = owner

            def visit_Name(self, value: ast.Name) -> ast.expr:
                return self.owner._resolve(value)

            def visit_Attribute(self, value: ast.Attribute) -> ast.expr:
                value = self.generic_visit(value)
                assert isinstance(value, ast.Attribute)
                return self.owner._resolve(value)

        return AnnotationRewriter(self).visit(node)

    def visit_Call(self, node: ast.Call) -> ast.expr:
        node = self.generic_visit(node)
        assert isinstance(node, ast.Call)
        node.func = self._resolve(node.func)
        return node

    def visit_arg(self, node: ast.arg) -> ast.arg:
        node.annotation = self._rewrite_annotation(node.annotation)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        original_name = node.name
        node.name = self.emitted_names.get((self.module.path, original_name), original_name)
        node.args = self.visit(node.args)
        node.returns = self._rewrite_annotation(node.returns)
        node.body = [self.visit(statement) for statement in node.body]
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        original_name = node.name
        node.name = self.emitted_names[(self.module.path, original_name)]
        rewritten: list[ast.stmt] = []
        for statement in node.body:
            if isinstance(statement, ast.FunctionDef):
                statement.args = self.visit(statement.args)
                statement.returns = self._rewrite_annotation(statement.returns)
                statement.body = [self.visit(value) for value in statement.body]
                rewritten.append(statement)
            else:
                rewritten.append(self.visit(statement))
        node.body = rewritten
        return node

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AST:
        node.annotation = self._rewrite_annotation(node.annotation)
        if node.value is not None:
            node.value = self.visit(node.value)
        return node

    def _error(self, node: ast.AST, message: str) -> None:
        raise CompileError(
            message,
            SourceLocation(
                str(self.module.path),
                getattr(node, "lineno", 1),
                getattr(node, "col_offset", 0) + 1,
            ),
        )


class ModuleGraph:
    """Loads project-local DSL modules as source without executing Python."""

    def __init__(
        self,
        input_path: str | Path,
        specializations: tuple[SpecializationAssignment, ...] = (),
        entry: str | None = None,
    ):
        self.input_path = Path(input_path).resolve()
        # Type identities are module names relative to the project root, so only a real root makes them portable.
        # Without one, imports and display paths fall back to the input's directory.
        self.project_root = project_root(self.input_path.parent)
        self.search_root = self.project_root or self.input_path.parent
        self.specializations = specializations
        self.specialization_values = {assignment.name: assignment.value for assignment in specializations}
        self.type_assignments = {
            assignment.name: str(assignment.value) for assignment in specializations if assignment.type == "type"
        }
        self.entry = entry
        self.modules: dict[Path, _Module] = {}
        self.order: list[_Module] = []
        self._abi_structs: dict[str, tuple[_Module, ast.ClassDef]] = {}
        self.reachable_symbols: set[tuple[Path, str]] = set()
        self._resolving_symbols: list[tuple[Path, str]] = []

    def discover(self) -> _Module:
        root = self._load_module(self.input_path, self._module_name(self.input_path))
        if self.entry is not None:
            seeds = (self.entry,)
        else:
            seeds = tuple(
                name
                for name, definition in root.definitions.items()
                if (
                    isinstance(definition, ast.FunctionDef)
                    and any(_decorator_name(item) in _FUNCTION_DECORATORS for item in definition.decorator_list)
                )
                or _is_dsl_class(definition)
            )
        for seed in (*seeds, *root.specializations):
            self._touch_symbol(root, seed)
        # A type value's defining module joins the graph as if imported, so its source is a dependency of the key.
        for identity in self.type_assignments.values():
            declaration = self._declared_type(root, root.tree, identity).declaration
            if declaration is not None:
                self._touch_symbol(declaration[0], declaration[1].name)
        return root

    def declared_specializations(self) -> dict[str, str]:
        """Maps every reachable specialization name to its kind; requires `discover`."""
        declared: dict[str, str] = {}
        for module in self.modules.values():
            for local, (name, kind) in module.specializations.items():
                if (module.path, local) not in self.reachable_symbols:
                    continue
                previous = declared.setdefault(name, kind)
                if previous != kind:
                    self._error(module, module.tree, f"specialization '{name}' is declared as {previous} and {kind}")
        return declared

    def _selected_entry_symbol(self, root: _Module) -> tuple[_Module, str] | None:
        if self.entry is None:
            return None
        if self.entry in root.definitions:
            return root, self.entry
        imported = root.symbol_imports.get(self.entry)
        if imported is None:
            imported = self._materialize_import(root, self.entry)
        if imported is None or imported[1] is None:
            return None
        return imported[0], imported[1]

    def load(self) -> LoadedProject:
        root = self.discover()
        declared = self.declared_specializations()
        declared_features = {name for name, kind in declared.items() if kind == "bool"}
        unknown = sorted(assignment.name for assignment in self.specializations if assignment.name not in declared)
        if unknown:
            self._error(root, root.tree, "requested undeclared specialization(s): " + ", ".join(unknown))
        for assignment in self.specializations:
            if declared[assignment.name] != assignment.type:
                self._error(
                    root,
                    root.tree,
                    f"specialization '{assignment.name}' is declared as {declared[assignment.name]}, "
                    f"not {assignment.type}",
                )
        emitted_names = self._emitted_names(root)
        selected_entry_symbol = self._selected_entry_symbol(root)
        selected_entry_key = (
            (selected_entry_symbol[0].path, selected_entry_symbol[1]) if selected_entry_symbol is not None else None
        )
        if selected_entry_symbol is not None and self.entry is not None:
            emitted_names[(selected_entry_symbol[0].path, selected_entry_symbol[1])] = self.entry

        body: list[ast.stmt] = []
        function_sources: dict[str, tuple[_Module, ast.FunctionDef]] = {}
        entry_names: set[str] = set()
        host_function_names: set[str] = set()
        for module in self.order:
            rewriter = _ReferenceRewriter(module, emitted_names)
            for original_name, definition in module.definitions.items():
                if (module.path, original_name) not in self.reachable_symbols:
                    continue
                if isinstance(definition, ast.FunctionDef) and original_name in module.host_functions:
                    emitted_name = emitted_names[(module.path, original_name)]
                    function_sources[emitted_name] = (module, definition)
                    host_function_names.add(emitted_name)
                    continue
                if isinstance(definition, ast.ClassDef) and not _is_dsl_class(definition):
                    continue
                transformed = copy.deepcopy(definition)
                transformed = _Specializer(self, module, emitted_names).visit(transformed)
                assert isinstance(transformed, (ast.FunctionDef, ast.ClassDef))
                if isinstance(transformed, ast.FunctionDef):
                    is_entry = any(
                        _decorator_name(decorator) in _STAGE_DECORATORS for decorator in transformed.decorator_list
                    )
                    is_selected_entry = selected_entry_key == (module.path, original_name)
                    if is_entry:
                        entry_names.add(emitted_names[(module.path, original_name)])
                    if module is not root and not is_selected_entry:
                        # Imported entries are retained as private functions.
                        # The call-graph check still remembers their original
                        # stage identity and rejects calls to them.
                        transformed.decorator_list = [ast.Name(id="func", ctx=ast.Load())]
                transformed = rewriter.visit(transformed)
                assert isinstance(transformed, (ast.FunctionDef, ast.ClassDef))
                if isinstance(transformed, ast.FunctionDef):
                    function_sources[transformed.name] = (module, definition)
                body.append(transformed)

        # Analysis runs on hoisted methods so pruning and call-graph checks see calls hidden behind receiver syntax.
        # The emitted source keeps methods inside their classes: the frontend hoists them again and needs the
        # method tables to resolve calls on receivers whose type is known only after inference.
        analysis = normalize_struct_methods(ast.Module(body=copy.deepcopy(body), type_ignores=[]), str(root.path))
        analysis_body = analysis.body
        for statement in analysis_body:
            if isinstance(statement, ast.FunctionDef):
                function_sources.setdefault(statement.name, (root, statement))
        if self.entry is not None:
            reachable = self._reachable_functions(analysis_body, self.entry, root, method_tables(analysis))
            analysis_body = [
                statement
                for statement in analysis_body
                if not isinstance(statement, ast.FunctionDef) or statement.name in reachable
            ]
            body = self._prune_emitted(body, reachable)
        self._validate_call_graph(analysis_body, function_sources, entry_names, host_function_names)
        combined = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
        dependency_records = tuple(
            (
                module.path,
                self._display_path(module.path),
                source_text_digest(module.source),
            )
            for module in sorted(self.modules.values(), key=lambda value: self._display_path(value.path))
        )
        return LoadedProject(
            ast.unparse(combined) + "\n",
            tuple((display_path, digest) for _, display_path, digest in dependency_records),
            tuple(sorted(declared_features)),
            tuple((path, digest) for path, _, digest in dependency_records),
        )

    def _emitted_names(self, root: _Module) -> dict[tuple[Path, str], str]:
        """Names definitions in the combined source: imported helpers carry their module prefix, while a Struct keeps
        its declared name from any module, because host Values, reflection, and manifests name a Struct by its class
        name. Two reachable Structs with one name therefore conflict."""

        emitted_names: dict[tuple[Path, str], str] = {}
        owners: dict[str, Path] = {}
        for module in self.order:
            prefix = "" if module is root else self._symbol_prefix(module)
            for symbol, definition in module.definitions.items():
                emitted = symbol if not prefix or _is_struct_declaration(definition) else f"{prefix}{symbol}"
                emitted_names[(module.path, symbol)] = emitted
                if (module.path, symbol) not in self.reachable_symbols:
                    continue
                owner = owners.setdefault(emitted, module.path)
                if owner != module.path:
                    self._error(
                        module,
                        definition,
                        f"conflicting declarations named '{emitted}' in {self._display_path(owner)} and "
                        f"{self._display_path(module.path)}; a Struct keeps its declared name across modules",
                    )
        return emitted_names

    def _load_module(self, path: Path, name: str) -> _Module:
        path = path.resolve()
        if path in self.modules:
            return self.modules[path]
        cached = _module_graph_cache.restore_parsed_module(path)
        if cached is not None:
            cached.name = name
            self.modules[path] = cached
            self.order.append(cached)
            return cached
        try:
            source = path.read_text(encoding="utf-8")
        except OSError as error:
            raise CompileError(str(error), SourceLocation(str(path), 1, 1)) from None
        try:
            tree = ast.parse(source, filename=str(path), type_comments=False)
        except SyntaxError as error:
            raise CompileError(
                error.msg,
                SourceLocation(str(path), error.lineno or 1, error.offset or 1),
            ) from None

        module = _Module(path, name, source, tree)
        self.modules[path] = module
        self._collect_definitions(module)
        self._index_imports(module)
        self.order.append(module)
        _module_graph_cache.store_parsed_module(module)
        return module

    def _collect_definitions(self, module: _Module) -> None:
        for node in module.tree.body:
            if isinstance(node, ast.Assign):
                self._collect_specialization(module, node)
                if (
                    len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, (int, float, bool))
                ):
                    module.constants[node.targets[0].id] = node.value.value
                continue
            if not isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                continue
            if node.name in module.definitions or module.declares_specialization(node.name):
                self._error(module, node, f"duplicate DSL symbol '{node.name}'")
            if isinstance(node, ast.FunctionDef):
                decorators = [_decorator_name(decorator) for decorator in node.decorator_list]
                dsl_decorators = [decorator for decorator in decorators if decorator in _FUNCTION_DECORATORS]
                if not dsl_decorators:
                    module.host_functions.add(node.name)
                elif len(decorators) != 1:
                    self._error(
                        module,
                        node,
                        "DSL functions require exactly one of @func, @vertex, @fragment, or @kernel",
                    )
            module.definitions[node.name] = node

    def _collect_specialization(self, module: _Module, node: ast.Assign) -> None:
        if not isinstance(node.value, ast.Call):
            return
        declaration = (_dotted_name(node.value.func) or "").split(".")[-1]
        if declaration not in {"feature", "specialization"}:
            return
        expected_arguments = 1 if declaration == "feature" else 2
        if (
            len(node.targets) != 1
            or not isinstance(node.targets[0], ast.Name)
            or len(node.value.args) != expected_arguments
            or node.value.keywords
            or not isinstance(node.value.args[0], ast.Constant)
            or not isinstance(node.value.args[0].value, str)
        ):
            self._error(
                module,
                node,
                'specialization declaration must be NAME = feature("NAME") or NAME = specialization("NAME", TYPE)',
            )
        kind = "bool" if declaration == "feature" else (_dotted_name(node.value.args[1]) or "").split(".")[-1]
        if kind not in _SPECIALIZATION_KINDS:
            self._error(module, node, "specialization type must be bool, i32, u32, f32, f64, or vd.type")
        local_name = node.targets[0].id
        specialization_name = node.value.args[0].value
        if not specialization_name or local_name in module.definitions or module.declares_specialization(local_name):
            self._error(module, node, f"invalid or duplicate specialization declaration '{local_name}'")
        module.specializations[local_name] = (specialization_name, kind)

    def _index_imports(self, module: _Module) -> None:
        for node in module.tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top_level = alias.name.split(".", 1)[0]
                    if top_level in _HOST_MODULES:
                        continue
                    local_name = alias.asname or alias.name
                    self._add_import_spec(
                        module,
                        _ImportSpec(node, alias.name, 0, local_name),
                    )
            elif isinstance(node, ast.ImportFrom):
                imported_module = node.module or ""
                if node.level == 0 and imported_module.split(".", 1)[0] in _HOST_MODULES:
                    continue
                for alias in node.names:
                    if alias.name == "*":
                        self._error(module, alias, "project-local DSL imports must name symbols explicitly")
                    local_name = alias.asname or alias.name
                    self._add_import_spec(
                        module,
                        _ImportSpec(alias, imported_module, node.level, local_name, alias.name),
                    )

    def _add_import_spec(self, module: _Module, spec: _ImportSpec) -> None:
        if (
            spec.local_name in module.definitions
            or module.declares_specialization(spec.local_name)
            or spec.local_name in module.import_specs
        ):
            self._error(module, spec.node, f"duplicate imported symbol '{spec.local_name}'")
        module.import_specs[spec.local_name] = spec

    def _materialize_import(self, module: _Module, local_name: str) -> tuple[_Module, str | None] | None:
        imported_symbol = module.symbol_imports.get(local_name)
        if imported_symbol is not None:
            return imported_symbol
        imported_module = module.module_imports.get(local_name)
        if imported_module is not None:
            return imported_module, None
        spec = module.import_specs.get(local_name)
        if spec is None:
            return None
        target_path = self._find_local_path(module, spec.module_name, spec.level)
        if target_path is None:
            if spec.level:
                spelling = "." * spec.level + spec.module_name
                self._error(module, spec.node, f"cannot resolve project-local DSL import '{spelling}'")
            return None
        target_name = spec.module_name or target_path.stem
        target = self._load_module(target_path, target_name)
        if spec.symbol_name is None:
            self._add_module_import(module, local_name, target, spec.node)
            return target, None
        if spec.symbol_name not in target.definitions and not target.declares_specialization(spec.symbol_name):
            self._error(module, spec.node, f"module '{target.name}' has no symbol '{spec.symbol_name}'")
        self._add_symbol_import(module, local_name, target, spec.symbol_name, spec.node)
        return target, spec.symbol_name

    def _touch_symbol(self, module: _Module, name: str) -> None:
        key = (module.path, name)
        if key in self._resolving_symbols:
            cycle_start = self._resolving_symbols.index(key)
            cycle_keys = self._resolving_symbols[cycle_start:] + [key]
            cycle_paths = [path for path, _ in cycle_keys]
            if len(set(cycle_paths)) > 1:
                cycle = " -> ".join(self._display_path(path) for path in cycle_paths)
                self._error(module, module.tree, f"DSL import cycle: {cycle}")
            return
        if key in self.reachable_symbols:
            return
        self.reachable_symbols.add(key)
        self._resolving_symbols.append(key)
        try:
            definition = module.definitions.get(name)
            if definition is not None:
                if not (isinstance(definition, ast.FunctionDef) and name in module.host_functions) and not (
                    isinstance(definition, ast.ClassDef) and not _is_dsl_class(definition)
                ):
                    self._follow_references(module, definition)
                return
            if module.declares_specialization(name) or name in module.constants:
                return
            imported = self._materialize_import(module, name)
            if imported is not None and imported[1] is not None:
                self._touch_symbol(imported[0], imported[1])
        finally:
            self._resolving_symbols.pop()

    def _follow_references(self, module: _Module, node: ast.AST) -> None:
        for value in ast.walk(node):
            if isinstance(value, ast.Attribute):
                dotted = _dotted_name(value) or ""
                root, separator, remainder = dotted.partition(".")
                if separator and root in module.import_specs:
                    imported = self._materialize_import(module, root)
                    if imported is not None and imported[1] is None:
                        symbol = remainder.split(".", 1)[0]
                        if symbol in imported[0].definitions or imported[0].declares_specialization(symbol):
                            self._touch_symbol(imported[0], symbol)
            elif isinstance(value, ast.Name) and isinstance(value.ctx, ast.Load):
                if (
                    value.id in module.definitions
                    or module.declares_specialization(value.id)
                    or value.id in module.constants
                    or value.id in module.import_specs
                ):
                    self._touch_symbol(module, value.id)

    def _find_local_path(self, module: _Module, dotted: str, level: int) -> Path | None:
        parts = tuple(part for part in dotted.split(".") if part)
        bases: list[Path] = []
        if level:
            base = module.path.parent
            for _ in range(level - 1):
                base = base.parent
            bases.append(base)
        else:
            bases.extend((module.path.parent, self.search_root))
        for base in dict.fromkeys(bases):
            candidate = base.joinpath(*parts)
            paths = (candidate.with_suffix(".py"), candidate / "__init__.py")
            for path in paths:
                if path.is_file():
                    return path.resolve()
        return None

    def _add_symbol_import(self, module: _Module, local_name: str, target: _Module, symbol: str, node: ast.AST) -> None:
        if (
            local_name in module.definitions
            or module.declares_specialization(local_name)
            or local_name in module.symbol_imports
            or local_name in module.module_imports
        ):
            self._error(module, node, f"duplicate imported symbol '{local_name}'")
        module.symbol_imports[local_name] = (target, symbol)

    def _add_module_import(self, module: _Module, local_name: str, target: _Module, node: ast.AST) -> None:
        if (
            local_name in module.definitions
            or module.declares_specialization(local_name)
            or local_name in module.symbol_imports
            or local_name in module.module_imports
        ):
            self._error(module, node, f"duplicate imported symbol '{local_name}'")
        module.module_imports[local_name] = target

    def _resolve_specialization(self, module: _Module, node: ast.AST) -> tuple[str, str] | None:
        """Returns the (name, kind) of the specialization `node` names, if any."""

        if isinstance(node, ast.Name):
            local = module.specializations.get(node.id)
            if local is not None:
                return local
            imported = module.symbol_imports.get(node.id)
            if imported:
                target, symbol = imported
                return target.specializations.get(symbol)
            return None
        if isinstance(node, ast.Attribute):
            dotted = _dotted_name(node) or ""
            for alias, target in module.module_imports.items():
                prefix = alias + "."
                if dotted.startswith(prefix):
                    return target.specializations.get(dotted.removeprefix(prefix))
        return None

    def _scalar_value(self, module: _Module, node: ast.AST, name: str, kind: str) -> bool | int | float:
        if kind == "bool":
            return self.specialization_values.get(name, False) is True
        value = self.specialization_values.get(name)
        if value is None:
            self._error(module, node, f"missing specialization '{name}'")
        assert isinstance(value, (int, float))
        return value

    def _type_value(self, module: _Module, node: ast.AST, name: str) -> _TypeValue:
        identity = self.type_assignments.get(name)
        if identity is None:
            self._error(module, node, f"missing type specialization '{name}'")
        return self._declared_type(module, node, identity)

    def _bound_type(self, module: _Module, node: ast.AST) -> _TypeValue | None:
        """The type value bound to the type specialization `node` names, if it names one."""

        declaration = self._resolve_specialization(module, node)
        if declaration is None or declaration[1] != "type":
            return None
        return self._type_value(module, node, declaration[0])

    def _declared_type(self, module: _Module, node: ast.AST, identity: str) -> _TypeValue:
        """Resolves a canonical type identity: a scalar name, or `<module>:<Class>` loaded from its defining source."""

        if identity in SCALAR_TYPES:
            return _TypeValue(identity)
        module_name, separator, class_name = identity.partition(":")
        source = (
            project_module_path(module_name, self.project_root) if separator and self.project_root is not None else None
        )
        if source is None or not source.is_file():
            self._error(module, node, f"type value has no portable identity: '{identity}' names no source module")
        owner = self._load_module(source, module_name)
        declaration = owner.definitions.get(class_name)
        if not isinstance(declaration, ast.ClassDef) or not _is_dsl_class(declaration):
            self._error(
                module, node, f"type specialization requires a declared type or scalar type; '{identity}' is not one"
            )
        return _TypeValue(identity, (owner, declaration))

    def _materialize_reference(self, module: _Module, node: ast.AST) -> str:
        """Materializes the import a dotted reference starts with; returns the spelling after that first name."""

        head, _, member = (_dotted_name(node) or "").partition(".")
        if head in module.import_specs:
            self._materialize_import(module, head)
        return member

    def _type_reference(self, module: _Module, node: ast.AST) -> str | None:
        """The identity of the scalar type or declared class `node` names, materializing imports on the way."""

        spelling = _dotted_name(node) or ""
        if spelling.split(".")[-1] in SCALAR_TYPES:
            return spelling.split(".")[-1]
        member = self._materialize_reference(module, node)
        resolved = self._resolve_class(module, node) if "." not in member else None
        if resolved is None or not _is_dsl_class(resolved[1]):
            return None
        owner, declaration = resolved
        module_name = project_module_name(owner.path, self.project_root) if self.project_root is not None else None
        if module_name is None:
            self._error(
                module,
                node,
                f"type value has no portable identity: '{declaration.name}' is not defined under a project root",
            )
        return f"{module_name}:{declaration.name}"

    def _parse_abi_type(self, module: _Module, node: ast.AST) -> ConcreteType:
        if isinstance(node, ast.Subscript) and (_dotted_name(node.value) or "").split(".")[-1] == "Annotated":
            items = list(node.slice.elts) if isinstance(node.slice, ast.Tuple) else [node.slice]
            if not items:
                raise ValueError("Annotated interface type is empty")
            return self._parse_abi_type(module, items[0])

        spelling = (_dotted_name(node) or "").split(".")[-1]
        scalar = SCALAR_ALIASES.get(spelling, spelling)
        if scalar in SCALAR_TYPES:
            return ConcreteType("scalar", scalar)
        bound = self._bound_type(module, node)
        if bound is not None and bound.declaration is None:
            return ConcreteType("scalar", bound.identity)
        structure = self._resolve_struct_type(module, node)
        if structure is not None:
            owner, declaration = structure
            identity = f"{owner.path.as_posix()}::{declaration.name}"
            self._abi_structs[identity] = structure
            return ConcreteType("struct", identity)
        if not isinstance(node, ast.Subscript):
            raise ValueError(f"'{spelling or ast.dump(node)}' is not an ABI-stable interface Value")

        constructor = (_dotted_name(node.value) or "").split(".")[-1]
        arguments = list(node.slice.elts) if isinstance(node.slice, ast.Tuple) else [node.slice]
        if constructor == "Tuple":
            if not arguments:
                raise ValueError("Tuple interface Value requires at least one element")
            return ConcreteType("tuple", "Tuple", tuple(self._parse_abi_type(module, item) for item in arguments))
        if constructor not in {"Tensor", "Vector", "Matrix"}:
            raise ValueError(f"'{constructor}' is not an ABI-stable interface Value")
        rank = {"Vector": 1, "Matrix": 2}.get(constructor)
        if not arguments:
            raise ValueError(f"{constructor} requires an element type")
        element = self._parse_abi_type(module, arguments[0])
        if constructor == "Tensor" and len(arguments) == 2 and isinstance(arguments[1], ast.Tuple):
            dimensions = list(arguments[1].elts)
        else:
            dimensions = arguments[1:]
        if rank is not None and len(dimensions) != rank:
            raise ValueError(f"{constructor} requires {rank} static dimension(s)")
        shape: list[int] = []
        for dimension in dimensions:
            value = self._constant_int(module, dimension)
            if value is None or value <= 0:
                raise ValueError(f"{constructor} interface dimensions must be positive integer literals")
            shape.append(value)
        return ConcreteType("tensor", "Tensor", (element, *shape))

    def _resolve_struct_type(self, module: _Module, node: ast.AST) -> tuple[_Module, ast.ClassDef] | None:
        resolved = self._resolve_class(module, node)
        if resolved is None or not _is_struct_declaration(resolved[1]):
            return None
        return resolved

    def _resolve_class(self, module: _Module, node: ast.AST) -> tuple[_Module, ast.ClassDef] | None:
        bound = self._bound_type(module, node)
        if bound is not None:
            return bound.declaration
        if isinstance(node, ast.Name):
            local = module.definitions.get(node.id)
            if isinstance(local, ast.ClassDef):
                return module, local
            imported = module.symbol_imports.get(node.id)
            if imported is not None:
                owner, name = imported
                declaration = owner.definitions.get(name)
                if isinstance(declaration, ast.ClassDef):
                    return owner, declaration
            return None
        if isinstance(node, ast.Attribute):
            dotted = _dotted_name(node) or ""
            for alias, owner in module.module_imports.items():
                prefix = alias + "."
                if not dotted.startswith(prefix):
                    continue
                name = dotted.removeprefix(prefix)
                declaration = owner.definitions.get(name)
                if isinstance(declaration, ast.ClassDef):
                    return owner, declaration
        return None

    def _struct_abi_fields(self, identity: str) -> tuple[tuple[str, ConcreteType], ...]:
        resolved = self._abi_structs.get(identity)
        if resolved is None:
            raise ValueError(f"unresolved Struct '{identity}'")
        module, declaration = resolved
        fields: list[tuple[str, ConcreteType]] = []
        for statement in declaration.body:
            if not isinstance(statement, ast.AnnAssign) or not isinstance(statement.target, ast.Name):
                continue
            fields.append((statement.target.id, self._parse_abi_type(module, statement.annotation)))
        return tuple(fields)

    @staticmethod
    def _constant_int(module: _Module, node: ast.AST) -> int | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.Name):
            value = module.constants.get(node.id)
            return value if isinstance(value, int) and not isinstance(value, bool) else None
        return None

    def _reachable_functions(
        self, body: list[ast.stmt], entry: str, root: _Module, methods: dict[str, dict[str, str]]
    ) -> set[str]:
        functions = {statement.name: statement for statement in body if isinstance(statement, ast.FunctionDef)}
        selected = functions.get(entry)
        if selected is None or not any(
            _decorator_name(decorator) in _STAGE_DECORATORS for decorator in selected.decorator_list
        ):
            self._error(root, root.tree, f"unknown shader entry point '{entry}'")
        # A method call left on its receiver resolves only after inference, so it keeps every same-named method.
        helpers_by_method: dict[str, set[str]] = {}
        for table in methods.values():
            for method, helper in table.items():
                helpers_by_method.setdefault(method, set()).add(helper)
        reachable = {entry}
        pending = [entry]
        while pending:
            current = pending.pop()
            for node in ast.walk(functions[current]):
                if not isinstance(node, ast.Call):
                    continue
                if isinstance(node.func, ast.Name):
                    callees = {node.func.id}
                elif isinstance(node.func, ast.Attribute):
                    callees = helpers_by_method.get(node.func.attr, set())
                else:
                    continue
                for callee in callees:
                    if callee in functions and callee not in reachable:
                        reachable.add(callee)
                        pending.append(callee)
        return reachable

    @staticmethod
    def _prune_emitted(body: list[ast.stmt], reachable: set[str]) -> list[ast.stmt]:
        pruned: list[ast.stmt] = []
        for statement in body:
            if isinstance(statement, ast.FunctionDef):
                if statement.name in reachable:
                    pruned.append(statement)
            elif _is_dsl_class(statement):
                assert isinstance(statement, ast.ClassDef)
                statement.body = [
                    member
                    for member in statement.body
                    if not isinstance(member, ast.FunctionDef) or f"{statement.name}__{member.name}" in reachable
                ] or [ast.Pass()]
                pruned.append(statement)
        return pruned

    def _validate_call_graph(
        self,
        body: list[ast.stmt],
        function_sources: dict[str, tuple[_Module, ast.FunctionDef]],
        entry_names: set[str],
        host_function_names: set[str],
    ) -> None:
        calls: dict[str, set[str]] = {name: set() for name in function_sources}
        call_nodes: dict[tuple[str, str], ast.Call] = {}
        function_nodes = {statement.name: statement for statement in body if isinstance(statement, ast.FunctionDef)}
        shared_functions = {name for name, node in function_nodes.items() if self._is_shared_function(node)}
        for statement in body:
            if not isinstance(statement, ast.FunctionDef):
                continue
            if statement.name in shared_functions:
                self._validate_shared_surface(function_sources[statement.name][0], statement)
            for node in ast.walk(statement):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in calls:
                    if node.func.id in host_function_names:
                        module, _ = function_sources[statement.name]
                        self._error(module, node, f"shader helper '{node.func.id}' requires @func")
                    calls[statement.name].add(node.func.id)
                    call_nodes[(statement.name, node.func.id)] = node
                    if statement.name in shared_functions and node.func.id not in shared_functions:
                        module, _ = function_sources[statement.name]
                        self._error(
                            module,
                            node,
                            f"shared function '{statement.name}' cannot call device-only function '{node.func.id}'",
                        )
                    if node.func.id in entry_names:
                        module, _ = function_sources[statement.name]
                        self._error(module, node, f"entry function '{node.func.id}' cannot be called")

        states: dict[str, int] = {}
        stack: list[str] = []

        def visit(name: str) -> None:
            states[name] = 1
            stack.append(name)
            for callee in sorted(calls[name]):
                if states.get(callee, 0) == 0:
                    visit(callee)
                elif states.get(callee) == 1:
                    cycle = stack[stack.index(callee) :] + [callee]
                    module, _ = function_sources[name]
                    self._error(
                        module,
                        call_nodes[(name, callee)],
                        f"recursive DSL call graph: {' -> '.join(cycle)}",
                    )
            stack.pop()
            states[name] = 2

        for name in sorted(calls):
            if states.get(name, 0) == 0:
                visit(name)

    @staticmethod
    def _is_shared_function(node: ast.FunctionDef) -> bool:
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call) or _decorator_name(decorator) != "func":
                continue
            return any(
                keyword.arg == "shared" and isinstance(keyword.value, ast.Constant) and keyword.value.value is True
                for keyword in decorator.keywords
            )
        return False

    def _validate_shared_surface(self, module: _Module, node: ast.FunctionDef) -> None:
        device_only_names = DEVICE_ONLY_TYPE_NAMES | DEVICE_ONLY_OPERATION_NAMES
        annotations: list[ast.expr] = [
            argument.annotation for argument in node.args.args if argument.annotation is not None
        ]
        if node.returns is not None:
            annotations.append(node.returns)
        for annotation in annotations:
            for value in ast.walk(annotation):
                name = (_dotted_name(value) or "").split(".")[-1]
                if name in device_only_names:
                    self._error(
                        module, value, f"shared function '{node.name}' uses device-only type or annotation '{name}'"
                    )
        for value in ast.walk(node):
            if not isinstance(value, ast.Call):
                continue
            name = (_dotted_name(value.func) or "").split(".")[-1]
            if name in DEVICE_ONLY_OPERATION_NAMES:
                self._error(module, value, f"shared function '{node.name}' uses device-only operation '{name}'")

    def _symbol_prefix(self, module: _Module) -> str:
        relative = self._display_path(module.path).removesuffix(".py")
        normalized = re.sub(r"[^A-Za-z0-9_]", "_", relative)
        return f"__vernon_{normalized}__"

    def _display_path(self, path: Path) -> str:
        try:
            return path.relative_to(self.search_root).as_posix()
        except ValueError:
            return path.as_posix()

    def _module_name(self, path: Path) -> str:
        return self._display_path(path).removesuffix(".py").replace("/", ".")

    @staticmethod
    def _error(module: _Module, node: ast.AST, message: str) -> NoReturn:
        raise CompileError(
            message,
            SourceLocation(
                str(module.path),
                getattr(node, "lineno", 1),
                getattr(node, "col_offset", 0) + 1,
            ),
        )


def _load_project_graph(
    input_path: str | Path, specializations: tuple[SpecializationAssignment, ...], entry: str | None
) -> tuple[ModuleGraph, _Module]:
    graph = ModuleGraph(input_path, specializations, entry)
    if _module_graph_cache.restore(graph):
        return graph, graph.modules[graph.input_path]
    root = graph.discover()
    _module_graph_cache.store(graph)
    return graph, root


def load_project(
    input_path: str | Path,
    specializations: tuple[SpecializationAssignment, ...] = (),
    entry: str | None = None,
) -> LoadedProject:
    graph, _ = _load_project_graph(input_path, specializations, entry)
    return graph.load()


def _private_project_root(input_path: str | Path) -> tuple[ModuleGraph, _Module]:
    """A graph that only `input_path` has joined; resolving through it materializes imports, so it never touches
    cached discovery state."""

    graph = ModuleGraph(input_path)
    return graph, graph._load_module(graph.input_path, graph._module_name(graph.input_path))


def project_specialization(input_path: str | Path, expression: ast.expr) -> tuple[str, str] | None:
    """The (name, kind) of the specialization a module-level expression of `input_path` names, local or imported."""

    graph, root = _private_project_root(input_path)
    graph._materialize_reference(root, expression)
    return graph._resolve_specialization(root, expression)


def project_type_identity(input_path: str | Path, expression: ast.expr) -> str | None:
    """The canonical identity of the type a module-level expression of `input_path` names, read without running it."""

    graph, root = _private_project_root(input_path)
    return graph._type_reference(root, expression)


def resolve_project_entry(input_path: str | Path, entry: str) -> ProjectEntryResolution | None:
    graph, root = _load_project_graph(input_path, (), entry)
    selected = graph._selected_entry_symbol(root)
    if selected is None:
        return None
    definition = selected[0].definitions.get(selected[1])
    if not isinstance(definition, ast.FunctionDef):
        return None
    decorators = frozenset(_decorator_name(decorator) for decorator in definition.decorator_list)
    return ProjectEntryResolution(entry, selected[0].path, decorators)


def clear_project_cache() -> None:
    _module_graph_cache.clear()
