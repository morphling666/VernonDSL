from __future__ import annotations

import ast
import copy
import re
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock

from .diagnostics import CompileError, SourceLocation
from .frontend.abi import attribute_layout
from .frontend.model import ConcreteType
from .language.ast_utils import decorator_name as _decorator_name
from .language.ast_utils import dotted_name as _dotted_name
from .language.scalar_types import SCALAR_ALIASES, SCALAR_TYPES
from .language.syntax import ENTRY_DECORATORS, FUNCTION_DECORATORS
from .shader_contracts import DEVICE_ONLY_OPERATION_NAMES, DEVICE_ONLY_TYPE_NAMES
from .source_identity import source_file_digest, source_text_digest
from .struct_methods import normalize_struct_methods

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
_STAGE_DECORATORS = ENTRY_DECORATORS
_FUNCTION_DECORATORS = FUNCTION_DECORATORS


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
    features: dict[str, str] = field(default_factory=dict)
    constants: dict[str, int | float | bool] = field(default_factory=dict)
    import_specs: dict[str, _ImportSpec] = field(default_factory=dict)
    symbol_imports: dict[str, tuple["_Module", str]] = field(default_factory=dict)
    module_imports: dict[str, "_Module"] = field(default_factory=dict)


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
    def __init__(self) -> None:
        self._entries: dict[tuple[Path, str | None, frozenset[str]], _ModuleGraphSnapshot] = {}
        self._parsed_modules: dict[Path, tuple[str, _Module]] = {}
        self._lock = RLock()

    def restore(self, graph: "ModuleGraph") -> bool:
        key = (graph.input_path, graph.entry, graph.enabled_features)
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
        key = (graph.input_path, graph.entry, graph.enabled_features)
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


class _FeatureSpecializer(ast.NodeTransformer):
    def __init__(self, graph: "ModuleGraph", module: _Module, enabled_features: frozenset[str]):
        self.graph = graph
        self.module = module
        self.enabled_features = enabled_features
        self.disabled_arguments: set[str] = set()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        self._infer_interface_locations(node)
        kept_arguments: list[ast.arg] = []
        disabled: set[str] = set()
        for argument in node.args.args:
            conditional = self._conditional_type(argument.annotation)
            if conditional is None:
                kept_arguments.append(argument)
                continue
            feature_name, wrapped_type = conditional
            if feature_name in self.enabled_features:
                argument.annotation = wrapped_type
                kept_arguments.append(argument)
            else:
                disabled.add(argument.arg)
        node.args.args = kept_arguments
        node.body = self._specialize_statements(node.body)
        for value in ast.walk(ast.Module(body=node.body, type_ignores=[])):
            if isinstance(value, ast.Name) and value.id in disabled:
                self.graph._error(self.module, value, f"disabled value '{value.id}' is used outside its feature guard")
        return node

    def _infer_interface_locations(self, node: ast.FunctionDef) -> None:
        if not any(_decorator_name(decorator) in _STAGE_DECORATORS for decorator in node.decorator_list):
            return
        candidates: list[tuple[ast.arg, ast.expr, int, int | None]] = []
        for argument in node.args.args:
            annotation = argument.annotation
            if annotation is None:
                continue
            interface_type = self._unwrap_when(annotation)
            if self._has_metadata(interface_type, {"builtin", "uniform", "resource"}):
                continue
            candidates.append(
                (
                    argument,
                    interface_type,
                    self._location_span(interface_type),
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

    def _location_span(self, annotation: ast.expr) -> int:
        items = self._annotated_items(annotation)
        value = items[0] if items else annotation
        try:
            value_type = self.graph._parse_abi_type(self.module, value)
            return attribute_layout(value_type, self.graph._struct_abi_fields).location_span
        except ValueError as error:
            constructor_node = value.value if isinstance(value, ast.Subscript) else value
            spelling = (_dotted_name(constructor_node) or "").split(".")[-1]
            scalar = SCALAR_ALIASES.get(spelling, spelling)
            if spelling in DEVICE_ONLY_TYPE_NAMES or scalar == "bool":
                # These types need typed, stage-specific validation after
                # feature specialization; they never expand recursively here.
                return 1
            self.graph._error(self.module, value, str(error))
            raise AssertionError("unreachable") from None

    def visit_If(self, node: ast.If) -> ast.AST | list[ast.stmt]:
        condition = self._feature_condition(node.test)
        if condition is None:
            return self.generic_visit(node)
        feature_name, positive = condition
        enabled = feature_name in self.enabled_features
        selected = node.body if enabled == positive else node.orelse
        return self._specialize_statements(selected)

    def visit_Name(self, node: ast.Name) -> ast.expr:
        feature_name = self.graph._resolve_feature_name(self.module, node)
        if feature_name is not None:
            return ast.copy_location(ast.Constant(value=feature_name in self.enabled_features), node)
        if isinstance(node.ctx, ast.Load) and node.id in self.module.constants:
            return ast.copy_location(ast.Constant(value=self.module.constants[node.id]), node)
        return node

    def visit_Attribute(self, node: ast.Attribute) -> ast.expr:
        feature_name = self.graph._resolve_feature_name(self.module, node)
        if feature_name is not None:
            return ast.copy_location(ast.Constant(value=feature_name in self.enabled_features), node)
        return self.generic_visit(node)

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

    def _conditional_type(self, annotation: ast.expr | None) -> tuple[str, ast.expr] | None:
        if not isinstance(annotation, ast.Subscript) or (_dotted_name(annotation.value) or "").split(".")[-1] != "When":
            return None
        items = list(annotation.slice.elts) if isinstance(annotation.slice, ast.Tuple) else [annotation.slice]
        if len(items) != 2:
            self.graph._error(self.module, annotation, "When requires a feature and a type")
        feature_name = self.graph._resolve_feature_name(self.module, items[0])
        if feature_name is None:
            self.graph._error(self.module, items[0], "When condition must be a declared feature")
        return feature_name, items[1]

    def _feature_condition(self, expression: ast.expr) -> tuple[str, bool] | None:
        positive = True
        value = expression
        if isinstance(expression, ast.UnaryOp) and isinstance(expression.op, ast.Not):
            positive = False
            value = expression.operand
        feature_name = self.graph._resolve_feature_name(self.module, value)
        return (feature_name, positive) if feature_name is not None else None


class _ReferenceRewriter(ast.NodeTransformer):
    def __init__(self, module: _Module, emitted_names: dict[tuple[Path, str], str]):
        self.module = module
        self.emitted_names = emitted_names

    def _resolve(self, node: ast.expr) -> ast.expr:
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

    def __init__(self, input_path: str | Path, enabled_features: tuple[str, ...] = (), entry: str | None = None):
        self.input_path = Path(input_path).resolve()
        self.project_root = self._project_root(self.input_path.parent)
        self.enabled_features = frozenset(enabled_features)
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
                or (
                    isinstance(definition, ast.ClassDef)
                    and any(_decorator_name(item) == "struct" for item in definition.decorator_list)
                )
            )
        for seed in (*seeds, *root.features):
            self._touch_symbol(root, seed)
        return root

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
        declared_features = {
            feature
            for module in self.modules.values()
            for name, feature in module.features.items()
            if (module.path, name) in self.reachable_symbols
        }
        unknown_features = self.enabled_features - declared_features
        if unknown_features:
            self._error(root, root.tree, "requested undeclared feature(s): " + ", ".join(sorted(unknown_features)))
        emitted_names: dict[tuple[Path, str], str] = {}
        for module in self.order:
            prefix = "" if module is root else self._symbol_prefix(module)
            for symbol in module.definitions:
                emitted_names[(module.path, symbol)] = f"{prefix}{symbol}" if prefix else symbol
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
                if isinstance(definition, ast.ClassDef) and not any(
                    _decorator_name(decorator) == "struct" for decorator in definition.decorator_list
                ):
                    continue
                transformed = copy.deepcopy(definition)
                transformed = _FeatureSpecializer(self, module, self.enabled_features).visit(transformed)
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

        normalized = normalize_struct_methods(ast.Module(body=body, type_ignores=[]), str(root.path))
        body = normalized.body
        for statement in body:
            if isinstance(statement, ast.FunctionDef):
                function_sources.setdefault(statement.name, (root, statement))
        # Method normalization exposes calls hidden behind receiver syntax.
        # Entry pruning must see those calls or it can remove helpers that are
        # reachable only through a struct method.
        if self.entry is not None:
            body = self._prune_to_entry(body, self.entry, root)
        self._validate_call_graph(body, function_sources, entry_names, host_function_names)
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
                self._collect_feature(module, node)
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
            if node.name in module.definitions or node.name in module.features:
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

    def _collect_feature(self, module: _Module, node: ast.Assign) -> None:
        if not isinstance(node.value, ast.Call) or (_dotted_name(node.value.func) or "").split(".")[-1] != "feature":
            return
        if (
            len(node.targets) != 1
            or not isinstance(node.targets[0], ast.Name)
            or len(node.value.args) != 1
            or node.value.keywords
            or not isinstance(node.value.args[0], ast.Constant)
            or not isinstance(node.value.args[0].value, str)
        ):
            self._error(module, node, 'feature declaration must be NAME = feature("NAME")')
        local_name = node.targets[0].id
        feature_name = node.value.args[0].value
        if not feature_name or local_name in module.definitions or local_name in module.features:
            self._error(module, node, f"invalid or duplicate feature declaration '{local_name}'")
        module.features[local_name] = feature_name

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
            or spec.local_name in module.features
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
        if spec.symbol_name not in target.definitions and spec.symbol_name not in target.features:
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
                    isinstance(definition, ast.ClassDef)
                    and not any(_decorator_name(item) == "struct" for item in definition.decorator_list)
                ):
                    self._follow_references(module, definition)
                return
            if name in module.features or name in module.constants:
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
                        if symbol in imported[0].definitions or symbol in imported[0].features:
                            self._touch_symbol(imported[0], symbol)
            elif isinstance(value, ast.Name) and isinstance(value.ctx, ast.Load):
                if (
                    value.id in module.definitions
                    or value.id in module.features
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
            bases.extend((module.path.parent, self.project_root))
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
            or local_name in module.features
            or local_name in module.symbol_imports
            or local_name in module.module_imports
        ):
            self._error(module, node, f"duplicate imported symbol '{local_name}'")
        module.symbol_imports[local_name] = (target, symbol)

    def _add_module_import(self, module: _Module, local_name: str, target: _Module, node: ast.AST) -> None:
        if (
            local_name in module.definitions
            or local_name in module.features
            or local_name in module.symbol_imports
            or local_name in module.module_imports
        ):
            self._error(module, node, f"duplicate imported symbol '{local_name}'")
        module.module_imports[local_name] = target

    def _resolve_feature_name(self, module: _Module, node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            if node.id in module.features:
                return module.features[node.id]
            imported = module.symbol_imports.get(node.id)
            if imported:
                target, symbol = imported
                return target.features.get(symbol)
            return None
        if isinstance(node, ast.Attribute):
            dotted = _dotted_name(node) or ""
            for alias, target in module.module_imports.items():
                prefix = alias + "."
                if dotted.startswith(prefix):
                    symbol = dotted.removeprefix(prefix)
                    return target.features.get(symbol)
        return None

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
        if not dimensions:
            raise ValueError(f"{constructor} interface Value requires a positive static shape")
        shape: list[int] = []
        for dimension in dimensions:
            value = self._constant_int(module, dimension)
            if value is None or value <= 0:
                raise ValueError(f"{constructor} interface dimensions must be positive integer literals")
            shape.append(value)
        return ConcreteType("tensor", "Tensor", (element, *shape))

    def _resolve_struct_type(self, module: _Module, node: ast.AST) -> tuple[_Module, ast.ClassDef] | None:
        if isinstance(node, ast.Name):
            local = module.definitions.get(node.id)
            if isinstance(local, ast.ClassDef) and any(
                _decorator_name(item) == "struct" for item in local.decorator_list
            ):
                return module, local
            imported = module.symbol_imports.get(node.id)
            if imported is not None:
                owner, name = imported
                declaration = owner.definitions.get(name)
                if isinstance(declaration, ast.ClassDef) and any(
                    _decorator_name(item) == "struct" for item in declaration.decorator_list
                ):
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
                if isinstance(declaration, ast.ClassDef) and any(
                    _decorator_name(item) == "struct" for item in declaration.decorator_list
                ):
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

    def _prune_to_entry(self, body: list[ast.stmt], entry: str, root: _Module) -> list[ast.stmt]:
        functions = {statement.name: statement for statement in body if isinstance(statement, ast.FunctionDef)}
        selected = functions.get(entry)
        if selected is None or not any(
            _decorator_name(decorator) in _STAGE_DECORATORS for decorator in selected.decorator_list
        ):
            self._error(root, root.tree, f"unknown shader entry point '{entry}'")
        reachable = {entry}
        pending = [entry]
        while pending:
            current = pending.pop()
            for node in ast.walk(functions[current]):
                if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                    continue
                callee = node.func.id
                if callee in functions and callee not in reachable:
                    reachable.add(callee)
                    pending.append(callee)
        return [
            statement
            for statement in body
            if (
                (isinstance(statement, ast.FunctionDef) and statement.name in reachable)
                or (
                    isinstance(statement, ast.ClassDef)
                    and any(_decorator_name(decorator) == "struct" for decorator in statement.decorator_list)
                )
            )
        ]

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
            return path.relative_to(self.project_root).as_posix()
        except ValueError:
            return path.as_posix()

    def _module_name(self, path: Path) -> str:
        return self._display_path(path).removesuffix(".py").replace("/", ".")

    @staticmethod
    def _project_root(start: Path) -> Path:
        for candidate in (start, *start.parents):
            if (candidate / "pyproject.toml").is_file():
                return candidate.resolve()
        return start.resolve()

    @staticmethod
    def _error(module: _Module, node: ast.AST, message: str) -> None:
        raise CompileError(
            message,
            SourceLocation(
                str(module.path),
                getattr(node, "lineno", 1),
                getattr(node, "col_offset", 0) + 1,
            ),
        )


def _load_project_graph(
    input_path: str | Path, enabled_features: tuple[str, ...], entry: str | None
) -> tuple[ModuleGraph, _Module]:
    graph = ModuleGraph(input_path, enabled_features, entry)
    if _module_graph_cache.restore(graph):
        return graph, graph.modules[graph.input_path]
    root = graph.discover()
    _module_graph_cache.store(graph)
    return graph, root


def load_project(
    input_path: str | Path, enabled_features: tuple[str, ...] = (), entry: str | None = None
) -> LoadedProject:
    graph, _ = _load_project_graph(input_path, enabled_features, entry)
    return graph.load()


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
