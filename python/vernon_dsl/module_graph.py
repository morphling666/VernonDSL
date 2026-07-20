from __future__ import annotations

import ast
import copy
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

from .diagnostics import CompileError, SourceLocation

_HOST_MODULES = {"__future__", "typing", "vernon_dsl"}
_STAGE_DECORATORS = {"vertex", "fragment", "compute", "kernel"}


def _dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else None
    return None


def _decorator_name(node: ast.expr) -> str:
    target = node.func if isinstance(node, ast.Call) else node
    return (_dotted_name(target) or "").split(".")[-1]


@dataclass
class _Module:
    path: Path
    name: str
    source: str
    tree: ast.Module
    definitions: dict[str, ast.FunctionDef
                      | ast.ClassDef] = field(default_factory=dict)
    features: dict[str, str] = field(default_factory=dict)
    constants: dict[str, int | float | bool] = field(default_factory=dict)
    symbol_imports: dict[str, tuple["_Module",
                                    str]] = field(default_factory=dict)
    module_imports: dict[str, "_Module"] = field(default_factory=dict)


@dataclass(frozen=True)
class LoadedProject:
    source: str
    dependencies: tuple[tuple[str, str], ...]
    features: tuple[str, ...]


class _FeatureSpecializer(ast.NodeTransformer):

    def __init__(self, graph: "ModuleGraph", module: _Module,
                 enabled_features: frozenset[str]):
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
                self.graph._error(
                    self.module, value,
                    f"disabled value '{value.id}' is used outside its feature guard"
                )
        return node

    def _infer_interface_locations(self, node: ast.FunctionDef) -> None:
        if not any(
                _decorator_name(decorator) in _STAGE_DECORATORS
                for decorator in node.decorator_list):
            return
        occupied: set[int] = set()
        next_location = 0
        for argument in node.args.args:
            annotation = argument.annotation
            if annotation is None:
                continue
            interface_type = self._unwrap_when(annotation)
            if self._has_metadata(interface_type,
                                  {"builtin", "uniform", "resource"}):
                continue
            span = self._location_span(interface_type)
            explicit = self._explicit_location(interface_type)
            location = explicit if explicit is not None else next_location
            slots = set(range(location, location + span))
            overlap = occupied & slots
            if overlap:
                self.graph._error(
                    self.module, annotation,
                    f"interface location overlap at {min(overlap)}")
            occupied.update(slots)
            next_location = max(next_location, location + span)
            if explicit is None:
                located_type = self._insert_location(interface_type, location)
                if interface_type is annotation:
                    argument.annotation = located_type
                else:
                    when_items = list(annotation.slice.elts) if isinstance(
                        annotation, ast.Subscript) and isinstance(
                            annotation.slice, ast.Tuple) else [
                                annotation.slice
                            ]
                    when_items[1] = located_type
                    assert isinstance(annotation, ast.Subscript)
                    annotation.slice = ast.Tuple(elts=when_items,
                                                 ctx=ast.Load())

    def _unwrap_when(self, annotation: ast.expr) -> ast.expr:
        if isinstance(annotation, ast.Subscript) and (_dotted_name(
                annotation.value) or "").split(".")[-1] == "When":
            items = list(annotation.slice.elts) if isinstance(
                annotation.slice, ast.Tuple) else [annotation.slice]
            if len(items) == 2:
                return items[1]
        return annotation

    @staticmethod
    def _annotated_items(annotation: ast.expr) -> list[ast.expr] | None:
        if not isinstance(annotation, ast.Subscript) or (_dotted_name(
                annotation.value) or "").split(".")[-1] != "Annotated":
            return None
        return list(annotation.slice.elts) if isinstance(
            annotation.slice, ast.Tuple) else [annotation.slice]

    def _has_metadata(self, annotation: ast.expr, kinds: set[str]) -> bool:
        items = self._annotated_items(annotation)
        if items is None:
            return False
        return any(
            isinstance(item, ast.Call) and (
                _dotted_name(item.func) or "").split(".")[-1] in kinds
            for item in items[1:])

    def _explicit_location(self, annotation: ast.expr) -> int | None:
        items = self._annotated_items(annotation)
        if items is None:
            return None
        for item in items[1:]:
            if not isinstance(item, ast.Call):
                continue
            kind = (_dotted_name(item.func) or "").split(".")[-1]
            if kind == "location" and len(item.args) == 1 and isinstance(
                    item.args[0], ast.Constant) and isinstance(
                        item.args[0].value, int):
                return item.args[0].value
            if kind == "instance":
                if item.args and isinstance(item.args[0],
                                            ast.Constant) and isinstance(
                                                item.args[0].value, int):
                    return item.args[0].value
                for keyword in item.keywords:
                    if keyword.arg == "location" and isinstance(
                            keyword.value, ast.Constant) and isinstance(
                                keyword.value.value, int):
                        return keyword.value.value
        return None

    def _insert_location(self, annotation: ast.expr,
                         location: int) -> ast.expr:
        items = self._annotated_items(annotation)
        location_value = ast.Constant(value=location)
        if items is None:
            return ast.copy_location(
                ast.Subscript(value=ast.Name(id="Annotated", ctx=ast.Load()),
                              slice=ast.Tuple(elts=[
                                  copy.deepcopy(annotation),
                                  ast.Call(func=ast.Name(id="location",
                                                         ctx=ast.Load()),
                                           args=[location_value],
                                           keywords=[])
                              ],
                                              ctx=ast.Load()),
                              ctx=ast.Load()), annotation)
        for item in items[1:]:
            if isinstance(item, ast.Call) and (_dotted_name(
                    item.func) or "").split(".")[-1] == "instance":
                item.keywords.append(
                    ast.keyword(arg="location", value=location_value))
                return annotation
        items.append(
            ast.Call(func=ast.Name(id="location", ctx=ast.Load()),
                     args=[location_value],
                     keywords=[]))
        annotation.slice = ast.Tuple(elts=items, ctx=ast.Load())
        return annotation

    def _location_span(self, annotation: ast.expr) -> int:
        items = self._annotated_items(annotation)
        value = items[0] if items else annotation
        if not isinstance(value, ast.Subscript):
            return 1
        constructor = (_dotted_name(value.value) or "").split(".")[-1]
        arguments = list(value.slice.elts) if isinstance(
            value.slice, ast.Tuple) else [value.slice]
        if constructor in {"mat2", "mat3", "mat4"}:
            return int(constructor[-1])
        if constructor == "mat" and len(arguments) >= 2 and isinstance(
                arguments[1], ast.Constant) and isinstance(
                    arguments[1].value, int):
            return arguments[1].value
        if constructor == "Tensor" and len(arguments) >= 2:
            shape = arguments[1].elts if isinstance(
                arguments[1], ast.Tuple) else arguments[1:]
            if len(shape) == 2 and isinstance(shape[1],
                                              ast.Constant) and isinstance(
                                                  shape[1].value, int):
                return shape[1].value
        if constructor == "Array" and len(arguments) == 2 and isinstance(
                arguments[1], ast.Constant) and isinstance(
                    arguments[1].value, int):
            return self._location_span(arguments[0]) * arguments[1].value
        return 1

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
            return ast.copy_location(
                ast.Constant(value=feature_name in self.enabled_features),
                node)
        if isinstance(node.ctx, ast.Load) and node.id in self.module.constants:
            return ast.copy_location(
                ast.Constant(value=self.module.constants[node.id]), node)
        return node

    def visit_Attribute(self, node: ast.Attribute) -> ast.expr:
        feature_name = self.graph._resolve_feature_name(self.module, node)
        if feature_name is not None:
            return ast.copy_location(
                ast.Constant(value=feature_name in self.enabled_features),
                node)
        return self.generic_visit(node)

    def _specialize_statements(self,
                               statements: list[ast.stmt]) -> list[ast.stmt]:
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

    def _conditional_type(
            self, annotation: ast.expr | None) -> tuple[str, ast.expr] | None:
        if not isinstance(annotation, ast.Subscript) or (_dotted_name(
                annotation.value) or "").split(".")[-1] != "When":
            return None
        items = list(annotation.slice.elts) if isinstance(
            annotation.slice, ast.Tuple) else [annotation.slice]
        if len(items) != 2:
            self.graph._error(self.module, annotation,
                              "When requires a feature and a type")
        feature_name = self.graph._resolve_feature_name(self.module, items[0])
        if feature_name is None:
            self.graph._error(self.module, items[0],
                              "When condition must be a declared feature")
        return feature_name, items[1]

    def _feature_condition(self,
                           expression: ast.expr) -> tuple[str, bool] | None:
        positive = True
        value = expression
        if isinstance(expression, ast.UnaryOp) and isinstance(
                expression.op, ast.Not):
            positive = False
            value = expression.operand
        feature_name = self.graph._resolve_feature_name(self.module, value)
        return (feature_name, positive) if feature_name is not None else None


class _ReferenceRewriter(ast.NodeTransformer):

    def __init__(self, module: _Module, emitted_names: dict[tuple[Path, str],
                                                            str]):
        self.module = module
        self.emitted_names = emitted_names

    def _resolve(self, node: ast.expr) -> ast.expr:
        if isinstance(node, ast.Name):
            imported = self.module.symbol_imports.get(node.id)
            if imported:
                target, symbol = imported
                return ast.copy_location(
                    ast.Name(id=self.emitted_names[(target.path, symbol)],
                             ctx=node.ctx), node)
            if node.id in self.module.definitions:
                return ast.copy_location(
                    ast.Name(
                        id=self.emitted_names[(self.module.path, node.id)],
                        ctx=node.ctx,
                    ), node)
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
                    self._error(
                        node,
                        f"module '{target.name}' has no symbol '{symbol}'")
                return ast.copy_location(
                    ast.Name(id=self.emitted_names[(target.path, symbol)],
                             ctx=node.ctx), node)
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
        node.name = self.emitted_names[(self.module.path, original_name)]
        node.args = self.visit(node.args)
        node.returns = self._rewrite_annotation(node.returns)
        node.body = [self.visit(statement) for statement in node.body]
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        original_name = node.name
        node.name = self.emitted_names[(self.module.path, original_name)]
        node.body = [self.visit(statement) for statement in node.body]
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

    def __init__(self,
                 input_path: str | Path,
                 enabled_features: tuple[str, ...] = (),
                 entry: str | None = None):
        self.input_path = Path(input_path).resolve()
        self.project_root = self._project_root(self.input_path.parent)
        self.enabled_features = frozenset(enabled_features)
        self.entry = entry
        self.modules: dict[Path, _Module] = {}
        self.order: list[_Module] = []
        self.loading: list[Path] = []

    def load(self) -> LoadedProject:
        root = self._load_module(self.input_path,
                                 self._module_name(self.input_path))
        declared_features = {
            feature
            for module in self.modules.values()
            for feature in module.features.values()
        }
        unknown_features = self.enabled_features - declared_features
        if unknown_features:
            self._error(
                root, root.tree, "requested undeclared feature(s): " +
                ", ".join(sorted(unknown_features)))
        emitted_names: dict[tuple[Path, str], str] = {}
        for module in self.order:
            prefix = "" if module is root else self._symbol_prefix(module)
            for symbol in module.definitions:
                emitted_names[(
                    module.path,
                    symbol)] = f"{prefix}{symbol}" if prefix else symbol

        body: list[ast.stmt] = []
        function_sources: dict[str, tuple[_Module, ast.FunctionDef]] = {}
        entry_names: set[str] = set()
        for module in self.order:
            rewriter = _ReferenceRewriter(module, emitted_names)
            for original_name, definition in module.definitions.items():
                transformed = copy.deepcopy(definition)
                transformed = _FeatureSpecializer(
                    self, module, self.enabled_features).visit(transformed)
                assert isinstance(transformed, (ast.FunctionDef, ast.ClassDef))
                if isinstance(transformed, ast.FunctionDef):
                    if module is not root:
                        transformed.decorator_list = []
                    elif any(
                            _decorator_name(decorator) in _STAGE_DECORATORS
                            for decorator in transformed.decorator_list):
                        entry_names.add(emitted_names[(module.path,
                                                       original_name)])
                transformed = rewriter.visit(transformed)
                assert isinstance(transformed, (ast.FunctionDef, ast.ClassDef))
                body.append(transformed)
                if isinstance(transformed, ast.FunctionDef):
                    function_sources[transformed.name] = (module, definition)

        if self.entry is not None:
            body = self._prune_to_entry(body, self.entry, root)
        self._validate_call_graph(body, function_sources, entry_names)
        combined = ast.fix_missing_locations(
            ast.Module(body=body, type_ignores=[]))
        dependencies = tuple(
            (self._display_path(module.path),
             hashlib.sha256(module.source.encode("utf-8")).hexdigest())
            for module in sorted(
                self.modules.values(),
                key=lambda value: self._display_path(value.path)))
        return LoadedProject(
            ast.unparse(combined) + "\n", dependencies,
            tuple(sorted(declared_features)))

    def _load_module(self, path: Path, name: str) -> _Module:
        path = path.resolve()
        if path in self.loading:
            cycle_paths = self.loading[self.loading.index(path):] + [path]
            cycle = " -> ".join(
                self._display_path(item) for item in cycle_paths)
            raise CompileError(
                f"DSL import cycle: {cycle}",
                SourceLocation(str(path), 1, 1),
            )
        if path in self.modules:
            return self.modules[path]
        try:
            source = path.read_text(encoding="utf-8")
        except OSError as error:
            raise CompileError(str(error), SourceLocation(str(path), 1,
                                                          1)) from None
        try:
            tree = ast.parse(source, filename=str(path), type_comments=False)
        except SyntaxError as error:
            raise CompileError(
                error.msg,
                SourceLocation(str(path), error.lineno or 1, error.offset
                               or 1),
            ) from None

        module = _Module(path, name, source, tree)
        self.modules[path] = module
        self.loading.append(path)
        self._collect_definitions(module)
        self._resolve_imports(module)
        self.loading.pop()
        self.order.append(module)
        return module

    def _collect_definitions(self, module: _Module) -> None:
        for node in module.tree.body:
            if isinstance(node, ast.Assign):
                self._collect_feature(module, node)
                if (len(node.targets) == 1
                        and isinstance(node.targets[0], ast.Name)
                        and isinstance(node.value, ast.Constant)
                        and isinstance(node.value.value, (int, float, bool))):
                    module.constants[node.targets[0].id] = node.value.value
                continue
            if not isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                continue
            if node.name in module.definitions or node.name in module.features:
                self._error(module, node,
                            f"duplicate DSL symbol '{node.name}'")
            module.definitions[node.name] = node

    def _collect_feature(self, module: _Module, node: ast.Assign) -> None:
        if not isinstance(node.value, ast.Call) or (_dotted_name(
                node.value.func) or "").split(".")[-1] != "feature":
            return
        if len(
                node.targets
        ) != 1 or not isinstance(node.targets[0], ast.Name) or len(
                node.value.args) != 1 or node.value.keywords or not isinstance(
                    node.value.args[0], ast.Constant) or not isinstance(
                        node.value.args[0].value, str):
            self._error(
                module, node,
                "feature declaration must be NAME = feature(\"NAME\")")
        local_name = node.targets[0].id
        feature_name = node.value.args[0].value
        if not feature_name or local_name in module.definitions or local_name in module.features:
            self._error(
                module, node,
                f"invalid or duplicate feature declaration '{local_name}'")
        module.features[local_name] = feature_name

    def _resolve_imports(self, module: _Module) -> None:
        for node in module.tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top_level = alias.name.split(".", 1)[0]
                    if top_level in _HOST_MODULES:
                        continue
                    target_path = self._resolve_path(module, alias.name, 0,
                                                     node)
                    target = self._load_module(target_path, alias.name)
                    local_name = alias.asname or alias.name
                    self._add_module_import(module, local_name, target, node)
            elif isinstance(node, ast.ImportFrom):
                imported_module = node.module or ""
                if node.level == 0 and imported_module.split(
                        ".", 1)[0] in _HOST_MODULES:
                    continue
                target_path = self._resolve_path(module, imported_module,
                                                 node.level, node)
                target_name = imported_module or target_path.stem
                target = self._load_module(target_path, target_name)
                for alias in node.names:
                    if alias.name == "*":
                        self._error(
                            module, alias,
                            "project-local DSL imports must name symbols explicitly"
                        )
                    if alias.name not in target.definitions and alias.name not in target.features:
                        self._error(
                            module, alias,
                            f"module '{target.name}' has no symbol '{alias.name}'"
                        )
                    local_name = alias.asname or alias.name
                    self._add_symbol_import(module, local_name, target,
                                            alias.name, alias)

    def _resolve_path(self, module: _Module, dotted: str, level: int,
                      node: ast.AST) -> Path:
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
        spelling = "." * level + dotted
        self._error(module, node,
                    f"cannot resolve project-local DSL import '{spelling}'")
        raise AssertionError("unreachable")

    def _add_symbol_import(self, module: _Module, local_name: str,
                           target: _Module, symbol: str,
                           node: ast.AST) -> None:
        if local_name in module.definitions or local_name in module.features or local_name in module.symbol_imports or local_name in module.module_imports:
            self._error(module, node,
                        f"duplicate imported symbol '{local_name}'")
        module.symbol_imports[local_name] = (target, symbol)

    def _add_module_import(self, module: _Module, local_name: str,
                           target: _Module, node: ast.AST) -> None:
        if local_name in module.definitions or local_name in module.features or local_name in module.symbol_imports or local_name in module.module_imports:
            self._error(module, node,
                        f"duplicate imported symbol '{local_name}'")
        module.module_imports[local_name] = target

    def _resolve_feature_name(self, module: _Module,
                              node: ast.AST) -> str | None:
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

    def _prune_to_entry(self, body: list[ast.stmt], entry: str,
                        root: _Module) -> list[ast.stmt]:
        functions = {
            statement.name: statement
            for statement in body if isinstance(statement, ast.FunctionDef)
        }
        selected = functions.get(entry)
        if selected is None or not any(
                _decorator_name(decorator) in _STAGE_DECORATORS
                for decorator in selected.decorator_list):
            self._error(root, root.tree,
                        f"unknown shader entry point '{entry}'")
        reachable = {entry}
        pending = [entry]
        while pending:
            current = pending.pop()
            for node in ast.walk(functions[current]):
                if not isinstance(node, ast.Call) or not isinstance(
                        node.func, ast.Name):
                    continue
                callee = node.func.id
                if callee in functions and callee not in reachable:
                    reachable.add(callee)
                    pending.append(callee)
        return [
            statement for statement in body
            if not isinstance(statement, ast.FunctionDef)
            or statement.name in reachable
        ]

    def _validate_call_graph(
        self,
        body: list[ast.stmt],
        function_sources: dict[str, tuple[_Module, ast.FunctionDef]],
        entry_names: set[str],
    ) -> None:
        calls: dict[str, set[str]] = {name: set() for name in function_sources}
        call_nodes: dict[tuple[str, str], ast.Call] = {}
        for statement in body:
            if not isinstance(statement, ast.FunctionDef):
                continue
            for node in ast.walk(statement):
                if isinstance(node, ast.Call) and isinstance(
                        node.func, ast.Name) and node.func.id in calls:
                    calls[statement.name].add(node.func.id)
                    call_nodes[(statement.name, node.func.id)] = node
                    if node.func.id in entry_names:
                        module, _ = function_sources[statement.name]
                        self._error(
                            module, node,
                            f"entry function '{node.func.id}' cannot be called"
                        )

        states: dict[str, int] = {}
        stack: list[str] = []

        def visit(name: str) -> None:
            states[name] = 1
            stack.append(name)
            for callee in sorted(calls[name]):
                if states.get(callee, 0) == 0:
                    visit(callee)
                elif states.get(callee) == 1:
                    cycle = stack[stack.index(callee):] + [callee]
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


def load_project(input_path: str | Path,
                 enabled_features: tuple[str, ...] = (),
                 entry: str | None = None) -> LoadedProject:
    return ModuleGraph(input_path, enabled_features, entry).load()
