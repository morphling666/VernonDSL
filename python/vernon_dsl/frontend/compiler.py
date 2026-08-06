from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable

from ..ad import ProgramTransformSpec
from ..diagnostics import CompileError, SourceLocation
from ..language.ast_utils import dotted_name
from ..language.stage_registry import ENTRY_DECORATORS, GRAPHICS_STAGES, STAGE_BY_DECORATOR
from ..module_graph import clear_project_cache, load_project
from ..struct_methods import normalize_struct_methods
from .autodiff import AutodiffProgram, build_program_graph
from .autodiff_profiles import AutodiffProfilePlan, build_autodiff_profile_plan
from .cache import frontend_cache
from .emission import emit_mlir_module
from .lowering import _FunctionEmitter
from .lowering_types import DslType, FunctionSignature, ModuleContext
from .model import ConcreteType, StorageEffect, StorageOwnerKind, StorageRegionKind, is_abi_stable_value
from .monomorphize import infer_and_monomorphize_helpers
from .request import FrontendCompileRequest, FrontendCompileResult
from .specialization import specialize_frontend_source
from .type_parser import AnnotatedType, TypeParser


class Compiler:
    """Compiles a restricted Python source string without importing or executing it."""

    def compile(
        self,
        source: str,
        filename: str = "<string>",
        dependencies: tuple[tuple[str, str], ...] = (),
        declared_features: tuple[str, ...] = (),
        enabled_features: tuple[str, ...] = (),
        runtime_entry: str | None = None,
        runtime_workgroup_size: tuple[int, int, int] | None = None,
        program_transform: ProgramTransformSpec | None = None,
    ) -> str:
        """Compile an already loaded module without entry-specialization caching."""
        try:
            module = ast.parse(source, filename=filename, type_comments=False)
        except SyntaxError as error:
            raise CompileError(
                error.msg,
                SourceLocation(filename, error.lineno or 1, error.offset or 1),
            ) from None
        module = normalize_struct_methods(module, filename)
        context = ModuleContext(filename, runtime_entry=runtime_entry)
        self._collect_struct_names(module, context)
        type_parser = TypeParser(context)
        self._collect_structs(module, context, type_parser)
        self._structs = tuple(
            (
                name,
                tuple((field_name, annotation.type) for field_name, annotation in fields),
            )
            for name, fields in sorted(context.structs.items())
        )
        self._validate_entry_annotations(module, context)
        module = infer_and_monomorphize_helpers(
            module,
            lambda annotation: type_parser.parse(annotation).type,
            lambda annotation: type_parser.parse(annotation).metadata,
            context.error,
            tuple(sorted(enabled_features)),
        )
        self._helper_specializations = tuple(getattr(module, "_vernon_helper_specializations", ()))
        self._typed_functions = tuple(getattr(module, "_vernon_typed_functions", ()))
        self._entry_workgroup_size: tuple[int, int, int] | None = None
        context.typed_functions = {function.symbol: function for function in self._typed_functions}
        self._collect_signatures(module, context, type_parser)
        self._program_graph: AutodiffProgram | None = None
        self._autodiff_profiles: AutodiffProfilePlan | None = None
        if program_transform is not None:
            self._validate_program_transform(program_transform, runtime_entry, context)
            assert runtime_entry is not None
            typed_entry = context.typed_functions[runtime_entry]
            entry_node = next(
                node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == runtime_entry
            )
            entry_stage, declared_workgroup_size = self._decorator(entry_node, context)
            if entry_stage != "compute":
                raise context.error(entry_node, "autodiff currently requires a compute entry")
            struct_fields = {
                name: tuple((field_name, annotation.type) for field_name, annotation in fields)
                for name, fields in context.structs.items()
            }
            self._program_graph = build_program_graph(
                typed_entry,
                context.typed_functions,
                program_transform.wrt,
                struct_fields,
                program_transform.derivative_rules_version,
                runtime_workgroup_size or declared_workgroup_size,
                context.error,
            )
            self._autodiff_profiles = build_autodiff_profile_plan(
                program_transform,
                self._program_graph,
            )

        def emit_function(node: ast.FunctionDef) -> list[str]:
            stage, workgroup_size = self._decorator(node, context)
            if node.name == runtime_entry and runtime_workgroup_size is not None:
                if stage != "compute":
                    raise context.error(node, "runtime workgroup size applies only to compute entries")
                workgroup_size = runtime_workgroup_size
            if node.name == runtime_entry:
                self._entry_workgroup_size = workgroup_size
            annotations = [type_parser.parse(argument.annotation) for argument in node.args.args]
            return _FunctionEmitter(
                context,
                context.signatures[node.name],
                annotations,
                context.result_annotations[node.name],
                context.typed_functions[node.name],
                stage,
                workgroup_size,
            ).emit()

        return emit_mlir_module(
            module,
            dependencies=dependencies,
            declared_features=declared_features,
            enabled_features=enabled_features,
            structs=context.structs,
            emit_function=emit_function,
            error=context.error,
            program_transform=program_transform,
            program_graph=self._program_graph,
            autodiff_profiles=self._autodiff_profiles,
        )

    def compile_file(
        self,
        input_path: str | Path,
        *,
        features: Iterable[str] = (),
        entry: str | None = None,
        program_transform: ProgramTransformSpec | None = None,
    ) -> str:
        """Compile one entry with caching, or an uncached whole module when entry is omitted."""
        path = Path(input_path)
        enabled_features = tuple(sorted(set(features)))
        if entry is None:
            if program_transform is not None:
                raise ValueError("program transforms require one specialized entry")
            project = load_project(path, enabled_features, entry)
            return self.compile(project.source, str(path), project.dependencies, project.features, enabled_features)
        return self.compile_request(
            FrontendCompileRequest(path, entry, enabled_features, program_transform=program_transform)
        ).mlir

    def compile_request(self, request: FrontendCompileRequest) -> FrontendCompileResult:
        cached = frontend_cache.get(request)
        if cached is not None:
            self._helper_specializations = cached.helper_specializations
            self._typed_functions = cached.typed_functions
            self._program_graph = cached.program_graph
            self._autodiff_profiles = cached.autodiff_profiles
            self._entry_workgroup_size = cached.entry_workgroup_size
            return cached

        project = load_project(request.source_path, request.enabled_features, request.entry)
        specialized_source = specialize_frontend_source(project.source, request)
        mlir = self.compile(
            specialized_source,
            filename=str(request.source_path),
            dependencies=project.dependencies,
            declared_features=project.features,
            enabled_features=request.enabled_features,
            runtime_entry=request.entry,
            runtime_workgroup_size=request.workgroup_size,
            program_transform=request.program_transform,
        )
        result = FrontendCompileResult(
            mlir,
            specialized_source,
            project.dependencies,
            project.features,
            request,
            self._helper_specializations,
            self._typed_functions,
            self._program_graph,
            self._autodiff_profiles,
            self._entry_workgroup_size,
            self._structs,
        )
        frontend_cache.put(request, result, project.dependency_files)
        return result

    @staticmethod
    def clear_cache() -> None:
        frontend_cache.clear()
        clear_project_cache()

    @staticmethod
    def _validate_program_transform(
        transform: ProgramTransformSpec,
        runtime_entry: str | None,
        context: ModuleContext,
    ) -> None:
        if runtime_entry is None:
            raise ValueError("program transforms require one specialized entry")
        function = context.typed_functions.get(runtime_entry)
        if function is None:
            raise ValueError(f"program transform entry '{runtime_entry}' has no typed function")
        unsupported_effects = [
            effect
            for effect in function.effects
            if not isinstance(effect, StorageEffect)
            or effect.owner.kind is not StorageOwnerKind.PARAMETER
            or effect.region.kind not in {StorageRegionKind.ELEMENT, StorageRegionKind.UNKNOWN}
        ]
        if unsupported_effects:
            raise context.error(
                function.source,
                "Storage VJP requires static effects or analyzable parallel TensorView parameter effects",
            )
        for parameter in function.parameters:
            if parameter.type.kind != "tensor_view":
                continue
            shape = parameter.type.arguments[1]
            if not isinstance(shape, tuple) or any(not isinstance(extent, int) or extent <= 0 for extent in shape):
                raise context.error(
                    function.source,
                    f"initial Storage VJP requires a positive static shape for TensorView '{parameter.name}'",
                )
        if function.result_type is None:
            raise context.error(function.source, "VJP transform requires a differentiable program result")

        def floating_leaves(value_type: ConcreteType) -> int:
            if value_type.kind == "scalar":
                return int(value_type.is_float)
            if value_type.kind in {"tensor", "tensor_view"}:
                element = value_type.arguments[0]
                return floating_leaves(element) if isinstance(element, ConcreteType) else 0
            if value_type.kind == "tuple":
                return sum(floating_leaves(item) for item in value_type.arguments if isinstance(item, ConcreteType))
            if value_type.kind == "struct":
                return sum(floating_leaves(field.type) for _, field in context.structs[value_type.name])
            return 0

        if not floating_leaves(function.result_type):
            raise context.error(function.source, "VJP program result has no differentiable floating Value leaves")
        parameters = {parameter.name: parameter.type for parameter in function.parameters}
        for path in transform.wrt:
            components = path.split(".")
            value_type = parameters.get(components[0])
            if value_type is None:
                raise context.error(function.source, f"VJP wrt path '{path}' names no entry parameter")
            for component in components[1:]:
                if value_type.kind == "struct":
                    fields = dict(context.structs[value_type.name])
                    field = fields.get(component)
                    if field is None:
                        raise context.error(function.source, f"VJP wrt path '{path}' names no Struct field")
                    value_type = field.type
                elif value_type.kind == "tuple" and component.isdigit():
                    index = int(component)
                    if index >= len(value_type.arguments) or not isinstance(value_type.arguments[index], ConcreteType):
                        raise context.error(function.source, f"VJP wrt path '{path}' has an invalid Tuple index")
                    value_type = value_type.arguments[index]
                else:
                    raise context.error(function.source, f"VJP wrt path '{path}' does not resolve to a Value")
            if value_type.kind in {"struct", "tuple"}:
                raise context.error(
                    function.source,
                    f"VJP wrt path '{path}' must resolve to one differentiable Scalar or Tensor leaf (or TensorView)",
                )
            if not floating_leaves(value_type):
                raise context.error(function.source, f"VJP wrt path '{path}' has no differentiable floating leaves")

    @staticmethod
    def _validate_entry_annotations(module: ast.Module, context: ModuleContext) -> None:
        for node in module.body:
            if not isinstance(node, ast.FunctionDef):
                continue
            decorators = {
                (dotted_name(item.func if isinstance(item, ast.Call) else item) or "").split(".")[-1]
                for item in node.decorator_list
            }
            if not decorators & ENTRY_DECORATORS:
                continue
            if node.returns is None:
                raise context.error(
                    node,
                    f"entry function '{node.name}' requires a result annotation",
                )
            for argument in node.args.args:
                if argument.annotation is None:
                    raise context.error(
                        argument,
                        f"entry argument '{argument.arg}' requires a type annotation",
                    )

    @staticmethod
    def _collect_struct_names(module: ast.Module, context: ModuleContext) -> None:
        for node in module.body:
            if isinstance(node, ast.ClassDef):
                decorators = {
                    (
                        dotted_name(decorator.func) if isinstance(decorator, ast.Call) else dotted_name(decorator) or ""
                    ).split(".")[-1]
                    for decorator in node.decorator_list
                }
                if "struct" not in decorators:
                    raise context.error(node, "DSL classes must use @struct")
                context.structs[node.name] = ()

    @staticmethod
    def _collect_structs(module: ast.Module, context: ModuleContext, parser: TypeParser) -> None:
        for node in module.body:
            if not isinstance(node, ast.ClassDef):
                continue
            fields: list[tuple[str, AnnotatedType]] = []
            for statement in node.body:
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
                    raise context.error(statement, "@struct bodies may contain only annotation-only fields")
                fields.append((statement.target.id, parser.parse(statement.annotation)))
            context.structs[node.name] = tuple(fields)

        def field_types(name: str) -> tuple[ConcreteType, ...]:
            return tuple(annotation.type for _, annotation in context.structs[name])

        for node in module.body:
            if not isinstance(node, ast.ClassDef):
                continue
            for statement, (field_name, annotation) in zip(
                (item for item in node.body if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)),
                context.structs[node.name],
                strict=True,
            ):
                if not is_abi_stable_value(annotation.type, field_types):
                    raise context.error(
                        statement.annotation,
                        f"Struct field '{node.name}.{field_name}' must be an ABI-stable Value",
                    )

    @staticmethod
    def _collect_signatures(module: ast.Module, context: ModuleContext, parser: TypeParser) -> None:
        for node in module.body:
            if not isinstance(node, ast.FunctionDef):
                continue
            shared = Compiler._is_shared_function(node)
            if shared:
                context.shared_functions.add(node.name)
            if (
                node.args.posonlyargs
                or node.args.kwonlyargs
                or node.args.vararg
                or node.args.kwarg
                or node.args.defaults
            ):
                raise context.error(node, "DSL functions support only required positional arguments")
            arguments: list[DslType] = []
            for argument in node.args.args:
                if argument.annotation is None:
                    raise context.error(argument, f"argument '{argument.arg}' requires a type annotation")
                annotation = parser.parse(argument.annotation)
                if annotation.type.kind == "tensor_storage":
                    raise context.error(
                        argument,
                        "TensorStorage is a host-runtime owner; device parameters must use TensorView",
                    )
                if shared and (
                    annotation.metadata
                    or annotation.type.kind in {"sampler", "tensor_storage", "tensor_view", "texture"}
                ):
                    raise context.error(argument, f"shared function '{node.name}' uses a device-only argument")
                arguments.append(annotation.type)
            result = None
            result_annotation = None
            if node.returns is not None and not (isinstance(node.returns, ast.Constant) and node.returns.value is None):
                result_annotation = parser.parse(node.returns)
                result = result_annotation.type
                if shared and (
                    result_annotation.metadata or result.kind in {"sampler", "tensor_storage", "tensor_view", "texture"}
                ):
                    raise context.error(node.returns, f"shared function '{node.name}' uses a device-only result")
            context.signatures[node.name] = FunctionSignature(tuple(arguments), result)
            context.result_annotations[node.name] = result_annotation

    @staticmethod
    def _is_shared_function(node: ast.FunctionDef) -> bool:
        if len(node.decorator_list) != 1:
            return False
        decorator = node.decorator_list[0]
        if not isinstance(decorator, ast.Call) or (dotted_name(decorator.func) or "").split(".")[-1] != "func":
            return False
        return any(
            keyword.arg == "shared" and isinstance(keyword.value, ast.Constant) and keyword.value.value is True
            for keyword in decorator.keywords
        )

    @staticmethod
    def _decorator(node: ast.FunctionDef, context: ModuleContext) -> tuple[str | None, tuple[int, int, int] | None]:
        stage = None
        workgroup_size = None
        function_kind = None
        for decorator in node.decorator_list:
            if isinstance(decorator, (ast.Name, ast.Attribute)):
                name = (dotted_name(decorator) or "").split(".")[-1]
                definition = STAGE_BY_DECORATOR.get(name)
                if definition is not None and definition.kind in GRAPHICS_STAGES:
                    function_kind = name
                    stage = definition.kind
                elif name == "kernel":
                    function_kind = "compute"
                    stage = "compute"
                    workgroup_size = (1, 1, 1)
                elif name == "func":
                    function_kind = "func"
                else:
                    raise context.error(decorator, f"unknown DSL decorator '{name}'")
            elif isinstance(decorator, ast.Call) and (dotted_name(decorator.func) or "").split(".")[-1] == "kernel":
                if function_kind is not None:
                    raise context.error(decorator, "DSL functions require exactly one function decorator")
                function_kind = "compute"
                stage = "compute"
                values = None
                for keyword in decorator.keywords:
                    if keyword.arg == "workgroup_size":
                        values = keyword.value
                    else:
                        raise context.error(keyword, f"unknown kernel option '{keyword.arg}'")
                if decorator.args or not isinstance(values, ast.Tuple) or len(values.elts) != 3:
                    raise context.error(decorator, "kernel requires workgroup_size=(x, y, z)")
                parsed: list[int] = []
                for value in values.elts:
                    if not isinstance(value, ast.Constant) or not isinstance(value.value, int) or value.value <= 0:
                        raise context.error(value, "workgroup dimensions must be positive integer literals")
                    parsed.append(value.value)
                workgroup_size = tuple(parsed)  # type: ignore[assignment]
            elif isinstance(decorator, ast.Call) and (dotted_name(decorator.func) or "").split(".")[-1] == "func":
                if function_kind is not None:
                    raise context.error(decorator, "DSL functions require exactly one function decorator")
                function_kind = "func"
                if decorator.args or len(decorator.keywords) != 1:
                    raise context.error(decorator, "@func accepts only shared=True")
                keyword = decorator.keywords[0]
                if (
                    keyword.arg != "shared"
                    or not isinstance(keyword.value, ast.Constant)
                    or keyword.value.value is not True
                ):
                    raise context.error(decorator, "@func accepts only shared=True")
            else:
                raise context.error(decorator, "unsupported decorator syntax")
            if len(node.decorator_list) > 1:
                raise context.error(decorator, "DSL functions require exactly one function decorator")
        if function_kind is None:
            expected = ", ".join(f"@{name}" for name in sorted(ENTRY_DECORATORS))
            raise context.error(node, f"DSL functions require @func or one of {expected}")
        return stage, workgroup_size


def compile_source(source: str, filename: str = "<string>") -> str:
    return Compiler().compile(source, filename)


def compile_file(
    input_path: str | Path,
    *,
    features: Iterable[str] = (),
    entry: str | None = None,
    program_transform: ProgramTransformSpec | None = None,
) -> str:
    return Compiler().compile_file(input_path, features=features, entry=entry, program_transform=program_transform)
